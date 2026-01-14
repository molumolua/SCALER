# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm


from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


import environment.generate_problem_from_environment
from omegaconf import OmegaConf, open_dict
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer,apply_kl_penalty, compute_advantage, compute_response_mask
from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
from verl.utils.dataset.inmemory_dataset import InMemoryRLHFDataset
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
import json
import os
import random
from .difficulty_control import DifficultyControl
import math
from collections import deque
def get_train_prompt(question):
    prompt_list=[]
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."
    prompt_list.append(
        {"content":system_prompt, "role": "system"}
    )
    prompt_list.append(
        {"content": question, "role": "user"}
    )
    return prompt_list

def _rand_round_to_int(x):
    lo, hi = math.floor(x), math.ceil(x)
    p_hi = x - lo
    value = hi if random.random() < p_hi else lo
    return value

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        
        if self.config.trainer.get("enable_windows_sample",False):
            self.windows_environment = list(self.train_configs.keys())[:self.config.data.num_environment_per_step]
            # self.windows_environment = list(random.sample(list(self.train_configs.keys()),self.config.data.num_environment_per_step))
            windows_cover = {k:False for k in self.train_configs.keys()}
        else:
            self.windows_environment = []
            self.env_queue=deque([])
            
        # load checkpoint before doing anything
        self._load_checkpoint()
        self._load_train_configs()
        self._load_environment_windows()
        
        if self.global_steps == 0:
            self.init_env_queue()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        
        start_idx = 0
        show_environment_name =random.sample(list(self.train_configs.keys()),8)
        
            
        for epoch in range(self.config.trainer.total_epochs):
            problems = self.generate_one_batch_problems(num_problems=self.config.data.num_environment_per_step,
                                             batch_size=self.config.data.train_batch_size,
                                             train_configs=self.train_configs,
                                             start_idx = start_idx)
            
            if self.config.trainer.get("enable_windows_sample",False):
                # show_environment_name = self.windows_environment
                for problem in problems:
                    windows_cover[problem['problem_name']]=True
            print(problems[0])
            print("problem len:",len(problems))
            train_dataloader = self.get_train_dataloader_from_list(problems)
            start_idx = start_idx +self.config.data.train_batch_size
            for batch_dict in train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
                    
                    if self.config.algorithm.update_train_configs:
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                        
                        logger_data = {}
                        # Collect the sequence reward for each trajectory
                        prompt_name2metric_vals = defaultdict(lambda: defaultdict(list))
                        prompt_idx2metric_vals = defaultdict(list)
                        for problem_name, idx,distance, metric_val in zip(
                                new_batch.non_tensor_batch['problem_name'],
                                new_batch.non_tensor_batch['index'],
                                new_batch.non_tensor_batch['distance'],
                                new_batch.non_tensor_batch[metric_name]):
                            prompt_name2metric_vals[problem_name][distance].append(metric_val)
                            prompt_idx2metric_vals[idx].append(metric_val)
                            
                        efficiency_sample_num = 0
                        efficiency_metric_sum = 0
                        for idx,metric_vals in prompt_idx2metric_vals.items():
                            if np.std(metric_vals) > 0:
                                efficiency_sample_num += 1
                                efficiency_metric_sum += np.average(metric_vals)
                                
                        logger_data[f"dapo/efficiency_sample_num"]=efficiency_sample_num
                        logger_data[f"dapo/efficiency_metric_avg"]=efficiency_metric_sum/efficiency_sample_num if efficiency_sample_num >0 else 0
                        
                        # 计算平均值：结果是 dict: problem_name -> distance -> avg_metric
                        prompt_name2metric_avg_len = {}
                        for problem_name, dist_dict in prompt_name2metric_vals.items():
                            prompt_name2metric_avg_len[problem_name] = {}
                            for distance, vals in dist_dict.items():
                                prompt_name2metric_avg_len[problem_name][distance] = (np.average(vals),len(vals))
                        

                        problem_name2metric_list=self.update_train_configs(prompt_name2metric_avg_len,self.global_steps)
                        
                        
                        for problem_name, metric_list in problem_name2metric_list.items():
                            if problem_name in show_environment_name:
                                logger_data[f"environment/{problem_name}/ema"]=metric_list[0]
                                logger_data[f"environment/{problem_name}/avg_correct"]=metric_list[1]
                                logger_data[f"environment/{problem_name}/k"]=metric_list[2]
                        total_distance_norm = 0
                        total_distance = 0   
                        total_weight = 0
                        for problem_name,train_config in self.train_configs.items():
                            distance =train_config['params']['difficulty'].state['d']
                            max_distance = train_config['params']['difficulty'].dmax
                            total_distance += distance
                            total_distance_norm += distance/max_distance if max_distance >0 else 0
                            
                            weight = train_config['params']['difficulty'].get_weight(self.global_steps)
                            total_weight += weight
                            
                            
                            if problem_name in show_environment_name:
                                logger_data[f"environment/{problem_name}/distance"]=distance
                                logger_data[f"environment/{problem_name}/weight"]=weight
                                
                        logger_data[f"environment_all/avg_weight"]=total_weight/ len(self.train_configs)
                        logger_data[f"environment_all/avg_distance"]=total_distance / len(self.train_configs)
                        logger_data[f"environment_all/avg_norm_distance"]=total_distance_norm / len(self.train_configs)
                        if self.config.trainer.get("enable_windows_sample",False):
                            logger_data[f"environment_all/windows_cover"]=sum(windows_cover.values())
                        logger.log(data=logger_data, step=self.global_steps,backend="wandb")
                    batch = new_batch
                        
                

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()
                        self._save_train_configs()
                        self._save_environment_windows()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)

    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type
        if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
            dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                                f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset")
        else:
            dataset_cls = RLHFDataset
            
        with open(self.config.data.setting_filename, 'r') as file:
            self.train_configs = json.load(file,object_hook=DifficultyControl.json_object_hook)
            

        self.val_dataset = dataset_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            collate_fn=collate_fn,
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False)

        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."


        total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
    
    
    def init_env_queue(self):
        # 1) 训练配置的“全体环境”顺序：默认沿用 dict 插入顺序（更可控）
        all_names = list(self.train_configs.keys())

        # 2) windows_environment 可能有重复/脏数据：去重并过滤不存在的
        seen = set()
        win = []
        for n in self.windows_environment:
            if n in self.train_configs and n not in seen:
                seen.add(n)
                win.append(n)

        win_set = set(win)

        # 3) 队列构造：非窗口里的在前，窗口里的放后
        others = [n for n in all_names if n not in win_set]
        self.env_queue = deque(others + win)
        
    def _distribute_batch_size(self, batch_size, num_groups):
        # 根据batch_size和num_groups，生成一个分配数组，并进行随机调整
        base_size = batch_size // num_groups
        remainder = batch_size % num_groups
        
        # 初始化分配数组，每个组先分配base_size
        batch_distribution = [base_size] * num_groups
        
        # 将剩余部分分配到前remainder个组
        for i in range(remainder):
            batch_distribution[i] += 1
        
        # 返回分配结果
        return batch_distribution
   

    def _pick_from_queue(self) -> str | None:
        """从队首开始轮转，找到第一个不在 exclude 里的；找到后该元素已经被旋到队尾。"""
        n = len(self.env_queue)
        if n == 0:
            return None

        cand = self.env_queue[0]
        self.env_queue.rotate(-1)  # cand 被放到队尾（相当于 popleft + append）
        return cand

    def update_environment_windows(self, train_configs):
        for idx, name in enumerate(self.windows_environment):
            cfg = train_configs[name]
            dc = cfg["params"]["difficulty"]
            windows_state = dc.get_windows_state()

            # 检查是否满足更新条件
            if (windows_state['zero_correct_count'] >= self.config.trainer.get("windows_continous_zero_correct_limit") or
                windows_state['max_distance_count'] >= self.config.trainer.get("windows_continous_max_distance_limit") or
                windows_state['distance_slope'] <= 0):
                dc.empty_histroy()
                # 从其他train_configs中随机选一个
                other_names = list(train_configs.keys())
                other_names.remove(name)  # 排除当前正在处理的配置
                new_name = random.choice(other_names)
                # new_name = self._pick_from_queue()
                
                # 更新当前配置在windows_environment中的位置
                self.windows_environment[idx] = new_name
                
                print(f"Updated {name} at index {idx} with {new_name} due to condition trigger.")

            
            
    def generate_one_batch_problems(self, num_problems, batch_size, train_configs, start_idx,fixed_sample_names=None):
        """
        num_problems: 本次希望选多少个不同的 problem_name
        batch_size:   总的样本数（会按 _distribute_batch_size 均分到这些 problem 上）
        train_configs: dict[problem_name -> train_config]
        """
        # fixed_sample_names= [
        #     "p02832 AtCoder Beginner Contest 148 - Brick Break",
        #     "p03021 AtCoder Grand Contest 034 - Complete Compress",
        #     "p03512 Code Festival Team Relay (Parallel) - Akashic Records",
        #     "p02464 Set Intersection",
        #     "1219_H. Function Composition",
        #     "p02435 Stack",
        #     "p01410 Dangerous Tower",
        #     "1299_C. Water Balance",
        #     "285_C. Building Permutation",
        #     "315_E. Sereja and Subsequences",
        #     "1362_E. Johnny and Grandmaster",
        #     "p03530 CODE FESTIVAL 2017 Final - Poor Penguin",
        #     "1025_C. Plasticine zebra",
        #     "1003_F. Abbreviation",
        #     "1113_B. Sasha and Magnetic Machines",
        #     "1340_C. Nastya and Unexpected Guest",
        #     "1399_B. Gifts Fixing",
        #     "1490_G. Old Floppy Drive ",
        #     "525_A. Vitaliy and Pie",
        #     "p03718 AtCoder Regular Contest 074 - Lotus Leaves",
        #     "278_A. Circle Line",
        #     "1223_D. Sequence Sorting",
        #     "830_B. Cards Sorting",
        #     "p01735 Optimal alpha beta pruning",
        #     "207_B3. Military Trainings",
        #     "608_D. Zuma",
        #     "p01075 One-Time Path",
        #     "1067_E. Random Forest Rank",
        #     "933_A. A Twisty Movement",
        #     "686_A. Free Ice Cream",
        #     "p03190 AtCoder Grand Contest 030 - Less than 3",
        #     "158_D. Ice Sculptures",
        # ]

        if fixed_sample_names != None:
            fixed_sample_items = [(name, train_configs[name]) for name in fixed_sample_names]
            fixed_problems_to_train = dict(fixed_sample_items)
            
            num_problems -= len(fixed_sample_items)
        
        all_items = list(train_configs.items())
        if self.config.trainer.get('enable_windows_sample',False):
            self.update_environment_windows(train_configs)
            sampled_items = [(name, train_configs[name]) for name in self.windows_environment]
            problems_to_train = dict(sampled_items)
            
        # ==== 1. 选择这一次要训练的 problem 集合 ====
        elif self.config.trainer.get('enable_weighted_sample',False):
            # 按 DifficultyControl 的权重来做“无放回加权采样”
            names = [name for name, _ in all_items]
            cfgs = [cfg for _, cfg in all_items]

            weights = []
            for cfg in cfgs:
                dc = cfg["params"]["difficulty"]  # 这里假设是 DifficultyControl 实例
                w = 1.0
                if hasattr(dc, "get_weight"):
                    w = dc.get_weight(self.global_steps)
                # 防止出现负数或 NaN
                if not isinstance(w, (int, float)) or math.isnan(w):
                    w = 0.0
                weights.append(max(0.0, float(w)))

            # 如果全部权重都是 0，就退化成均匀采样
            if sum(weights) <= 0:
                sampled_items = random.sample(all_items, num_problems)
            else:
                # 简单的“按权重无放回采样”
                indices = list(range(len(names)))
                chosen_names = []
                for _ in range(min(num_problems, len(indices))):
                    total_w = sum(weights[i] for i in indices)
                    if total_w <= 0:
                        # 剩下全是 0 权重时，均匀选一个
                        j = random.choice(indices)
                    else:
                        r = random.random() * total_w
                        acc = 0.0
                        j = indices[-1]
                        for idx in indices:
                            acc += weights[idx]
                            if acc >= r:
                                j = idx
                                break
                    chosen_names.append(names[j])
                    indices.remove(j)

                sampled_items = [(name, train_configs[name]) for name in chosen_names]

            problems_to_train = dict(sampled_items)

        else:
            # 原来的行为：从所有题目里等概率随机选 num_problems 个
            sampled_items = random.sample(all_items, num_problems)
            problems_to_train = dict(sampled_items)
            
        if fixed_sample_names:
            problems_to_train.update(fixed_problems_to_train)  
        
        # 这一步之后，problems_to_train 的大小就是实际要训练的 problem 个数
        problem_names = list(problems_to_train.keys())
        num_problems = len(problem_names)

        # ==== 2. 按 batch_size 均匀分配给这些 problem ====
        batch_distribution = self._distribute_batch_size(batch_size, num_problems)

        problems = []
        
        # 对problem_names进行随机打乱（避免固定顺序带来的偏差）
        random.shuffle(problem_names)
        
        for idx, problem_name in enumerate(problem_names):
            # 根据batch_distribution确定当前train_config需要生成多少问题
            num_problems_for_current = batch_distribution[idx]  
            # 获取对应的train_config
            train_config = problems_to_train[problem_name]
            # 生成问题
            problems_for_current = self.generate_problems_for_setting(
                problem_name,
                train_config,
                num=num_problems_for_current,
                start_idx=start_idx
            )
            problems.extend(problems_for_current)
            # 更新start_idx以确保后续问题有不同的索引
            start_idx += num_problems_for_current


        return problems  
        
    def generate_problems_for_setting(self,problem_name,train_config,num=1,start_idx=0):
        problems = []
        train_config_params = train_config['params']
        difficulty_control = train_config_params['difficulty'] 
        
        dist_list = difficulty_control.propose_distances(num)
        for idx,distance in enumerate(dist_list,1):

            if isinstance(distance,int):
                problem_distance=distance
            else:
                problem_distance = _rand_round_to_int(distance)

            
            problem_description,solution=environment.generate_problem_from_environment.get_problems(train_config,[problem_distance],sandboxfusion_url=self.config.data.sandboxfusion_url,with_instruction=self.config.data.with_instruction)[0]
            
            problems.append({
                "prompt":get_train_prompt(problem_description),
                "reward_model":{
                    "ground_truth":str(solution)
                },
                "extra_info":{
                    "index":idx+start_idx,
                    "problem_name":problem_name,
                    "output_type":train_config.get('output_type','number')
                },
                "data_source":"scaler",
                "problem_name":problem_name,
                "distance":problem_distance,
                "index":idx+start_idx,
            })
            
        assert len(problems) == num
        return problems


    def get_train_dataloader_from_list(self,data_list):
        train_dataset = InMemoryRLHFDataset(
            data_list=data_list,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data
        )

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            seed=self.config.data.get('seed')
            if not seed:
                seed=1 
            train_dataloader_generator.manual_seed(seed)
            sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=train_dataset)

        train_dataloader = StatefulDataLoader(dataset=train_dataset,
                                                   batch_size=self.config.data.get('gen_batch_size',
                                                                                   self.config.data.train_batch_size),
                                                   num_workers=8,
                                                   drop_last=False,
                                                   sampler=sampler,
                                                   collate_fn=collate_fn)

        return train_dataloader
        
        
    def update_train_configs(self,prompt_name2distance_metric_avg_len_dict,global_steps):
        problem_name2metric_list={}
        for problem_name,distance_metric_avg_len_dict in prompt_name2distance_metric_avg_len_dict.items():
            difficulty_control = self.train_configs[problem_name]['params']['difficulty']
            problem_name2metric_list[problem_name]=difficulty_control.update(distance_metric_avg_len_dict,global_steps)
            
        return problem_name2metric_list
    
    def _save_train_configs(self):
        """
        Saves the current training configurations to a JSON file.
        """
        # 1) 规范化并绝对化 checkpoint 根目录
        checkpoint_folder = self.config.trainer.default_local_dir
        checkpoint_folder = os.path.expanduser(os.path.expandvars(checkpoint_folder))
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        checkpoint_folder = os.path.normpath(checkpoint_folder)

        # 2) 选择 global_step 目录；若找不到就用 global_step_0（注意无前导斜杠）
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if not global_step_folder or not os.path.isdir(global_step_folder):
            global_step_folder = os.path.join(checkpoint_folder, "global_step_0")

        # 3) 确保目录存在
        os.makedirs(global_step_folder, exist_ok=True)

        config_filename = os.path.join(global_step_folder, "train_configs.json")
        with open(config_filename, 'w') as file:
            json.dump(self.train_configs, file,default=DifficultyControl.json_default, indent=4)
        print(f"Training configurations saved to {config_filename}")

    def _save_environment_windows(self):
        """
        Saves the current training configurations to a JSON file.
        """
        # 1) 规范化并绝对化 checkpoint 根目录
        checkpoint_folder = self.config.trainer.default_local_dir
        checkpoint_folder = os.path.expanduser(os.path.expandvars(checkpoint_folder))
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        checkpoint_folder = os.path.normpath(checkpoint_folder)

        # 2) 选择 global_step 目录；若找不到就用 global_step_0（注意无前导斜杠）
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if not global_step_folder or not os.path.isdir(global_step_folder):
            global_step_folder = os.path.join(checkpoint_folder, "global_step_0")

        # 3) 确保目录存在
        os.makedirs(global_step_folder, exist_ok=True)

        save_json ={
            "windows_environment":self.windows_environment,
            "env_queue":list(self.env_queue)
        }
        config_filename = os.path.join(global_step_folder, "environment_windows.json")
        with open(config_filename, 'w') as file:
            json.dump(save_json, file, indent=4)
        print(f"Training environment windows saved to {config_filename}")
        
        
        
        
    def _load_train_configs(self):
        checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)

        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if not global_step_folder or not os.path.isdir(global_step_folder):
            print(f"No valid checkpoint folder under: {checkpoint_folder}, start from initial.")
            return 

        config_filename = os.path.join(global_step_folder, "train_configs.json")
        if not os.path.isfile(config_filename):
            print(f"Config file not found: {config_filename}, start from initial.")
            return 

        with open(config_filename, "r", encoding="utf-8") as f:
            cfg = json.load(f,object_hook=DifficultyControl.json_object_hook)

        self.train_configs = cfg
            
        print(f"Training configurations loaded from {config_filename}.")
    
    def _load_environment_windows(self):
        checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)

        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if not global_step_folder or not os.path.isdir(global_step_folder):
            print(f"No valid checkpoint folder under: {checkpoint_folder}, start from initial.")
            return 

        config_filename = os.path.join(global_step_folder, "environment_windows.json")
        if not os.path.isfile(config_filename):
            print(f"Config file not found: {config_filename}, start from initial.")
            return 

        with open(config_filename, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        
        if isinstance(cfg,dict):
            self.windows_environment = cfg['windows_environment']
            self.env_queue = deque(cfg['env_queue'])
        else:
            self.windows_environment = cfg
            self.env_queue = deque([])
            
        print(f"Training environment windows loaded from {config_filename}.")
        print(f"cfg type is {type(cfg)}.")