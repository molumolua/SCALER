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
import datetime
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

def to_py(obj):
    """递归把对象转成 JSON 可序列化的纯 Python 类型。"""
    # numpy 标量
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # tolist() 会返回纯 Python 标量/列表
    # torch 张量
    if _HAS_TORCH:
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
    # 基本类型
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # 容器类型：递归处理
    if isinstance(obj, dict):
        return {str(to_py(k)): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_py(x) for x in obj]
    # 字节串
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    # 兜底：转字符串
    return str(obj)
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

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._load_train_configs()

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

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )

        timing_raw = defaultdict(float)
        num_gen_batches = 0
        
        start_idx = 0
        save_json_list = []
        for epoch in range(1):
            problems = self.generate_one_batch_problems(num_problems=len(self.train_configs),
                                             batch_size=len(self.train_configs),
                                             train_configs=self.train_configs,
                                             start_idx = start_idx)
            print(problems[0])
            print("problem len:",len(problems))
            train_dataloader = self.get_train_dataloader_from_list(problems)
            start_idx = start_idx +len(self.train_configs)
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

                        for problem_name,prompt,distance,metric_val,solution_str,model_answer,prompt_ids,attention_mask in zip(
                                new_batch.non_tensor_batch['problem_name'],
                                new_batch.non_tensor_batch['raw_prompt'],
                                new_batch.non_tensor_batch['distance'],
                                new_batch.non_tensor_batch[metric_name],
                                new_batch.non_tensor_batch['solution'],
                                new_batch.non_tensor_batch['answer'],
                                new_batch.batch["prompts"],
                                new_batch.batch["attention_mask"]
                                ):
                            prompt_length = prompt_ids.shape[-1]
                            valid_response_length = attention_mask[prompt_length:].sum()
                            save_json_list.append({
                                "prompt":prompt,
                                "problem_name":problem_name,
                                "distance":distance,
                                "reward":metric_val,
                                "response":model_answer,
                                "ground_truth":solution_str,
                                "response_length":valid_response_length
                            })
              
                            
        base_path = "/inspire/hdd/global_user/xucaijun-253108120121/verl/recipe/environment/save_json_list.json"

        # 生成时间戳，比如 20251106_142530
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 拆分出主文件名和后缀，再拼回去
        root, ext = os.path.splitext(base_path)
        file_path = f"{root}_{timestamp}{ext}"
                
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([to_py(x) for x in save_json_list], f, ensure_ascii=False, indent=2)

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
    
    def generate_one_batch_problems(self, num_problems,batch_size, train_configs,start_idx):
        
        # 获取train_configs的problem_name和对应的train_config
        problems_to_train = random.sample(list(train_configs.items()), num_problems)
        problems_to_train = dict(problems_to_train)

        problem_names = list(problems_to_train.keys())
        
        # 计算每个train_config分配的任务数
        batch_distribution = self._distribute_batch_size(batch_size, num_problems)

        problems = []
        
        # 对problem_names进行随机打乱
        random.shuffle(problem_names)
        
        for idx, problem_name in enumerate(problem_names):
            # 根据batch_distribution确定当前train_config需要生成多少问题
            num_problems_for_current = batch_distribution[idx]  
            # 获取对应的train_config
            train_config = problems_to_train[problem_name]
            # 生成问题
            problems_for_current = self.generate_problems_for_setting(problem_name, train_config, num=num_problems_for_current, start_idx=start_idx)
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

            
            problem_description,solution=environment.generate_problem_from_environment.get_problems(train_config,[problem_distance],sandboxfusion_url=self.config.data.sandboxfusion_url)[0]
            
            problems.append({
                "prompt":get_train_prompt(problem_description),
                "reward_model":{
                    "ground_truth":str(solution)
                },
                "extra_info":{
                    "index":idx+start_idx,
                    "problem_name":problem_name,
                },
                "data_source":"forge",
                "problem_name":problem_name,
                "distance":problem_distance,
                "solution":solution
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
                                                                                   len(self.train_configs)),
                                                   num_workers=8,
                                                   drop_last=False,
                                                   sampler=sampler,
                                                   collate_fn=collate_fn)

        return train_dataloader
        
        
    def update_train_configs(self,prompt_name2distance_metric_avg_len_dict):
        problem_name2metric_list={}
        for problem_name,distance_metric_avg_len_dict in prompt_name2distance_metric_avg_len_dict.items():
            difficulty_control = self.train_configs[problem_name]['params']['difficulty']
            problem_name2metric_list[problem_name]=difficulty_control.update(distance_metric_avg_len_dict)
            
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
            