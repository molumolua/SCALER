#!/usr/bin/env bash
set -euxo pipefail
export WANDB_MODE=offline
model_name="Qwen3-4B-Base"

num_gpus=8
tensor_model_parallel_size=4


enable_weighted_sample=False
val_before_train=False
with_instruction=True






epoch=10000
project_name='ACL2026'

lr_warmup_steps=20
lr=1e-6
test_and_save_freq=20
n_resp_per_prompt=8
train_prompt_bsz=64
train_prompt_mini_bsz=64
num_environment_per_step=64
sandboxfusion_url="http://localhost:8080/run_code"

entropy_coeff=0

exp_name=${exp_name:-"small-W-${enable_weighted_sample}-Instruction-ne-${num_environment_per_step}-model-${model_name}-lr-${lr}-bsz-${train_prompt_bsz}-n_resp-${n_resp_per_prompt}-mini-${train_prompt_mini_bsz}"}
# exp_name=${exp_name:-"None-test-data-True-select-False-batch-size-192-64-64-1-7-0-7-replay-0-entropy_coeff-0-dataset-think-DeepMath-103K-model-Qwen2.5-7B"}
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 3))
max_response_length=$((1024 * 8))

enable_overlong_buffer=False
overlong_buffer_len=0
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"
# loss_agg_mode="seq-mean-token-sum"
enable_filter_groups=False
filter_groups_metric=acc

# # Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/inspire/hdd/global_user/xucaijun-253108120121/verl"}
MODEL_PATH=${MODEL_PATH:-"/inspire/hdd/global_user/xucaijun-253108120121/Model/${model_name}"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"/inspire/hdd/global_user/xucaijun-253108120121/Code/FORGE_THIRD/FORGE_train_data_small.json"}
TEST_FILE=${TEST_FILE:-["/inspire/hdd/global_user/xucaijun-253108120121/Dataset-P/bbeh_data.parquet","/inspire/hdd/global_user/xucaijun-253108120121/Dataset-P/think_MATH-500_MATH-500-processed.parquet","/inspire/hdd/global_user/xucaijun-253108120121/Dataset-P/think_amc23_amc23_test.parquet","/inspire/hdd/global_user/xucaijun-253108120121/Dataset-P/think_aime24_aime24_test.parquet","/inspire/hdd/global_user/xucaijun-253108120121/Dataset-P/MMLU-Pro-Valid.parquet","/inspire/hdd/global_user/xucaijun-253108120121/Dataset-P/GPQA-Diamond-Test.parquet"]}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

val_temperature=0.6
val_top_p=0.95

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
max_num_gen_batches=100





# h100 / h200 config
# offload=False
# ref_offload=False
# gpu_memory_utilization=0.85
# actor_ppo_max_token_len=$((2*(max_prompt_length + max_response_length)))
# infer_ppo_max_token_len=$((2*(max_prompt_length + max_response_length)))

offload=True
ref_offload=True
gpu_memory_utilization=0.8
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))

PYTHONUNBUFFERED=1 python3 -m recipe.environment.main_dapo \
    +data.setting_filename="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=512 \
    +data.num_environment_per_step=${num_environment_per_step} \
    +data.with_instruction=${with_instruction} \
    +data.sandboxfusion_url="${sandboxfusion_url}" \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    +algorithm.update_train_configs=True \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${num_gpus}" \
    trainer.nnodes=1 \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_and_save_freq} \
    trainer.save_freq=${test_and_save_freq} \
    trainer.total_epochs=${epoch} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.max_actor_ckpt_to_keep=1  \
    trainer.total_training_steps=${epoch} \
    +trainer.enable_weighted_sample=${enable_weighted_sample}
    

