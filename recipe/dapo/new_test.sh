#!/usr/bin/env bash
set -euxo pipefail
export CUDA_VISIBLE_DEVICES=4,5
export RAY_TMPDIR="/data2/xucaijun/raytmp"

dataset_name="think-mui-MATH-3000"
model_name="Qwen2.5-7B"
offload=True
num_gpus=2

test_and_save_freq=20
lr_warmup_steps=10
lr=1e-6
entropy_coeff=0
epoch=1000
# score_mode="None"
project_name='Balance-Debug'
enable_dataset_update=True
enable_dynamic_batch_size=True
enable_dynamic_sample=False
n_resp_per_prompt=8  #采样次数
sum_max=7
sum_min=1
filter_max=7
filter_min=0
replay_size=0
sort_prompt_bsz=32
train_prompt_bsz=32
gen_prompt_bsz=32
train_prompt_mini_bsz=32

exp_name=${exp_name:-"Dynamic-d-${enable_dataset_update}-ds-${enable_dynamic_batch_size}-s-${enable_dynamic_sample}-r-${replay_size}-b-${gen_prompt_bsz}-${sort_prompt_bsz}-${train_prompt_bsz}-${sum_min}${sum_max}${filter_min}${filter_max}-dataset-${dataset_name}-model-${model_name}"}
# exp_name=${exp_name:-"None-test-data-True-select-False-batch-size-192-64-64-1-7-0-7-replay-0-entropy_coeff-0-dataset-think-DeepMath-103K-model-Qwen2.5-7B"}
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 ))
max_response_length=$((1024 * 7))

enable_overlong_buffer=False
overlong_buffer_len=0
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"
# loss_agg_mode="seq-mean-token-sum"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=100

# # Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/Data-Select-verl"}
MODEL_PATH=${MODEL_PATH:-"/data2/models/Qwen/${model_name}"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"/data2/xucaijun/DAPO_verl/verl/data/${dataset_name}.parquet"}
TEST_FILE=${TEST_FILE:-["/data2/xucaijun/DAPO_verl/verl/data/think_MATH-500_MATH-500-processed.parquet","/data2/xucaijun/DAPO_verl/verl/data/think_aime24_aime24_test.parquet","/data2/xucaijun/DAPO_verl/verl/data/think_amc23_amc23_test.parquet"]}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

val_temperature=0.6
val_top_p=0.95

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

expect_bsz=$((train_prompt_bsz * n_resp_per_prompt))

PYTHONUNBUFFERED=1 python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    +data.sort_prompt_bsz=${sort_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    +actor_rollout_ref.actor.expect_train_size=${expect_bsz} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${num_gpus} \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','swanlab'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${num_gpus}" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=${test_and_save_freq} \
    trainer.save_freq=${test_and_save_freq} \
    trainer.total_epochs=${epoch} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.sum_max=${sum_max} \
    +trainer.sum_min=${sum_min} \
    +trainer.filter_max=${filter_max} \
    +trainer.filter_min=${filter_min} \
    +trainer.max_actor_ckpt_to_keep=1 \
    +trainer.enable_dataset_update=${enable_dataset_update} \
    +trainer.replay_buffer_size=${replay_size} \
    +trainer.enable_dynamic_batch_size=${enable_dynamic_batch_size} \
    +trainer.enable_dynamic_sample=${enable_dynamic_sample} \

