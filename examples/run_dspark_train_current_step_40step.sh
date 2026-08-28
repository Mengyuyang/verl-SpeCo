#!/usr/bin/env bash
set -euo pipefail

project_name='verl_grpo_example_dsparktemp1_train'
exp_name='qwen3_8b_dspark_train_current_step_temp1_vllm_npu'

# Reproduce the effective train (17).log recipe after cross-step collection was
# removed: collect and train on RL steps 5, 10, 15, ... using current-step data.
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

export PYTHONPATH=/efs_rl/z00876269/Speculative_Decoding_new/vllm-ascend:${PYTHONPATH:-}
export PYTHONPATH=/efs_rl/z00876269/Speculative_Decoding_new/vllm:${PYTHONPATH}
export PYTHONPATH=/efs_rl/z00876269/Speculative_Decoding_new/verl:${PYTHONPATH}
export PYTHONPATH=/efs_rl/z00876269/Speculative_Decoding_new/verl-SpeCo:${PYTHONPATH}

case "${LD_PRELOAD:-}" in
    *libjemalloc*) ;;
    *)
        if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
            export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        elif [ -f /usr/lib64/libjemalloc.so.2 ]; then
            export LD_PRELOAD="/usr/lib64/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        fi
        ;;
esac

export MALLOC_CONF="${MALLOC_CONF:-narenas:8,thp:never,metadata_thp:disabled,dirty_decay_ms:0,muzzy_decay_ms:0}"
export SPECO_JEMALLOC_RECLAIM_MODE="${SPECO_JEMALLOC_RECLAIM_MODE:-purge}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

gen_tp=2
train_sp=4
ppo_gpus_per_node="${SPECO_ACCELERATOR_COUNT:-16}"
ray_num_cpus="${SPECO_RAY_NUM_CPUS:-64}"
ray_worker_soft_limit="${SPECO_RAY_WORKER_SOFT_LIMIT:-16}"
spec_verify_tokens="${SPECO_DSPARK_VERIFY_TOKENS:-7}"
drafter_lr="${SPECO_DRAFTER_LR:-5e-6}"
validation_batch_size="${SPECO_VALIDATION_BATCH_SIZE:-8}"
# Conservative defaults bound MRV2 KV/graph admission and the DSpark
# auxiliary-hidden concat workspace. Override them only after an NPU memory A/B.
rollout_max_num_seqs="${SPECO_ROLLOUT_MAX_NUM_SEQS:-128}"
rollout_max_num_batched_tokens="${SPECO_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
rollout_gpu_memory_utilization="${SPECO_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"

RUN_ROOT="${RUN_ROOT:-/efs_rl/z00876269/Speculative_Decoding_new}"
MODEL_PATH="${MODEL_PATH:-/efs_rl/z00886395/models/Qwen3-8B}"
TRAIN_FILE="${TRAIN_FILE:-/efs_rl/z00886395/datasets/dapo-math-17k.parquet}"
TEST_FILE="${TEST_FILE:-/efs_rl/z00886395/datasets/aime-2024.parquet}"
DRAFTER_PATH="${DRAFTER_PATH:-/efs_rl/z00886395/models/dspark_qwen3_8b_block7}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CKPTS_DIR="${CKPTS_DIR:-${RUN_ROOT}/checkpoints_baseline_matched_temp1/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs/${exp_name}}"
RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"
LOG_FILE="${RUN_LOG_DIR}/train.log"

mkdir -p "${RUN_LOG_DIR}" "${CKPTS_DIR}"

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
if [ -f "${SCRIPT_SOURCE}" ]; then
    cp -f "${SCRIPT_SOURCE}" "${RUN_LOG_DIR}/launch.sh"
fi

{
    printf '%q ' "$0" "$@"
    printf '\n'
} > "${RUN_LOG_DIR}/launch_cmd.txt"

cat > "${RUN_LOG_DIR}/run_info.txt" <<INFO
RUN_ID=${RUN_ID}
RUN_LOG_DIR=${RUN_LOG_DIR}
LOG_FILE=${LOG_FILE}
CKPTS_DIR=${CKPTS_DIR}
MODEL_PATH=${MODEL_PATH}
DRAFTER_PATH=${DRAFTER_PATH}
TRAIN_FILE=${TRAIN_FILE}
TEST_FILE=${TEST_FILE}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}
ppo_gpus_per_node=${ppo_gpus_per_node}
gen_tp=${gen_tp}
train_sp=${train_sp}
spec_verify_tokens=${spec_verify_tokens}
drafter_lr=${drafter_lr}
validation_batch_size=${validation_batch_size}
rollout_max_num_seqs=${rollout_max_num_seqs}
rollout_max_num_batched_tokens=${rollout_max_num_batched_tokens}
rollout_gpu_memory_utilization=${rollout_gpu_memory_utilization}
VLLM_USE_V1=${VLLM_USE_V1}
VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER}
DSPARK_ENABLED=True
DSPARK_TRAINING_ENABLED=True
DRAFTER_DATA_SCOPE=current_training_step
INFO

exec > >(tee -a "${LOG_FILE}") 2>&1
set -x

echo "Logging to: ${LOG_FILE}"

on_exit() {
    rc=$?
    set +x
    echo "============================================================"
    echo "[DSPARK_TRAIN] exit_code=${rc}"
    echo "[DSPARK_TRAIN] run_dir=${RUN_LOG_DIR}"
    echo "[DSPARK_TRAIN] log_file=${LOG_FILE}"
    echo "[DSPARK_TRAIN] checkpoint_dir=${CKPTS_DIR}"
    echo "============================================================"
}
trap on_exit EXIT

[ -e "${MODEL_PATH}" ] || { echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}"; exit 2; }
[ -e "${DRAFTER_PATH}" ] || { echo "ERROR: DRAFTER_PATH not found: ${DRAFTER_PATH}"; exit 2; }
[ -f "${TRAIN_FILE}" ] || { echo "ERROR: TRAIN_FILE not found: ${TRAIN_FILE}"; exit 2; }
[ -f "${TEST_FILE}" ] || { echo "ERROR: TEST_FILE not found: ${TEST_FILE}"; exit 2; }

VISIBLE_NPU_COUNT=$(awk -F',' '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")
echo "[DSPARK_TRAIN] visible_npus=${VISIBLE_NPU_COUNT}, trainer.n_gpus_per_node=${ppo_gpus_per_node}"

if [ "${VISIBLE_NPU_COUNT}" -ne "${ppo_gpus_per_node}" ]; then
    echo "WARNING: visible NPU count (${VISIBLE_NPU_COUNT}) != trainer.n_gpus_per_node (${ppo_gpus_per_node})"
fi

PYTHONUNBUFFERED=1 python3 -m verl_speco.main \
    algorithm.adv_estimator=grpo \
    transfer_queue.enable=False \
    ray_kwargs.ray_init.num_cpus=${ray_num_cpus} \
    +ray_kwargs.ray_init._system_config.prestart_worker_first_driver=false \
    +ray_kwargs.ray_init._system_config.num_workers_soft_limit=${ray_worker_soft_limit} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=256 \
    data.truncation='error' \
    actor_rollout_ref.rollout.temperature=1 \
    +actor_rollout_ref.rollout.repetition_penalty=1 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=512 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path=${DRAFTER_PATH} \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=7 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=32 \
    actor_rollout_ref.rollout.drafter.training.dspark_max_window=512 \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=restricted_ce \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=7 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=5 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_hidden_layers=5 \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_rank=256 \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_head_type=vanilla \
    actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=False \
    actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=0.1 \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0.9 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha=0.0 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=0.0 \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=${spec_verify_tokens} \
    actor_rollout_ref.rollout.drafter.training.use_data_buffer=True \
    actor_rollout_ref.rollout.drafter.training.sample_last_n_steps=0 \
    actor_rollout_ref.rollout.drafter.training.step=8 \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.drafter.training.lr=${drafter_lr} \
    actor_rollout_ref.rollout.drafter.training.lr_scheduler_type=constant \
    actor_rollout_ref.rollout.drafter.training.max_collect_samples_per_step_per_replica=16 \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_mode=random \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=512 \
    actor_rollout_ref.rollout.drafter.training.hidden_state_random_seed_by_step=True \
    actor_rollout_ref.rollout.drafter.training.max_collect_tokens_per_step_per_replica=16384 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=5 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=5 \
    actor_rollout_ref.rollout.drafter.training.validation_batch_size=${validation_batch_size} \
    actor_rollout_ref.rollout.drafter.training.publish_quality_gate_enable=True \
    actor_rollout_ref.rollout.drafter.training.publish_quality_gate_holdout_ratio=0.2 \
    actor_rollout_ref.rollout.drafter.training.publish_quality_gate_holdout_samples=4 \
    actor_rollout_ref.rollout.drafter.training.publish_quality_gate_min_proxy_delta=0.0 \
    actor_rollout_ref.rollout.drafter.training.publish_quality_gate_max_front_accuracy_drop=0.002 \
    actor_rollout_ref.rollout.drafter.training.publish_quality_gate_max_loss_increase_ratio=0.01 \
    actor_rollout_ref.rollout.drafter.training.publish_quality_gate_meaningful_proxy_delta=0.01 \
    actor_rollout_ref.rollout.drafter.training.publish_quality_gate_rejection_patience=3 \
    actor_rollout_ref.rollout.drafter.training.publish_quality_gate_plateau_patience=4 \
    actor_rollout_ref.rollout.drafter.training.publish_async=False \
    actor_rollout_ref.rollout.drafter.training.publish_dtype=bf16 \
    actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation=True \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=False \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=True \
    actor_rollout_ref.rollout.load_format='auto' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${ppo_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.total_training_steps=40 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=6 \
    "$@"
