#!/usr/bin/env bash
set -euo pipefail

# A3 single-node topology: 16 visible NPUs form 8 rollout replicas at TP=2.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
expected_npu_count=16
IFS=',' read -r -a visible_npus <<< "${ASCEND_RT_VISIBLE_DEVICES}"
declare -A seen_npus=()
for npu_id in "${visible_npus[@]}"; do
    if [[ ! "${npu_id}" =~ ^[0-9]+$ ]]; then
        echo "Invalid NPU id in ASCEND_RT_VISIBLE_DEVICES: ${npu_id@Q}" >&2
        exit 2
    fi
    if [[ -n "${seen_npus[${npu_id}]:-}" ]]; then
        echo "Duplicate NPU id in ASCEND_RT_VISIBLE_DEVICES: ${npu_id}" >&2
        exit 2
    fi
    seen_npus["${npu_id}"]=1
done
visible_npu_count="${#visible_npus[@]}"
if [[ "${visible_npu_count}" -ne "${expected_npu_count}" ]]; then
    echo "A3 test requires ${expected_npu_count} visible NPUs, got ${visible_npu_count}: ${ASCEND_RT_VISIBLE_DEVICES}" >&2
    exit 2
fi

gen_tp=2
rollout_dp=1
rollout_pp=1
train_sp=4
rollout_replica_width=$((gen_tp * rollout_dp * rollout_pp))
if ((visible_npu_count % rollout_replica_width != 0)); then
    echo "NPU count ${visible_npu_count} is not divisible by rollout TP*DP*PP=${rollout_replica_width}" >&2
    exit 2
fi
if ((visible_npu_count % train_sp != 0)); then
    echo "NPU count ${visible_npu_count} is not divisible by actor/ref SP=${train_sp}" >&2
    exit 2
fi
ppo_gpus_per_node="${visible_npu_count}"
rollout_replicas=$((visible_npu_count / rollout_replica_width))

# Keep the fail-fast topology checks authoritative. Experiment knobs may be
# appended at launch, but the A3 resource/parallelism contract may not.
for override do
    override_key="${override%%=*}"
    override_key="${override_key#+}"
    override_key="${override_key#+}"
    case "${override_key}" in
        trainer.n_gpus_per_node|trainer.nnodes|trainer.use_v1|\
        actor_rollout_ref.rollout.tensor_model_parallel_size|\
        actor_rollout_ref.rollout.data_parallel_size|\
        actor_rollout_ref.rollout.pipeline_model_parallel_size|\
        actor_rollout_ref.actor.ulysses_sequence_parallel_size|\
        actor_rollout_ref.ref.ulysses_sequence_parallel_size|\
        actor_rollout_ref.rollout.calculate_log_probs|\
        actor_rollout_ref.rollout.drafter.enable|\
        actor_rollout_ref.rollout.drafter.enable_drafter_training|\
        actor_rollout_ref.rollout.drafter.model_path|\
        actor_rollout_ref.rollout.drafter.speculative_algorithm|\
        actor_rollout_ref.rollout.drafter.vllm.speculative_config_overrides.method|\
        actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens|\
        actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha|\
        actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha|\
        actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_with_markov|\
        actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation|\
        actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dynamic_spec_config.method|\
        actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dynamic_spec_config.method_params.initial_verify_budget_per_req|\
        actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dynamic_spec_config.method_params.budget_update_interval|\
        actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dynamic_spec_config.method_params.budget_threshold|\
        trainer.resume_mode)
            echo "Refusing protected A3/confidence override: ${override}" >&2
            exit 2
            ;;
    esac
done

project_name='verl_grpo_example_dsparktemp1_confidence_runtimefix'
exp_name='qwen3_8b_dsparktemp1_confidence_runtimefix_a3_16npu'
run_id="${SPECO_RUN_ID:-$(date +"%Y%m%d_%H%M%S")}"

# Defaults below are copied from the user's working A3 experiment. Every path
# can still be overridden through an environment variable.
RUN_ROOT="${RUN_ROOT:-/efs_rl/z00876269/Speculative_Decoding_new}"
MODEL_PATH="${MODEL_PATH:-/efs_rl/z00886395/models/Qwen3-8B}"
TRAIN_FILE="${TRAIN_FILE:-/efs_rl/z00886395/datasets/dapo-math-17k.parquet}"
TEST_FILE="${TEST_FILE:-/efs_rl/z00886395/datasets/aime-2024.parquet}"
DRAFTER_SOURCE_PATH="${DRAFTER_SOURCE_PATH:-/efs_rl/z00886395/models/dspark_qwen3_8b_block7}"
VERL_ROOT="${VERL_ROOT:-/efs_rl/z00876269/0625stable/verl}"
SPECO_ROOT="${SPECO_ROOT:-${RUN_ROOT}/verl-SpeCo}"
VLLM_ROOT="${VLLM_ROOT:-${RUN_ROOT}/vllm}"
VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-${RUN_ROOT}/vllm-ascend}"
CKPTS_DIR="${CKPTS_DIR:-${RUN_ROOT}/checkpoints_dspark_confidence_runtimefix_a3_16npu/${run_id}}"
DRAFTER_BOOTSTRAP_PATH="${DRAFTER_BOOTSTRAP_PATH:-${CKPTS_DIR}/bootstrap_drafter}"
DRAFTER_CHECKPOINT_DIR="${DRAFTER_CHECKPOINT_DIR:-${CKPTS_DIR}/drafter}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${exp_name}_${run_id}.log}"

mkdir -p "${LOG_DIR}" "${CKPTS_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
set -x

echo "Logging to: ${LOG_FILE}"
echo "A3 topology: visible_npus=${visible_npu_count}, rollout_tp=${gen_tp}, rollout_replicas=${rollout_replicas}, actor_ref_sp=${train_sp}"
if [[ -n "${RAY_ADDRESS:-}" && "${SPECO_ALLOW_EXISTING_RAY:-0}" != "1" ]]; then
    echo "RAY_ADDRESS is already set (${RAY_ADDRESS}); refusing to attach this exclusive 16-NPU test to an existing Ray cluster. Unset it, or set SPECO_ALLOW_EXISTING_RAY=1 only after verifying the cluster owns all 16 NPUs." >&2
    exit 2
fi

for required_path in \
    "${MODEL_PATH}" \
    "${DRAFTER_SOURCE_PATH}" \
    "${VERL_ROOT}/verl" \
    "${SPECO_ROOT}/verl_speco" \
    "${VLLM_ROOT}/vllm" \
    "${VLLM_ASCEND_ROOT}/vllm_ascend"; do
    if [[ ! -d "${required_path}" ]]; then
        echo "Required directory is missing: ${required_path}" >&2
        exit 2
    fi
done
for required_file in "${TRAIN_FILE}" "${TEST_FILE}" "${DRAFTER_SOURCE_PATH}/config.json"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file is missing: ${required_file}" >&2
        exit 2
    fi
done

# Require the runtime-revision fix while allowing later commits on this branch.
SPECO_REQUIRED_COMMIT="${SPECO_REQUIRED_COMMIT:-e86468011fa7a4eea9d0482bd5664392462a10d3}"
if ! git -C "${SPECO_ROOT}" merge-base --is-ancestor "${SPECO_REQUIRED_COMMIT}" HEAD; then
    echo "SPECO_ROOT does not contain required runtime fix ${SPECO_REQUIRED_COMMIT}: ${SPECO_ROOT}" >&2
    exit 2
fi
echo "verl SHA: $(git -C "${VERL_ROOT}" rev-parse HEAD)"
echo "verl-SpeCo SHA: $(git -C "${SPECO_ROOT}" rev-parse HEAD)"
echo "vllm SHA: $(git -C "${VLLM_ROOT}" rev-parse HEAD)"
echo "vllm-ascend SHA: $(git -C "${VLLM_ASCEND_ROOT}" rev-parse HEAD)"

export PYTHONPATH="${VLLM_ROOT}:${VLLM_ASCEND_ROOT}:${VERL_ROOT}:${SPECO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# release/v0.9.0 reports a development-version suffix while the branch is in
# active development. The compatibility gate accepts that suffix but still
# rejects other release lines and missing APIs before Ray starts.
export VERL_SPECO_STRICT_VERL=1
export VERL_SPECO_EXPECTED_VERL_ROOT="${VERL_ROOT}"
# Confidence-based dynamic verification in vllm-ascend#13216 uses the vLLM V1
# model runner. This is independent of verl's trainer.use_v1 switch below.
export VLLM_USE_V2_MODEL_RUNNER=0

case "${LD_PRELOAD:-}" in
    *libjemalloc*) ;;
    *)
        if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
            export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
        elif [[ -f /usr/lib64/libjemalloc.so.2 ]]; then
            export LD_PRELOAD="/usr/lib64/libjemalloc.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
        fi
        ;;
esac
export MALLOC_CONF="${MALLOC_CONF:-narenas:8,thp:never,metadata_thp:disabled,dirty_decay_ms:0,muzzy_decay_ms:0}"
export SPECO_JEMALLOC_RECLAIM_MODE="${SPECO_JEMALLOC_RECLAIM_MODE:-purge}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

initial_verify_budget="${SPECO_INITIAL_VERIFY_BUDGET:-5}"
budget_update_interval="${SPECO_BUDGET_UPDATE_INTERVAL:-50}"
budget_threshold="${SPECO_BUDGET_THRESHOLD:-0.7}"
spec_verify_tokens=7
if [[ ! "${initial_verify_budget}" =~ ^[1-9][0-9]*$ ]] \
    || ((initial_verify_budget < 1 || initial_verify_budget > spec_verify_tokens)); then
    echo "SPECO_INITIAL_VERIFY_BUDGET must be in [1, ${spec_verify_tokens}], got ${initial_verify_budget}" >&2
    exit 2
fi
if [[ ! "${budget_update_interval}" =~ ^[1-9][0-9]*$ ]] || ((budget_update_interval < 1)); then
    echo "SPECO_BUDGET_UPDATE_INTERVAL must be a positive integer, got ${budget_update_interval}" >&2
    exit 2
fi
if ! python3 - "${budget_threshold}" <<'PY'
import math
import sys

threshold = float(sys.argv[1])
if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
    raise SystemExit(1)
PY
then
    echo "SPECO_BUDGET_THRESHOLD must be finite and in (0, 1), got ${budget_threshold}" >&2
    exit 2
fi

# Build an immutable runtime view that preserves the released block7
# confidence tensors while normalizing the config/index expected by vLLM.
# Missing, malformed, or non-finite confidence weights fail before Ray starts.
DRAFTER_PATH="$(
    python3 -m verl_speco.integration.dspark_confidence_bootstrap \
        --source "${DRAFTER_SOURCE_PATH}" \
        --output "${DRAFTER_BOOTSTRAP_PATH}" \
        --link-mode "${SPECO_BOOTSTRAP_LINK_MODE:-symlink}"
)"
export DRAFTER_PATH

export SPECO_EXPECTED_NPU_COUNT="${expected_npu_count}"
export SPECO_EXPECTED_SPECO_ROOT="${SPECO_ROOT}"
export SPECO_EXPECTED_VLLM_ROOT="${VLLM_ROOT}"
export SPECO_EXPECTED_VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT}"
export SPECO_INITIAL_VERIFY_BUDGET_VALUE="${initial_verify_budget}"
export SPECO_BUDGET_UPDATE_INTERVAL_VALUE="${budget_update_interval}"
export SPECO_BUDGET_THRESHOLD_VALUE="${budget_threshold}"
export SPECO_SPEC_VERIFY_TOKENS_VALUE="${spec_verify_tokens}"
python3 - <<'PY'
import math
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import verl
import verl_speco
import vllm
import vllm_ascend

from vllm.config.speculative import SpeculativeConfig
from vllm_ascend.ascend_config import DynamicSpecConfig
from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from verl_speco.integration.compat import check_compatible_verl
from verl_speco.integration.vllm_runtime import (
    _validate_vllm_dynamic_dspark_confidence_config,
)


def require_import_under(module, expected_env: str) -> None:
    expected = Path(os.environ[expected_env]).resolve()
    actual = Path(module.__file__).resolve()
    if expected not in actual.parents:
        raise RuntimeError(f"unexpected {module.__name__} import: {actual}; expected under {expected}")
    print(f"verified {module.__name__} import: {actual}")


require_import_under(verl, "VERL_SPECO_EXPECTED_VERL_ROOT")
require_import_under(verl_speco, "SPECO_EXPECTED_SPECO_ROOT")
require_import_under(vllm, "SPECO_EXPECTED_VLLM_ROOT")
require_import_under(vllm_ascend, "SPECO_EXPECTED_VLLM_ASCEND_ROOT")
compatibility = check_compatible_verl(strict=True)
if compatibility.missing_api:
    raise RuntimeError(
        "verl checkout is missing APIs required by verl-SpeCo: "
        + ", ".join(compatibility.missing_api)
    )
print(
    "verified verl API contract: "
    f"version={compatibility.version!r}, supported_metadata={compatibility.supported}"
)
initial_budget = int(os.environ["SPECO_INITIAL_VERIFY_BUDGET_VALUE"])
update_interval = int(os.environ["SPECO_BUDGET_UPDATE_INTERVAL_VALUE"])
budget_threshold = float(os.environ["SPECO_BUDGET_THRESHOLD_VALUE"])
spec_tokens = int(os.environ["SPECO_SPEC_VERIFY_TOKENS_VALUE"])
dynamic_config = DynamicSpecConfig(
    {
        "method": "dspark",
        "method_params": {
            "initial_verify_budget_per_req": initial_budget,
            "budget_update_interval": update_interval,
            "budget_threshold": budget_threshold,
        },
    }
)
if dynamic_config.method != "dspark":
    raise RuntimeError("vllm-ascend does not accept dynamic_spec_config.method=dspark")
if not callable(getattr(SpeculativeConfig, "use_dspark", None)):
    raise RuntimeError("vLLM SpeculativeConfig.use_dspark is missing")
if not callable(getattr(AscendQwen3DSparkForCausalLM, "confidence_logits", None)):
    raise RuntimeError("vllm-ascend Qwen3 DSpark confidence head is missing")
for method_name in (
    "update_num_verify_tokens",
    "_compute_verify_budget",
    "_allocate_verify_budget",
):
    if not callable(getattr(AscendDSparkProposer, method_name, None)):
        raise RuntimeError(
            f"vllm-ascend dynamic DSpark proposer API is missing: {method_name}"
        )
print(
    "verified vLLM/vllm-ascend dynamic DSpark APIs: "
    f"vllm={getattr(vllm, '__version__', 'unknown')!r}"
)
if not 1 <= initial_budget <= spec_tokens:
    raise RuntimeError(
        f"initial verify budget must be in [1, {spec_tokens}], got {initial_budget}"
    )
if update_interval <= 0:
    raise RuntimeError(f"budget update interval must be positive, got {update_interval}")
if not math.isfinite(budget_threshold) or not 0.0 < budget_threshold < 1.0:
    raise RuntimeError(
        f"budget threshold must be finite and in (0, 1), got {budget_threshold}"
    )
print(
    "verified dynamic DSpark policy: "
    f"initial_budget={initial_budget}, update_interval={update_interval}, "
    f"threshold={budget_threshold}, max_verify_tokens={spec_tokens}"
)
expected_npus = int(os.environ["SPECO_EXPECTED_NPU_COUNT"])
actual_npus = int(torch.npu.device_count())
if actual_npus != expected_npus:
    raise RuntimeError(
        f"torch.npu sees {actual_npus} devices, but this A3 job requires {expected_npus}; "
        f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!r}"
    )
print(f"verified torch.npu device count: {actual_npus}")
_validate_vllm_dynamic_dspark_confidence_config(os.environ["DRAFTER_PATH"])
print(f"verified dynamic DSpark confidence checkpoint: {os.environ['DRAFTER_PATH']}")
PY

ray_num_cpus="${SPECO_RAY_NUM_CPUS:-64}"
ray_worker_soft_limit="${SPECO_RAY_WORKER_SOFT_LIMIT:-16}"

# Keep generation on the token-only path. Confidence supervision is collected
# by the separate old-logprob forward hook every 10 steps. Since
# rollout_correction.bypass_mode remains false, rollout logprobs are diagnostics
# only and would materialize per-token logprob objects during every long rollout.
PYTHONUNBUFFERED=1 python3 -m verl_speco.main \
    algorithm.adv_estimator=grpo \
    trainer.use_v1=False \
    transfer_queue.enable=False \
    ray_kwargs.ray_init.num_cpus="${ray_num_cpus}" \
    +ray_kwargs.ray_init._system_config.prestart_worker_first_driver=false \
    +ray_kwargs.ray_init._system_config.num_workers_soft_limit="${ray_worker_soft_limit}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=256 \
    data.truncation=error \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${gen_tp}" \
    actor_rollout_ref.rollout.data_parallel_size="${rollout_dp}" \
    actor_rollout_ref.rollout.pipeline_model_parallel_size="${rollout_pp}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${train_sp}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${train_sp}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=512 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dynamic_spec_config.method=dspark \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dynamic_spec_config.method_params.initial_verify_budget_per_req="${initial_verify_budget}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dynamic_spec_config.method_params.budget_update_interval="${budget_update_interval}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.dynamic_spec_config.method_params.budget_threshold="${budget_threshold}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.drafter.enable=True \
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True \
    actor_rollout_ref.rollout.drafter.model_path="${DRAFTER_PATH}" \
    actor_rollout_ref.rollout.drafter.checkpoint_path="${DRAFTER_CHECKPOINT_DIR}" \
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK \
    +actor_rollout_ref.rollout.drafter.vllm.speculative_config_overrides.method=dspark \
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True \
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook \
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=7 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=32 \
    actor_rollout_ref.rollout.drafter.training.dspark_max_window=512 \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=full_vocab \
    actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=4 \
    actor_rollout_ref.rollout.drafter.training.dspark_hard_sample_ratio=0.3 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=5 \
    actor_rollout_ref.rollout.drafter.training.dspark_num_hidden_layers=5 \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_rank=256 \
    actor_rollout_ref.rollout.drafter.training.dspark_markov_head_type=vanilla \
    actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=False \
    actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=0.1 \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0.9 \
    actor_rollout_ref.rollout.drafter.training.dspark_l1_chunk_size=128 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_alpha=1.0 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=1.0 \
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_head_with_markov=True \
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log=False \
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log_first_n=2 \
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log_interval=100 \
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1 \
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens="${spec_verify_tokens}" \
    actor_rollout_ref.rollout.drafter.training.step=20 \
    actor_rollout_ref.rollout.drafter.training.max_collect_samples_per_step_per_replica=16 \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=512 \
    actor_rollout_ref.rollout.drafter.training.max_collect_tokens_per_step_per_replica=16384 \
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=10 \
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=10 \
    actor_rollout_ref.rollout.drafter.training.lr=1e-5 \
    actor_rollout_ref.rollout.drafter.training.lr_decay_steps=200 \
    actor_rollout_ref.rollout.drafter.training.min_lr_ratio=0.1 \
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_mode=front \
    actor_rollout_ref.rollout.drafter.training.publish_async=False \
    actor_rollout_ref.rollout.drafter.training.publish_dtype=bf16 \
    actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=512 \
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation=True \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=False \
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=True \
    actor_rollout_ref.rollout.drafter.training.save_full_drafter_checkpoint=True \
    actor_rollout_ref.rollout.drafter.training.batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.drafter.training.sample_last_n_steps=8 \
    actor_rollout_ref.rollout.drafter.training.train_batches_per_cycle=8 \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.actor.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${ppo_gpus_per_node}" \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_training_steps=200 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=6 \
    "$@"
