# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted(
    [
        *(ROOT / "examples").glob("*.sh"),
        *(ROOT / "examples" / "dynamic").glob("*.sh"),
    ]
)


def _require_working_bash() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    probe = subprocess.run([bash, "--version"], capture_output=True, check=False)
    if probe.returncode != 0:
        pytest.skip("bash is present but not usable in this environment")
    return bash


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_example_shell_syntax_is_valid(script: Path) -> None:
    bash = _require_working_bash()
    subprocess.run([bash, "-n", str(script)], check=True)


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_example_keeps_speco_entrypoint_and_required_drafter_switches(
    script: Path,
) -> None:
    source = script.read_text(encoding="utf-8")

    assert (
        "python3 -m verl_speco.main" in source or "python -m verl_speco.main" in source
    )
    assert "actor_rollout_ref.rollout.drafter.enable=" in source
    assert "actor_rollout_ref.rollout.drafter.enable_drafter_training=" in source
    assert "actor_rollout_ref.rollout.drafter.model_path=" in source
    assert "actor_rollout_ref.rollout.drafter.speculative_algorithm=" in source
    assert (
        "actor_rollout_ref.rollout.drafter.training.collect_interval_steps=" in source
    )
    assert (
        "actor_rollout_ref.rollout.drafter.training.training_interval_steps=" in source
    )
    assert "actor_rollout_ref.rollout.drafter.training.publish_async=" in source


def test_vllm_eagle3_example_keeps_runtime_agnostic_training_switches() -> None:
    source = (ROOT / "examples" / "run_qwen3-8b_drafter_eagle3_vllm.sh").read_text(
        encoding="utf-8"
    )

    assert "actor_rollout_ref.rollout.name=vllm" in source
    assert 'actor_rollout_ref.rollout.drafter.speculative_algorithm="EAGLE3"' in source
    assert (
        "actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True"
        in source
    )
    assert "actor_rollout_ref.rollout.drafter.training.use_logits=False" in source


def test_sglang_examples_request_sglang_rollout() -> None:
    for script in (ROOT / "examples").glob("*sglang*.sh"):
        source = script.read_text(encoding="utf-8")
        assert "actor_rollout_ref.rollout.name=sglang" in source


def test_npu_vllm_example_keeps_explicit_graph_settings() -> None:
    source = (ROOT / "examples" / "run_qwen3-8b_drafter_eagle3_vllm_npu.sh").read_text(
        encoding="utf-8"
    )

    assert 'cudagraph_mode="FULL_DECODE_ONLY"' in source
    assert "cudagraph_capture_sizes=" in source
    assert "max_cudagraph_capture_size=" in source


def test_dspark_dynamic_npu_example_trains_and_serves_confidence_head() -> None:
    source = (
        ROOT / "examples" / "dynamic" / "run_qwen3-8b_drafter_dspark_vllm_npu.sh"
    ).read_text(encoding="utf-8")

    assert "dspark_confidence_head_alpha=1.0" in source
    assert "dspark_confidence_loss_alpha=1.0" in source
    assert "dspark_confidence_head_with_markov=True" in source
    assert "speculative_config_overrides.method=dspark" in source
    assert "additional_config.dynamic_spec_config.method=dspark" in source
    assert "initial_verify_budget_per_req=5" in source
    assert "budget_update_interval=50" in source
    assert "budget_threshold=0.7" in source
    assert "VLLM_USE_V2_MODEL_RUNNER=0" in source
    assert "VERL_SPECO_EXPECTED_VERL_ROOT" in source
    assert "VERL_SPECO_STRICT_VERL=1" in source
    assert "actor_rollout_ref.rollout.calculate_log_probs=False" in source
    assert "actor_rollout_ref.rollout.calculate_log_probs=True" not in source
    assert "trainer.resume_mode=disable" in source
    assert "trainer.val_before_train=True" in source


def test_dspark_a3_16npu_script_is_runnable_and_keeps_test_contract() -> None:
    script = (
        ROOT
        / "examples"
        / "dynamic"
        / "run_qwen3-8b_drafter_dsparktemp1_confidence_a3_16npu.sh"
    )
    source = script.read_text(encoding="utf-8")

    default_devices = re.search(r"ASCEND_RT_VISIBLE_DEVICES:-([0-9,]+)", source)
    assert default_devices is not None
    assert default_devices.group(1).split(",") == [str(index) for index in range(16)]
    assert "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES" in source
    assert 'ray_worker_soft_limit="${SPECO_RAY_WORKER_SOFT_LIMIT:-16}"' in source
    assert "trainer.n_gpus_per_node=16" in source
    assert "trainer.nnodes=1" in source
    assert "gen_tp=2" in source
    assert "rollout_dp=1" in source
    assert "rollout_pp=1" in source
    assert "train_sp=4" in source

    assert "/path/to/" not in source
    assert "/efs_rl/z00886395/models/Qwen3-8B" in source
    assert "/efs_rl/z00886395/datasets/dapo-math-17k.parquet" in source
    assert "/efs_rl/z00886395/datasets/aime-2024.parquet" in source
    assert "/efs_rl/z00886395/models/dspark_qwen3_8b_block7" in source
    assert "/efs_rl/z00876269/0625stable/verl" in source
    assert "/efs_rl/z00876269/Speculative_Decoding/verl" not in source
    assert 'SPECO_ROOT="${SPECO_ROOT:-${RUN_ROOT}/verl-SpeCo}"' in source
    assert 'VLLM_ROOT="${VLLM_ROOT:-${RUN_ROOT}/vllm}"' in source
    assert 'VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-${RUN_ROOT}/vllm-ascend}"' in source
    assert (
        'export PYTHONPATH="${VLLM_ROOT}:${VLLM_ASCEND_ROOT}:${VERL_ROOT}:'
        '${SPECO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' in source
    )

    assert "verl_speco.integration.dspark_confidence_bootstrap" in source
    assert "export VERL_SPECO_STRICT_VERL=1" in source
    assert 'export SOC_VERSION="${SOC_VERSION:-ascend910_9391}"' in source
    assert "export VLLM_USE_V2_MODEL_RUNNER=0" in source
    assert "VLLM_USE_V2_MODEL_RUNNER=1" not in source
    assert 'SPECO_BUDGET_UPDATE_INTERVAL:-16' in source
    assert 'SPECO_BUDGET_THRESHOLD:-0.3' in source
    assert 'SPECO_MIN_VERIFY_TOKENS:-1' in source
    assert "method_params.min_verify_tokens" in source
    assert "dspark_ce_loss_alpha=0.1" in source
    assert "dspark_l1_loss_alpha=0.9" in source
    assert "dspark_confidence_head_alpha=1.0" in source
    assert "dspark_confidence_loss_alpha=1.0" in source
    assert "dspark_confidence_loss_alpha=0.0" not in source
    assert "dspark_confidence_head_with_markov=True" in source
    assert "speculative_config_overrides.method=dspark" in source
    assert "additional_config.dynamic_spec_config.method=dspark" in source
    assert "actor_rollout_ref.rollout.calculate_log_probs=False" in source
    assert "actor_rollout_ref.rollout.calculate_log_probs=True" not in source
    assert (
        "actor_rollout_ref.actor.fsdp_config."
        "use_no_sync_for_gradient_accumulation=False" in source
    )
    assert "draft_update_pause_generation=True" in source
    assert "actor_rollout_ref.rollout.drafter.training.collect_interval_steps=2" in source
    assert "actor_rollout_ref.rollout.drafter.training.training_interval_steps=2" in source
    assert "trainer.resume_mode=disable" in source
    assert "trainer.use_v1=False" in source
    assert "trainer.val_before_train=True" in source
    assert "data.filter_overlong_prompts=False" in source
    assert source.count('"$@"') == 1
    assert source.rfind('"$@"') > source.rfind("trainer.total_epochs=6")

    # Keep the launch script close to the compact experiment-script style.
    assert len(source.splitlines()) < 260
    assert "Refusing protected A3/confidence override" not in source
    assert "def load_class_node(" not in source
    assert "SPECO_EXPECTED_NPU_COUNT" not in source
