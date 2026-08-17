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

import asyncio
import json
import re
import sys
import types
from inspect import getsource
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl_speco.integration.vllm_runtime import (
    SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX,
    SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS,
    SPECO_VLLM_WORKER_EXTENSION_CLS,
    SpecoVLLMColocateWorkerExtension,
    SpecoVLLMWeightSyncCompatExtension,
    _describe_vllm_draft_logits,
    _new_vllm_spec_decode_stats,
    _normalize_dflash_target_layer_aliases,
    _record_vllm_spec_decode_scheduler_stats,
    _speco_can_use_npu_target_staging,
    _speco_npu_target_staging,
    _speco_npu_target_staging_decision,
    _speco_persistent_weight_shm_name,
    _validate_vllm_pause_ack,
    _validate_vllm_dflash_drafter_config,
    _validate_vllm_dynamic_dspark_confidence_config,
    _vllm_ascend_has_dspark_pr11153_k_query_runtime,
    _vllm_spec_decode_stats_to_metrics,
    attach_update_draft_weights_to_rollout,
    build_vllm_speculative_config_from_drafter,
    configure_vllm_runtime_from_config,
    patch_transformers_attention_layer_type_constants,
    patch_verl_bucketed_weight_transfer_npu_staging,
    patch_verl_bucketed_weight_transfer_shm_reuse,
    speco_vllm_update_draft_weights,
)


def _save_safetensors(tensors, path) -> None:
    save_file = pytest.importorskip("safetensors.torch").save_file
    save_file(tensors, path)


def _drafter(**overrides):
    config = {
        "enable": True,
        "enable_drafter_training": True,
        "speculative_algorithm": "EAGLE3",
        "model_path": "/models/drafter",
        "rollout": {"spec_steps": 3},
        "training": {},
        "vllm": {},
    }
    config.update(overrides)
    return config


def test_vllm_speculative_config_maps_eagle3_contract() -> None:
    config = build_vllm_speculative_config_from_drafter(_drafter())

    assert config == {
        "draft_sample_method": "greedy",
        "method": "eagle3",
        "model": "/models/drafter",
        "num_speculative_tokens": 3,
    }


def test_vllm_fresh_training_does_not_load_checkpoint_output_root() -> None:
    config = build_vllm_speculative_config_from_drafter(
        _drafter(checkpoint_path="/checkpoints/run/drafter")
    )

    assert config["model"] == "/models/drafter"


def test_vllm_checkpoint_path_remains_a_fallback_without_model_path() -> None:
    config = build_vllm_speculative_config_from_drafter(
        _drafter(model_path=None, checkpoint_path="/checkpoints/draft_step_10")
    )

    assert config["model"] == "/checkpoints/draft_step_10"


def test_vllm_worker_extension_constructs_without_wake_up_fallback() -> None:
    extension = SpecoVLLMColocateWorkerExtension()

    assert isinstance(extension, SpecoVLLMColocateWorkerExtension)


class _FakeDraftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.online = torch.nn.Parameter(torch.tensor([3.0]))
        self.model.confidence_head = torch.nn.Linear(1, 1)
        self.lm_head = torch.nn.Linear(2, 3, bias=False)


def _runtime_extension(*, method: str = "dspark"):
    extension = SpecoVLLMColocateWorkerExtension()
    draft = _FakeDraftModel()
    proposer = SimpleNamespace(
        model=draft,
        speculative_config=SimpleNamespace(method=method),
    )
    target = SimpleNamespace(lm_head=torch.nn.Linear(2, 3, bias=False))
    extension.model_runner = SimpleNamespace(
        drafter=proposer,
        model=target,
        get_model=lambda: target,
    )
    return extension, draft, target


def test_vllm_level1_wake_never_reloads_draft_checkpoint(monkeypatch) -> None:
    extension, _, _ = _runtime_extension()
    reload_calls = []
    monkeypatch.setattr(
        extension,
        "_speco_reload_draft_from_checkpoint",
        lambda: reload_calls.append(True) or 1,
    )

    assert extension._speco_prepare_draft_for_sleep(1) == 0
    assert extension._speco_restore_draft_for_wake(["weights"]) == (None, 0)
    assert reload_calls == []


def test_vllm_level2_wake_restores_exact_online_revision() -> None:
    extension, draft, _ = _runtime_extension()
    extension._speco_draft_runtime_revision = 4
    expected = {
        name: tensor.detach().clone()
        for name, tensor in (
            list(draft.named_parameters()) + list(draft.named_buffers())
        )
    }

    assert extension._speco_prepare_draft_for_sleep(2) == len(expected)
    with torch.no_grad():
        for tensor in draft.parameters():
            tensor.zero_()

    assert extension._speco_restore_draft_for_wake(["weights"]) == (
        "snapshot",
        len(expected),
    )
    assert extension._speco_draft_runtime_revision == 4
    for name, tensor in draft.named_parameters():
        torch.testing.assert_close(tensor, expected[name])


def test_vllm_level2_missing_online_snapshot_fails_closed(monkeypatch) -> None:
    extension, _, _ = _runtime_extension()
    extension._speco_draft_runtime_revision = 2
    extension._speco_draft_level2_restore_pending = True
    extension._speco_draft_level2_snapshot = None
    monkeypatch.setattr(extension, "_speco_reload_draft_from_checkpoint", lambda: 64)

    with pytest.raises(RuntimeError, match="Refusing to roll back"):
        extension._speco_restore_draft_for_wake(["weights"])


def test_vllm_dspark_target_sync_updates_only_outer_lm_head() -> None:
    extension, draft, target = _runtime_extension()
    with torch.no_grad():
        draft.lm_head.weight.zero_()
        draft.model.online.fill_(7.0)
        draft.model.confidence_head.weight.fill_(11.0)
        target.lm_head.weight.fill_(5.0)
    online_before = draft.model.online.detach().clone()
    confidence_before = draft.model.confidence_head.weight.detach().clone()

    assert extension._speco_sync_dspark_lm_head_from_target() == 1

    torch.testing.assert_close(draft.lm_head.weight, target.lm_head.weight)
    torch.testing.assert_close(draft.model.online, online_before)
    torch.testing.assert_close(
        draft.model.confidence_head.weight, confidence_before
    )


def test_vllm_dspark_target_sync_accepts_ascend_dflash_method_alias(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "VERL_SPECO_SGLANG_DRAFTER_CONFIG",
        json.dumps({"speculative_algorithm": "DSPARK"}),
    )
    extension, draft, target = _runtime_extension(method="dflash")
    with torch.no_grad():
        draft.lm_head.weight.zero_()
        target.lm_head.weight.fill_(9.0)

    assert extension._speco_sync_dspark_lm_head_from_target() == 1
    torch.testing.assert_close(draft.lm_head.weight, target.lm_head.weight)


def test_vllm_dspark_hot_update_validates_loader_result_and_confidence_pair() -> None:
    requested = [
        "fc.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    ]
    assert (
        SpecoVLLMColocateWorkerExtension._speco_validate_loaded_draft_weights(
            requested, set(requested), draft_method="dspark"
        )
        == 3
    )
    with pytest.raises(RuntimeError, match="complete online update"):
        SpecoVLLMColocateWorkerExtension._speco_validate_loaded_draft_weights(
            requested,
            {"fc.weight", "confidence_head.proj.weight"},
            draft_method="dspark",
        )
    with pytest.raises(RuntimeError, match="atomic pair"):
        SpecoVLLMColocateWorkerExtension._speco_validate_loaded_draft_weights(
            ["fc.weight", "confidence_head.proj.weight"],
            {"fc.weight", "confidence_head.proj.weight"},
            draft_method="dspark",
        )
    with pytest.raises(RuntimeError, match="every revision"):
        SpecoVLLMColocateWorkerExtension._speco_validate_loaded_draft_weights(
            ["fc.weight"],
            {"fc.weight"},
            draft_method="dspark",
            require_confidence_pair=True,
        )


def test_vllm_dspark_confidence_revision_is_required_by_dynamic_contract(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "VERL_SPECO_SGLANG_DRAFTER_CONFIG",
        json.dumps({"_speco_require_confidence_revision": True}),
    )
    extension, _, _ = _runtime_extension(method="dspark")

    assert extension._speco_dspark_confidence_revision_required() is True


def test_vllm_target_sync_source_has_no_unconditional_checkpoint_reload() -> None:
    source = getsource(SpecoVLLMColocateWorkerExtension.update_weights_from_ipc)

    assert "_speco_reload_draft_from_checkpoint" not in source
    assert "_speco_sync_dspark_lm_head_from_target" in source


def test_vllm_weight_sync_extension_has_stable_runtime_path() -> None:
    assert SPECO_VLLM_WEIGHT_SYNC_WORKER_EXTENSION_CLS.endswith(
        ".SpecoVLLMWeightSyncCompatExtension"
    )
    source = getsource(SpecoVLLMWeightSyncCompatExtension.update_weights_from_ipc)
    assert source.index(
        "patch_verl_bucketed_weight_transfer_rebuild_ipc()"
    ) < source.index("super().update_weights_from_ipc(")
    assert source.index(
        "patch_verl_bucketed_weight_transfer_shm_reuse()"
    ) < source.index("super().update_weights_from_ipc(")
    assert source.index(
        "patch_verl_bucketed_weight_transfer_npu_staging()"
    ) < source.index("super().update_weights_from_ipc(")
    assert "with _speco_npu_target_staging(" in source


def test_vllm_npu_staging_is_guarded_and_preserves_upstream_fallback() -> None:
    guard_source = getsource(_speco_npu_target_staging_decision)
    context_source = getsource(_speco_npu_target_staging)
    patch_source = getsource(patch_verl_bucketed_weight_transfer_npu_staging)

    assert "not use_shm" in guard_source
    assert "peft_config is not None" in guard_source
    assert "not _speco_is_npu_vllm_worker(worker)" in guard_source
    assert 'getattr(vllm_config, "quant_config", None)' in guard_source
    assert "quant_config is not None" in guard_source
    assert "return original_receive(self, on_bucket_received)" in patch_source
    assert "on_bucket_received(weights, is_last)" in patch_source
    assert "SPECO_VLLM_NPU_STAGING_COPY_CHUNK_BYTES" in patch_source
    assert "staging_buffer[start:end].copy_(" in patch_source
    assert "self.buffer[start:end], non_blocking=False" in patch_source
    assert "get_torch_device().synchronize()" in patch_source
    assert "NPU staging decision" in context_source
    assert "flush=True" in context_source
    assert "return enabled" in getsource(_speco_can_use_npu_target_staging)


def test_vllm_weight_shm_name_is_stable_and_channel_scoped() -> None:
    handle = "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-0.sock"

    assert _speco_persistent_weight_shm_name(
        handle, 2048 << 20
    ) == _speco_persistent_weight_shm_name(handle, 2048 << 20)
    assert _speco_persistent_weight_shm_name(
        handle, 2048 << 20
    ) != _speco_persistent_weight_shm_name(handle, 512 << 20)
    assert _speco_persistent_weight_shm_name(
        handle, 2048 << 20
    ) != _speco_persistent_weight_shm_name(
        handle.replace("rank-0", "rank-1"), 2048 << 20
    )


def test_vllm_weight_shm_patch_reuses_mapping_and_preserves_ipc_path() -> None:
    created = []

    class FakeShm:
        def __init__(self, size: int):
            self.size = size
            self.buf = bytearray(size)
            self.close_count = 0
            self.unlink_count = 0

        def close(self):
            self.close_count += 1

        def unlink(self):
            self.unlink_count += 1

    class FakeTorch:
        uint8 = "uint8"

        @staticmethod
        def frombuffer(buffer, dtype):
            assert dtype == FakeTorch.uint8
            return buffer

    class FakeSocket:
        def __init__(self, incoming=None):
            self.metadata = []
            self.incoming = incoming

        def send_pyobj(self, value):
            self.metadata.append(value)

        def recv(self):
            return b""

        def recv_pyobj(self):
            return self.incoming

        def send(self, value):
            self.metadata.append(value)

    class FakeSender:
        def __init__(self, *, use_shm: bool):
            self.use_shm = use_shm
            self.zmq_handle = "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-0.sock"
            self.bucket_size = 64
            self.socket = FakeSocket()
            self.buffer = None
            self.shm = None
            self.upstream_init_called = False
            self.upstream_cleanup_called = False

        def _init_buffer(self):
            self.upstream_init_called = True

        def _cleanup(self):
            self.upstream_cleanup_called = True
            self.buffer = None
            self.shm = None

    class FakeReceiver:
        def __init__(self, *, use_shm: bool, metadata):
            self.use_shm = use_shm
            self.socket = FakeSocket(metadata)
            self.buffer = None
            self.shm = None
            self.upstream_init_called = False
            self.upstream_cleanup_called = False

        def _init_buffer(self):
            self.upstream_init_called = True

        def _cleanup(self):
            self.upstream_cleanup_called = True
            self.buffer = None
            self.shm = None

    def create_shared_memory(size, name):
        shm = FakeShm(size)
        created.append((name, shm))
        return shm

    module = SimpleNamespace(
        BucketedWeightSender=FakeSender,
        BucketedWeightReceiver=FakeReceiver,
        create_shared_memory=create_shared_memory,
        rebuild_shared_memory=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected attach")
        ),
        torch=FakeTorch,
    )

    assert patch_verl_bucketed_weight_transfer_shm_reuse(module) is True
    assert patch_verl_bucketed_weight_transfer_shm_reuse(module) is False
    first = FakeSender(use_shm=True)
    first._init_buffer()
    first_buffer = first.buffer
    first._cleanup()
    second = FakeSender(use_shm=True)
    second._init_buffer()

    assert len(created) == 1
    assert second.buffer is first_buffer
    assert created[0][1].close_count == 0
    assert created[0][1].unlink_count == 0
    assert first.upstream_cleanup_called is True

    receiver = FakeReceiver(use_shm=True, metadata=second.socket.metadata[0])
    receiver._init_buffer()
    assert receiver.buffer is first_buffer
    receiver._cleanup()
    assert receiver.upstream_cleanup_called is True

    ipc_sender = FakeSender(use_shm=False)
    ipc_sender._init_buffer()
    assert ipc_sender.upstream_init_called is True

    second._cleanup()
    module._speco_cleanup_persistent_weight_shm()
    assert created[0][1].close_count == 1
    assert created[0][1].unlink_count == 1


def test_vllm_draft_logits_diagnostic_handles_missing_and_non_tensor_values() -> None:
    assert _describe_vllm_draft_logits(None, missing=True) == "missing"
    assert _describe_vllm_draft_logits(None) == "None(greedy)"
    assert _describe_vllm_draft_logits("MISSING") == "str"


def test_vllm_speculative_config_maps_dflash_contract() -> None:
    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DFLASH",
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
        )
    )

    assert config == {
        "draft_sample_method": "greedy",
        "method": "dflash",
        "model": "/models/drafter",
        "num_speculative_tokens": 16,
    }


def test_vllm_speculative_config_maps_dspark_to_native_gpu_contract(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint",
        lambda: False,
    )

    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3DSparkModel"],
          "markov_head_type": "vanilla",
          "target_layer_ids": [1, 9, 17, 25, 33]
        }
        """,
        encoding="utf-8",
    )

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
        )
    )

    assert config == {
        "draft_sample_method": "greedy",
        "method": "dspark",
        "model": str(model_path),
        "num_speculative_tokens": 16,
    }


def test_vllm_speculative_config_maps_dspark_to_dflash_on_npu_contract(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3DSparkModel"],
          "markov_head_type": "vanilla",
          "target_layer_ids": [1, 9, 17, 25, 33]
        }
        """,
        encoding="utf-8",
    )

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
        )
    )

    assert config == {
        "draft_sample_method": "greedy",
        "method": "dflash",
        "model": str(model_path),
        "num_speculative_tokens": 16,
    }


def test_vllm_dynamic_dspark_uses_native_method_on_npu_contract(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3DSparkModel"],
          "markov_head_type": "vanilla",
          "enable_confidence_head": true,
          "confidence_head_with_markov": true,
          "hidden_size": 8,
          "markov_rank": 4,
          "target_layer_ids": [1, 9, 17, 25, 33]
        }
        """,
        encoding="utf-8",
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.confidence_head.proj.weight": "model-00001-of-00001.safetensors",
                    "model.confidence_head.proj.bias": "model-00001-of-00001.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    confidence_shard = model_path / "model-00001-of-00001.safetensors"
    _save_safetensors(
        {
            "model.confidence_head.proj.weight": torch.zeros(1, 12),
            "model.confidence_head.proj.bias": torch.zeros(1),
        },
        confidence_shard,
    )
    rollout_cfg = {
        "engine_kwargs": {
            "vllm": {
                "additional_config": {
                    "dynamic_spec_config": {
                        "method": "dspark",
                        "method_params": {
                            "initial_verify_budget_per_req": 5,
                            "budget_update_interval": 50,
                            "budget_threshold": 0.7,
                        },
                    }
                }
            }
        }
    }

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 1, "spec_verify_tokens": 7},
        ),
        rollout_cfg=rollout_cfg,
    )

    assert config["method"] == "dspark"
    assert config["num_speculative_tokens"] == 7


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (
            {
                "architectures": ["Qwen3DSparkModel"],
                "markov_head_type": "vanilla",
                "enable_confidence_head": False,
            },
            "enable_confidence_head=true",
        ),
        (
            {
                "architectures": ["Qwen3DSparkModel"],
                "markov_head_type": "vanilla",
                "enable_confidence_head": True,
                "confidence_head_with_markov": False,
            },
            "confidence_head_with_markov=true",
        ),
    ],
)
def test_vllm_dynamic_dspark_rejects_incompatible_confidence_topology(
    tmp_path, config, error
) -> None:
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        _validate_vllm_dynamic_dspark_confidence_config(model_path)


def test_vllm_dynamic_dspark_requires_resolved_local_checkpoint() -> None:
    with pytest.raises(ValueError, match="resolved local drafter checkpoint"):
        _validate_vllm_dynamic_dspark_confidence_config("org/remote-dspark")


def test_vllm_dynamic_dspark_rejects_config_without_confidence_weights(
    tmp_path,
) -> None:
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3DSparkModel"],
                "markov_head_type": "vanilla",
                "enable_confidence_head": True,
                "confidence_head_with_markov": True,
            }
        ),
        encoding="utf-8",
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.fc.weight": "model.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="complete confidence_head"):
        _validate_vllm_dynamic_dspark_confidence_config(model_path)


def test_vllm_dynamic_dspark_rejects_wrong_confidence_tensor_shape(
    tmp_path,
) -> None:
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "enable_confidence_head": True,
                "confidence_head_with_markov": True,
                "hidden_size": 8,
                "markov_rank": 4,
            }
        ),
        encoding="utf-8",
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.confidence_head.proj.weight": "model.safetensors",
                    "model.confidence_head.proj.bias": "model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    _save_safetensors(
        {
            "model.confidence_head.proj.weight": torch.zeros(1, 8),
            "model.confidence_head.proj.bias": torch.zeros(1),
        },
        model_path / "model.safetensors",
    )

    with pytest.raises(ValueError, match="tensor shapes disagree"):
        _validate_vllm_dynamic_dspark_confidence_config(model_path)


def test_vllm_dynamic_dspark_rejects_static_dflash_override(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla", '
        '"enable_confidence_head": true, "confidence_head_with_markov": true, '
        '"hidden_size": 8, "markov_rank": 4}',
        encoding="utf-8",
    )
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.confidence_head.proj.weight": "model.safetensors",
                    "model.confidence_head.proj.bias": "model.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    _save_safetensors(
        {
            "model.confidence_head.proj.weight": torch.zeros(1, 12),
            "model.confidence_head.proj.bias": torch.zeros(1),
        },
        model_path / "model.safetensors",
    )
    rollout_cfg = {
        "engine_kwargs": {
            "vllm": {"additional_config": {"dynamic_spec_config": {"method": "dspark"}}}
        }
    }

    with pytest.raises(ValueError, match="requires speculative_config.method=dspark"):
        build_vllm_speculative_config_from_drafter(
            _drafter(
                speculative_algorithm="DSPARK",
                model_path=str(model_path),
                rollout={"spec_steps": 1, "spec_verify_tokens": 7},
                vllm={"speculative_config_overrides": {"method": "dflash"}},
            ),
            rollout_cfg=rollout_cfg,
        )


def test_vllm_dspark_gpu_probabilistic_sampling_requires_override(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint",
        lambda: False,
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    config = build_vllm_speculative_config_from_drafter(
        _drafter(
            speculative_algorithm="DSPARK",
            model_path=str(model_path),
            rollout={"spec_steps": 3, "spec_verify_tokens": 16},
            vllm={
                "speculative_config_overrides": {"draft_sample_method": "probabilistic"}
            },
        )
    )

    assert config["method"] == "dspark"
    assert config["draft_sample_method"] == "probabilistic"


def test_vllm_dflash_validator_rejects_dspark_when_algorithm_is_dflash(
    tmp_path,
) -> None:
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vLLM DFlash requires"):
        _validate_vllm_dflash_drafter_config(model_path, algorithm="DFLASH")


def test_vllm_dspark_validator_accepts_markov_head_config(tmp_path) -> None:
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )

    _validate_vllm_dflash_drafter_config(model_path, algorithm="DSPARK")


def test_vllm_dspark_config_aliases_are_dflash_compatible() -> None:
    config = {
        "architectures": ["DFlashDSparkDraftModel"],
        "markov_head_type": "vanilla",
        "mask_token_id": 151669,
        "target_layer_ids": [1, 9, 17, 25, 33],
    }

    assert _normalize_dflash_target_layer_aliases(config) is True

    assert config["dflash_config"] == {
        "target_layer_ids": [1, 9, 17, 25, 33],
        "mask_token_id": 151669,
    }
    assert config["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]


def _install_fake_vllm_ascend_modules(monkeypatch, dflash_cls, proposer_cls) -> None:
    root_module = types.ModuleType("vllm_ascend")
    spec_decode_module = types.ModuleType("vllm_ascend.spec_decode")
    dflash_module = types.ModuleType("vllm_ascend.spec_decode.dflash_proposer")
    proposer_module = types.ModuleType("vllm_ascend.spec_decode.llm_base_proposer")

    dflash_module.AscendDflashProposer = dflash_cls
    proposer_module.AscendSpecDecodeBaseProposer = proposer_cls
    spec_decode_module.dflash_proposer = dflash_module
    spec_decode_module.llm_base_proposer = proposer_module
    root_module.spec_decode = spec_decode_module

    monkeypatch.setitem(sys.modules, "vllm_ascend", root_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode", spec_decode_module)
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.spec_decode.dflash_proposer", dflash_module
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.spec_decode.llm_base_proposer", proposer_module
    )


class _FakePR11153DflashProposer:
    def _num_query_per_req(self):
        return (
            self.num_speculative_tokens
            if self._is_dspark
            else 1 + self.num_speculative_tokens
        )

    def set_inputs_first_pass(self):
        return self._num_query_per_req(), "IS_DSPARK"


class _FakePR11153SpecDecodeBaseProposer:
    def _run_merged_draft(self):
        if hasattr(
            self.speculative_config.draft_model_config.hf_config, "markov_head_type"
        ):
            blk = self.num_speculative_tokens
            draft_token_ids = self.model.model.markov_head
            return draft_token_ids[:, 1:] if blk else None
        return None


class _FakeOldDSparkDflashProposer:
    def set_inputs_first_pass(self):
        return 1 + self.num_speculative_tokens


class _FakeOldDSparkSpecDecodeBaseProposer:
    def _run_merged_draft(self):
        if hasattr(
            self.speculative_config.draft_model_config.hf_config, "markov_head_type"
        ):
            blk = self.num_speculative_tokens + 1
            draft_token_ids = self.model.model.markov_head
            return draft_token_ids[:, 1:] if blk else None
        return None


def test_vllm_ascend_dspark_runtime_detector_accepts_pr11153_k_query(
    monkeypatch,
) -> None:
    _install_fake_vllm_ascend_modules(
        monkeypatch,
        _FakePR11153DflashProposer,
        _FakePR11153SpecDecodeBaseProposer,
    )

    assert _vllm_ascend_has_dspark_pr11153_k_query_runtime() is True


def test_vllm_ascend_dspark_runtime_detector_rejects_old_full_block_layout(
    monkeypatch,
) -> None:
    _install_fake_vllm_ascend_modules(
        monkeypatch,
        _FakeOldDSparkDflashProposer,
        _FakeOldDSparkSpecDecodeBaseProposer,
    )

    assert _vllm_ascend_has_dspark_pr11153_k_query_runtime() is False


def test_vllm_runtime_injects_native_config_and_worker_extension(monkeypatch) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.install_upstream_vllm_runtime_bridge",
        lambda: True,
    )
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "drafter": _drafter(),
                "engine_kwargs": {"vllm": {}},
            }
        }
    }

    configure_vllm_runtime_from_config(config)

    engine_kwargs = config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]
    assert engine_kwargs["speculative_config"]["method"] == "eagle3"
    assert engine_kwargs["worker_extension_cls"] == SPECO_VLLM_WORKER_EXTENSION_CLS


def test_vllm_runtime_injects_dspark_as_dflash_on_npu_and_worker_extension(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.install_upstream_vllm_runtime_bridge",
        lambda: True,
    )
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime._is_vllm_ascend_runtime_hint", lambda: True
    )
    model_path = tmp_path / "dspark-drafter"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"architectures": ["Qwen3DSparkModel"], "markov_head_type": "vanilla"}',
        encoding="utf-8",
    )
    config = {
        "actor_rollout_ref": {
            "rollout": {
                "name": "vllm",
                "drafter": _drafter(
                    speculative_algorithm="DSPARK",
                    model_path=str(model_path),
                    rollout={"spec_steps": 3, "spec_verify_tokens": 16},
                ),
                "engine_kwargs": {"vllm": {}},
            }
        }
    }

    configure_vllm_runtime_from_config(config)

    engine_kwargs = config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]
    assert engine_kwargs["speculative_config"]["method"] == "dflash"
    assert engine_kwargs["speculative_config"]["num_speculative_tokens"] == 16
    assert engine_kwargs["worker_extension_cls"] == SPECO_VLLM_WORKER_EXTENSION_CLS


def test_transformers_attention_layer_type_constants_compat(monkeypatch) -> None:
    transformers_module = types.ModuleType("transformers")
    configuration_utils_module = types.ModuleType("transformers.configuration_utils")
    transformers_module.configuration_utils = configuration_utils_module
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(
        sys.modules, "transformers.configuration_utils", configuration_utils_module
    )

    assert patch_transformers_attention_layer_type_constants() is True
    assert configuration_utils_module.ALLOWED_LAYER_TYPES
    assert (
        configuration_utils_module.ALLOWED_LAYER_TYPES
        == configuration_utils_module.ALLOWED_ATTENTION_LAYER_TYPES
    )
    assert patch_transformers_attention_layer_type_constants() is False


def test_import_compat_runs_before_vllm_worker_extension_import() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "verl_speco"
        / "integration"
        / "vllm_runtime.py"
    ).read_text(encoding="utf-8")

    extension_import_match = re.search(
        r"from\s+verl\.workers\.rollout\.vllm_rollout\.utils\s+import\s+"
        r"\(?\s*vLLMColocateWorkerExtension",
        source,
    )
    assert extension_import_match is not None
    extension_import = extension_import_match.start()
    assert (
        source.index("\npatch_transformers_attention_layer_type_constants()\n")
        < extension_import
    )
    assert source.index("\ninstall_verl_npu_vllm_import_compat()\n") < extension_import


def test_vllm_acceptance_stats_keep_stable_transport_keys() -> None:
    stats = _new_vllm_spec_decode_stats()
    scheduler_stats = SimpleNamespace(
        spec_decoding_stats=SimpleNamespace(num_drafts=4, num_accepted_tokens=7)
    )

    _record_vllm_spec_decode_scheduler_stats(stats, scheduler_stats)

    assert _vllm_spec_decode_stats_to_metrics(stats) == {
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_drafts": 4.0,
        f"{SPECO_VLLM_SPEC_DECODE_EXTRA_PREFIX}_accepted_tokens": 7.0,
    }


def test_trainer_keeps_public_acceptance_metric_name() -> None:
    trainer_source = (
        Path(__file__).resolve().parents[2]
        / "verl_speco"
        / "trainer"
        / "speco_ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert '"drafter/spec_decode/mean_acceptance_length"' in trainer_source


def test_vllm_draft_update_attachment_is_idempotent() -> None:
    rollout = SimpleNamespace()

    assert attach_update_draft_weights_to_rollout(rollout) is rollout
    first = rollout.update_draft_weights
    assert first.__func__ is speco_vllm_update_draft_weights
    assert attach_update_draft_weights_to_rollout(rollout).update_draft_weights == first


@pytest.mark.parametrize("weights", [None, [], {}])
def test_vllm_draft_update_rejects_empty_publish_payload(weights) -> None:
    with pytest.raises(RuntimeError, match="empty drafter publish payload"):
        asyncio.run(
            speco_vllm_update_draft_weights(
                SimpleNamespace(),
                weights,
                global_steps=10,
            )
        )


def test_vllm_pause_ack_rejects_nested_server_error() -> None:
    _validate_vllm_pause_ack(
        {
            "aborted_count": 0,
            "request_ids": [],
            "server_results": [{"aborted_count": 0, "request_ids": []}],
        }
    )

    with pytest.raises(RuntimeError, match="was not safely paused"):
        _validate_vllm_pause_ack(
            {
                "aborted_count": 0,
                "request_ids": [],
                "server_results": [
                    {"aborted_count": 0, "request_ids": [], "error": "pause failed"}
                ],
            }
        )

    with pytest.raises(RuntimeError, match="was not safely paused"):
        _validate_vllm_pause_ack({})

    with pytest.raises(RuntimeError, match="was not safely paused"):
        _validate_vllm_pause_ack({"success": False})


def test_vllm_draft_update_does_not_send_before_pause_ack(monkeypatch) -> None:
    events = []

    class _Sender:
        def __init__(self, **kwargs):
            del kwargs

        async def async_send_weights(self, weights):
            list(weights)
            events.append("send")

    class _RemoteMethod:
        async def remote(self, *args, **kwargs):
            del args, kwargs
            events.append("abort")
            return {
                "aborted_count": 0,
                "request_ids": [],
                "server_results": [{"error": "engine pause failed"}],
            }

    async def _execute_method(*args, **kwargs):
        del args, kwargs
        events.append("receiver")

    bucket_module = types.ModuleType(
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"
    )
    bucket_module.BucketedWeightSender = _Sender
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer",
        bucket_module,
    )

    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.patch_verl_bucketed_weight_transfer_shm_reuse",
        lambda: False,
    )
    monkeypatch.setenv(
        "VERL_SPECO_SGLANG_DRAFTER_CONFIG",
        json.dumps({"training": {"draft_update_pause_generation": True}}),
    )
    adapter = SimpleNamespace(
        rollout_rank=0,
        replica_rank=0,
        node_rank=0,
        server_handle=SimpleNamespace(abort_all_requests=_RemoteMethod()),
        config=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1)
        ),
        use_shm=False,
        zmq_handle="ipc:///tmp/test-speco-target.sock",
        _execute_method=_execute_method,
    )

    with pytest.raises(RuntimeError, match="was not safely paused"):
        asyncio.run(
            speco_vllm_update_draft_weights(
                adapter,
                [("fc.weight", torch.zeros(1))],
                global_steps=10,
            )
        )

    assert events == ["abort"]


def test_vllm_failed_draft_update_keeps_generation_paused(monkeypatch) -> None:
    events = []

    class _RemoteMethod:
        def __init__(self, name):
            self.name = name

        async def remote(self, *args, **kwargs):
            events.append((self.name, args, kwargs))
            if self.name == "abort_all_requests":
                return {"aborted_count": 0, "request_ids": []}

    class _Sender:
        def __init__(self, **kwargs):
            del kwargs

        async def async_send_weights(self, weights):
            list(weights)

    async def _failed_update():
        raise RuntimeError("mixed revision")

    async def _execute_method(*args, **kwargs):
        del args, kwargs
        return _failed_update()

    bucket_module = types.ModuleType(
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"
    )
    bucket_module.BucketedWeightSender = _Sender
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer",
        bucket_module,
    )
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.patch_verl_bucketed_weight_transfer_rebuild_ipc",
        lambda: False,
    )
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.patch_verl_bucketed_weight_transfer_shm_reuse",
        lambda: False,
    )
    monkeypatch.setenv(
        "VERL_SPECO_SGLANG_DRAFTER_CONFIG",
        json.dumps({"training": {"draft_update_pause_generation": True}}),
    )
    adapter = SimpleNamespace(
        rollout_rank=0,
        replica_rank=0,
        node_rank=0,
        server_handle=SimpleNamespace(
            abort_all_requests=_RemoteMethod("abort_all_requests"),
            clear_kv_cache=_RemoteMethod("clear_kv_cache"),
            set_global_steps=_RemoteMethod("set_global_steps"),
            resume_generation=_RemoteMethod("resume_generation"),
        ),
        config=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1)
        ),
        use_shm=False,
        zmq_handle="ipc:///tmp/test-speco-target.sock",
        _execute_method=_execute_method,
    )

    with pytest.raises(RuntimeError, match="mixed revision"):
        asyncio.run(
            speco_vllm_update_draft_weights(
                adapter,
                [("fc.weight", torch.zeros(1))],
                global_steps=10,
            )
        )

    assert [name for name, *_ in events] == ["abort_all_requests"]


@pytest.mark.parametrize(
    ("received_weights", "draft_model", "message"),
    [
        ([], SimpleNamespace(model=object()), "received zero tensors"),
        (
            [("fc.weight", torch.zeros(1))],
            None,
            "speculative model is unavailable",
        ),
        (
            [("fc.weight", torch.zeros(1))],
            SimpleNamespace(),
            "has no inner model",
        ),
    ],
)
def test_vllm_draft_ipc_update_fails_closed_before_revision_commit(
    monkeypatch, received_weights, draft_model, message
) -> None:
    class _Receiver:
        def __init__(self, **kwargs):
            del kwargs

        def receive_weights(self, on_bucket_received):
            if received_weights:
                on_bucket_received(received_weights, True)

    bucket_module = types.ModuleType(
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"
    )
    bucket_module.BucketedWeightReceiver = _Receiver
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer",
        bucket_module,
    )
    platform_module = types.ModuleType("vllm.platforms")
    platform_module.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform_module)
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.patch_verl_bucketed_weight_transfer_rebuild_ipc",
        lambda: False,
    )
    monkeypatch.setattr(
        "verl_speco.integration.vllm_runtime.patch_verl_bucketed_weight_transfer_shm_reuse",
        lambda: False,
    )

    extension = SpecoVLLMColocateWorkerExtension()
    extension.device = torch.device("cpu")
    extension.local_rank = 0
    extension._speco_resolve_draft_model = lambda: (draft_model, None)
    extension._speco_draft_method = lambda: "dspark"

    with pytest.raises(RuntimeError, match=message):
        extension.update_draft_weights_from_ipc(use_shm=False)

    assert getattr(extension, "_speco_draft_runtime_revision", 0) == 0
