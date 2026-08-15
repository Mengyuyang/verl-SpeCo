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
from types import SimpleNamespace

import pytest

from verl_speco.integration import rollout_publish


class _FakeObjectRef:
    pass


class _FakeRay:
    ObjectRef = _FakeObjectRef

    @staticmethod
    def get(value):
        return {"resolved": value}


def test_materialize_direct_and_object_ref_payloads(monkeypatch) -> None:
    monkeypatch.setattr(rollout_publish, "_ray_module", lambda: _FakeRay)
    direct = {"weight": 1}
    ref = _FakeObjectRef()

    assert rollout_publish.materialize_draft_weights_payload(direct) == (direct, False)
    assert rollout_publish.materialize_draft_weights_payload(ref) == (
        {"resolved": ref},
        True,
    )
    assert rollout_publish.materialize_draft_weights_payload({"weights_ref": ref}) == (
        {"resolved": ref},
        True,
    )


def test_rollout_backend_and_drafter_gates_support_both_config_shapes() -> None:
    assert rollout_publish.rollout_backend_name({"rollout": {"name": "vllm"}}) == "vllm"
    assert (
        rollout_publish.rollout_backend_name(
            {"actor_rollout_ref": {"rollout": {"name": "sglang"}}}
        )
        == "sglang"
    )
    assert rollout_publish.drafter_rollout_enabled(
        {"actor_rollout_ref": {"rollout": {"drafter": {"enable": True}}}}
    )
    assert not rollout_publish.drafter_rollout_enabled(
        {"actor_rollout_ref": {"rollout": {"drafter": {"enable": False}}}}
    )


def test_publish_state_filter_keeps_eagle3_trainable_lm_head() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="publish state filtering needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="eagle3")
    trainer.training_device_mesh = None
    trainer._frozen_param_names = ["target_model."]
    trainer.model = SimpleNamespace(
        state_dict=lambda: {
            "embed_tokens.weight": torch.ones(2, 2),
            "target_model.fc.weight": torch.ones(2, 2),
            "lm_head.weight": torch.ones(2, 2),
            "midlayer.fc.weight": torch.ones(2, 2),
            "t2d": torch.ones(2, dtype=torch.bool),
        }
    )

    assert set(trainer._get_trainable_state_dict()) == {
        "lm_head.weight",
        "midlayer.fc.weight",
    }


def test_publish_state_filter_skips_non_eagle_lm_head() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="publish state filtering needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dflash")
    trainer.training_device_mesh = None
    trainer._frozen_param_names = []
    trainer.model = SimpleNamespace(
        state_dict=lambda: {
            "lm_head.weight": torch.ones(2, 2),
            "draft_model.fc.weight": torch.ones(2, 2),
        }
    )

    assert set(trainer._get_trainable_state_dict()) == {"draft_model.fc.weight"}


def test_publish_state_filter_excludes_block_drafter_embedding() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="publish state filtering needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.training_device_mesh = None
    trainer._frozen_param_names = []
    trainer.model = SimpleNamespace(
        state_dict=lambda: {
            "draft_model.embed_tokens.weight": torch.ones(2, 2),
            "draft_model.fc.weight": torch.ones(2, 2),
            "draft_model.confidence_head.proj.weight": torch.ones(1, 4),
            "draft_model.confidence_head.proj.bias": torch.ones(1),
        }
    )

    assert set(trainer._get_trainable_state_dict()) == {
        "draft_model.fc.weight",
        "draft_model.confidence_head.proj.weight",
        "draft_model.confidence_head.proj.bias",
    }


@pytest.mark.parametrize("state_dict", [None, {}])
def test_publish_snapshot_rejects_empty_state_dict(state_dict) -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="publish snapshot state needs the trainer dependency stack",
    )
    trainer = base_trainer.DrafterBaseTrainer.__new__(
        base_trainer.DrafterBaseTrainer
    )
    trainer.rank = 0
    trainer._pending_publish_state_dict = {"stale": object()}
    trainer._pending_publish_step = 9
    trainer._pending_publish_ready = True
    trainer.get_model_state_dict = lambda: state_dict

    assert trainer.prepare_model_state_dict_for_publish(10) is False
    assert trainer.pop_model_state_dict_for_publish(10) == (False, None)
    assert trainer._pending_publish_state_dict is None
    assert trainer._pending_publish_step is None
    assert trainer._pending_publish_ready is False


def test_target_lm_head_device_helper_handles_dflash_style_backend() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="target lm_head device contract needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    class _FakeHead:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(device)
            return self

    head = _FakeHead()
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(target_lm_head=head)

    assert trainer._move_target_lm_head("cpu") is True
    assert head.devices == ["cpu"]


def test_target_lm_head_device_helper_preserves_eagle_backend() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="target model device contract needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    class _FakeHead:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(device)
            return self

    head = _FakeHead()
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(target_model=head, target_lm_head=None)

    assert trainer._move_target_lm_head("npu:0") is True
    assert head.devices == ["npu:0"]


@pytest.mark.parametrize(
    "release_method",
    ["release_training_memory_after_activation", "cleanup_training"],
)
def test_idle_drafter_lifecycle_offloads_dspark_target_lm_head(
    monkeypatch, release_method
) -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="target lm_head lifecycle contract needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    class _FakeHead:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(device)
            return self

    head = _FakeHead()
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.rank = 3
    trainer.backend = SimpleNamespace(
        model_type="dspark", target_model=None, target_lm_head=head
    )
    trainer.model = None
    trainer.optimizer = None
    trainer._training_active = True
    trainer._training_initialized = True
    monkeypatch.setattr(base_trainer, "device_name", "cpu")

    if release_method == "cleanup_training":
        trainer._pending_checkpoint_future = None
        trainer._pending_full_checkpoint_future = None
        trainer.skip_heavy_cleanup_after_drafter_training = False
        trainer._get_sp_group = lambda: None
        trainer._get_dp_group = lambda: None
        trainer.training_device_mesh = None
        trainer.collected_data = []
        trainer.data_buffer = []
        trainer._full_checkpoint_executor = None
        trainer._last_ckpt_step = 0
        trainer.training_steps = 1
        asyncio.run(trainer.cleanup_training(clear_data=True))
    else:
        asyncio.run(trainer.release_training_memory_after_activation())

    assert head.devices == ["cpu"]


def test_dspark_pretrained_export_strips_only_training_wrapper_prefix() -> None:
    torch = pytest.importorskip("torch")
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="checkpoint export needs the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.training_device_mesh = None
    trainer.model = SimpleNamespace(
        draft_model=SimpleNamespace(),
        state_dict=lambda: {
            "draft_model.fc.weight": torch.ones(2, 4),
            "draft_model.hidden_norm.weight": torch.ones(2),
            "draft_model.norm.weight": torch.ones(2),
            "draft_model.markov_head.markov_w1.weight": torch.ones(2, 2),
            "draft_model.confidence_head.proj.weight": torch.ones(1, 4),
            "draft_model.confidence_head.proj.bias": torch.ones(1),
        },
    )

    exported_state = trainer._get_pretrained_export_state_dict()

    assert set(exported_state) == {
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "markov_head.markov_w1.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    }


def test_dspark_training_metrics_report_confidence_loss() -> None:
    base_trainer = pytest.importorskip(
        "verl_speco.trainer.base_trainer",
        reason="confidence metrics need the trainer dependency stack",
    )
    DrafterBaseTrainer = base_trainer.DrafterBaseTrainer

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training={"dspark_block_size": 2})
        )
    )
    trainer._training_metric_sums = {
        "dspark/confidence_loss_sum": 3.0,
        "dspark/confidence_weighted_token_count": 2.0,
        "dspark/confidence_target_sum": 1.6,
        "dspark/confidence_prediction_sum": 1.4,
        "dspark/confidence_abs_error_sum": 0.4,
        "dspark/confidence_signed_error_sum": -0.2,
        "dspark/confidence_prefix_target_sum": 1.2,
        "dspark/confidence_prefix_prediction_sum": 1.0,
        "dspark/confidence_prefix_weight_sum": 2.0,
    }
    trainer._training_metric_steps = 1
    trainer.optimizer_steps_total = 0
    trainer.optimizer = None

    metrics = trainer.get_training_metrics()

    assert metrics["dspark/confidence_loss"] == pytest.approx(1.5)
    assert metrics["dspark/confidence_weighted_token_count"] == pytest.approx(2.0)
    assert metrics["dspark/confidence_target_mean"] == pytest.approx(0.8)
    assert metrics["dspark/confidence_prediction_mean"] == pytest.approx(0.7)
    assert metrics["dspark/confidence_mae"] == pytest.approx(0.2)
    assert metrics["dspark/confidence_bias"] == pytest.approx(-0.1)
    assert metrics["dspark/confidence_prefix_target_mean"] == pytest.approx(0.6)
    assert metrics["dspark/confidence_prefix_prediction_mean"] == pytest.approx(0.5)
    assert metrics["dspark/confidence_cumprod_bias"] == pytest.approx(-0.1)
