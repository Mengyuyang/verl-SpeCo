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

from types import SimpleNamespace
from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")
base_trainer = pytest.importorskip(
    "verl_speco.trainer.base_trainer",
    reason="drafter quality gate needs the trainer dependency stack",
)

DrafterBaseTrainer = base_trainer.DrafterBaseTrainer


def _training_config(**values):
    return SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training=values),
        )
    )


def _trainer_with_optimizer() -> DrafterBaseTrainer:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.model = torch.nn.Linear(3, 2)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.02)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=1, gamma=0.5
    )
    trainer.training_steps = 7
    trainer.optimizer_steps_total = 11
    trainer._quality_gate_holdout_items = []
    trainer._quality_gate_holdout_item_ids = set()
    trainer._quality_gate_model_snapshot = None
    trainer._quality_gate_optimizer_snapshot = None
    trainer._quality_gate_optimizer_group_snapshot = None
    trainer._quality_gate_scheduler_snapshot = None
    trainer._quality_gate_counter_snapshot = None
    trainer._quality_gate_before_metrics = None
    return trainer


def _optimizer_step(trainer: DrafterBaseTrainer, value: float) -> None:
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model(torch.full((2, 3), value)).sum().backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()


def test_accept_length_proxy_uses_prefix_survival_probability() -> None:
    correct = torch.tensor([8.0, 6.0, 5.0])
    counts = torch.tensor([10.0, 10.0, 10.0])

    proxy, front_accuracy = DrafterBaseTrainer._quality_gate_accept_length_proxy(
        correct, counts
    )

    assert front_accuracy == pytest.approx(0.8)
    assert proxy == pytest.approx(0.8 + (0.8 * 0.6) + (0.8 * 0.6 * 0.5))


def test_quality_gate_rollback_restores_model_optimizer_scheduler_and_counters() -> None:
    trainer = _trainer_with_optimizer()
    _optimizer_step(trainer, 1.0)
    trainer._snapshot_quality_gate_state()

    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    first_parameter = next(trainer.model.parameters())
    expected_exp_avg = trainer.optimizer.state[first_parameter]["exp_avg"].clone()
    expected_lr = trainer.optimizer.param_groups[0]["lr"]

    _optimizer_step(trainer, 3.0)
    trainer.training_steps += 1
    trainer.optimizer_steps_total += 1

    trainer.finalize_publish_quality_gate(commit=False)

    for name, parameter in trainer.model.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name])
    torch.testing.assert_close(
        trainer.optimizer.state[first_parameter]["exp_avg"], expected_exp_avg
    )
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert trainer.training_steps == 7
    assert trainer.optimizer_steps_total == 11
    assert trainer._quality_gate_model_snapshot is None


def test_quality_gate_commit_preserves_candidate_and_releases_rollback_snapshot() -> None:
    trainer = _trainer_with_optimizer()
    _optimizer_step(trainer, 1.0)
    trainer._snapshot_quality_gate_state()
    _optimizer_step(trainer, 3.0)
    candidate = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    trainer.finalize_publish_quality_gate(commit=True)

    for name, parameter in trainer.model.named_parameters():
        torch.testing.assert_close(parameter, candidate[name])
    assert trainer._quality_gate_model_snapshot is None
    assert trainer._quality_gate_optimizer_snapshot is None


def test_training_plan_quality_gate_flag_overrides_static_disabled_default() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.config = _training_config(publish_quality_gate_enable=False)

    assert not trainer._publish_quality_gate_enabled()
    assert trainer._publish_quality_gate_enabled({"quality_gate_enabled": True})


@pytest.mark.parametrize("model_type", ["eagle3", "dflash", "dspark", "domino"])
def test_hidden_state_block_training_status_requires_current_actor_step(
    model_type: str,
) -> None:
    class Buffer:
        requested_windows: ClassVar[list[int]] = []

        @staticmethod
        def get_all_data():
            return [{"step": 5}, {"step": 7}]

        @classmethod
        def get_data_from_last_n_steps(cls, steps):
            cls.requested_windows.append(steps)
            return [{"step": 7}] if steps == 0 else [{"step": 5}, {"step": 7}]

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.current_rl_step = 7
    trainer.config = _training_config(use_logits=False)
    trainer.backend = SimpleNamespace(model_type=model_type)
    trainer.collected_data = [{"step": 7}]
    trainer.use_data_buffer = True
    trainer.data_buffer = Buffer()
    trainer.batch_size = 1
    trainer.buffer_version = 3

    status = trainer.get_training_data_status(sample_last_n_steps=20)

    assert status["same_step_data_required"] is True
    assert status["min_sample_step"] == 7
    assert status["max_sample_step"] == 7
    assert status["oldest_sample_step"] == 7
    assert status["newest_sample_step"] == 7
    assert Buffer.requested_windows == [0]


def test_logits_training_can_reuse_recent_block_drafter_buffer() -> None:
    class Buffer:
        requested_windows: ClassVar[list[int]] = []

        @staticmethod
        def get_all_data():
            return [{"step": 5}, {"step": 7}]

        @classmethod
        def get_data_from_last_n_steps(cls, steps):
            cls.requested_windows.append(steps)
            return [{"step": 5}, {"step": 7}]

    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.current_rl_step = 7
    trainer.config = _training_config(use_logits=True)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.collected_data = [{"step": 7}]
    trainer.use_data_buffer = True
    trainer.data_buffer = Buffer()
    trainer.batch_size = 1
    trainer.buffer_version = 3

    status = trainer.get_training_data_status(sample_last_n_steps=2)

    assert status["same_step_data_required"] is False
    assert status["min_sample_step"] == 5
    assert status["max_sample_step"] == 7
    assert Buffer.requested_windows == [2]
