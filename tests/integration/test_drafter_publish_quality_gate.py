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

import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
base_trainer = pytest.importorskip(
    "verl_speco.trainer.base_trainer",
    reason="drafter quality gate needs the trainer dependency stack",
)
speco_worker = pytest.importorskip(
    "verl_speco.workers.speco_worker",
    reason="drafter quality gate patience needs the worker dependency stack",
)

DrafterBaseTrainer = base_trainer.DrafterBaseTrainer
SpecoWorker = speco_worker.SpecoWorker


def _training_config(**values):
    return SimpleNamespace(
        rollout=SimpleNamespace(
            drafter=SimpleNamespace(training=values),
        )
    )


def test_accept_length_proxy_uses_prefix_survival_probability() -> None:
    correct = torch.tensor([8.0, 6.0, 5.0])
    counts = torch.tensor([10.0, 10.0, 10.0])

    proxy, front_accuracy = DrafterBaseTrainer._quality_gate_accept_length_proxy(
        correct, counts
    )

    assert front_accuracy == pytest.approx(0.8)
    assert proxy == pytest.approx(0.8 + (0.8 * 0.6) + (0.8 * 0.6 * 0.5))


def test_quality_gate_rollback_restores_model_optimizer_scheduler_and_counters() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.model = torch.nn.Linear(3, 2)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.02)
    trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer, step_size=1, gamma=0.5
    )
    trainer.training_steps = 7
    trainer.optimizer_steps_total = 11

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model(torch.ones(2, 3)).sum().backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer._snapshot_quality_gate_state()

    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    first_parameter = next(trainer.model.parameters())
    expected_exp_avg = trainer.optimizer.state[first_parameter]["exp_avg"].clone()
    expected_lr = trainer.optimizer.param_groups[0]["lr"]

    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.model(torch.full((2, 3), 3.0)).sum().backward()
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.training_steps += 1
    trainer.optimizer_steps_total += 1

    trainer._restore_quality_gate_state()

    for name, parameter in trainer.model.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name])
    torch.testing.assert_close(
        trainer.optimizer.state[first_parameter]["exp_avg"], expected_exp_avg
    )
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    assert trainer.training_steps == 7
    assert trainer.optimizer_steps_total == 11


def test_zero_hard_sample_ratio_is_the_uniform_random_path() -> None:
    trainer = DrafterBaseTrainer.__new__(DrafterBaseTrainer)
    trainer.backend = SimpleNamespace(model_type="dspark")
    trainer.config = _training_config(dspark_hard_sample_ratio=0.0)
    available = [
        {"index": index, "_verl_accept_len": float(index)} for index in range(10)
    ]

    selected = trainer._sample_training_items(available, 4, random.Random(1234))
    expected = random.Random(1234).sample(available, 4)

    assert selected == expected


def test_quality_gate_patience_freezes_after_three_rejections() -> None:
    worker = SpecoWorker.__new__(SpecoWorker)
    worker.config = _training_config(
        publish_quality_gate_rejection_patience=3,
        publish_quality_gate_plateau_patience=4,
        publish_quality_gate_meaningful_proxy_delta=0.01,
    )
    worker._quality_gate_consecutive_rejections = 0
    worker._quality_gate_plateau_updates = 0
    worker._quality_gate_frozen = False

    for _ in range(3):
        worker._update_quality_gate_patience(
            {"enabled": True, "approved": False}, trained=True
        )

    assert worker._quality_gate_consecutive_rejections == 3
    assert worker._quality_gate_frozen is True


def test_quality_gate_patience_freezes_after_four_plateau_updates() -> None:
    worker = SpecoWorker.__new__(SpecoWorker)
    worker.config = _training_config(
        publish_quality_gate_rejection_patience=3,
        publish_quality_gate_plateau_patience=4,
        publish_quality_gate_meaningful_proxy_delta=0.01,
    )
    worker._quality_gate_consecutive_rejections = 0
    worker._quality_gate_plateau_updates = 0
    worker._quality_gate_frozen = False

    for _ in range(4):
        worker._update_quality_gate_patience(
            {"enabled": True, "approved": True, "proxy_delta": 0.005},
            trained=True,
        )

    assert worker._quality_gate_plateau_updates == 4
    assert worker._quality_gate_frozen is True
