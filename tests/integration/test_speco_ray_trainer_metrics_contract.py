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

import math

import pytest


def test_rollout_actor_logprob_metrics_use_masked_log_space_quantiles() -> None:
    torch = pytest.importorskip("torch")
    module = pytest.importorskip(
        "verl_speco.trainer.speco_ray_trainer",
        reason="driver metric contract needs the verl/Ray dependency stack",
    )
    batch = {
        "rollout_log_probs": torch.tensor([[-10.0, -8.0, -4.0, -3.0]]),
        "old_log_probs": torch.tensor([[-7.0, -8.5, -2.0, -99.0]]),
        "response_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
    }

    metrics = module._speco_rollout_actor_logprob_metrics(batch)

    assert metrics["training/rollout_actor_abs_logprob_delta_mean"] == pytest.approx(
        (3.0 + 0.5 + 2.0) / 3.0
    )
    assert metrics["training/rollout_actor_abs_logprob_delta_max"] == pytest.approx(3.0)
    assert metrics[
        "training/rollout_actor_abs_logprob_delta_gt_1p0_fraction"
    ] == pytest.approx(2.0 / 3.0)
    assert metrics["training/rollout_actor_logprob_nonfinite_tokens"] == 0


def test_logprob_metric_attachment_does_not_mutate_policy_batch() -> None:
    torch = pytest.importorskip("torch")
    module = pytest.importorskip(
        "verl_speco.trainer.speco_ray_trainer",
        reason="driver metric contract needs the verl/Ray dependency stack",
    )

    class _FakeDataProto:
        def __init__(self, batch):
            self.batch = batch
            self.meta_info = {}

        def union(self, _other):
            raise AssertionError("diagnostics must not call mutating DataProto.union")

    source_batch = _FakeDataProto(
        {
            "rollout_log_probs": torch.tensor([[-2.0, -3.0]]),
            "response_mask": torch.ones((1, 2), dtype=torch.bool),
            "input_ids": torch.tensor([[1, 2]]),
        }
    )
    old_log_prob = _FakeDataProto(
        {
            "old_log_probs": torch.tensor([[-2.5, -2.0]]),
            "entropys": torch.tensor([[0.1, 0.2]]),
            "routed_experts": torch.tensor([[3, 4]]),
        }
    )
    source_keys_before = set(source_batch.batch)

    returned = module._speco_attach_rollout_actor_logprob_metrics(
        old_log_prob, source_batch
    )

    assert returned is old_log_prob
    assert set(source_batch.batch) == source_keys_before
    assert "old_log_probs" not in source_batch.batch
    assert "entropys" not in source_batch.batch
    assert "routed_experts" not in source_batch.batch
    assert old_log_prob.meta_info["metrics"][
        "training/rollout_actor_abs_logprob_delta_max"
    ] == pytest.approx(1.0)


def test_logprob_diagnostics_are_optional_when_rollout_uses_token_only_path() -> None:
    torch = pytest.importorskip("torch")
    module = pytest.importorskip(
        "verl_speco.trainer.speco_ray_trainer",
        reason="driver metric contract needs the verl/Ray dependency stack",
    )

    class _FakeDataProto:
        def __init__(self, batch):
            self.batch = batch
            self.meta_info = {}

    source_batch = _FakeDataProto(
        {
            "response_mask": torch.ones((1, 2), dtype=torch.bool),
            "input_ids": torch.tensor([[1, 2]]),
        }
    )
    old_log_prob = _FakeDataProto({"old_log_probs": torch.tensor([[-2.5, -2.0]])})

    returned = module._speco_attach_rollout_actor_logprob_metrics(
        old_log_prob, source_batch
    )

    assert returned is old_log_prob
    assert old_log_prob.meta_info == {}


def test_driver_exposes_finite_drafter_backend_metrics_without_rank_sum() -> None:
    module = pytest.importorskip(
        "verl_speco.trainer.speco_ray_trainer",
        reason="driver metric contract needs the verl/Ray dependency stack",
    )

    metrics = module._speco_aggregate_drafter_training_metrics(
        [
            {
                "dspark/confidence_loss": 0.8,
                "dspark/confidence_target_mean": 0.7,
                "drafter/current_lr": 1.0e-5,
                "drafter/optimizer_steps_total": 9,
                "unrelated": 99,
            },
            {
                "dspark/confidence_loss": 0.8,
                "dspark/confidence_target_mean": 0.7,
                "drafter/current_lr": 1.0e-5,
                "drafter/optimizer_steps_total": 10,
            },
            {
                "dspark/confidence_loss": math.nan,
                "drafter/current_lr": math.inf,
            },
        ]
    )

    assert metrics["dspark/confidence_loss"] == pytest.approx(0.8)
    assert metrics["dspark/confidence_target_mean"] == pytest.approx(0.7)
    assert metrics["drafter/current_lr"] == pytest.approx(1.0e-5)
    assert metrics["drafter/optimizer_steps_total"] == pytest.approx(9.5)
    assert metrics["drafter/metric_invalid_count"] == pytest.approx(2)
    assert metrics["drafter/metric_desync_count"] >= 2
    assert metrics[
        "drafter/worker_metric_min/drafter/optimizer_steps_total"
    ] == pytest.approx(9)
    assert metrics[
        "drafter/worker_metric_max/drafter/optimizer_steps_total"
    ] == pytest.approx(10)
    assert "unrelated" not in metrics
