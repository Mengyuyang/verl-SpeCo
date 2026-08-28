# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("torch")

from verl_speco.workers.speco_worker import SpecoWorker


class _FakeTrainer:
    def __init__(self, *, data_version: int) -> None:
        self.buffer_version = 3
        self._target_lm_head_weight_step = 4
        self.optimizer_steps_total = 0
        self.data_version = data_version
        self.activation_calls = 0
        self.status_kwargs = []

    def get_training_data_status(self, **kwargs):
        self.status_kwargs.append(kwargs)
        return {
            "trainable_batches": 1,
            "trainable_samples": 4,
            "data_version": self.data_version,
        }

    def get_publish_quality_gate_preflight_status(self, _training_plan):
        return {
            "enabled": False,
            "ready": True,
            "candidate_samples": 4,
            "holdout_samples": 0,
            "training_samples": 4,
        }

    async def activate_training_model(self) -> bool:
        self.activation_calls += 1
        return True


def _worker(*, data_version: int) -> SpecoWorker:
    worker = SpecoWorker.__new__(SpecoWorker)
    worker.enable_drafter = True
    worker.in_drafter_train_group = True
    worker._rank = 0
    worker.worker_incarnation = "worker-0"
    worker.last_global_step = 4
    worker.device_name = "cpu"
    worker.trainer = _FakeTrainer(data_version=data_version)
    worker._prepared_training_plan_id = None
    worker._prepared_training_data_version = None
    worker._prepared_training_target_version = None
    worker._pending_training_candidate_plan_id = None
    worker._pending_training_candidate_step = None
    worker._pending_training_candidate_successful_steps = 0
    worker._pending_training_candidate_publish = False
    return worker


def _plan() -> dict[str, object]:
    return {
        "launch": True,
        "execution_strategy": "sync",
        "source_global_step": 4,
        "plan_id": "plan-4",
        "data_version": 4,
        "required_target_version": 4,
        "sample_last_n_steps": 2,
        "require_full_batch": False,
        "min_sample_step": 2,
        "max_sample_step": 4,
        "data_filter_reason": "recent_buffer_window",
        "min_batches": 1,
        "worker_snapshots": {
            "0": {
                "worker_incarnation": "worker-0",
                "buffer_version": 3,
                "data_version": 4,
            }
        },
    }


def test_worker_preflight_rejects_changed_data_version_before_activation() -> None:
    worker = _worker(data_version=5)

    result = asyncio.run(worker.preflight_drafter_training(_plan()))

    assert not result["ready"]
    assert result["reason"] == "data_version_changed"
    assert result["data_version"] == 5
    assert worker.trainer.activation_calls == 0
    assert worker._prepared_training_plan_id is None


def test_worker_preflight_records_actual_versions_for_training_result() -> None:
    worker = _worker(data_version=4)

    result = asyncio.run(worker.preflight_drafter_training(_plan()))

    assert result["ready"]
    assert result["data_version"] == 4
    assert result["target_version"] == 4
    assert worker._prepared_training_plan_id == "plan-4"
    assert worker._prepared_training_data_version == 4
    assert worker._prepared_training_target_version == 4
    assert worker.trainer.status_kwargs[-1]["min_sample_step"] == 2
    assert worker.trainer.status_kwargs[-1]["max_sample_step"] == 4


def test_worker_preflight_rejects_quality_gate_without_shared_holdout_before_activation() -> None:
    worker = _worker(data_version=4)
    worker.trainer.get_publish_quality_gate_preflight_status = lambda _plan: {
        "enabled": True,
        "ready": False,
        "reason": "insufficient_quality_gate_samples",
        "candidate_samples": 1,
        "holdout_samples": 1,
        "training_samples": 0,
    }

    result = asyncio.run(worker.preflight_drafter_training(_plan()))

    assert not result["ready"]
    assert result["reason"] == "insufficient_quality_gate_samples"
    assert result["quality_gate_enabled"] is True
    assert result["quality_gate_training_samples"] == 0
    assert worker.trainer.activation_calls == 0
    assert worker._prepared_training_plan_id is None


class _CandidateTrainer:
    def __init__(self, *, snapshot_cached: bool) -> None:
        self.optimizer_steps_total = 9
        self.snapshot_cached = snapshot_cached
        self.finalize_calls = []
        self.snapshot_calls = []
        self.clear_calls = 0
        self.cleanup_calls = []

    def finalize_publish_quality_gate(self, *, commit: bool) -> None:
        self.finalize_calls.append(commit)

    def prepare_model_state_dict_for_publish(self, step: int, *, cache: bool) -> bool:
        self.snapshot_calls.append((step, cache))
        return bool(cache and self.snapshot_cached)

    def clear_pending_publish_state_dict(self) -> None:
        self.clear_calls += 1

    async def cleanup_training(self, *, clear_data: bool) -> None:
        self.cleanup_calls.append(clear_data)


def _candidate_worker(*, leader: bool) -> SpecoWorker:
    worker = SpecoWorker.__new__(SpecoWorker)
    worker.enable_drafter = True
    worker.in_drafter_train_group = True
    worker._rank = 0 if leader else 1
    worker.worker_incarnation = f"worker-{worker.rank}"
    worker.is_global_publish_leader = leader
    worker.device_name = "cpu"
    worker.trainer = _CandidateTrainer(snapshot_cached=leader)
    worker.last_trained_step = None
    worker._pending_training_candidate_plan_id = "plan-4"
    worker._pending_training_candidate_step = 4
    worker._pending_training_candidate_successful_steps = 3
    worker._pending_training_candidate_publish = True
    return worker


def test_candidate_commit_collectively_builds_snapshot_but_only_leader_caches() -> None:
    leader = _candidate_worker(leader=True)
    follower = _candidate_worker(leader=False)

    leader_result = asyncio.run(
        leader.finalize_drafter_training_candidate("plan-4", commit=True)
    )
    follower_result = asyncio.run(
        follower.finalize_drafter_training_candidate("plan-4", commit=True)
    )

    assert leader_result["committed"]
    assert leader_result["publish_snapshot_cached"] == 1
    assert leader.trainer.snapshot_calls == [(4, True)]
    assert follower_result["committed"]
    assert follower_result["publish_snapshot_cached"] == 0
    # FSDP rank0_only state-dict export is collective, so every training rank
    # must enter even though only the global leader keeps a non-empty snapshot.
    assert follower.trainer.snapshot_calls == [(4, False)]
    assert leader.last_trained_step == 4
    assert follower.last_trained_step == 4
    assert leader.trainer.cleanup_calls == [True]
    assert follower.trainer.cleanup_calls == [True]


def test_candidate_rollback_is_uniform_and_does_not_create_publish_snapshot() -> None:
    leader = _candidate_worker(leader=True)

    result = asyncio.run(
        leader.finalize_drafter_training_candidate("plan-4", commit=False)
    )

    assert result["rolled_back"]
    assert not result["committed"]
    assert result["publish_snapshot_cached"] == 0
    assert leader.trainer.finalize_calls == [False]
    assert leader.trainer.snapshot_calls == []
    assert leader.trainer.clear_calls == 1
    assert leader.last_trained_step is None
    assert leader.trainer.cleanup_calls == [False]


def test_candidate_finalize_rejects_a_stale_plan_without_mutating_state() -> None:
    worker = _candidate_worker(leader=True)

    result = asyncio.run(
        worker.finalize_drafter_training_candidate("stale-plan", commit=True)
    )

    assert result["participating"]
    assert not result["finalized"]
    assert result["reason"] == "candidate_plan_mismatch"
    assert worker.trainer.finalize_calls == []
    assert worker._pending_training_candidate_plan_id == "plan-4"
