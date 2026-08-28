# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock, call

import pytest

from verl_speco.trainer.scheduler import (
    AfterActorUpdateContext,
    AfterWeightUpdateContext,
    BeforeActorUpdateContext,
    DrafterExecutionStrategy,
    DrafterRuntimeState,
    DrafterScheduleConfig,
    DrafterScheduleContext,
    DrafterScheduler,
    PublishOutcome,
    PublishPlan,
    TrainingOutcome,
    TrainingPlan,
)
from verl_speco.trainer.scheduler.execution_strategy import ExecutionOutcome


def _training_plan(*, launch: bool = True) -> TrainingPlan:
    return TrainingPlan(
        launch=launch,
        reason="training_ready" if launch else "interval_not_reached",
        interval_matched=launch,
        execution_strategy=DrafterExecutionStrategy.SYNC,
        source_global_step=4,
        max_batches=1 if launch else 0,
        publish_after_success=True,
    )


def _quality_plan() -> TrainingPlan:
    return replace(
        _training_plan(),
        plan_id="quality-plan-4",
        quality_gate_enabled=True,
        worker_snapshots={"0": {}, "1": {}},
    )


def _quality_worker_result(
    worker_id: str,
    *,
    approved: bool,
    candidate_pending: bool = True,
    optimizer_step: int = 8,
    proxy_delta: float = 0.02,
) -> dict[str, object]:
    return {
        "trained": True,
        "triggered": True,
        "worker_id": worker_id,
        "worker_incarnation": f"worker-{worker_id}",
        "plan_id": "quality-plan-4",
        "source_global_step": 4,
        "data_version": None,
        "target_version": None,
        "successful_steps": 1,
        "attempted_steps": 1,
        "optimizer_step": optimizer_step,
        "publish_snapshot_cached": 0,
        "is_publish_leader": worker_id == "0",
        "publish_approved": approved,
        "candidate_pending": candidate_pending,
        "dspark/top1_acc": 0.4 + (0.2 * int(worker_id)),
        "drafter/current_lr": 5e-6,
        "drafter/quality_gate_accept_length_proxy_delta": proxy_delta,
    }


def _quality_outcome(results: list[dict[str, object]]) -> TrainingOutcome:
    plan = _quality_plan()
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    runtime_state.mark_running()
    return TrainingOutcome.from_execution(
        ExecutionOutcome(raw_results=results, elapsed_sec=0.2),
        runtime_state=runtime_state,
        plan=plan,
    )


def test_before_actor_update_plans_then_prepares() -> None:
    scheduler = DrafterScheduler()
    plan = _training_plan()
    scheduler.prepare_training_plan = Mock(return_value=plan)
    scheduler.prepare_training_execution = Mock(
        return_value={"drafter/target_lm_head_synced": 1}
    )
    context = BeforeActorUpdateContext(
        schedule_context=DrafterScheduleContext(
            global_step=4,
            training_mode="online",
            collected_samples_this_step=1,
            oldlogprob_collection_requested=False,
        ),
        config=DrafterScheduleConfig(),
    )

    outcome = scheduler.on_before_actor_update(context)

    assert outcome.training_plan is plan
    assert outcome.metrics["drafter/target_lm_head_synced"] == 1
    scheduler.prepare_training_plan.assert_called_once()
    scheduler.prepare_training_execution.assert_called_once_with(plan)


def test_after_actor_update_executes_prepared_plan() -> None:
    scheduler = DrafterScheduler()
    execution = ExecutionOutcome(
        raw_results=[
            {
                "trained": True,
                "successful_steps": 1,
                "attempted_steps": 1,
            }
        ],
        elapsed_sec=0.5,
    )
    scheduler.execute_training_plan = Mock(return_value=execution)
    plan = _training_plan()
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    runtime_state.mark_running()

    outcome = scheduler.on_after_actor_update(
        AfterActorUpdateContext(plan, runtime_state)
    )

    assert outcome.training_execution.trained
    assert outcome.training_execution.successful_steps == 1
    scheduler.execute_training_plan.assert_called_once_with(
        plan, runtime_state=runtime_state
    )


def test_after_actor_update_skips_inactive_plan() -> None:
    scheduler = DrafterScheduler()
    scheduler.execute_training_plan = Mock()

    outcome = scheduler.on_after_actor_update(
        AfterActorUpdateContext(_training_plan(launch=False), DrafterRuntimeState())
    )

    assert outcome.training_execution is None
    scheduler.execute_training_plan.assert_not_called()


def test_after_actor_update_normalizes_preflight_failure_and_resets_runtime() -> None:
    scheduler = DrafterScheduler()
    plan = _training_plan()
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[{"ready": False, "reason": "buffer_version_changed"}],
            elapsed_sec=0.1,
            launched=False,
            reason="worker_preflight_failed",
        )
    )

    outcome = scheduler.on_after_actor_update(
        AfterActorUpdateContext(plan, runtime_state)
    )

    assert not outcome.training_execution.trained
    assert outcome.training_execution.reason == "worker_preflight_failed"
    assert runtime_state.status.name == "IDLE"


def test_after_actor_update_rejects_inconsistent_worker_training_results() -> None:
    scheduler = DrafterScheduler()
    plan = replace(
        _training_plan(),
        plan_id="plan-4",
        worker_snapshots={"0": {}, "1": {}},
    )
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    runtime_state.mark_running()

    def result(worker_id: str, optimizer_step: int):
        return {
            "trained": True,
            "triggered": True,
            "worker_id": worker_id,
            "worker_incarnation": f"worker-{worker_id}",
            "plan_id": "plan-4",
            "source_global_step": 4,
            "data_version": None,
            "target_version": None,
            "successful_steps": 1,
            "attempted_steps": 1,
            "optimizer_step": optimizer_step,
            "publish_snapshot_cached": 1,
            "is_publish_leader": worker_id == "0",
        }

    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[result("0", 8), result("1", 9)],
            elapsed_sec=0.2,
        )
    )

    outcome = scheduler.on_after_actor_update(
        AfterActorUpdateContext(plan, runtime_state)
    )

    assert not outcome.training_execution.trained
    assert outcome.training_execution.reason == "worker_result_inconsistent"
    assert outcome.metrics["drafter/train_worker_results_consistent"] == 0
    assert runtime_state.status.name == "IDLE"


def test_after_actor_update_requires_snapshot_only_from_publish_leader() -> None:
    scheduler = DrafterScheduler()
    plan = replace(
        _training_plan(),
        plan_id="plan-4",
        worker_snapshots={"0": {}, "1": {}},
    )
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    runtime_state.mark_running()

    def result(worker_id: str, *, is_publish_leader: bool, snapshot_ready: bool):
        return {
            "trained": True,
            "triggered": True,
            "worker_id": worker_id,
            "worker_incarnation": f"worker-{worker_id}",
            "plan_id": "plan-4",
            "source_global_step": 4,
            "data_version": None,
            "target_version": None,
            "successful_steps": 1,
            "attempted_steps": 1,
            "optimizer_step": 8,
            "publish_snapshot_cached": int(snapshot_ready),
            "is_publish_leader": is_publish_leader,
        }

    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[
                result("0", is_publish_leader=True, snapshot_ready=True),
                result("1", is_publish_leader=False, snapshot_ready=False),
            ],
            elapsed_sec=0.2,
        )
    )

    outcome = scheduler.on_after_actor_update(
        AfterActorUpdateContext(plan, runtime_state)
    )

    assert outcome.training_execution.trained
    assert outcome.metrics["drafter/train_worker_results_consistent"] == 1
    assert outcome.metrics["drafter/train_publish_leader_count"] == 1
    assert outcome.metrics["drafter/train_publish_leader_snapshot_ready"] == 1


def test_quality_gate_all_workers_approve_and_metrics_are_aggregated() -> None:
    outcome = _quality_outcome(
        [
            _quality_worker_result("0", approved=True, proxy_delta=0.02),
            _quality_worker_result("1", approved=True, proxy_delta=0.04),
        ]
    )

    assert outcome.trained
    assert outcome.publish_approved
    assert not outcome.candidate_finalized
    assert outcome.metrics["drafter/train_publish_votes_consistent"] == 1
    assert outcome.metrics["drafter/publish_approved"] == 1
    assert outcome.metrics["dspark/top1_acc"] == 0.5
    assert outcome.metrics[
        "drafter/quality_gate_accept_length_proxy_delta"
    ] == 0.03


def test_quality_gate_single_worker_rejection_keeps_training_fact_and_blocks_publish() -> None:
    outcome = _quality_outcome(
        [
            _quality_worker_result("0", approved=True),
            _quality_worker_result("1", approved=False),
        ]
    )

    assert outcome.trained
    assert not outcome.publish_approved
    assert outcome.metrics["drafter/train_publish_votes_consistent"] == 0
    assert outcome.metrics["drafter/quality_gate_rejected"] == 1


def test_quality_gate_consistent_rejection_rolls_back_without_hiding_optimizer_work() -> None:
    outcome = _quality_outcome(
        [
            _quality_worker_result("0", approved=False),
            _quality_worker_result("1", approved=False),
        ]
    )

    assert outcome.trained
    assert not outcome.publish_approved
    assert outcome.metrics["drafter/train_publish_votes_consistent"] == 1
    assert outcome.metrics["drafter/quality_gate_rejected"] == 1


def test_quality_gate_commit_requires_one_leader_snapshot() -> None:
    plan = _quality_plan()
    outcome = _quality_outcome(
        [
            _quality_worker_result("0", approved=True),
            _quality_worker_result("1", approved=True),
        ]
    )
    finalization = [
        {
            "participating": True,
            "finalized": True,
            "committed": True,
            "rolled_back": False,
            "worker_id": worker_id,
            "plan_id": plan.plan_id,
            "is_publish_leader": worker_id == "0",
            "publish_snapshot_cached": worker_id == "0",
        }
        for worker_id in ("0", "1")
    ]

    committed = outcome.with_candidate_finalization(
        finalization,
        plan=plan,
        commit=True,
    )

    assert committed.candidate_finalized
    assert committed.publish_approved
    assert committed.reason == "candidate_committed"
    assert committed.metrics["drafter/candidate_committed"] == 1
    assert committed.metrics["drafter/train_publish_leader_count"] == 1
    assert committed.metrics["drafter/train_publish_leader_snapshot_ready"] == 1

    finalization[0]["publish_snapshot_cached"] = False
    with pytest.raises(RuntimeError, match="not consistent across workers"):
        outcome.with_candidate_finalization(
            finalization,
            plan=plan,
            commit=True,
        )


def test_quality_gate_rollback_finalizes_every_worker_without_snapshot() -> None:
    plan = _quality_plan()
    outcome = _quality_outcome(
        [
            _quality_worker_result("0", approved=False),
            _quality_worker_result("1", approved=False),
        ]
    )
    finalization = [
        {
            "participating": True,
            "finalized": True,
            "committed": False,
            "rolled_back": True,
            "worker_id": worker_id,
            "plan_id": plan.plan_id,
            "is_publish_leader": worker_id == "0",
            "publish_snapshot_cached": False,
        }
        for worker_id in ("0", "1")
    ]

    rolled_back = outcome.with_candidate_finalization(
        finalization,
        plan=plan,
        commit=False,
    )

    assert rolled_back.candidate_finalized
    assert not rolled_back.publish_approved
    assert rolled_back.reason == "candidate_rolled_back"
    assert rolled_back.metrics["drafter/candidate_rolled_back"] == 1


def test_scheduler_commits_only_after_all_workers_approve() -> None:
    plan = _quality_plan()
    executor = Mock()
    executor.finalize_training_candidate.return_value = [
        {
            "participating": True,
            "finalized": True,
            "committed": True,
            "rolled_back": False,
            "worker_id": worker_id,
            "plan_id": plan.plan_id,
            "is_publish_leader": worker_id == "0",
            "publish_snapshot_cached": worker_id == "0",
        }
        for worker_id in ("0", "1")
    ]
    scheduler = DrafterScheduler(worker_executor=executor)
    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[
                _quality_worker_result("0", approved=True),
                _quality_worker_result("1", approved=True),
            ],
            elapsed_sec=0.2,
        )
    )
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    runtime_state.mark_running()

    event = scheduler.on_after_actor_update(AfterActorUpdateContext(plan, runtime_state))

    executor.finalize_training_candidate.assert_called_once_with(plan, commit=True)
    assert event.training_execution.candidate_finalized
    assert event.training_execution.publish_approved


def test_scheduler_rolls_back_every_worker_when_one_vote_rejects() -> None:
    plan = _quality_plan()
    executor = Mock()
    executor.finalize_training_candidate.return_value = [
        {
            "participating": True,
            "finalized": True,
            "committed": False,
            "rolled_back": True,
            "worker_id": worker_id,
            "plan_id": plan.plan_id,
            "is_publish_leader": worker_id == "0",
            "publish_snapshot_cached": False,
        }
        for worker_id in ("0", "1")
    ]
    scheduler = DrafterScheduler(worker_executor=executor)
    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[
                _quality_worker_result("0", approved=True),
                _quality_worker_result("1", approved=False),
            ],
            elapsed_sec=0.2,
        )
    )
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    runtime_state.mark_running()

    event = scheduler.on_after_actor_update(AfterActorUpdateContext(plan, runtime_state))

    executor.finalize_training_candidate.assert_called_once_with(plan, commit=False)
    assert event.training_execution.trained
    assert not event.training_execution.publish_approved
    assert event.training_execution.reason == "candidate_rolled_back"


def test_gate_disabled_training_still_uses_transactional_candidate_finalize() -> None:
    plan = replace(_quality_plan(), quality_gate_enabled=False)
    executor = Mock()
    executor.finalize_training_candidate.return_value = [
        {
            "participating": True,
            "finalized": True,
            "committed": True,
            "rolled_back": False,
            "worker_id": worker_id,
            "plan_id": plan.plan_id,
            "is_publish_leader": worker_id == "0",
            "publish_snapshot_cached": worker_id == "0",
        }
        for worker_id in ("0", "1")
    ]
    scheduler = DrafterScheduler(worker_executor=executor)
    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[
                _quality_worker_result("0", approved=True),
                _quality_worker_result("1", approved=True),
            ],
            elapsed_sec=0.2,
        )
    )
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    runtime_state.mark_running()

    event = scheduler.on_after_actor_update(AfterActorUpdateContext(plan, runtime_state))

    executor.finalize_training_candidate.assert_called_once_with(plan, commit=True)
    assert event.training_execution.candidate_finalized
    assert event.training_execution.publish_approved
    assert not scheduler.quality_gate_frozen


def test_candidate_commit_failure_triggers_best_effort_global_rollback() -> None:
    plan = _quality_plan()
    rollback_results = [
        {
            "participating": True,
            "finalized": True,
            "committed": False,
            "rolled_back": True,
            "worker_id": worker_id,
            "plan_id": plan.plan_id,
        }
        for worker_id in ("0", "1")
    ]
    executor = Mock()
    executor.finalize_training_candidate.side_effect = [
        RuntimeError("snapshot gather failed"),
        rollback_results,
    ]
    scheduler = DrafterScheduler(worker_executor=executor)
    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[
                _quality_worker_result("0", approved=True),
                _quality_worker_result("1", approved=True),
            ],
            elapsed_sec=0.2,
        )
    )
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    runtime_state.mark_running()

    with pytest.raises(RuntimeError, match="snapshot gather failed"):
        scheduler.on_after_actor_update(AfterActorUpdateContext(plan, runtime_state))

    assert executor.finalize_training_candidate.call_args_list == [
        call(plan, commit=True),
        call(plan, commit=False),
    ]


def test_after_actor_update_allows_target_version_when_not_required() -> None:
    scheduler = DrafterScheduler()
    plan = replace(
        _training_plan(),
        plan_id="plan-4",
        required_target_version=None,
        worker_snapshots={"0": {}, "1": {}},
    )
    runtime_state = DrafterRuntimeState()
    runtime_state.submit(plan, started_at=0.0)
    runtime_state.mark_running()

    def result(worker_id: str):
        return {
            "trained": True,
            "triggered": True,
            "worker_id": worker_id,
            "worker_incarnation": f"worker-{worker_id}",
            "plan_id": "plan-4",
            "source_global_step": 4,
            "data_version": None,
            "target_version": 4,
            "successful_steps": 1,
            "attempted_steps": 1,
            "optimizer_step": 8,
            "publish_snapshot_cached": int(worker_id == "0"),
            "is_publish_leader": worker_id == "0",
        }

    scheduler.execute_training_plan = Mock(
        return_value=ExecutionOutcome(
            raw_results=[result("0"), result("1")],
            elapsed_sec=0.2,
        )
    )

    outcome = scheduler.on_after_actor_update(
        AfterActorUpdateContext(plan, runtime_state)
    )

    assert outcome.training_execution.trained
    assert outcome.training_execution.reason == "completed"
    assert outcome.metrics["drafter/train_target_versions_consistent"] == 1


def test_after_weight_update_plans_then_publishes() -> None:
    scheduler = DrafterScheduler()
    publish_plan = PublishPlan(
        publish=True,
        reason="publish_interval_reached",
        interval_matched=True,
        source_global_step=4,
    )
    publish_outcome = PublishOutcome(attempted=True, published=True)
    scheduler.plan_publish = Mock(return_value=publish_plan)
    scheduler.execute_publish_plan = Mock(return_value=publish_outcome)
    training_plan = _training_plan()

    outcome = scheduler.on_after_weight_update(
        AfterWeightUpdateContext(
            global_step=4,
            drafter_trained=True,
            config=DrafterScheduleConfig(),
            training_plan=training_plan,
        )
    )

    assert outcome.publish_plan is publish_plan
    assert outcome.publish_outcome is publish_outcome
    assert outcome.metrics["drafter/published"] == 1
    scheduler.execute_publish_plan.assert_called_once_with(publish_plan)
