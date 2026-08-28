# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Normalized multi-worker outcome for one drafter training event."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verl_speco.trainer.scheduler.drafter_runtime_state import (
    DrafterRuntimeState,
    DrafterRuntimeStatus,
)
from verl_speco.trainer.scheduler.execution_strategy import ExecutionOutcome
from verl_speco.trainer.scheduler.schedule_types import (
    TrainingPlan,
    TrainingResult,
    _as_float,
    _as_int,
)


def _metric_float(value: object) -> float | None:
    try:
        return _as_float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class TrainingOutcome:
    trained: bool
    successful_steps: int
    worker_results: list[TrainingResult]
    raw_results: list[Any]
    elapsed_sec: float
    reason: str
    metrics: dict[str, float | int]
    publish_approved: bool = False
    candidate_finalized: bool = False

    @classmethod
    def from_execution(
        cls,
        execution: ExecutionOutcome,
        *,
        runtime_state: DrafterRuntimeState,
        plan: TrainingPlan,
    ) -> "TrainingOutcome":
        normalized_results: list[dict[str, object]] = []
        for result in execution.raw_results:
            if isinstance(result, dict):
                normalized_results.append(result)
            else:
                trained = bool(result)
                normalized_results.append(
                    {
                        "trained": trained,
                        "triggered": trained,
                        "attempted_steps": int(trained),
                        "successful_steps": int(trained),
                        "elapsed_sec": 0.0,
                        "reason": "legacy_bool_result",
                    }
                )

        trained = any(
            bool(result.get("trained", False)) for result in normalized_results
        )
        successful_steps = max(
            (
                _as_int(result.get("successful_steps", 0))
                for result in normalized_results
            ),
            default=0,
        )
        worker_results = [
            TrainingResult.from_mapping(result) for result in normalized_results
        ]
        participating_results = [
            TrainingResult.from_mapping(result)
            for result in normalized_results
            if bool(result.get("triggered", False))
        ]
        participating_mappings = [
            result
            for result in normalized_results
            if bool(result.get("triggered", False))
        ]
        candidate_protocol_used = bool(participating_mappings) and all(
            "candidate_pending" in result and "publish_approved" in result
            for result in participating_mappings
        )
        expected_worker_ids = set((plan.worker_snapshots or {}).keys())
        actual_worker_ids = {result.worker_id for result in participating_results}
        strict_consistency = bool(expected_worker_ids)
        worker_ids_consistent = actual_worker_ids == expected_worker_ids and len(
            participating_results
        ) == len(expected_worker_ids)
        incarnations_consistent = all(
            result.worker_incarnation for result in participating_results
        )
        plan_ids_consistent = all(
            result.plan_id == plan.plan_id for result in participating_results
        )
        source_steps_consistent = all(
            result.source_global_step == _as_int(plan.source_global_step)
            for result in participating_results
        )
        data_versions_consistent = all(
            result.data_version == plan.data_version for result in participating_results
        )
        target_versions_consistent = all(
            plan.required_target_version is None
            or result.target_version == plan.required_target_version
            for result in participating_results
        )
        trained_consistent = (
            len({result.trained for result in participating_results}) == 1
        )
        successful_steps_consistent = (
            len({result.successful_steps for result in participating_results}) == 1
        )
        optimizer_steps_consistent = (
            len({result.optimizer_step for result in participating_results}) == 1
        )
        candidate_pending_consistent = not candidate_protocol_used or all(
            result.candidate_pending == result.trained
            for result in participating_results
        )
        publish_votes_consistent = not candidate_protocol_used or (
            len({result.publish_approved for result in participating_results}) == 1
        )
        legacy_publish_leaders = [
            result for result in participating_results if result.is_publish_leader
        ]
        legacy_snapshot_consistent = (
            candidate_protocol_used
            or not (plan.publish_after_success and trained)
            or (
                len(legacy_publish_leaders) == 1
                and legacy_publish_leaders[0].snapshot_ready
            )
        )
        execution_consistent = not strict_consistency or (
            worker_ids_consistent
            and incarnations_consistent
            and plan_ids_consistent
            and source_steps_consistent
            and data_versions_consistent
            and target_versions_consistent
            and trained_consistent
            and successful_steps_consistent
            and optimizer_steps_consistent
        )
        quality_consistent = (
            candidate_pending_consistent
            and publish_votes_consistent
            and legacy_snapshot_consistent
        )
        result_consistent = execution_consistent and quality_consistent
        if not execution_consistent:
            trained = False
        if candidate_protocol_used:
            publish_approved = bool(
                trained
                and execution_consistent
                and quality_consistent
                and all(result.publish_approved for result in participating_results)
                and all(result.candidate_pending for result in participating_results)
            )
        else:
            publish_approved = bool(
                trained and execution_consistent and quality_consistent
            )
        metrics: dict[str, float | int] = {
            "drafter/trained": int(trained),
            "drafter/publish_approved": int(publish_approved),
            "drafter/candidate_pending": int(
                bool(participating_results)
                and all(result.candidate_pending for result in participating_results)
            ),
            "drafter/train_successful_steps_max": successful_steps,
            "drafter/train_no_trainable_batch": int(
                any(
                    result.get("reason") == "no_trainable_batch"
                    for result in normalized_results
                )
            ),
            "drafter/train_activation_failed": int(
                any(
                    result.get("reason") == "activation_failed"
                    for result in normalized_results
                )
            ),
            "drafter/train_attempted_batches_max": max(
                (result.attempted_batches for result in worker_results), default=0
            ),
            "drafter/train_buffer_size_before_min": min(
                (result.buffer_size_before for result in worker_results), default=0
            ),
            "drafter/train_buffer_size_after_min": min(
                (result.buffer_size_after for result in worker_results), default=0
            ),
            "drafter/train_optimizer_step_max": max(
                (result.optimizer_step for result in worker_results), default=0
            ),
            "drafter/train_worker_results_consistent": int(result_consistent),
            "drafter/train_worker_execution_consistent": int(execution_consistent),
            "drafter/train_worker_ids_consistent": int(worker_ids_consistent),
            "drafter/train_worker_incarnations_consistent": int(
                incarnations_consistent
            ),
            "drafter/train_plan_ids_consistent": int(plan_ids_consistent),
            "drafter/train_source_steps_consistent": int(source_steps_consistent),
            "drafter/train_data_versions_consistent": int(data_versions_consistent),
            "drafter/train_target_versions_consistent": int(target_versions_consistent),
            "drafter/train_successful_steps_consistent": int(
                successful_steps_consistent
            ),
            "drafter/train_optimizer_steps_consistent": int(optimizer_steps_consistent),
            "drafter/train_candidate_pending_consistent": int(
                candidate_pending_consistent
            ),
            "drafter/train_publish_votes_consistent": int(publish_votes_consistent),
            "drafter/train_publish_leader_count": len(legacy_publish_leaders),
            "drafter/train_publish_leader_snapshot_ready": int(
                len(legacy_publish_leaders) == 1
                and legacy_publish_leaders[0].snapshot_ready
            ),
            "drafter/quality_gate_rejected": int(
                trained and not publish_approved and plan.quality_gate_enabled
            ),
        }
        quality_prefixes = ("dspark/", "dflash/", "domino/")
        quality_keys = {
            key
            for result in normalized_results
            for key in result
            if key.startswith(quality_prefixes)
            or key in {"drafter/current_lr", "drafter/optimizer_steps_total"}
            or key.startswith("drafter/quality_gate_")
        }
        for key in sorted(quality_keys):
            values = [
                value
                for result in normalized_results
                if (value := _metric_float(result.get(key))) is not None
            ]
            if not values:
                continue
            if key in {
                "drafter/quality_gate_enabled",
                "drafter/quality_gate_ready",
                "drafter/quality_gate_approved",
            }:
                metrics[key] = min(values)
            elif key == "drafter/quality_gate_rejected":
                metrics[key] = max(values)
            else:
                metrics[key] = sum(values) / len(values)
        for key in (
            "timing_s/drafter_prepare_batch",
            "timing_s/drafter_forward_loss",
            "timing_s/drafter_reduce_loss",
            "timing_s/drafter_backward",
            "timing_s/drafter_optimizer",
            "timing_s/drafter_publish_snapshot",
            "quality_gate_before_elapsed_sec",
            "quality_gate_after_elapsed_sec",
            "activation_elapsed_sec",
            "training_loop_elapsed_sec",
            "cleanup_elapsed_sec",
            "elapsed_sec",
        ):
            values = [
                value
                for result in normalized_results
                if (value := _metric_float(result.get(key))) is not None
            ]
            if values:
                metric_key = {
                    "activation_elapsed_sec": "timing_s/drafter_worker_activation",
                    "training_loop_elapsed_sec": "timing_s/drafter_worker_training_loop",
                    "cleanup_elapsed_sec": "timing_s/drafter_worker_cleanup",
                    "elapsed_sec": "timing_s/drafter_worker_elapsed",
                    "quality_gate_before_elapsed_sec": (
                        "timing_s/drafter_quality_gate_before"
                    ),
                    "quality_gate_after_elapsed_sec": (
                        "timing_s/drafter_quality_gate_after"
                    ),
                }.get(key, key)
                metrics[metric_key] = max(values)

        metrics["timing_s/drafter_train_rpc"] = execution.elapsed_sec
        if not execution_consistent:
            outcome_reason = "worker_result_inconsistent"
        elif not quality_consistent:
            outcome_reason = "quality_vote_inconsistent"
        else:
            outcome_reason = execution.reason
        if (
            runtime_state.status is DrafterRuntimeStatus.RUNNING
            and execution_consistent
        ):
            runtime_state.mark_completed(
                completed_batches=successful_steps,
                elapsed_sec=execution.elapsed_sec,
            )
        elif runtime_state.status in {
            DrafterRuntimeStatus.SUBMITTED,
            DrafterRuntimeStatus.RUNNING,
        }:
            runtime_state.mark_failed(outcome_reason)
        else:
            raise RuntimeError(
                "Drafter training execution returned with unexpected runtime state "
                f"{runtime_state.status.name}"
            )
        metrics.update(runtime_state.metrics())
        runtime_state.reset()
        return cls(
            trained=trained,
            publish_approved=publish_approved,
            candidate_finalized=False,
            successful_steps=successful_steps,
            worker_results=worker_results,
            raw_results=execution.raw_results,
            elapsed_sec=execution.elapsed_sec,
            reason=outcome_reason,
            metrics=metrics,
        )

    def with_candidate_finalization(
        self,
        raw_results: list[Any],
        *,
        plan: TrainingPlan,
        commit: bool,
    ) -> "TrainingOutcome":
        participating = [
            result
            for result in raw_results
            if isinstance(result, dict) and bool(result.get("participating", False))
        ]
        expected_worker_ids = set((plan.worker_snapshots or {}).keys())
        actual_worker_ids = {
            str(result.get("worker_id", result.get("rank", "")))
            for result in participating
        }
        finalized = bool(expected_worker_ids) and (
            actual_worker_ids == expected_worker_ids
            and len(participating) == len(expected_worker_ids)
            and all(bool(result.get("finalized", False)) for result in participating)
            and all(result.get("plan_id") == plan.plan_id for result in participating)
            and all(
                bool(result.get("committed", False)) == bool(commit)
                for result in participating
            )
            and all(
                bool(result.get("rolled_back", False)) == (not bool(commit))
                for result in participating
            )
        )
        publish_leaders = [
            result
            for result in participating
            if bool(result.get("is_publish_leader", False))
        ]
        snapshot_consistent = not (commit and plan.publish_after_success) or (
            len(publish_leaders) == 1
            and bool(publish_leaders[0].get("publish_snapshot_cached", False))
        )
        finalized = finalized and snapshot_consistent
        if not finalized:
            raise RuntimeError(
                "Drafter candidate finalization was not consistent across workers: "
                f"commit={commit}, expected_workers={sorted(expected_worker_ids)}, "
                f"results={participating[:3]}"
            )

        metrics = dict(self.metrics)
        metrics.update(
            {
                "drafter/candidate_finalized": 1,
                "drafter/candidate_committed": int(commit),
                "drafter/candidate_rolled_back": int(not commit),
                "drafter/train_publish_leader_count": len(publish_leaders),
                "drafter/train_publish_leader_snapshot_ready": int(
                    len(publish_leaders) == 1
                    and bool(publish_leaders[0].get("publish_snapshot_cached", False))
                ),
            }
        )
        finalize_elapsed = [
            value
            for result in participating
            if (value := _metric_float(result.get("elapsed_sec"))) is not None
        ]
        if finalize_elapsed:
            metrics["timing_s/drafter_candidate_finalize"] = max(finalize_elapsed)
        cleanup_elapsed = [
            value
            for result in participating
            if (value := _metric_float(result.get("cleanup_elapsed_sec"))) is not None
        ]
        if cleanup_elapsed:
            metrics["timing_s/drafter_worker_cleanup"] = max(cleanup_elapsed)
        metrics["drafter/publish_approved"] = int(commit and self.publish_approved)
        return TrainingOutcome(
            trained=self.trained,
            publish_approved=bool(commit and self.publish_approved),
            candidate_finalized=True,
            successful_steps=self.successful_steps,
            worker_results=self.worker_results,
            raw_results=self.raw_results,
            elapsed_sec=self.elapsed_sec,
            reason=("candidate_committed" if commit else "candidate_rolled_back"),
            metrics=metrics,
        )
