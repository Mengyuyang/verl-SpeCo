# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training budget policies for drafter scheduling."""

from __future__ import annotations

from typing import Protocol

from verl_speco.trainer.sampling import (
    epoch_training_budget,
    quality_gate_holdout_count,
)
from verl_speco.trainer.scheduler.schedule_types import (
    DrafterScheduleConfig,
    DrafterScheduleContext,
    TrainingBudget,
)


class TrainingBudgetPolicy(Protocol):
    def make_budget(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingBudget: ...


class SyncTrainingBudgetPolicy:
    """Build one shared synchronous optimizer budget for every worker.

    Legacy sampling keeps the configured ``step`` count.  Epoch sampling uses
    the conservative data count aggregated across workers, subtracts the
    quality-gate holdout, and caps the result by ``step``.
    """

    def make_budget(
        self,
        context: DrafterScheduleContext,
        config: DrafterScheduleConfig,
    ) -> TrainingBudget:
        configured_max_batches = max(config.train_batches_per_trigger, 0)
        max_batches = configured_max_batches
        samples_per_epoch = 0
        batches_per_epoch = 0
        planned_epochs = 0
        if config.sample_without_replacement:
            data_status = context.data_status
            sample_count = data_status.trainable_samples if data_status else 0
            batch_size = data_status.batch_size_per_gpu if data_status else 1
            holdout_count = 0
            if config.publish_quality_gate_enable:
                holdout_count = quality_gate_holdout_count(
                    sample_count,
                    holdout_ratio=config.publish_quality_gate_holdout_ratio,
                    holdout_limit=config.publish_quality_gate_holdout_samples,
                )
            (
                max_batches,
                samples_per_epoch,
                batches_per_epoch,
                planned_epochs,
            ) = epoch_training_budget(
                sample_count=sample_count,
                batch_size=batch_size,
                max_epochs=config.max_epochs_per_trigger,
                max_batches=configured_max_batches,
                require_full_batch=config.require_full_batch,
                holdout_count=holdout_count,
            )
        return TrainingBudget(
            max_batches=max_batches,
            min_batches=max(config.min_trainable_batches, 1),
            deadline_ts=None,
            require_full_batch=config.require_full_batch,
            sample_last_n_steps=config.sample_last_n_steps,
            reason="sync_budget_ready" if max_batches > 0 else "no_training_budget",
            sample_without_replacement=config.sample_without_replacement,
            samples_per_epoch=samples_per_epoch,
            batches_per_epoch=batches_per_epoch,
            planned_epochs=planned_epochs,
        )
