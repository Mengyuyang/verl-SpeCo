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
"""Pure helpers for bounded online-drafter epoch sampling."""

from __future__ import annotations

import math
import random
from typing import TypeVar

T = TypeVar("T")


def quality_gate_holdout_count(
    sample_count: int,
    *,
    holdout_ratio: float,
    holdout_limit: int,
) -> int:
    """Return the shared holdout size used by scheduling and gate setup."""

    count = max(int(sample_count), 0)
    if count <= 1:
        return 0
    return min(
        max(1, round(count * max(float(holdout_ratio), 0.0))),
        max(int(holdout_limit), 1),
        count - 1,
    )


def epoch_training_budget(
    *,
    sample_count: int,
    batch_size: int,
    max_epochs: int,
    max_batches: int,
    require_full_batch: bool,
    holdout_count: int = 0,
) -> tuple[int, int, int, int]:
    """Return ``(batches, samples/epoch, batches/epoch, planned_epochs)``.

    ``max_batches`` remains a hard compatibility and safety cap.  The returned
    sample count is conservative, so every distributed worker can execute the
    same number of optimizer steps even when local pools have different sizes.
    """

    batch_size = max(int(batch_size), 1)
    usable_samples = max(int(sample_count) - max(int(holdout_count), 0), 0)
    if require_full_batch:
        batches_per_epoch = usable_samples // batch_size
        samples_per_epoch = batches_per_epoch * batch_size
    else:
        batches_per_epoch = math.ceil(usable_samples / batch_size)
        samples_per_epoch = usable_samples

    epoch_limit = max(int(max_epochs), 0)
    batch_limit = max(int(max_batches), 0)
    planned_batches = min(batch_limit, batches_per_epoch * epoch_limit)
    planned_epochs = (
        math.ceil(planned_batches / batches_per_epoch)
        if planned_batches > 0 and batches_per_epoch > 0
        else 0
    )
    return planned_batches, samples_per_epoch, batches_per_epoch, planned_epochs


def build_epoch_batches(
    items: list[T],
    *,
    batch_size: int,
    samples_per_epoch: int,
    max_epochs: int,
    max_batches: int,
    seed: int,
    require_full_batch: bool = False,
) -> list[list[T]]:
    """Shuffle and traverse each epoch without replacement.

    A worker may have more local data than the distributed conservative sample
    count.  In that case each epoch uses a deterministic shuffled subset of the
    requested size.  No item is repeated inside an epoch.
    """

    batch_size = max(int(batch_size), 1)
    epoch_samples = min(max(int(samples_per_epoch), 0), len(items))
    epoch_limit = max(int(max_epochs), 0)
    batch_limit = max(int(max_batches), 0)
    if epoch_samples <= 0 or epoch_limit <= 0 or batch_limit <= 0:
        return []

    batches: list[list[T]] = []
    for epoch in range(epoch_limit):
        order = list(items)
        random.Random(int(seed) + epoch * 1_000_003).shuffle(order)
        order = order[:epoch_samples]
        for offset in range(0, len(order), batch_size):
            batch = order[offset : offset + batch_size]
            if require_full_batch and len(batch) < batch_size:
                continue
            batches.append(batch)
            if len(batches) >= batch_limit:
                return batches
    return batches
