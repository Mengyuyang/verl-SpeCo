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

from verl_speco.trainer.sampling import (
    build_epoch_batches,
    epoch_training_budget,
    quality_gate_holdout_count,
)


def test_quality_gate_holdout_matches_online_gate_formula() -> None:
    assert quality_gate_holdout_count(10, holdout_ratio=0.2, holdout_limit=4) == 2
    assert quality_gate_holdout_count(16, holdout_ratio=0.2, holdout_limit=4) == 3
    assert quality_gate_holdout_count(1, holdout_ratio=0.2, holdout_limit=4) == 0


def test_epoch_budget_includes_partial_batch_and_caps_steps() -> None:
    assert epoch_training_budget(
        sample_count=16,
        batch_size=4,
        max_epochs=2,
        max_batches=8,
        require_full_batch=False,
        holdout_count=3,
    ) == (8, 13, 4, 2)


def test_epoch_batches_cover_every_item_before_repeating() -> None:
    batches = build_epoch_batches(
        list(range(8)),
        batch_size=4,
        samples_per_epoch=8,
        max_epochs=2,
        max_batches=4,
        seed=17,
    )

    assert len(batches) == 4
    assert sorted(item for batch in batches[:2] for item in batch) == list(range(8))
    assert sorted(item for batch in batches[2:] for item in batch) == list(range(8))
    assert batches == build_epoch_batches(
        list(range(8)),
        batch_size=4,
        samples_per_epoch=8,
        max_epochs=2,
        max_batches=4,
        seed=17,
    )


def test_epoch_batches_keep_last_partial_batch() -> None:
    batches = build_epoch_batches(
        list(range(13)),
        batch_size=4,
        samples_per_epoch=13,
        max_epochs=2,
        max_batches=8,
        seed=23,
    )

    assert [len(batch) for batch in batches] == [4, 4, 4, 1, 4, 4, 4, 1]
