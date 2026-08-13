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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from verl_speco.integration.dspark_confidence_bootstrap import (
    bootstrap_dspark_confidence_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]


def _valid_confidence_state() -> dict[str, torch.Tensor]:
    return {
        "confidence_head.proj.weight": torch.arange(6, dtype=torch.float32).reshape(
            1, 6
        ),
        "confidence_head.proj.bias": torch.tensor([-0.25], dtype=torch.float32),
    }


def _write_config(path: Path) -> None:
    config = {
        "architectures": ["Qwen3DSparkModel"],
        "hidden_size": 4,
        "markov_rank": 2,
        "enable_confidence_head": False,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_source_checkpoint(
    path: Path, *, confidence_state: dict[str, torch.Tensor] | None = None
) -> None:
    save_file = pytest.importorskip("safetensors.torch").save_file
    path.mkdir()
    _write_config(path)
    state = {
        "fc.weight": torch.ones(4, 8, dtype=torch.bfloat16),
        "hidden_norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "norm.weight": torch.ones(4, dtype=torch.bfloat16),
    }
    state.update(
        _valid_confidence_state() if confidence_state is None else confidence_state
    )
    save_file(state, path / "model.safetensors")


def test_bootstrap_preserves_confidence_head_and_never_mutates_source(tmp_path) -> None:
    load_file = pytest.importorskip("safetensors.torch").load_file
    save_file = pytest.importorskip("safetensors.torch").save_file
    source = tmp_path / "source"
    output = tmp_path / "runtime"
    confidence = _valid_confidence_state()
    _write_source_checkpoint(source, confidence_state=confidence)
    save_file(
        {"stale.weight": torch.ones(1, dtype=torch.bfloat16)},
        source / "stale.safetensors",
    )
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    result = bootstrap_dspark_confidence_checkpoint(source, output, link_mode="copy")

    assert result == output.resolve()
    assert not (source / "model.safetensors.index.json").exists()
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    assert source_config["enable_confidence_head"] is False

    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert output_config["enable_confidence_head"] is True
    assert output_config["confidence_head_with_markov"] is True
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert index["weight_map"]["fc.weight"] == "speco-base-model.safetensors"
    assert (
        index["weight_map"]["confidence_head.proj.weight"]
        == "speco-base-model.safetensors"
    )
    assert not (output / "model.safetensors").exists()
    assert not (output / "stale.safetensors").exists()
    assert (output / "tokenizer_config.json").is_file()

    runtime_state = load_file(output / "speco-base-model.safetensors")
    assert torch.equal(
        runtime_state["confidence_head.proj.weight"],
        confidence["confidence_head.proj.weight"],
    )
    assert torch.equal(
        runtime_state["confidence_head.proj.bias"],
        confidence["confidence_head.proj.bias"],
    )
    marker = json.loads(
        (output / "speco_confidence_bootstrap.json").read_text(encoding="utf-8")
    )
    assert marker["initialization"] == "preserved_existing_confidence_head"

    # Re-running is deliberately idempotent and validates the existing view.
    assert (
        bootstrap_dspark_confidence_checkpoint(source, output, link_mode="copy")
        == output.resolve()
    )


@pytest.mark.parametrize(
    "confidence_state",
    [
        {},
        {"confidence_head.proj.weight": torch.zeros(1, 6)},
    ],
    ids=["missing", "partial"],
)
def test_bootstrap_rejects_missing_or_partial_confidence_pair(
    tmp_path, confidence_state
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "runtime"
    _write_source_checkpoint(source, confidence_state=confidence_state)

    with pytest.raises(ValueError, match="must contain exactly one existing"):
        bootstrap_dspark_confidence_checkpoint(source, output, link_mode="copy")


def test_bootstrap_rejects_wrong_existing_confidence_shape(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "runtime"
    _write_source_checkpoint(
        source,
        confidence_state={
            "confidence_head.proj.weight": torch.zeros(1, 4),
            "confidence_head.proj.bias": torch.zeros(1),
        },
    )

    with pytest.raises(ValueError, match="topology mismatch"):
        bootstrap_dspark_confidence_checkpoint(source, output, link_mode="copy")


def test_bootstrap_rejects_nonfinite_confidence_head(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "runtime"
    _write_source_checkpoint(
        source,
        confidence_state={
            "confidence_head.proj.weight": torch.full((1, 6), float("nan")),
            "confidence_head.proj.bias": torch.tensor([float("inf")]),
        },
    )

    with pytest.raises(ValueError, match="non-finite tensors"):
        bootstrap_dspark_confidence_checkpoint(source, output, link_mode="copy")


def test_bootstrap_converts_a_sharded_bin_index_and_drops_stale_size(tmp_path) -> None:
    load_file = pytest.importorskip("safetensors.torch").load_file
    source = tmp_path / "source"
    output = tmp_path / "runtime"
    source.mkdir()
    _write_config(source)
    shard_one = {
        "fc.weight": torch.ones(4, 8),
        "confidence_head.proj.weight": torch.ones(1, 6),
    }
    shard_two = {
        "norm.weight": torch.ones(4),
        "confidence_head.proj.bias": torch.zeros(1),
    }
    torch.save(shard_one, source / "pytorch_model-00001-of-00002.bin")
    torch.save(shard_two, source / "pytorch_model-00002-of-00002.bin")
    weight_map = {key: "pytorch_model-00001-of-00002.bin" for key in shard_one}
    weight_map.update({key: "pytorch_model-00002-of-00002.bin" for key in shard_two})
    (source / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": 123, "format": "pt"}, "weight_map": weight_map}
        ),
        encoding="utf-8",
    )

    bootstrap_dspark_confidence_checkpoint(source, output, link_mode="copy")

    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert index["metadata"] == {"format": "pt"}
    assert index["weight_map"]["fc.weight"].endswith(".bin.safetensors")
    converted = load_file(output / index["weight_map"]["fc.weight"])
    assert set(converted) == set(shard_one)


def test_bootstrap_rejects_index_and_shard_key_disagreement(tmp_path) -> None:
    save_file = pytest.importorskip("safetensors.torch").save_file
    source = tmp_path / "source"
    output = tmp_path / "runtime"
    source.mkdir()
    _write_config(source)
    state = {
        "fc.weight": torch.ones(4, 8),
        "unindexed.weight": torch.ones(1),
        **_valid_confidence_state(),
    }
    save_file(state, source / "model-00001-of-00001.safetensors")
    weight_map = {
        key: "model-00001-of-00001.safetensors"
        for key in state
        if key != "unindexed.weight"
    }
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    with pytest.raises(KeyError, match="extra=.*unindexed.weight"):
        bootstrap_dspark_confidence_checkpoint(source, output, link_mode="copy")


@pytest.mark.skipif(os.name == "nt", reason="Windows symlinks require host policy")
def test_bootstrap_symlink_mode_keeps_large_source_weights_immutable(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "runtime"
    _write_source_checkpoint(source)

    bootstrap_dspark_confidence_checkpoint(source, output, link_mode="symlink")

    linked_base = output / "speco-base-model.safetensors"
    assert linked_base.is_symlink()
    assert linked_base.resolve() == (source / "model.safetensors").resolve()


def test_module_cli_writes_only_the_runtime_path_to_stdout(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "runtime"
    _write_source_checkpoint(source)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "verl_speco.integration.dspark_confidence_bootstrap",
            "--source",
            str(source),
            "--output",
            str(output),
            "--link-mode",
            "copy",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [str(output.resolve())]
    assert "initialization=preserved_existing_confidence_head" in result.stderr
