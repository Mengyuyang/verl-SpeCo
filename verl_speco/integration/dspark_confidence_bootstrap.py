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
"""Create a runtime-safe view of a confidence-enabled DSpark checkpoint.

The source checkpoint is never modified. Existing safetensors shards are linked
or copied into a new directory, the existing confidence tensors are validated,
and the config is normalized for the vLLM-Ascend loader. Missing confidence
weights fail closed: a synthetic constant head cannot represent DSpark's
position-wise dynamic verify-budget policy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_MARKER_NAME = "speco_confidence_bootstrap.json"
_CONFIDENCE_WEIGHT = "confidence_head.proj.weight"
_CONFIDENCE_BIAS = "confidence_head.proj.bias"


@dataclass(frozen=True)
class _CheckpointLayout:
    weight_map: dict[str, str]
    metadata: dict[str, Any]
    referenced_files: tuple[Path, ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _safe_relative_weight_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe checkpoint weight path: {raw!r}")
    return path


def _torch_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if (
        isinstance(loaded, dict)
        and isinstance(loaded.get("state_dict"), dict)
        and all(torch.is_tensor(value) for value in loaded["state_dict"].values())
    ):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict) or not all(
        torch.is_tensor(value) for value in loaded.values()
    ):
        raise TypeError(f"Checkpoint file does not contain a tensor state dict: {path}")
    return {str(key): value for key, value in loaded.items()}


def _safetensor_keys(path: Path) -> list[str]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("DSpark confidence bootstrap requires safetensors") from exc
    with safe_open(path, framework="pt", device="cpu") as handle:
        # safetensors.safe_open exposes keys(), but is not itself iterable.
        return [str(key) for key in handle.keys()]  # noqa: SIM118


def _discover_checkpoint(source: Path) -> _CheckpointLayout:
    index_candidates = (
        source / "model.safetensors.index.json",
        source / "pytorch_model.bin.index.json",
    )
    for index_path in index_candidates:
        if not index_path.is_file():
            continue
        index = _read_json(index_path)
        raw_map = index.get("weight_map")
        if not isinstance(raw_map, dict) or not raw_map:
            raise ValueError(f"Checkpoint index has no weight_map: {index_path}")
        weight_map = {str(key): str(value) for key, value in raw_map.items()}
        referenced = []
        for filename in sorted(set(weight_map.values())):
            relative = _safe_relative_weight_path(filename)
            checkpoint_file = source / relative
            if not checkpoint_file.is_file():
                raise FileNotFoundError(
                    f"Checkpoint index references a missing file: {checkpoint_file}"
                )
            referenced.append(relative)
        metadata = index.get("metadata")
        return _CheckpointLayout(
            weight_map=weight_map,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            referenced_files=tuple(referenced),
        )

    safetensors_path = source / "model.safetensors"
    if safetensors_path.is_file():
        keys = _safetensor_keys(safetensors_path)
        if not keys:
            raise ValueError(f"Checkpoint is empty: {safetensors_path}")
        return _CheckpointLayout(
            weight_map={key: safetensors_path.name for key in keys},
            metadata={},
            referenced_files=(Path(safetensors_path.name),),
        )

    pytorch_path = source / "pytorch_model.bin"
    if pytorch_path.is_file():
        state_dict = _torch_state_dict(pytorch_path)
        if not state_dict:
            raise ValueError(f"Checkpoint is empty: {pytorch_path}")
        return _CheckpointLayout(
            weight_map={key: pytorch_path.name for key in state_dict},
            metadata={},
            referenced_files=(Path(pytorch_path.name),),
        )

    raise FileNotFoundError(
        "DSpark source checkpoint must contain model.safetensors, "
        "model.safetensors.index.json, pytorch_model.bin, or "
        f"pytorch_model.bin.index.json: {source}"
    )


def _matching_confidence_keys(weight_map: dict[str, str]) -> dict[str, list[str]]:
    return {
        suffix: sorted(key for key in weight_map if key.endswith(suffix))
        for suffix in (_CONFIDENCE_WEIGHT, _CONFIDENCE_BIAS)
    }


def _copy_auxiliary_files(
    source: Path, destination: Path, referenced_files: tuple[Path, ...]
) -> None:
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        _MARKER_NAME,
    }
    referenced_top_level = {path.parts[0] for path in referenced_files if path.parts}
    for child in source.iterdir():
        # Only the normalized safetensors files and index emitted below may be
        # visible as model weights. Copying an unreferenced standalone weight
        # file (for example a stale model.safetensors) lets some HF/vLLM
        # loaders bypass the index and silently omit the confidence shard.
        is_weight_file = child.suffix in {
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
        } or child.name in {"tf_model.h5", "flax_model.msgpack"}
        if (
            not child.is_file()
            or child.name in excluded
            or child.name in referenced_top_level
            or is_weight_file
        ):
            continue
        shutil.copy2(child, destination / child.name)


def _materialize_safetensors(
    *,
    source: Path,
    destination: Path,
    layout: _CheckpointLayout,
    link_mode: str,
) -> dict[str, str]:
    output_map: dict[str, str] = {}
    keys_by_file: dict[str, list[str]] = {}
    for key, filename in layout.weight_map.items():
        keys_by_file.setdefault(filename, []).append(key)

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("DSpark confidence bootstrap requires safetensors") from exc

    for filename, mapped_keys in sorted(keys_by_file.items()):
        relative = _safe_relative_weight_path(filename)
        input_path = source / relative
        if input_path.suffix == ".safetensors":
            # Do not leave a standalone model.safetensors next to the new index:
            # Transformers/vLLM loaders may prefer it and silently ignore the
            # confidence-only shard referenced by model.safetensors.index.json.
            output_relative = (
                relative.with_name("speco-base-model.safetensors")
                if relative.name == "model.safetensors"
                else relative
            )
            output_path = destination / output_relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if link_mode == "copy":
                shutil.copy2(input_path, output_path)
            else:
                os.symlink(input_path.resolve(), output_path)
            available = set(_safetensor_keys(input_path))
        else:
            output_relative = Path(f"{relative.as_posix()}.safetensors")
            output_path = destination / output_relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            state_dict = _torch_state_dict(input_path)
            available = set(state_dict)
            converted = {
                key: state_dict[key].detach().cpu().contiguous()
                for key in mapped_keys
                if key in state_dict
            }
            save_file(converted, output_path)

        expected_keys = set(mapped_keys)
        missing = sorted(expected_keys.difference(available))
        extra = sorted(available.difference(expected_keys))
        if missing or extra:
            raise KeyError(
                f"Checkpoint index/file mismatch in {input_path}: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        for key in mapped_keys:
            output_map[key] = output_relative.as_posix()
    return output_map


def _config_dimension(config: dict[str, Any], *names: str) -> int:
    for name in names:
        value = config.get(name)
        if value is not None:
            return int(value)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        for name in names:
            value = text_config.get(name)
            if value is not None:
                return int(value)
    return 0


def _confidence_shapes(
    root: Path, weight_map: dict[str, str]
) -> dict[str, tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("DSpark confidence bootstrap requires safetensors") from exc

    matches = _matching_confidence_keys(weight_map)
    if any(len(keys) != 1 for keys in matches.values()):
        raise ValueError(
            "Expected exactly one confidence_head.proj.{weight,bias} pair; "
            f"matches={matches}"
        )
    shapes: dict[str, tuple[int, ...]] = {}
    for suffix, keys in matches.items():
        key = keys[0]
        checkpoint_file = root / _safe_relative_weight_path(weight_map[key])
        if checkpoint_file.suffix != ".safetensors":
            raise ValueError(
                f"Confidence tensor is not in safetensors: {checkpoint_file}"
            )
        with safe_open(checkpoint_file, framework="pt", device="cpu") as handle:
            if key not in handle.keys():  # noqa: SIM118
                raise KeyError(f"Missing {key} from {checkpoint_file}")
            shapes[suffix] = tuple(
                int(dim) for dim in handle.get_slice(key).get_shape()
            )
    return shapes


def _confidence_tensors(
    root: Path, weight_map: dict[str, str]
) -> dict[str, torch.Tensor]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("DSpark confidence bootstrap requires safetensors") from exc

    matches = _matching_confidence_keys(weight_map)
    if any(len(keys) != 1 for keys in matches.values()):
        raise ValueError(
            "Expected exactly one confidence_head.proj.{weight,bias} pair; "
            f"matches={matches}"
        )
    tensors: dict[str, torch.Tensor] = {}
    for suffix, keys in matches.items():
        key = keys[0]
        checkpoint_file = root / _safe_relative_weight_path(weight_map[key])
        if checkpoint_file.suffix != ".safetensors":
            raise ValueError(
                f"Confidence tensor is not in safetensors: {checkpoint_file}"
            )
        with safe_open(checkpoint_file, framework="pt", device="cpu") as handle:
            if key not in handle.keys():  # noqa: SIM118
                raise KeyError(f"Missing {key} from {checkpoint_file}")
            tensors[suffix] = handle.get_tensor(key)
    return tensors


def _validate_output(output: Path, *, source: Path) -> None:
    marker = _read_json(output / _MARKER_NAME)
    if marker.get("source") != str(source.resolve()):
        raise ValueError(
            f"Existing bootstrap directory belongs to a different source: {output}"
        )
    if marker.get("initialization") != "preserved_existing_confidence_head":
        raise ValueError(
            "DSpark runtime view must preserve an existing confidence head; "
            f"got initialization={marker.get('initialization')!r}"
        )
    config = _read_json(output / "config.json")
    if config.get("enable_confidence_head") is not True:
        raise ValueError(
            f"Bootstrap config does not enable the confidence head: {output}"
        )
    if config.get("confidence_head_with_markov") is not True:
        raise ValueError(f"Bootstrap config is not Markov-conditioned: {output}")
    hidden_size = _config_dimension(config, "hidden_size")
    markov_rank = _config_dimension(config, "markov_rank", "dspark_markov_rank")
    layout = _discover_checkpoint(output)
    shapes = _confidence_shapes(output, layout.weight_map)
    expected = {
        _CONFIDENCE_WEIGHT: (1, hidden_size + markov_rank),
        _CONFIDENCE_BIAS: (1,),
    }
    if hidden_size <= 0 or markov_rank <= 0 or shapes != expected:
        raise ValueError(
            "Bootstrap confidence topology mismatch: "
            f"hidden_size={hidden_size} markov_rank={markov_rank} "
            f"expected={expected} actual={shapes}"
        )
    confidence = _confidence_tensors(output, layout.weight_map)
    nonfinite = [
        suffix
        for suffix, tensor in confidence.items()
        if not bool(torch.isfinite(tensor).all().item())
    ]
    if nonfinite:
        raise ValueError(
            "DSpark confidence checkpoint contains non-finite tensors: "
            + ", ".join(nonfinite)
        )


def bootstrap_dspark_confidence_checkpoint(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    link_mode: str = "symlink",
) -> Path:
    """Create or validate an idempotent confidence-enabled runtime view."""

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(
            f"DSpark source checkpoint does not exist: {source_path}"
        )
    if source_path == output_path:
        raise ValueError("Bootstrap output must differ from the source checkpoint")
    if link_mode not in {"symlink", "copy"}:
        raise ValueError(f"Unsupported link_mode: {link_mode!r}")

    if output_path.exists():
        _validate_output(output_path, source=source_path)
        return output_path

    config_path = source_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"DSpark source config.json is missing: {config_path}")
    config = _read_json(config_path)
    hidden_size = _config_dimension(config, "hidden_size")
    markov_rank = _config_dimension(config, "markov_rank", "dspark_markov_rank")
    if hidden_size <= 0 or markov_rank <= 0:
        raise ValueError(
            "Cannot build a Markov-conditioned confidence head without positive "
            f"hidden_size and markov_rank: hidden_size={hidden_size} "
            f"markov_rank={markov_rank} source={source_path}"
        )

    layout = _discover_checkpoint(source_path)
    confidence_matches = _matching_confidence_keys(layout.weight_map)
    present_counts = {suffix: len(keys) for suffix, keys in confidence_matches.items()}
    if any(count != 1 for count in present_counts.values()):
        raise ValueError(
            "Source checkpoint must contain exactly one existing "
            "confidence_head.proj.{weight,bias} pair; a synthetic head cannot "
            "represent DSpark's position-wise verify-budget policy: "
            f"matches={confidence_matches} source={source_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        _copy_auxiliary_files(source_path, temporary, layout.referenced_files)
        output_map = _materialize_safetensors(
            source=source_path,
            destination=temporary,
            layout=layout,
            link_mode=link_mode,
        )

        confidence_shapes = _confidence_shapes(temporary, output_map)
        expected_shapes = {
            _CONFIDENCE_WEIGHT: (1, hidden_size + markov_rank),
            _CONFIDENCE_BIAS: (1,),
        }
        if confidence_shapes != expected_shapes:
            raise ValueError(
                "Existing DSpark confidence-head topology mismatch: "
                f"expected={expected_shapes} actual={confidence_shapes}"
            )

        output_config = dict(config)
        # vLLM's DSpark loader reads these dimensions from the top-level config
        # even when the source model stores them under text_config.
        output_config["hidden_size"] = hidden_size
        output_config["markov_rank"] = markov_rank
        output_config["enable_confidence_head"] = True
        output_config["confidence_head_with_markov"] = True
        output_config["confidence_head_alpha"] = max(
            1.0, float(output_config.get("confidence_head_alpha", 0.0) or 0.0)
        )
        (temporary / "config.json").write_text(
            json.dumps(output_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        output_metadata = {
            key: value for key, value in layout.metadata.items() if key != "total_size"
        }
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": output_metadata, "weight_map": output_map},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        marker = {
            "schema_version": 1,
            "source": str(source_path),
            "initialization": "preserved_existing_confidence_head",
        }
        (temporary / _MARKER_NAME).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _validate_output(temporary, source=source_path)
        temporary.rename(output_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = bootstrap_dspark_confidence_checkpoint(
        args.source,
        args.output,
        link_mode=args.link_mode,
    )
    marker = _read_json(output / _MARKER_NAME)
    print(
        "DSpark confidence runtime view ready: "
        f"initialization={marker['initialization']} source={marker['source']}",
        file=sys.stderr,
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
