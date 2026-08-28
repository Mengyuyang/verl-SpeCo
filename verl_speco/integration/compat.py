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
"""Compatibility checks for the import-only verl dependency.

The supported upstream target is the ``release/v0.9.0`` API surface. A commit
may be used by CI to make a test reproducible, but it is not a runtime
requirement because the release branch can receive compatible fixes.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata

from packaging.version import InvalidVersion, Version

SUPPORTED_VERL_VERSION = "0.9.0"
SUPPORTED_VERL_BRANCH = "release/v0.9.0"
ALLOW_UNSUPPORTED_ENV = "VERL_SPECO_ALLOW_UNSUPPORTED_VERL"
STRICT_COMPAT_ENV = "VERL_SPECO_STRICT_VERL"

logger = logging.getLogger(__name__)

# These are the APIs imported by the core SPECO runner and trainer. Optional
# vLLM/SGLang APIs are checked when their corresponding rollout is enabled.
REQUIRED_VERL_API: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("verl.trainer.main_ppo", ("run_ppo",)),
    ("verl.trainer.main_ppo_v0", ("BaseTaskRunner",)),
    ("verl.trainer.ppo.ray_trainer", ("RayPPOTrainer",)),
    (
        "verl.trainer.ppo.utils",
        (
            "Role",
            "create_rl_dataset",
            "create_rl_sampler",
            "need_critic",
            "need_reference_policy",
        ),
    ),
    ("verl.utils.config", ("omega_conf_to_dataclass", "validate_config")),
    (
        "verl.utils.device",
        ("auto_set_device", "get_device_id", "get_device_name", "get_torch_device"),
    ),
    (
        "verl.utils.fsdp_utils",
        (
            "get_fsdp_full_state_dict",
            "get_fsdp_wrap_policy",
            "apply_fsdp2",
            "fsdp2_load_full_state_dict",
            "load_fsdp_model_to_gpu",
            "load_fsdp_optimizer",
            "offload_fsdp_model_to_cpu",
            "offload_fsdp_optimizer",
        ),
    ),
    ("verl.utils.dataset.rl_dataset", ("collate_fn",)),
    (
        "verl.utils.ulysses",
        ("get_ulysses_sequence_parallel_group", "set_ulysses_sequence_parallel_group"),
    ),
    (
        "verl.utils.tensordict_utils",
        ("assign_non_tensor", "assign_non_tensor_data", "get", "get_non_tensor_data"),
    ),
    ("verl.workers.engine_workers", ("ActorRolloutRefWorker", "TrainingWorker")),
    ("verl.workers.config", ("HFModelConfig",)),
    ("verl.workers.rollout.replica", ("RolloutReplica", "TokenOutput")),
    ("verl.single_controller.base", ("Worker",)),
    ("verl.single_controller.base.decorator", ("Dispatch", "register")),
    ("verl.single_controller.ray", ("RayClassWithInitArgs",)),
    ("verl.utils.ray_utils", ("auto_await", "parallel_put")),
    (
        "verl.utils.distributed",
        ("initialize_global_process_group_ray", "set_numa_affinity"),
    ),
    ("verl.workers.utils.padding", ("left_right_2_no_padding", "no_padding_2_padding")),
)


@dataclass(frozen=True)
class VerlCompatibility:
    """Resolved metadata for the installed verl package."""

    version: str | None
    commit_id: str | None
    requested_revision: str | None
    supported: bool
    reason: str
    missing_api: tuple[str, ...] = ()


def _read_distribution_vcs_info(
    distribution_name: str = "verl",
) -> tuple[str | None, str | None]:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return None, None

    raw_direct_url = distribution.read_text("direct_url.json")
    if not raw_direct_url:
        return None, None

    try:
        direct_url = json.loads(raw_direct_url)
    except json.JSONDecodeError:
        return None, None

    vcs_info = direct_url.get("vcs_info") or {}
    commit_id = vcs_info.get("commit_id")
    requested_revision = vcs_info.get("requested_revision")
    return (
        str(commit_id) if commit_id else None,
        str(requested_revision) if requested_revision else None,
    )


def _read_distribution_commit(distribution_name: str = "verl") -> str | None:
    """Return the installed commit for diagnostics, never for acceptance."""

    return _read_distribution_vcs_info(distribution_name)[0]


def _read_imported_verl_version() -> str | None:
    """Read the version from the same ``verl`` module Python imported.

    ``PYTHONPATH`` deployments can import verl from a source checkout while an
    older wheel's ``*.dist-info`` remains in site-packages. Prefer the imported
    module so version and API checks describe the dependency SPECO will run.
    """

    try:
        import verl
    except ImportError:
        verl = None

    if verl is not None:
        version = getattr(verl, "__version__", None)
        if version:
            return str(version)

    try:
        return metadata.version("verl")
    except metadata.PackageNotFoundError:
        return None


def _missing_required_api() -> tuple[str, ...]:
    missing: list[str] = []
    for module_name, symbols in REQUIRED_VERL_API:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            missing.append(
                f"{module_name} (import failed: {type(exc).__name__}: {exc})"
            )
            continue
        for symbol in symbols:
            if not hasattr(module, symbol):
                missing.append(f"{module_name}.{symbol}")
    return tuple(missing)


def _version_matches_release(version: str | None, allowed_version: str) -> bool:
    """Accept PEP 440 builds whose base version matches the supported release."""

    if version is None:
        return False
    try:
        return Version(version).base_version == Version(allowed_version).base_version
    except InvalidVersion:
        return False


def resolve_verl_compatibility(
    allowed_versions: Sequence[str] = (SUPPORTED_VERL_VERSION,),
) -> VerlCompatibility:
    """Return whether the importable verl matches the release API contract."""

    version = _read_imported_verl_version()
    commit_id, requested_revision = _read_distribution_vcs_info()
    missing_api = _missing_required_api()

    version_matches = any(
        _version_matches_release(version, allowed_version)
        for allowed_version in allowed_versions
    )
    branch_matches = requested_revision == SUPPORTED_VERL_BRANCH
    if (version_matches or branch_matches) and not missing_api:
        reason = f"matched {SUPPORTED_VERL_BRANCH} version/API contract"
        if branch_matches:
            reason = f"matched {SUPPORTED_VERL_BRANCH} branch/API contract"
        return VerlCompatibility(version, commit_id, requested_revision, True, reason)

    if os.getenv(ALLOW_UNSUPPORTED_ENV, "").lower() in {"1", "true", "yes"}:
        return VerlCompatibility(
            version,
            commit_id,
            requested_revision,
            True,
            "env override",
            missing_api,
        )

    reasons = []
    if not version_matches and not branch_matches:
        reasons.append(
            f"expected version {SUPPORTED_VERL_VERSION} or branch {SUPPORTED_VERL_BRANCH}"
        )
    if missing_api:
        reasons.append("missing/incompatible API: " + ", ".join(missing_api[:8]))
    return VerlCompatibility(
        version,
        commit_id,
        requested_revision,
        False,
        "; ".join(reasons) or "unknown compatibility failure",
        missing_api,
    )


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def check_compatible_verl(strict: bool | None = None) -> VerlCompatibility:
    """Warn by default when verl is outside the supported release API contract."""

    result = resolve_verl_compatibility()
    if result.supported:
        return result

    message = (
        f"SPECO requires import-only verl {SUPPORTED_VERL_BRANCH}; "
        f"found version={result.version!r}, requested_revision={result.requested_revision!r}, "
        f"commit_id={result.commit_id!r}: {result.reason}. "
        f"Set {STRICT_COMPAT_ENV}=1 to fail closed."
    )
    if strict is None:
        strict = _env_flag_enabled(STRICT_COMPAT_ENV)
    if strict:
        raise RuntimeError(message)

    logger.warning(message)
    return result


def assert_compatible_verl() -> VerlCompatibility:
    """Backward-compatible alias for the warning-only compatibility check."""

    return check_compatible_verl()
