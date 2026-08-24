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
"""TaskRunner hook for the SPECO trainer."""

import json
import logging
import os
import socket
from contextlib import contextmanager
from pprint import pprint

import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.utils import (
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
)
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.workers.config.model import HFModelConfig

logger = logging.getLogger(__name__)


def _serialize_drafter_config(config):
    try:
        drafter = OmegaConf.to_container(
            config.actor_rollout_ref.rollout.drafter, resolve=True
        )
    except Exception:  # noqa: BLE001
        return ""
    return json.dumps(drafter, sort_keys=True) if isinstance(drafter, dict) else ""


def _unwrap_ray_remote_actor_class(worker_cls):
    return getattr(worker_cls, "__ray_actor_class__", worker_cls)


def _remotify_like_worker_mapping_value(role_worker_cls, wrapped_cls):
    if hasattr(role_worker_cls, "__ray_actor_class__"):
        return ray.remote(wrapped_cls)
    return wrapped_cls


def _drafter_rollout_enabled(config) -> bool:
    try:
        drafter = config.actor_rollout_ref.rollout.get("drafter")
    except (AttributeError, TypeError):
        return False
    if drafter is None:
        return False
    if hasattr(drafter, "get"):
        return bool(drafter.get("enable", False))
    return bool(getattr(drafter, "enable", False))


def _rollout_name(config):
    try:
        return config.actor_rollout_ref.rollout.get("name")
    except (AttributeError, TypeError):
        return None


def _install_vllm_import_compat_for_task_runner(config) -> bool:
    if _rollout_name(config) != "vllm":
        return False
    from verl_speco.integration.verl_npu_vllm_compat import (
        install_verl_npu_vllm_import_compat,
    )

    return install_verl_npu_vllm_import_compat()


@contextmanager
def _prepare_no_drafter_runtime_config(config):
    from verl_speco.integration.vllm_runtime import install_upstream_vllm_runtime_bridge

    rollout_config = getattr(
        getattr(config, "actor_rollout_ref", None), "rollout", None
    )
    if rollout_config is not None and rollout_config.get("name") == "vllm":
        # Keep the no-drafter HTTP server on the same import-safe Ray actor
        # path as speculative rollout. This avoids hiding child-process import
        # failures behind Ray's TemporaryActor coroutine error.
        if not install_upstream_vllm_runtime_bridge():
            logger.warning(
                "SPECO no-drafter baseline could not install the vLLM server runtime bridge"
            )
        logger.info(
            "SPECO no-drafter baseline: preserving the configured vLLM scheduler "
            "and worker extension"
        )
    yield


class SpecoTaskRunner(BaseTaskRunner):
    """External TaskRunner that swaps in SpecoRayPPOTrainer.

    Adapted from verl v0.9.0
    ``verl/trainer/main_ppo_v0.py::TaskRunner.run``.
    """

    def add_actor_rollout_worker(self, config):
        worker_cls, ray_worker_group_cls = super().add_actor_rollout_worker(config)
        if _rollout_name(config) != "vllm":
            return worker_cls, ray_worker_group_cls

        from verl_speco.integration.verl_npu_vllm_compat import (
            VerlNPUVLLMImportCompatMixin,
        )

        raw_worker_cls = _unwrap_ray_remote_actor_class(worker_cls)
        if issubclass(raw_worker_cls, VerlNPUVLLMImportCompatMixin):
            return worker_cls, ray_worker_group_cls

        wrapped_cls = type(
            f"SpecoVLLMCompat{raw_worker_cls.__name__}",
            (VerlNPUVLLMImportCompatMixin, raw_worker_cls),
            {
                "__module__": __name__,
                "__doc__": raw_worker_cls.__doc__,
            },
        )
        for role, role_worker_cls in list(self.role_worker_mapping.items()):
            raw_role_worker_cls = _unwrap_ray_remote_actor_class(role_worker_cls)
            if role_worker_cls is worker_cls or raw_role_worker_cls is raw_worker_cls:
                self.role_worker_mapping[role] = _remotify_like_worker_mapping_value(
                    role_worker_cls, wrapped_cls
                )
        logger.warning(
            "SPECO vLLM worker import compatibility enabled: %s", wrapped_cls.__name__
        )
        return _remotify_like_worker_mapping_value(
            worker_cls, wrapped_cls
        ), ray_worker_group_cls

    def add_speco_drafter_worker(self, config):
        """Return the external SPECO drafter worker class when online training is enabled."""
        from verl_speco.workers import SpecoWorker

        enable_drafter = bool(
            config.actor_rollout_ref.rollout.drafter.enable
            and config.actor_rollout_ref.rollout.drafter.enable_drafter_training
        )
        if not enable_drafter:
            return None
        return ray.remote(SpecoWorker)

    def _with_speco_rollout_publish_mixin(self, worker_cls, config):
        from verl_speco.integration.rollout_publish import DraftWeightPublishMixin

        enable_drafter = bool(config.actor_rollout_ref.rollout.drafter.enable)
        raw_worker_cls = _unwrap_ray_remote_actor_class(worker_cls)
        if not enable_drafter or issubclass(raw_worker_cls, DraftWeightPublishMixin):
            return worker_cls

        wrapped_cls = type(
            f"Speco{raw_worker_cls.__name__}",
            (DraftWeightPublishMixin, raw_worker_cls),
            {
                "__module__": __name__,
                "__doc__": raw_worker_cls.__doc__,
                "_speco_sglang_drafter_config_env": _serialize_drafter_config(config),
            },
        )
        for role, role_worker_cls in list(self.role_worker_mapping.items()):
            raw_role_worker_cls = _unwrap_ray_remote_actor_class(role_worker_cls)
            if role_worker_cls is worker_cls or raw_role_worker_cls is raw_worker_cls:
                self.role_worker_mapping[role] = _remotify_like_worker_mapping_value(
                    role_worker_cls, wrapped_cls
                )
        return _remotify_like_worker_mapping_value(worker_cls, wrapped_cls)

    def run(self, config):
        # Ray actors do not share imported modules. Install this in the task
        # runner process before LLMServerManager imports verl's vLLM adapter.
        from verl_speco.integration.compat import check_compatible_verl

        check_compatible_verl()
        if bool(config.trainer.get("use_v1", False)):
            raise RuntimeError(
                "verl-SpeCo extends the legacy RayPPOTrainer on release/v0.9.0; "
                "set trainer.use_v1=false. The V1 trainer does not expose the "
                "online drafter training and atomic weight-publish hooks yet."
            )
        _install_vllm_import_compat_for_task_runner(config)
        if not _drafter_rollout_enabled(config):
            if _rollout_name(config) != "vllm":
                return super().run(config)
            # Keep the SPECO trainer's calculate_entropy=False old-logprob path.
            # The upstream legacy trainer forces entropy on here, which triggers a
            # costly torch.compile on NPU during the first training step.
            with _prepare_no_drafter_runtime_config(config):
                return self._run_with_speco_trainer(config)

        return self._run_with_speco_trainer(config)

    def _run_with_speco_trainer(self, config):
        from verl.utils.dataset.rl_dataset import collate_fn

        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        print(f"SpecoTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        actor_rollout_cls = self._with_speco_rollout_publish_mixin(
            actor_rollout_cls, config
        )
        self.add_critic_worker(config)
        speco_worker_cls = self.add_speco_drafter_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        model_config: HFModelConfig = omega_conf_to_dataclass(
            config.actor_rollout_ref.model
        )
        tokenizer = model_config.tokenizer
        processor = model_config.processor

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = SpecoRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            speco_worker_cls=speco_worker_cls,
        )

        trainer.init_workers()
        trainer.fit()
