"""Measured-posture observations and host-owned joint integrators for SimToolReal."""

import torch
from torch import nn

from action_bench.action_spaces import DeltaHandActionSpace
from action_bench.metrics import (
    HandActionMetrics,
    finalize_flush,
    representation_run_config,
)
from action_bench.settings import Study, build_study_hand

from .layout import SIMTOOLREAL_SHARPA_HOST_JOINT_NAMES, bind_simtoolreal_sharpa_layout


class SimToolRealAdapter(nn.Module):
    def __init__(self, study: Study):
        super().__init__()
        if study.benchmark != "simtoolreal":
            raise ValueError("adapter requires a SimToolReal study")
        runtime = build_study_hand(study, side="left")
        self.study = study
        self.hand_layout = runtime.hand_layout
        self.representation = runtime.representation
        self.action_space = runtime.action_space
        self.binding = bind_simtoolreal_sharpa_layout(self.hand_layout)
        self.host_joint_names = SIMTOOLREAL_SHARPA_HOST_JOINT_NAMES
        self.action_dim = self.action_space.action_dim
        self.is_delta = isinstance(self.action_space, DeltaHandActionSpace)
        self.measured_latent_dim = (
            study.k
            if study.method not in {"joint_absolute", "joint_delta", "eigendexplore"}
            else 0
        )
        self.latent_obs_dim = self.measured_latent_dim * (2 if self.is_delta else 1)
        self.metrics = None
        self._wired = False

    def _check(self, value, width):
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError("input shape differs from adapter contract")
        anchor = self.action_space.joint_lower
        if value.device != anchor.device or value.dtype != anchor.dtype:
            raise ValueError("input must match adapter device and dtype")

    def decode(self, hand_actions, prev_hand_targets):
        self._check(hand_actions, self.action_dim)
        self._check(prev_hand_targets, self.hand_layout.joint_count)
        if hand_actions.shape[0] != prev_hand_targets.shape[0]:
            raise ValueError("action and target batches differ")
        actions = hand_actions.clamp(-1.0, 1.0)
        previous = self.binding.pack(prev_hand_targets)
        target = (
            self.action_space.decode(actions, previous)
            if self.is_delta
            else self.action_space.decode(actions)
        )
        if self.metrics is not None:
            self.metrics.update_step(
                hand_actions, actions, target, previous, None, None, None, None, None
            )
        return self.binding.unpack(target)

    def encode_observation(self, hand_joint_pos, hand_prev_targets):
        self._check(hand_joint_pos, self.hand_layout.joint_count)
        self._check(hand_prev_targets, self.hand_layout.joint_count)
        if hand_joint_pos.shape != hand_prev_targets.shape:
            raise ValueError("posture and target batches differ")
        measured = self.binding.pack(hand_joint_pos)
        if self.metrics is not None:
            self.metrics.update_tracking(
                measured, self.binding.pack(hand_prev_targets), None, None, None
            )
        if not self.measured_latent_dim:
            return measured.new_empty((measured.shape[0], 0))
        latent = self.representation.encode_posture(measured)
        if self.is_delta:
            commanded = self.representation.encode_posture(
                self.binding.pack(hand_prev_targets)
            )
            return torch.cat((latent, commanded), dim=1)
        return latent

    def notify_hand_state_reset(self, env_indices):
        if self.metrics is not None:
            mask = torch.zeros(
                self.metrics.num_envs, dtype=torch.bool, device=env_indices.device
            )
            mask[env_indices] = True
            self.metrics.notify_reset(mask)

    def wire(self, cfg, *, device, dtype=torch.float32, metrics_detail="full"):
        if (
            self._wired
            or cfg.action.hand_action_transform is not None
            or cfg.obs.latent_obs_fn is not None
        ):
            raise ValueError("action/observation hooks are already wired")
        if cfg.action.hand_state_reset_fn is not None:
            raise ValueError("hand reset hook is already wired")
        self.to(device=device, dtype=dtype)
        kwargs = (
            dict(
                component_std=self.representation.component_std,
                coefficient_low=self.representation.coefficient_low,
                coefficient_high=self.representation.coefficient_high,
            )
            if self.measured_latent_dim
            else {}
        )
        self.metrics = HandActionMetrics(
            self.action_space.action_layout,
            tuple(self.hand_layout.joint_names),
            self.action_space.joint_lower,
            self.action_space.joint_upper,
            cfg.scene.num_envs,
            detail=metrics_detail,
            **kwargs,
        )
        cfg.action.hand_action_dim = self.action_dim
        cfg.action.hand_action_transform = self.decode
        cfg.action.hand_state_reset_fn = self.notify_hand_state_reset
        cfg.obs.num_latent_obs = self.latent_obs_dim
        cfg.obs.latent_obs_fn = self.encode_observation
        self._wired = True
        return self

    def representation_run_config(self):
        return representation_run_config(
            action_space=self.action_space,
            representation=self.representation,
            hand_layout=self.hand_layout,
        )

    def flush_action_metrics(self):
        return {} if self.metrics is None else finalize_flush(self.metrics.flush())
