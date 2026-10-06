"""Tensor-only target and observation transforms for the Python 3.8 host."""

from typing import Dict

import torch
from torch import nn

from action_bench.metrics import HandActionMetrics


class DextremeTorchScriptBundle(nn.Module):
    __constants__ = [
        "method",
        "action_dim",
        "joint_dim",
        "measured_latent_dim",
        "commanded_obs_dim",
        "metrics_num_envs",
        "eigen_dim",
    ]

    def __init__(self, runtime, binding, method: str, num_envs: int):
        super().__init__()
        space, rep = runtime.action_space, runtime.representation
        self.method = method
        self.action_dim = space.action_dim
        self.joint_dim = space.joint_dim
        self.eigen_dim = (
            rep.coordinate_dim if method not in ("joint_absolute", "joint_delta") else 0
        )
        self.measured_latent_dim = self.eigen_dim
        self.commanded_obs_dim = (
            self.joint_dim + self.eigen_dim
            if method in ("joint_delta", "eigen_delta_joint_delta")
            else 0
        )
        self.metrics_num_envs = num_envs
        self.register_buffer("joint_lower", space.joint_lower.detach().clone())
        self.register_buffer("joint_upper", space.joint_upper.detach().clone())
        self.register_buffer("layout_from_host", binding.layout_from_simulator.clone())
        self.register_buffer("host_from_layout", binding.simulator_from_layout.clone())
        self.register_buffer(
            "coordinate_signs", binding.coordinate_signs.float().clone()
        )
        self.register_buffer("midpoint", (self.joint_upper + self.joint_lower) * 0.5)
        self.register_buffer("half_range", (self.joint_upper - self.joint_lower) * 0.5)
        self.register_buffer(
            "encode_mean",
            rep.mean.clone() if self.eigen_dim else torch.zeros(self.joint_dim),
        )
        self.register_buffer(
            "components",
            rep.components.clone()
            if self.eigen_dim
            else torch.empty((0, self.joint_dim)),
        )
        self.register_buffer(
            "coefficient_low",
            rep.coefficient_low.clone() if self.eigen_dim else torch.empty(0),
        )
        self.register_buffer(
            "coefficient_high",
            rep.coefficient_high.clone() if self.eigen_dim else torch.empty(0),
        )
        self.register_buffer(
            "decode_matrix",
            space.decode_matrix.clone()
            if hasattr(space, "decode_matrix")
            else torch.empty((0, self.joint_dim)),
        )
        self.register_buffer(
            "residual_scale",
            space.residual_scale.clone()
            if hasattr(space, "residual_scale")
            else torch.tensor(0.0),
        )
        kwargs = (
            dict(
                component_std=rep.component_std,
                coefficient_low=rep.coefficient_low,
                coefficient_high=rep.coefficient_high,
            )
            if self.eigen_dim
            else {}
        )
        self.metrics = HandActionMetrics(
            space.action_layout,
            tuple(runtime.hand_layout.joint_names),
            self.joint_lower,
            self.joint_upper,
            num_envs,
            **kwargs,
        )

    def _check(self, value: torch.Tensor, width: int) -> None:
        if value.dim() != 2 or value.size(1) != width:
            raise RuntimeError("tensor shape differs from bundle contract")
        if (
            value.device != self.joint_lower.device
            or value.dtype != self.joint_lower.dtype
        ):
            raise RuntimeError("tensor device/dtype differs from bundle")

    def _pack(self, value: torch.Tensor) -> torch.Tensor:
        return value.index_select(1, self.layout_from_host) * self.coordinate_signs

    def _unpack(self, value: torch.Tensor) -> torch.Tensor:
        return (value * self.coordinate_signs).index_select(1, self.host_from_layout)

    def _coefficients(self, action: torch.Tensor) -> torch.Tensor:
        return torch.where(
            action < 0, -action * self.coefficient_low, action * self.coefficient_high
        )

    @torch.jit.export
    def decode(self, action: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        self._check(action, self.action_dim)
        self._check(previous, self.joint_dim)
        if action.size(0) != previous.size(0):
            raise RuntimeError("action and previous target batches differ")
        if self.method == "joint_absolute":
            target = self.midpoint + action * self.half_range
        elif self.method == "joint_delta" or self.method == "eigen_delta_joint_delta":
            target = self._pack(previous) + action @ self.decode_matrix
        elif self.method == "eigen_absolute":
            target = self.encode_mean + self._coefficients(action) @ self.components
        else:
            offset = self._coefficients(action[:, self.joint_dim :]) @ self.components
            target = (
                self.midpoint
                + action[:, : self.joint_dim] * self.half_range
                + self.residual_scale * offset
            )
        return self._unpack(
            torch.clamp(target, min=self.joint_lower, max=self.joint_upper)
        )

    @torch.jit.export
    def encode_obs(self, measured: torch.Tensor) -> torch.Tensor:
        self._check(measured, self.joint_dim)
        return (self._pack(measured) - self.encode_mean) @ self.components.T

    @torch.jit.export
    def commanded_obs(self, previous: torch.Tensor) -> torch.Tensor:
        self._check(previous, self.joint_dim)
        if self.commanded_obs_dim == 0:
            return previous.new_empty((previous.size(0), 0))
        canonical = self._pack(previous)
        normalized = (canonical - self.midpoint) / self.half_range
        if self.eigen_dim == 0:
            return normalized
        return torch.cat(
            (normalized, (canonical - self.encode_mean) @ self.components.T), 1
        )

    @torch.jit.export
    def metrics_update(
        self, action: torch.Tensor, previous: torch.Tensor, target: torch.Tensor
    ) -> None:
        self.metrics.update_step(
            action,
            torch.clamp(action, -1.0, 1.0),
            self._pack(target),
            self._pack(previous),
            None,
            None,
            None,
            None,
            None,
        )

    @torch.jit.export
    def metrics_update_measured(self, measured: torch.Tensor) -> None:
        self.metrics.update_tracking(self._pack(measured), None, None, None, None)

    @torch.jit.export
    def metrics_notify_reset(self, indices: torch.Tensor) -> None:
        mask = torch.zeros(
            self.metrics_num_envs, device=self.joint_lower.device, dtype=torch.bool
        )
        mask[indices] = True
        self.metrics.notify_reset(mask)

    @torch.jit.export
    def metrics_flush(self) -> Dict[str, torch.Tensor]:
        return self.metrics.flush()
