"""Fully batched runtime action spaces with explicit host-owned state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import nn

from action_bench.action_spaces.config import (
    ActionSpaceConfig,
    DecodedOffsetScaleConfig,
    EigenAbsoluteConfig,
    EigenDeltaJointDeltaConfig,
    JointAbsoluteConfig,
    JointAbsoluteEigenResidualConfig,
    JointDeltaConfig,
    JointScaleConfig,
    PCAScaleConfig,
)
from action_bench.hands import HandLayout
from action_bench.representations import (
    DirectRepresentation,
    PCARepresentation,
    Representation,
)


def _value_tensor(
    value: float | Sequence[float],
    dimension: int,
    *,
    name: str,
) -> torch.Tensor:
    if isinstance(value, (float, int)):
        return torch.full((dimension,), float(value), dtype=torch.float32)
    tensor = torch.tensor(tuple(value), dtype=torch.float32)
    if tensor.shape != (dimension,):
        raise ValueError(
            f"{name} vector must have length {dimension}, got {tensor.shape[0]}"
        )
    return tensor


def _joint_scale(config: JointScaleConfig, layout: HandLayout) -> torch.Tensor:
    scale = _value_tensor(config.value, layout.joint_count, name="joint scale")
    if config.unit == "joint_range_fraction":
        joint_range = torch.tensor(layout.joint_upper) - torch.tensor(
            layout.joint_lower
        )
        scale = scale * joint_range
    return scale


def _pca_scale(
    config: PCAScaleConfig,
    representation: PCARepresentation,
) -> torch.Tensor:
    scale = _value_tensor(
        config.value,
        representation.coordinate_dim,
        name="PCA scale",
    )
    if config.unit == "coefficient_span_fraction":
        span = representation.coefficient_span.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        dead = torch.nonzero(span <= 0.0, as_tuple=False).flatten().tolist()
        if dead:
            raise ValueError(
                f"PCA coefficient-span scaling selected zero-span coordinates: {dead}"
            )
        scale = scale * span
    return scale


def _decoded_offset_scale(config: DecodedOffsetScaleConfig) -> torch.Tensor:
    return torch.tensor(config.value, dtype=torch.float32)


class ActionBlock(NamedTuple):
    """One named contiguous slice of an action vector."""

    name: str
    start: int
    dimension: int


def _action_blocks(*parts: tuple[str, int]) -> tuple[ActionBlock, ...]:
    blocks: list[ActionBlock] = []
    start = 0
    for name, dimension in parts:
        blocks.append(ActionBlock(name, start, dimension))
        start += dimension
    return tuple(blocks)


class HandActionSpace(nn.Module):
    """Base for a single hand's fixed action layout."""

    def __init__(self, layout: HandLayout, action_dim: int):
        super().__init__()
        self.layout = layout
        self.action_dim = action_dim
        self.joint_dim = layout.joint_count
        lower, upper = layout.limits(device="cpu", dtype=torch.float32)
        self.register_buffer("joint_lower", lower, persistent=True)
        self.register_buffer("joint_upper", upper, persistent=True)

    def _check(
        self,
        value: torch.Tensor,
        *,
        dimension: int,
        name: str,
    ) -> None:
        if value.ndim < 1 or value.shape[-1] != dimension:
            raise ValueError(
                f"{name} expected last dimension {dimension}, got {tuple(value.shape)}"
            )
        if value.device != self.joint_lower.device:
            raise ValueError(
                f"{name} device mismatch: input is on {value.device}, "
                f"action space is on {self.joint_lower.device}"
            )
        if value.dtype != self.joint_lower.dtype:
            raise ValueError(
                f"{name} dtype mismatch: input is {value.dtype}, "
                f"action space is {self.joint_lower.dtype}"
            )

    def _check_action(self, action: torch.Tensor) -> None:
        self._check(action, dimension=self.action_dim, name="action")

    def _clamp(self, target: torch.Tensor) -> torch.Tensor:
        return torch.clamp(target, min=self.joint_lower, max=self.joint_upper)

    @property
    def action_layout(self) -> tuple[ActionBlock, ...]:
        raise NotImplementedError(
            f"{type(self).__name__} does not define an action layout"
        )


class DeltaHandActionSpace(HandActionSpace):
    """Delta space whose accumulator is supplied and owned by the host."""

    def decode_delta(self, action: torch.Tensor) -> torch.Tensor:
        self._check_action(action)
        return action @ self.decode_matrix

    def decode(
        self,
        action: torch.Tensor,
        previous_joint_target: torch.Tensor,
    ) -> torch.Tensor:
        self._check_action(action)
        self._check(
            previous_joint_target,
            dimension=self.joint_dim,
            name="previous_joint_target",
        )
        return self._clamp(previous_joint_target + self.decode_delta(action))


class JointAbsoluteSpace(HandActionSpace):
    def __init__(self, layout: HandLayout):
        super().__init__(layout, action_dim=layout.joint_count)
        self.register_buffer(
            "joint_midpoint",
            0.5 * (self.joint_lower + self.joint_upper),
            persistent=True,
        )
        self.register_buffer(
            "joint_half_range",
            0.5 * (self.joint_upper - self.joint_lower),
            persistent=True,
        )

    def decode(self, action: torch.Tensor) -> torch.Tensor:
        self._check_action(action)
        return self._clamp(self.joint_midpoint + action * self.joint_half_range)

    @property
    def action_layout(self) -> tuple[ActionBlock, ...]:
        return _action_blocks(("joint", self.joint_dim))


class JointAbsoluteEigenResidualSpace(JointAbsoluteSpace):
    def __init__(
        self,
        layout: HandLayout,
        representation: PCARepresentation,
        residual_scale: torch.Tensor,
    ):
        if representation.joint_dim != layout.joint_count:
            raise ValueError(
                "PCA representation joint dimension differs from hand layout"
            )
        _require_two_sided_absolute_coordinates(representation)
        super().__init__(layout)
        self.action_dim = layout.joint_count + representation.coordinate_dim
        self.representation = representation
        self.eigen_dim = representation.coordinate_dim
        coefficient_low = representation.coefficient_low
        coefficient_high = representation.coefficient_high
        self.register_buffer(
            "coefficient_low",
            coefficient_low.clone(),
            persistent=True,
        )
        self.register_buffer(
            "coefficient_high",
            coefficient_high.clone(),
            persistent=True,
        )
        zero = representation.coefficient_low.new_zeros(representation.coordinate_dim)
        self.register_buffer(
            "decode_center",
            representation.decode_posture(zero).detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "residual_scale",
            residual_scale,
            persistent=True,
        )

    def _coefficients(self, action: torch.Tensor) -> torch.Tensor:
        return torch.where(
            action < 0.0,
            (-action) * self.coefficient_low,
            action * self.coefficient_high,
        )

    def decode(self, action: torch.Tensor) -> torch.Tensor:
        self._check_action(action)
        joint_action = action[..., : self.joint_dim]
        eigen_action = action[..., self.joint_dim :]
        joint_target = self.joint_midpoint + joint_action * self.joint_half_range
        decoded = self.representation.decode_posture(self._coefficients(eigen_action))
        target = joint_target + self.residual_scale * (decoded - self.decode_center)
        return self._clamp(target)

    @property
    def action_layout(self) -> tuple[ActionBlock, ...]:
        return _action_blocks(
            ("joint", self.joint_dim),
            ("eigen", self.eigen_dim),
        )


class JointDeltaSpace(DeltaHandActionSpace):
    def __init__(self, layout: HandLayout, joint_scale: torch.Tensor):
        super().__init__(layout, action_dim=layout.joint_count)
        if joint_scale.shape != (layout.joint_count,):
            raise ValueError("joint_scale shape differs from hand layout")
        self.register_buffer(
            "decode_matrix",
            torch.diag(joint_scale),
            persistent=True,
        )

    @property
    def action_layout(self) -> tuple[ActionBlock, ...]:
        return _action_blocks(("joint_delta", self.joint_dim))


def _require_two_sided_absolute_coordinates(
    representation: PCARepresentation,
) -> None:
    invalid = (
        torch.nonzero(
            (representation.coefficient_low >= 0.0)
            | (representation.coefficient_high <= 0.0),
            as_tuple=False,
        )
        .flatten()
        .tolist()
    )
    if invalid:
        raise ValueError(
            "absolute PCA actions require demonstrated negative and positive "
            f"coefficient ranges; invalid coordinates: {invalid}"
        )


class EigenAbsoluteSpace(HandActionSpace):
    """Absolute Eigen posture: each action maps to its P1/P99 coefficient bound."""

    def __init__(self, layout: HandLayout, representation: PCARepresentation):
        if representation.joint_dim != layout.joint_count:
            raise ValueError(
                "PCA representation joint dimension differs from hand layout"
            )
        _require_two_sided_absolute_coordinates(representation)
        super().__init__(layout, action_dim=representation.coordinate_dim)
        self.representation = representation
        self.register_buffer(
            "coefficient_low",
            representation.coefficient_low.clone(),
            persistent=True,
        )
        self.register_buffer(
            "coefficient_high",
            representation.coefficient_high.clone(),
            persistent=True,
        )

    def decode(self, action: torch.Tensor) -> torch.Tensor:
        self._check_action(action)
        coefficients = torch.where(
            action < 0.0,
            (-action) * self.coefficient_low,
            action * self.coefficient_high,
        )
        return self._clamp(self.representation.decode_posture(coefficients))

    @property
    def action_layout(self) -> tuple[ActionBlock, ...]:
        return _action_blocks(("eigen", self.action_dim))


class EigenDeltaJointDeltaSpace(DeltaHandActionSpace):
    def __init__(
        self,
        layout: HandLayout,
        representation: PCARepresentation,
        eigen_scale: torch.Tensor,
        joint_scale: torch.Tensor,
    ):
        if representation.joint_dim != layout.joint_count:
            raise ValueError(
                "PCA representation joint dimension differs from hand layout"
            )
        super().__init__(
            layout,
            action_dim=representation.coordinate_dim + layout.joint_count,
        )
        matrix = torch.cat(
            (
                eigen_scale[:, None] * representation.components.detach(),
                torch.diag(joint_scale),
            ),
            dim=0,
        )
        self.register_buffer("decode_matrix", matrix, persistent=True)

    @property
    def action_layout(self) -> tuple[ActionBlock, ...]:
        return _action_blocks(
            ("eigen", self.action_dim - self.joint_dim),
            ("joint_delta", self.joint_dim),
        )


def build_hand_action_space(
    config: ActionSpaceConfig,
    *,
    hand_layout: HandLayout,
    representation: Representation,
    device: torch.device | str,
    dtype: torch.dtype,
) -> HandActionSpace:
    if not dtype.is_floating_point:
        raise TypeError("runtime dtype must be floating point")
    if isinstance(config, (JointAbsoluteConfig, JointDeltaConfig)):
        if not isinstance(representation, DirectRepresentation):
            raise TypeError("joint actions require DirectRepresentation")
        space = (
            JointAbsoluteSpace(hand_layout)
            if isinstance(config, JointAbsoluteConfig)
            else JointDeltaSpace(
                hand_layout, _joint_scale(config.joint_scale, hand_layout)
            )
        )
    else:
        if not isinstance(representation, PCARepresentation):
            raise TypeError("eigen actions require PCARepresentation")
        if isinstance(config, EigenAbsoluteConfig):
            space = EigenAbsoluteSpace(hand_layout, representation)
        elif isinstance(config, EigenDeltaJointDeltaConfig):
            space = EigenDeltaJointDeltaSpace(
                hand_layout,
                representation,
                _pca_scale(config.eigen_scale, representation),
                _joint_scale(config.joint_scale, hand_layout),
            )
        elif isinstance(config, JointAbsoluteEigenResidualConfig):
            space = JointAbsoluteEigenResidualSpace(
                hand_layout,
                representation,
                _decoded_offset_scale(config.residual_scale),
            )
        else:
            raise TypeError(f"unsupported action-space config: {type(config).__name__}")
    return space.to(device=device, dtype=dtype)
