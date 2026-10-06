"""The five retained target maps and their physical scale units."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ScaleValue = float | tuple[float, ...]


class _ScaleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: ScaleValue

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: ScaleValue) -> ScaleValue:
        values = (value,) if isinstance(value, float) else value
        if not values:
            raise ValueError("scale vector must not be empty")
        if any(not math.isfinite(item) or item <= 0.0 for item in values):
            raise ValueError("every scale value must be finite and positive")
        return value


class JointScaleConfig(_ScaleConfig):
    unit: Literal["joint_range_fraction", "raw"]


class PCAScaleConfig(_ScaleConfig):
    unit: Literal["coefficient_span_fraction", "raw"]


class DecodedOffsetScaleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: float = Field(ge=0.0, allow_inf_nan=False)
    unit: Literal["decoded_offset_fraction"]


class _ActionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JointAbsoluteConfig(_ActionConfig):
    type: Literal["joint_absolute"]


class JointDeltaConfig(_ActionConfig):
    type: Literal["joint_delta"]
    joint_scale: JointScaleConfig


class EigenAbsoluteConfig(_ActionConfig):
    type: Literal["eigen_absolute"]


class EigenDeltaJointDeltaConfig(_ActionConfig):
    type: Literal["eigen_delta_joint_delta"]
    eigen_scale: PCAScaleConfig
    joint_scale: JointScaleConfig


class JointAbsoluteEigenResidualConfig(_ActionConfig):
    type: Literal["joint_absolute_eigen_residual"]
    residual_scale: DecodedOffsetScaleConfig


ActionSpaceConfig = Annotated[
    JointAbsoluteConfig
    | JointDeltaConfig
    | EigenAbsoluteConfig
    | EigenDeltaJointDeltaConfig
    | JointAbsoluteEigenResidualConfig,
    Field(discriminator="type"),
]
