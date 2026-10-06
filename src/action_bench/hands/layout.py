"""Validated ordered joint layout for one independently controlled hand."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .binding import HandBinding

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HandLayoutSpec(BaseModel):
    """YAML schema for a hand's independent command coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    family: str = Field(min_length=1)
    side: Literal["left", "right"]
    joint_names: tuple[str, ...] = Field(min_length=1)
    joint_signs: tuple[Literal[-1, 1], ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]

    @model_validator(mode="after")
    def validate_coordinates(self) -> HandLayoutSpec:
        count = len(self.joint_names)
        if len(set(self.joint_names)) != count:
            duplicates = sorted(
                {name for name in self.joint_names if self.joint_names.count(name) > 1}
            )
            raise ValueError(f"joint_names contains duplicates: {duplicates}")
        if (
            len(self.joint_signs) != count
            or len(self.joint_lower) != count
            or len(self.joint_upper) != count
        ):
            raise ValueError(
                "joint_names, joint_signs, joint_lower, and joint_upper must "
                "have identical lengths"
            )
        for index, (lower, upper) in enumerate(
            zip(self.joint_lower, self.joint_upper, strict=True)
        ):
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError(f"joint limit {index} is not finite")
            if lower >= upper:
                raise ValueError(
                    f"joint {self.joint_names[index]!r} must satisfy lower < upper"
                )
        return self

    def to_layout(self) -> HandLayout:
        return HandLayout(
            id=self.id,
            family=self.family,
            side=self.side,
            joint_names=self.joint_names,
            joint_signs=self.joint_signs,
            joint_lower=self.joint_lower,
            joint_upper=self.joint_upper,
        )


@dataclass(frozen=True, slots=True)
class HandLayout:
    """Canonical finger coordinates plus an exact signed host-joint binding."""

    id: str
    family: str
    side: Literal["left", "right"]
    joint_names: tuple[str, ...]
    joint_signs: tuple[Literal[-1, 1], ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9_]*", self.id) is None:
            raise ValueError(f"invalid hand layout id: {self.id!r}")
        if not self.family:
            raise ValueError("hand family must not be empty")
        if self.side not in {"left", "right"}:
            raise ValueError(f"hand side must be left or right, got {self.side!r}")
        if not self.joint_names:
            raise ValueError("hand layout requires at least one joint")
        duplicates = sorted(
            name for name, count in Counter(self.joint_names).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"joint_names contains duplicates: {duplicates}")
        count = len(self.joint_names)
        if (
            len(self.joint_signs) != count
            or len(self.joint_lower) != count
            or len(self.joint_upper) != count
        ):
            raise ValueError(
                "joint_names, joint_signs, joint_lower, and joint_upper must "
                "have identical lengths"
            )
        if any(sign not in {-1, 1} for sign in self.joint_signs):
            raise ValueError("joint_signs entries must be exactly -1 or 1")
        for name, lower, upper in zip(
            self.joint_names,
            self.joint_lower,
            self.joint_upper,
            strict=True,
        ):
            if not name:
                raise ValueError("joint names must not be empty")
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError(f"joint {name!r} has nonfinite limits")
            if lower >= upper:
                raise ValueError(f"joint {name!r} must satisfy lower < upper")

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    def limits(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower = torch.tensor(self.joint_lower, device=device, dtype=dtype)
        upper = torch.tensor(self.joint_upper, device=device, dtype=dtype)
        return lower, upper

    def bind(self, simulator_joint_names: list[str] | tuple[str, ...]) -> "HandBinding":
        from action_bench.hands.binding import HandBinding

        return HandBinding(self, simulator_joint_names)
