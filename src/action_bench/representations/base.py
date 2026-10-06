"""Runtime representation contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Representation(nn.Module, ABC):
    """Maps between ordered hand joints and representation coordinates."""

    joint_dim: int
    coordinate_dim: int

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
        anchor = self._runtime_anchor
        if value.device != anchor.device:
            raise ValueError(
                f"{name} device mismatch: input is on {value.device}, "
                f"representation is on {anchor.device}"
            )
        if value.dtype != anchor.dtype:
            raise ValueError(
                f"{name} dtype mismatch: input is {value.dtype}, "
                f"representation is {anchor.dtype}"
            )

    @abstractmethod
    def encode_posture(self, joints: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode_posture(self, coordinates: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearDisplacementRepresentation(Representation, ABC):
    """Optional capability implemented by direct and linear PCA coordinates."""

    @abstractmethod
    def encode_displacement(self, joint_delta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode_displacement(self, coordinate_delta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
