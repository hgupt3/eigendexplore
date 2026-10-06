"""Identity joint representation."""

from __future__ import annotations

import torch

from action_bench.representations.base import LinearDisplacementRepresentation


class DirectRepresentation(LinearDisplacementRepresentation):
    def __init__(self, joint_dim: int):
        super().__init__()
        if joint_dim < 1:
            raise ValueError("joint_dim must be positive")
        self.joint_dim = joint_dim
        self.coordinate_dim = joint_dim
        self.register_buffer("_runtime_anchor", torch.empty(0), persistent=False)

    def encode_posture(self, joints: torch.Tensor) -> torch.Tensor:
        self._check(joints, dimension=self.joint_dim, name="encode_posture")
        return joints

    def decode_posture(self, coordinates: torch.Tensor) -> torch.Tensor:
        self._check(
            coordinates,
            dimension=self.coordinate_dim,
            name="decode_posture",
        )
        return coordinates

    def encode_displacement(self, joint_delta: torch.Tensor) -> torch.Tensor:
        self._check(
            joint_delta,
            dimension=self.joint_dim,
            name="encode_displacement",
        )
        return joint_delta

    def decode_displacement(self, coordinate_delta: torch.Tensor) -> torch.Tensor:
        self._check(
            coordinate_delta,
            dimension=self.coordinate_dim,
            name="decode_displacement",
        )
        return coordinate_delta
