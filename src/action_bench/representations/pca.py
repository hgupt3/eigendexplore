"""GPU-resident PCA/eigengrasp representation."""

from __future__ import annotations

import torch

from action_bench.artifacts import PCAArtifact
from action_bench.hands import HandLayout
from action_bench.representations.base import LinearDisplacementRepresentation


class PCARepresentation(LinearDisplacementRepresentation):
    def __init__(
        self,
        artifact: PCAArtifact,
        *,
        hand_layout: HandLayout,
        k: int,
    ):
        super().__init__()
        artifact.validate(hand_layout=hand_layout)
        if not 1 <= k <= artifact.manifest.component_count:
            raise ValueError(
                f"PCA k must be in [1, {artifact.manifest.component_count}], got {k}"
            )
        self.joint_dim = hand_layout.joint_count
        self.coordinate_dim = k
        self.artifact_id = artifact.manifest.id
        self.producer_revision = artifact.manifest.producer_revision
        self.dataset = artifact.manifest.dataset
        self.fit_split = artifact.manifest.fit.split
        self.split_protocol = artifact.manifest.fit.split_protocol
        self.fit_weighting = artifact.manifest.fit.weighting
        self.fit_summary = artifact.manifest.fit_summary
        self.source_validation_protocols = artifact.manifest.source_validation_protocols
        self.construction = artifact.manifest.construction
        self.canonical_hand_layout = artifact.manifest.canonical_hand_layout
        self.deployment_hand_layouts = tuple(
            deployment.hand_layout
            for deployment in artifact.manifest.deployment_layouts
        )
        self.register_buffer("_runtime_anchor", torch.empty(0), persistent=False)
        self.register_buffer("mean", artifact.mean.clone(), persistent=True)
        self.register_buffer(
            "components",
            artifact.components[:k].clone(),
            persistent=True,
        )
        self.register_buffer(
            "coefficient_low",
            artifact.coefficient_low[:k].clone(),
            persistent=True,
        )
        self.register_buffer(
            "coefficient_high",
            artifact.coefficient_high[:k].clone(),
            persistent=True,
        )
        self.register_buffer(
            "component_std",
            artifact.component_std[:k].clone(),
            persistent=True,
        )
        self.register_buffer(
            "explained_variance",
            artifact.explained_variance[:k].clone(),
            persistent=True,
        )

    @property
    def coefficient_span(self) -> torch.Tensor:
        return self.coefficient_high - self.coefficient_low

    def encode_posture(self, joints: torch.Tensor) -> torch.Tensor:
        self._check(joints, dimension=self.joint_dim, name="encode_posture")
        return (joints - self.mean) @ self.components.T

    def decode_posture(self, coordinates: torch.Tensor) -> torch.Tensor:
        self._check(
            coordinates,
            dimension=self.coordinate_dim,
            name="decode_posture",
        )
        return self.mean + coordinates @ self.components

    def encode_displacement(self, joint_delta: torch.Tensor) -> torch.Tensor:
        self._check(
            joint_delta,
            dimension=self.joint_dim,
            name="encode_displacement",
        )
        return joint_delta @ self.components.T

    def decode_displacement(self, coordinate_delta: torch.Tensor) -> torch.Tensor:
        self._check(
            coordinate_delta,
            dimension=self.coordinate_dim,
            name="decode_displacement",
        )
        return coordinate_delta @ self.components
