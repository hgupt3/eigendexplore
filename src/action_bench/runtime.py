"""Single-hand construction for simulator-independent rollout use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from action_bench.action_spaces import (
    ActionSpaceConfig,
    HandActionSpace,
    build_hand_action_space,
)
from action_bench.artifacts import PCAArtifact
from action_bench.hands import HandLayout
from action_bench.representations import (
    DirectRepresentation,
    PCARepresentation,
    Representation,
)
from action_bench.representations.config import (
    JointRepresentationConfig,
    PCARepresentationConfig,
    RepresentationConfig,
)


@dataclass(frozen=True, slots=True)
class RuntimeHand:
    hand_layout: HandLayout
    representation: Representation
    action_space: HandActionSpace


def build_runtime_hand(
    *,
    hand_layout: HandLayout,
    representation: RepresentationConfig,
    action_space: ActionSpaceConfig,
    artifact_path: Path | None = None,
    expected_artifact_id: str | None = None,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> RuntimeHand:
    """Build and place one hand's representation and action-space modules."""

    if not dtype.is_floating_point:
        raise TypeError(f"runtime dtype must be floating point, got {dtype}")
    if isinstance(representation, JointRepresentationConfig):
        if expected_artifact_id is not None:
            raise ValueError(
                "joint representation does not accept expected_artifact_id"
            )
        if artifact_path is not None:
            raise ValueError("joint representation does not accept artifact_path")
        module: Representation = DirectRepresentation(hand_layout.joint_count)
    elif isinstance(representation, PCARepresentationConfig):
        if expected_artifact_id is None:
            raise ValueError("PCA representation requires expected_artifact_id")
        if artifact_path is None:
            raise ValueError("PCA representation requires artifact_path")
        artifact = PCAArtifact.load(artifact_path, hand_layout=hand_layout)
        if artifact.manifest.construction != "mirrored_pair":
            raise ValueError("runtime PCA artifact must be a mirrored-pair artifact")
        if artifact.manifest.id != expected_artifact_id:
            raise ValueError(
                f"PCA artifact id {artifact.manifest.id!r} differs from "
                f"expected_artifact_id {expected_artifact_id!r}"
            )
        module = PCARepresentation(
            artifact,
            hand_layout=hand_layout,
            k=representation.k,
        )
    else:
        raise TypeError(
            f"unsupported representation config: {type(representation).__name__}"
        )

    compiled_action_space = build_hand_action_space(
        action_space,
        hand_layout=hand_layout,
        representation=module,
        device=device,
        dtype=dtype,
    )
    if not any(child is module for child in compiled_action_space.modules()):
        module.to(device=device, dtype=dtype)
    return RuntimeHand(
        hand_layout=hand_layout,
        representation=module,
        action_space=compiled_action_space,
    )
