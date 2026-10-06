"""Build-time serialization of action-representation run constants."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

_REPRESENTATION_TENSORS = (
    "explained_variance",
    "component_std",
    "coefficient_low",
    "coefficient_high",
)
_PROVENANCE_FIELDS = ("artifact_id", "dataset")
_SPACE_SCALES = (
    "joint_scale",
    "residual_scale",
)


def _tensor_list(value: Tensor) -> list[Any]:
    return value.detach().to(device="cpu").tolist()


def _single_hand_config(
    action_space: Any,
    representation: Any,
    hand_layout: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "action_layout": [
            {
                "name": str(block.name),
                "start": int(block.start),
                "dimension": int(block.dimension),
            }
            for block in action_space.action_layout
        ],
    }
    if hand_layout is not None and hasattr(hand_layout, "joint_names"):
        result["joint_names"] = list(hand_layout.joint_names)
    elif hasattr(action_space, "layout"):
        result["joint_names"] = list(action_space.layout.joint_names)
    if hasattr(action_space, "joint_lower"):
        result["joint_lower"] = _tensor_list(action_space.joint_lower)
    elif hand_layout is not None and hasattr(hand_layout, "joint_lower"):
        result["joint_lower"] = list(hand_layout.joint_lower)
    if hasattr(action_space, "joint_upper"):
        result["joint_upper"] = _tensor_list(action_space.joint_upper)
    elif hand_layout is not None and hasattr(hand_layout, "joint_upper"):
        result["joint_upper"] = list(hand_layout.joint_upper)

    _add_representation_config(result, representation)

    for name in _SPACE_SCALES:
        value = getattr(action_space, name, None)
        if isinstance(value, Tensor):
            result[name] = _tensor_list(value)
    decode_matrix = getattr(action_space, "decode_matrix", None)
    if isinstance(decode_matrix, Tensor) and decode_matrix.ndim == 2:
        result["decode_scale"] = _tensor_list(
            torch.linalg.vector_norm(decode_matrix.detach(), ord=2, dim=1)
        )
    return result


def _add_representation_config(result: dict[str, Any], representation: Any) -> None:
    if representation is None:
        return
    for name in _REPRESENTATION_TENSORS:
        value = getattr(representation, name, None)
        if isinstance(value, Tensor):
            result[name] = _tensor_list(value)
    for name in _PROVENANCE_FIELDS:
        value = getattr(representation, name, None)
        if value is not None:
            result[name] = value


def representation_run_config(
    action_space: Any, representation: Any = None, hand_layout: Any = None
) -> dict[str, Any]:
    """Return JSON-compatible constants for one hand's resolved runtime space."""
    return _single_hand_config(action_space, representation, hand_layout)
