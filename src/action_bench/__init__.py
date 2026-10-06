"""Stable runtime construction API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ActionSpaceConfig": ("action_bench.action_spaces", "ActionSpaceConfig"),
    "DirectRepresentation": ("action_bench.representations", "DirectRepresentation"),
    "EigenAbsoluteConfig": (
        "action_bench.action_spaces",
        "EigenAbsoluteConfig",
    ),
    "EigenDeltaJointDeltaConfig": (
        "action_bench.action_spaces",
        "EigenDeltaJointDeltaConfig",
    ),
    "HandBinding": ("action_bench.hands", "HandBinding"),
    "HandLayout": ("action_bench.hands", "HandLayout"),
    "JointAbsoluteConfig": ("action_bench.action_spaces", "JointAbsoluteConfig"),
    "JointAbsoluteEigenResidualConfig": (
        "action_bench.action_spaces",
        "JointAbsoluteEigenResidualConfig",
    ),
    "JointDeltaConfig": ("action_bench.action_spaces", "JointDeltaConfig"),
    "PCARepresentation": ("action_bench.representations", "PCARepresentation"),
    "Representation": ("action_bench.representations", "Representation"),
    "RepresentationConfig": (
        "action_bench.representations.config",
        "RepresentationConfig",
    ),
    "build_hand_action_space": (
        "action_bench.action_spaces",
        "build_hand_action_space",
    ),
    "build_runtime_hand": ("action_bench.runtime", "build_runtime_hand"),
    "available_hands": ("action_bench.catalog", "available_hands"),
    "load_basis": ("action_bench.catalog", "load_basis"),
    "load_eigendexplore": ("action_bench.exploration", "load_eigendexplore"),
    "load_custom_hand_layout": ("action_bench.hands", "load_custom_hand_layout"),
    "load_hand_layout": ("action_bench.hands", "load_hand_layout"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
