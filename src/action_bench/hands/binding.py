"""Strict exact-name permutation between host and action-bench joint order."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch
from torch import nn

from action_bench.hands.layout import HandLayout


class HandBinding(nn.Module):
    """A precomputed signed bijection between host and canonical coordinates."""

    def __init__(self, layout: HandLayout, simulator_joint_names: Sequence[str]):
        super().__init__()
        names = tuple(simulator_joint_names)
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ValueError(f"simulator joint names contain duplicates: {duplicates}")

        expected = set(layout.joint_names)
        actual = set(names)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise ValueError(
                f"joint binding for {layout.id!r} is not an exact match: "
                + ", ".join(details)
            )

        simulator_index = {name: index for index, name in enumerate(names)}
        layout_index = {name: index for index, name in enumerate(layout.joint_names)}
        self.layout = layout
        self.simulator_joint_names = names
        self.register_buffer(
            "layout_from_simulator",
            torch.tensor(
                [simulator_index[name] for name in layout.joint_names],
                dtype=torch.long,
            ),
            persistent=True,
        )
        self.register_buffer(
            "simulator_from_layout",
            torch.tensor(
                [layout_index[name] for name in names],
                dtype=torch.long,
            ),
            persistent=True,
        )
        self.register_buffer(
            "coordinate_signs",
            torch.tensor(layout.joint_signs, dtype=torch.int8),
            persistent=True,
        )

    def _check(self, values: torch.Tensor, expected: int, operation: str) -> None:
        if values.ndim < 1 or values.shape[-1] != expected:
            raise ValueError(
                f"{operation} expected last dimension {expected}, got {tuple(values.shape)}"
            )
        if values.device != self.layout_from_simulator.device:
            raise ValueError(
                f"{operation} device mismatch: values are on {values.device}, "
                f"binding is on {self.layout_from_simulator.device}"
            )

    def pack(self, simulator_values: torch.Tensor) -> torch.Tensor:
        """Reorder/sign-transform host values into canonical coordinates."""

        self._check(simulator_values, len(self.simulator_joint_names), "pack")
        return (
            simulator_values.index_select(-1, self.layout_from_simulator)
            * self.coordinate_signs
        )

    def unpack(self, layout_values: torch.Tensor) -> torch.Tensor:
        """Sign-transform/reorder canonical values into exact host order."""

        self._check(layout_values, self.layout.joint_count, "unpack")
        physical_layout_values = layout_values * self.coordinate_signs
        return physical_layout_values.index_select(-1, self.simulator_from_layout)
