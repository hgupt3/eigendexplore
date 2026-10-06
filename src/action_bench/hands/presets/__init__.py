"""Packaged hand layout loading."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import yaml

from action_bench.hands.layout import HandLayout, HandLayoutSpec


def _parse_layout(path: Path) -> HandLayout:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return HandLayoutSpec.model_validate(payload).to_layout()


def list_hand_layouts() -> tuple[str, ...]:
    root = files(__package__)
    return tuple(
        sorted(
            path.name.removesuffix(".yaml")
            for path in root.iterdir()
            if path.name.endswith(".yaml")
        )
    )


def load_hand_layout(layout_id: str) -> HandLayout:
    if layout_id not in list_hand_layouts():
        available = ", ".join(list_hand_layouts())
        raise KeyError(f"unknown hand layout {layout_id!r}; available: {available}")
    resource = files(__package__) / f"{layout_id}.yaml"
    with resource.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    layout = HandLayoutSpec.model_validate(payload).to_layout()
    if layout.id != layout_id:
        raise ValueError(
            f"packaged hand layout filename {layout_id!r} disagrees with id {layout.id!r}"
        )
    return layout


def load_custom_hand_layout(path: str | Path) -> HandLayout:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return _parse_layout(resolved)
