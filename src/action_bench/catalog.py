"""Exact lookup of the bundled EgoSuite PCA artifacts."""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml

from .artifacts import PCAArtifact
from .hands import load_hand_layout


@dataclass(frozen=True)
class BasisEntry:
    hand: str
    path: Path
    artifact_id: str
    k: int


def catalog() -> dict:
    return yaml.safe_load(
        files("action_bench").joinpath("assets/catalog.yaml").read_text()
    )


def available_hands() -> tuple[str, ...]:
    return tuple(catalog()["hands"])


def basis_entry(hand: str) -> BasisEntry:
    entry = catalog()["hands"][hand]
    path = Path(str(files("action_bench").joinpath("assets", hand)))
    return BasisEntry(hand, path, entry["artifact_id"], entry["k"])


def load_basis(hand: str, *, side: str = "right") -> PCAArtifact:
    entry = basis_entry(hand)
    artifact = PCAArtifact.load(
        entry.path, hand_layout=load_hand_layout(f"{hand}_{side}")
    )
    if artifact.manifest.id != entry.artifact_id:
        raise ValueError("artifact ID differs from packaged catalog")
    return artifact
