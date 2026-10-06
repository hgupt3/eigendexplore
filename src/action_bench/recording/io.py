"""Atomic capture publication and array validation, without simulator dependencies."""

import json
import os
import shutil
from pathlib import Path

import numpy as np

from .schema import validate_manifest


def _validate_channels(manifest, channels):
    validate_manifest(manifest)
    if set(manifest["channels"]) != set(channels):
        raise ValueError("capture channel set mismatch")
    for name, spec in manifest["channels"].items():
        array = channels[name]
        if array.shape != (
            manifest["num_frames"],
            *spec["shape"],
        ) or array.dtype != np.dtype(spec["dtype"]):
            raise ValueError(f"capture channel {name!r} shape or dtype mismatch")


def write_capture(output_root, manifest, channels):
    """Publish a complete bundle; never replace an existing epoch."""
    _validate_channels(manifest, channels)
    output_root = Path(output_root)
    name = f"step_{manifest['epoch']:06d}"
    target = output_root / name
    staging = output_root / (".staging." + name)
    output_root.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise FileExistsError(target)
    staging.mkdir()
    try:
        np.savez(staging / "channels.npz", **channels)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )
        if os.path.lexists(target):
            raise FileExistsError(target)
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def read_capture(path):
    """Read one bundle, rejecting inconsistent metadata and tensor arrays."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    with np.load(path / "channels.npz", allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate capture channel names")
        channels = {name: archive[name] for name in archive.files}
    _validate_channels(manifest, channels)
    return manifest, channels
