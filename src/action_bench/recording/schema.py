"""Portable JSON capture contract (also used by Python 3.8 simulator hosts)."""

import math
from pathlib import PurePosixPath, PureWindowsPath


def validate_manifest(manifest):
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != "action_bench.capture.v1"
    ):
        raise ValueError("unsupported capture format")
    for key in ("epoch", "global_step", "num_frames"):
        value = manifest[key]
        if type(value) is not int or value < (1 if key == "num_frames" else 0):
            raise ValueError(f"capture {key} must be a nonnegative integer")
    if not math.isfinite(manifest["control_dt"]) or manifest["control_dt"] <= 0:
        raise ValueError("capture control_dt must be positive and finite")
    for entity in (*manifest["robots"], *manifest["objects"]):
        path = entity["asset"]["relative_path"]
        windows = PureWindowsPath(path)
        if (
            not path
            or PurePosixPath(path).is_absolute()
            or windows.drive
            or windows.root
            or ".." in PurePosixPath(path).parts
        ):
            raise ValueError(
                "capture asset paths must stay inside their host repository"
            )
        pose = entity.get("base_pose")
        if pose is not None and (len(pose) != 7 or not all(map(math.isfinite, pose))):
            raise ValueError("capture base_pose must contain finite xyz+wxyz")
    for spec in manifest["channels"].values():
        if spec["dtype"] not in {"float32", "bool", "int64"} or any(
            type(n) is not int or n < 0 for n in spec["shape"]
        ):
            raise ValueError("invalid capture channel shape or dtype")
    reset = manifest["channels"].get("reset", {})
    if (
        reset.get("dtype") != "bool"
        or len(reset["shape"]) > 1
        or (reset["shape"] and reset["shape"][0] == 0)
    ):
        raise ValueError(
            "capture requires a scalar or per-environment bool reset channel"
        )
    return manifest
