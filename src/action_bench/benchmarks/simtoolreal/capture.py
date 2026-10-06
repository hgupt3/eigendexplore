"""SimToolReal feeder for simulator-independent training state captures."""

from __future__ import annotations

import filecmp
import math
import shutil
from pathlib import Path
from typing import Any

import torch

from action_bench.recording import TrainingStateCapture

SIMTOOLREAL_CAPTURE_WINDOW_STEPS = 1800
SIMTOOLREAL_CAPTURE_MAX_START_DELAY_STEPS = 900

_HOST_REPO = "simtoolreal"
_ROBOT_NAME = "kuka_sharpa"
_ROBOT_ASSET = (
    "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
)
_TABLE_ASSET = "assets/urdf/table_narrow.urdf"
_MAX_CAPTURE_ENVS = 4
_CHANNEL_DTYPES = {"reset": torch.bool}


def _require_host_root(host_repo_root: str | Path) -> Path:
    root = Path(host_repo_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(
            f"SimToolReal host repository is not a directory: {root}"
        )
    return root


def _host_asset_relative_path(raw_path: str, host_root: Path) -> str:
    path = Path(raw_path).expanduser()
    resolved = path.resolve() if path.is_absolute() else (host_root / path).resolve()
    try:
        relative = resolved.relative_to(host_root)
    except ValueError as error:
        raise ValueError(
            f"SimToolReal asset is outside the host repository: {resolved}"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return relative.as_posix()


def _object_asset_relative_path(
    raw_path: str,
    *,
    host_root: Path,
    experiment_id: str,
) -> str:
    """Resolve a checked-in asset or persist a generated URDF under the host root."""

    source_path = Path(raw_path).expanduser()
    source = (
        source_path.resolve()
        if source_path.is_absolute()
        else (host_root / source_path).resolve()
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        return source.relative_to(host_root).as_posix()
    except ValueError:
        pass

    run_id_path = Path(experiment_id)
    if (
        not experiment_id
        or experiment_id in {".", ".."}
        or run_id_path.name != experiment_id
        or len(run_id_path.parts) != 1
    ):
        raise ValueError("experiment id must be a single path-safe component")
    relative = (
        Path("outputs") / "action_bench_capture_assets" / experiment_id / source.name
    )
    destination = host_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        # Resumed runs reuse identical generated assets.
        if (
            destination.is_file()
            and not destination.is_symlink()
            and filecmp.cmp(source, destination, shallow=False)
        ):
            return relative.as_posix()
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    return relative.as_posix()


def _capture_env_count(value: int, *, num_envs: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("SimToolReal captured_envs must be an integer")
    if value < 1 or value > _MAX_CAPTURE_ENVS:
        raise ValueError(
            f"SimToolReal captured_envs must be between 1 and {_MAX_CAPTURE_ENVS}"
        )
    if value > num_envs:
        raise ValueError(
            f"SimToolReal cannot capture {value} envs from an environment with "
            f"{num_envs} envs"
        )
    return value


def _channel_specs(captured_envs: int) -> dict[str, tuple[int, ...]]:
    return {
        f"joint_pos_{_ROBOT_NAME}": (captured_envs, 29),
        f"joint_target_{_ROBOT_NAME}": (captured_envs, 29),
        "object_pose": (captured_envs, 7),
        "goal_pose": (captured_envs, 7),
        "table_pose": (captured_envs, 7),
        "reset": (captured_envs,),
    }


def _object_assets(
    inner: Any,
    host_root: Path,
    experiment_id: str,
    captured_envs: int,
) -> tuple[str, ...]:
    paths = tuple(inner._object_urdf_paths)
    index_tensor = inner._object_asset_index_per_env
    if index_tensor.ndim != 1 or index_tensor.shape[0] != inner.num_envs:
        raise ValueError("SimToolReal object asset indices have the wrong shape")
    asset_indices = tuple(
        int(value) for value in index_tensor[:captured_envs].detach().cpu().tolist()
    )
    resolved: dict[int, str] = {}
    assets: list[str] = []
    for env_index, asset_index in enumerate(asset_indices):
        if asset_index < 0 or asset_index >= len(paths):
            raise ValueError(
                f"SimToolReal env-{env_index} object asset index {asset_index} "
                "is out of range"
            )
        if asset_index not in resolved:
            resolved[asset_index] = _object_asset_relative_path(
                paths[asset_index],
                host_root=host_root,
                experiment_id=experiment_id,
            )
        assets.append(resolved[asset_index])
    return tuple(assets)


def _object_scales(
    inner: Any, captured_envs: int
) -> tuple[tuple[float, float, float], ...]:
    scales = inner._object_scale_per_env
    expected_shape = (inner.num_envs, 3)
    if tuple(scales.shape) != expected_shape:
        raise ValueError(
            f"SimToolReal object scales have shape {tuple(scales.shape)}; "
            f"expected {expected_shape}"
        )
    captured = tuple(
        tuple(float(value) for value in row)
        for row in scales[:captured_envs].detach().cpu().tolist()
    )
    for env_index, values in enumerate(captured):
        if len(values) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError(
                f"SimToolReal env-{env_index} object scale must contain three "
                "positive values"
            )
    return captured


def _robot_base_pose(inner: Any) -> tuple[float, ...]:
    initial_state = inner.robot.cfg.init_state
    values = tuple(float(value) for value in (*initial_state.pos, *initial_state.rot))
    if len(values) != 7 or any(not math.isfinite(value) for value in values):
        raise ValueError("SimToolReal robot base pose must contain finite xyz+wxyz")
    return values


def build_capture(
    env: Any,
    output_root: str | Path,
    *,
    experiment_id: str,
    host_repo_root: str | Path,
    captured_envs: int = _MAX_CAPTURE_ENVS,
    window_steps: int = SIMTOOLREAL_CAPTURE_WINDOW_STEPS,
    max_start_delay_steps: int = SIMTOOLREAL_CAPTURE_MAX_START_DELAY_STEPS,
) -> TrainingStateCapture:
    """Build a batched SimToolReal capture and its immutable manifest template."""

    inner = env.unwrapped
    captured_envs = _capture_env_count(captured_envs, num_envs=inner.num_envs)
    channel_specs = _channel_specs(captured_envs)
    host_root = _require_host_root(host_repo_root)
    source_device = inner.robot.data.joint_pos.device

    robot_asset = _host_asset_relative_path(inner.cfg.assets.robot_urdf, host_root)
    if robot_asset != _ROBOT_ASSET:
        raise ValueError(
            f"SimToolReal robot asset is {robot_asset!r}; expected {_ROBOT_ASSET!r}"
        )
    table_asset = _host_asset_relative_path(inner.cfg.assets.table_urdf, host_root)
    if table_asset != _TABLE_ASSET:
        raise ValueError(
            f"SimToolReal table asset is {table_asset!r}; expected {_TABLE_ASSET!r}"
        )
    object_assets = _object_assets(inner, host_root, experiment_id, captured_envs)
    object_scales = _object_scales(inner, captured_envs)
    control_dt = float(inner.step_dt)
    if not math.isclose(control_dt, 1.0 / 60.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"SimToolReal capture requires 60 Hz policy control; got dt={control_dt}"
        )

    manifest_template = {
        "schema_version": 1,
        "format": "action_bench.capture.v1",
        "benchmark": "simtoolreal",
        "experiment_id": experiment_id,
        "control_dt": control_dt,
        "robots": [
            {
                "name": _ROBOT_NAME,
                "asset": {
                    "host_repo": _HOST_REPO,
                    "relative_path": robot_asset,
                },
                "joint_names": tuple(inner.robot.data.joint_names),
                "base_pose": _robot_base_pose(inner),
            }
        ],
        "objects": [
            *[
                {
                    "name": f"object_env{env_index}",
                    "asset": {
                        "host_repo": _HOST_REPO,
                        "relative_path": object_asset,
                    },
                    "scale": None,
                    "articulated": False,
                }
                for env_index, object_asset in enumerate(object_assets)
            ],
            {
                "name": "table",
                "asset": {
                    "host_repo": _HOST_REPO,
                    "relative_path": table_asset,
                },
                "scale": None,
                "articulated": False,
            },
        ],
        "channels": {
            name: {
                "shape": shape,
                "dtype": "bool" if name == "reset" else "float32",
            }
            for name, shape in channel_specs.items()
        },
        "extras_meta": {
            "captured_envs": str(captured_envs),
            "object_pose_frame": "per_env_local",
            "goal_pose_frame": "per_env_local",
            "table_pose_frame": "per_env_local",
            "quaternion_order": "wxyz",
            "joint_target_stage": "delivered_post_moving_average",
            **{
                f"object_env{env_index}_scale_normalized": ",".join(
                    f"{value:.6g}" for value in scale
                )
                for env_index, scale in enumerate(object_scales)
            },
        },
    }
    return TrainingStateCapture(
        channel_specs=channel_specs,
        channel_dtypes=dict(_CHANNEL_DTYPES),
        window_steps=window_steps,
        max_start_delay_steps=max_start_delay_steps,
        manifest_template=manifest_template,
        output_root=Path(output_root),
        source_device=source_device,
    )


class CaptureEnvWrapper:
    """Append batched post-step state without changing the Gym transition."""

    def __init__(
        self,
        env: Any,
        capture: TrainingStateCapture,
        *,
        captured_envs: int = _MAX_CAPTURE_ENVS,
        close_timeout: float = 60.0,
    ) -> None:
        self.env = env
        self.capture = capture
        self.captured_envs = _capture_env_count(
            captured_envs, num_envs=env.unwrapped.num_envs
        )
        self.close_timeout = close_timeout

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step(self, action: Any) -> Any:
        transition = self.env.step(action)
        if not self.capture.active:
            return transition
        if not isinstance(transition, tuple) or len(transition) != 5:
            raise TypeError(
                "SimToolReal capture wrapper requires a Gymnasium transition"
            )
        inner = self.env.unwrapped
        terminated, truncated = transition[2], transition[3]
        captured = slice(0, self.captured_envs)
        origins = inner.scene.env_origins[captured]
        self.capture.append(
            {
                f"joint_pos_{_ROBOT_NAME}": inner.robot.data.joint_pos[captured],
                f"joint_target_{_ROBOT_NAME}": inner._cur_targets[captured],
                "object_pose": torch.cat(
                    (
                        inner.object.data.root_pos_w[captured] - origins,
                        inner.object.data.root_quat_w[captured],
                    ),
                    dim=1,
                ),
                "goal_pose": torch.cat(
                    (
                        inner.goal_viz.data.root_pos_w[captured] - origins,
                        inner.goal_viz.data.root_quat_w[captured],
                    ),
                    dim=1,
                ),
                "table_pose": torch.cat(
                    (
                        inner.table.data.root_pos_w[captured] - origins,
                        inner.table.data.root_quat_w[captured],
                    ),
                    dim=1,
                ),
                "reset": terminated[captured] | truncated[captured],
            }
        )
        return transition

    def close(self) -> Any:
        try:
            self.capture.close(timeout=self.close_timeout)
        finally:
            self.env.close()


__all__ = [
    "SIMTOOLREAL_CAPTURE_MAX_START_DELAY_STEPS",
    "SIMTOOLREAL_CAPTURE_WINDOW_STEPS",
    "CaptureEnvWrapper",
    "build_capture",
]


def configure_recording(env, observer, output_root, host_repo_root):
    """Preserve four environment views while recording only on request."""
    from action_bench.recording import StateCaptureObserver

    count = min(_MAX_CAPTURE_ENVS, env.unwrapped.num_envs)
    capture = build_capture(
        env,
        output_root,
        experiment_id=Path(output_root).parent.name,
        host_repo_root=host_repo_root,
        captured_envs=count,
    )
    return (
        CaptureEnvWrapper(env, capture, captured_envs=count),
        StateCaptureObserver(
            capture,
            (
                {"every_epochs": 400, "through_epoch": 2400},
                {"every_epochs": 800, "through_epoch": None},
            ),
            observer,
        ),
        capture,
    )
