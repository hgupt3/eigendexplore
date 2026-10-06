"""Render an Action-Bench state-capture bundle to an MP4 using URDF FK."""

from __future__ import annotations

import copy
import io
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import read_capture

CAMERA_PRESETS: dict[str, dict[str, object]] = {
    "simtoolreal": {
        "eye": (0.72, -0.70, 0.95),
        "target": (0.02, 0.12, 0.59),
        "fov": 45.0,
    },
    "dextreme": {
        "eye": (0.26, -0.55, 0.73),
        "target": (-0.06, -0.17, 0.54),
        "fov": 45.0,
    },
}

SIMTOOLREAL_GRID_SPACING = 1.6
DISPLAY_GRID = (
    (0.0, 0.0),
    (SIMTOOLREAL_GRID_SPACING, 0.0),
    (0.0, SIMTOOLREAL_GRID_SPACING),
    (SIMTOOLREAL_GRID_SPACING, SIMTOOLREAL_GRID_SPACING),
)

# Orientation-only tasks show the goal at a fixed world position to the side of
# the scene (its recorded position coincides with the object); only the
# recorded goal orientation animates.
GOAL_DISPLAY_OVERRIDES: dict[str, dict[str, object]] = {
    "dextreme": {"static_position": (-0.24, -0.18, 0.59), "opacity": 0.9},
}


@dataclass(frozen=True)
class _RenderedAsset:
    urdf: Any
    link_nodes: tuple[tuple[str, Any], ...]
    base_scale: np.ndarray


def _stabilize_opengl_context_keys() -> None:
    """Make PyOpenGL context handles usable as cache keys.

    PyOpenGL >= 3.1.7 returns a fresh ctypes pointer wrapper from
    ``contextdata.getContext()`` on every call. ctypes pointers compare by
    object identity, so pyrender's shader-program cache (keyed on the context)
    misses on every frame and recompiles every shader, making offscreen
    rendering ~20x slower. Normalizing the handle to its integer address
    restores stable keys.
    """

    import ctypes

    from OpenGL import contextdata

    original_get_context = contextdata.getContext
    if getattr(original_get_context, "_action_bench_stable", False):
        return

    def stable_get_context(context: object = None) -> object:
        resolved = original_get_context(context)
        if resolved is None or isinstance(resolved, int):
            return resolved
        try:
            address = ctypes.cast(resolved, ctypes.c_void_p).value
        except (TypeError, ctypes.ArgumentError):
            return resolved
        return resolved if address is None else address

    stable_get_context._action_bench_stable = True  # type: ignore[attr-defined]
    contextdata.getContext = stable_get_context


def _require_finite_pose(values: Sequence[float], *, label: str) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{label} must contain seven finite xyz+wxyz values")
    quaternion = pose[3:]
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-3):
        raise ValueError(f"{label} quaternion must be unit length")
    quaternion = quaternion / norm
    w, x, y, z = quaternion
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pose[:3]
    return matrix


def _camera_pose(
    eye_values: Sequence[float], target_values: Sequence[float]
) -> np.ndarray:
    eye = np.asarray(eye_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    if eye.shape != (3,) or target.shape != (3,):
        raise ValueError("camera eye and target must each contain three values")
    if not np.isfinite(eye).all() or not np.isfinite(target).all():
        raise ValueError("camera eye and target must be finite")
    backward = eye - target
    distance = float(np.linalg.norm(backward))
    if distance <= 0.0:
        raise ValueError("camera eye and target must differ")
    backward /= distance
    up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = np.cross(up, backward)
    if float(np.linalg.norm(right)) < 1e-8:
        up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    camera_up = np.cross(backward, right)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = camera_up
    pose[:3, 2] = backward
    pose[:3, 3] = eye
    return pose


def apply_reset_emphasis(frame: np.ndarray, *, active: bool) -> np.ndarray:
    """Draw a red reset border without changing the frame count."""

    image = np.asarray(frame)
    if not active:
        return image
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rendered frame must have uint8 shape (height, width, 3)")
    result = np.ascontiguousarray(image.copy())
    thickness = max(4, min(result.shape[:2]) // 45)
    result[:thickness, :, :] = (255, 24, 24)
    result[-thickness:, :, :] = (255, 24, 24)
    result[:, :thickness, :] = (255, 24, 24)
    result[:, -thickness:, :] = (255, 24, 24)
    return result


def apply_mesh_opacity(mesh: Any, *, opacity: float) -> Any:
    """Give a pyrender mesh independent blended materials at one opacity."""

    if not math.isfinite(opacity) or not 0.0 < opacity < 1.0:
        raise ValueError("ghost opacity must be between zero and one")
    if not mesh.primitives:
        raise ValueError("ghost mesh contains no primitives")
    for primitive in mesh.primitives:
        if primitive.material is None:
            raise ValueError("ghost mesh primitive has no material")
        material = copy.copy(primitive.material)
        factor = np.asarray(material.baseColorFactor, dtype=np.float64).copy()
        if factor.shape != (4,) or not np.isfinite(factor).all():
            raise ValueError("ghost mesh material base color is malformed")
        factor[3] *= opacity
        material.baseColorFactor = factor
        material.alphaMode = "BLEND"
        material.doubleSided = True
        primitive.material = material
    return mesh


def _nearest_link(
    node: str, *, parents: Mapping[str, str], link_names: set[str]
) -> str | None:
    current = node
    while True:
        if current in link_names:
            return current
        if current not in parents:
            return None
        current = parents[current]


def _add_urdf_geometry(
    *,
    render_scene: Any,
    urdf: Any,
    prefix: str,
    scale: Sequence[float] | None,
    opacity: float | None,
    pyrender: Any,
) -> _RenderedAsset:
    source_scene = urdf.scene
    link_names = set(urdf.link_map)
    parents = source_scene.graph.transforms.parents
    assignments: dict[str, list[str]] = {}
    for geometry_node in source_scene.graph.nodes_geometry:
        owner = _nearest_link(
            geometry_node,
            parents=parents,
            link_names=link_names,
        )
        if owner is None:
            raise ValueError(
                f"URDF geometry node {geometry_node!r} is not attached to a link"
            )
        assignments.setdefault(owner, []).append(geometry_node)
    if not assignments:
        raise ValueError(f"URDF asset {prefix!r} contains no renderable geometry")

    link_nodes: list[tuple[str, Any]] = []
    for link_name in urdf.link_map:
        geometry_nodes = assignments.get(link_name)
        if not geometry_nodes:
            continue
        link_world = np.asarray(source_scene.graph[link_name][0], dtype=np.float64)
        link_inverse = np.linalg.inv(link_world)
        link_node = pyrender.Node(name=f"{prefix}:{link_name}")
        render_scene.add_node(link_node)
        link_nodes.append((link_name, link_node))
        for geometry_node in geometry_nodes:
            geometry_world, geometry_name = source_scene.graph[geometry_node]
            geometry = source_scene.geometry[geometry_name]
            try:
                mesh = pyrender.Mesh.from_trimesh(geometry, smooth=True)
            except ValueError:
                mesh = pyrender.Mesh.from_trimesh(geometry, smooth=False)
            if opacity is not None:
                apply_mesh_opacity(mesh, opacity=opacity)
            render_scene.add(
                mesh,
                pose=link_inverse @ np.asarray(geometry_world, dtype=np.float64),
                name=f"{prefix}:{link_name}:{geometry_node}",
                parent_node=link_node,
            )

    scale_values = (
        np.ones(3, dtype=np.float64)
        if scale is None
        else np.asarray(scale, dtype=np.float64)
    )
    if (
        scale_values.shape != (3,)
        or not np.isfinite(scale_values).all()
        or np.any(scale_values <= 0.0)
    ):
        raise ValueError(
            f"URDF asset {prefix!r} scale must contain three positive values"
        )
    base_scale = np.eye(4, dtype=np.float64)
    base_scale[:3, :3] = np.diag(scale_values)
    return _RenderedAsset(
        urdf=urdf,
        link_nodes=tuple(link_nodes),
        base_scale=base_scale,
    )


def _set_asset_pose(
    render_scene: Any, rendered: _RenderedAsset, base_pose: np.ndarray, scale=None
) -> None:
    scaled_base = base_pose @ rendered.base_scale
    if scale is not None:
        scaled_base = scaled_base @ np.diag((*scale, 1.0))
    for link_name, link_node in rendered.link_nodes:
        link_pose = rendered.urdf.get_transform(link_name, rendered.urdf.base_link)
        render_scene.set_pose(
            link_node,
            pose=scaled_base @ link_pose,
        )


def _resolve_asset_roots(manifest, asset_roots):
    roots = {}
    for entry in (*manifest["robots"], *manifest["objects"]):
        host = entry["asset"]["host_repo"]
        if host not in asset_roots:
            raise ValueError(f"no asset root supplied for {host!r}")
        root = Path(asset_roots[host]).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        roots[host] = root
    return roots


def _asset_path(asset: Any, roots: Mapping[str, Path]) -> Path:
    root = roots[asset["host_repo"]]
    path = (root / asset["relative_path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"capture asset escapes host root: {asset['relative_path']!r}"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_urdf(
    asset: Any, roots: Mapping[str, Path], yourdfpy: Any, *, joint_names=()
) -> Any:
    path = _asset_path(asset, roots)
    tree = ET.parse(path)
    root = roots[asset["host_repo"]]
    # Host URDFs use package://<directory>/<file> relative to their asset folder.
    # Resolve those exact paths; yourdfpy otherwise strips the package directory.
    for mesh in tree.findall(".//visual/geometry/mesh"):
        filename = mesh.attrib["filename"]
        filename = filename.removeprefix("package://")
        # Some hosts prefix paths with the containing asset directory itself.
        first = Path(filename).parts[0]
        base = path.parent.parent if first == path.parent.name else path.parent
        resolved = (base / filename).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"URDF visual mesh escapes host root: {filename!r}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        mesh.set("filename", str(resolved))
    # Replay recorded mimic DOFs rather than replacing them with the URDF's
    # ideal coupling relation.
    recorded = set(joint_names)
    for joint in tree.findall("joint"):
        mimic = joint.find("mimic")
        if joint.attrib["name"] in recorded and mimic is not None:
            joint.remove(mimic)
    return yourdfpy.URDF.load(
        io.BytesIO(ET.tostring(tree.getroot())),
        mesh_dir=str(path.parent),
        build_scene_graph=True,
        load_meshes=True,
    )


def _required_channel(channels: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return channels[name]
    except KeyError as error:
        raise ValueError(f"capture is missing required channel {name!r}") from error


def _validate_pose_channel(values: np.ndarray, *, label: str, num_frames: int) -> None:
    expected_shape = (num_frames, 7)
    if values.shape != expected_shape:
        raise ValueError(f"{label} has shape {values.shape}; expected {expected_shape}")
    if values.dtype != np.float32:
        raise ValueError(f"{label} must have dtype float32")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains nonfinite values")
    norms = np.linalg.norm(values[:, 3:], axis=1)
    invalid = np.flatnonzero(~np.isclose(norms, 1.0, rtol=0.0, atol=1.0e-3))
    if invalid.size:
        raise ValueError(
            f"{label} has non-unit quaternions at frames {invalid[:10].tolist()}"
        )


def _validate_joint_names(*, name: str, recorded: tuple[str, ...], urdf: Any) -> None:
    if len(recorded) != len(set(recorded)):
        raise ValueError(f"robot {name!r} records duplicate joint names")
    urdf_names = tuple(urdf.actuated_joint_names)
    recorded_set = set(recorded)
    urdf_set = set(urdf_names)
    missing_from_urdf = sorted(recorded_set - urdf_set)
    unrecorded_urdf = sorted(urdf_set - recorded_set)
    if missing_from_urdf or unrecorded_urdf:
        raise ValueError(
            f"robot {name!r} joint mismatch: "
            f"missing_from_urdf={missing_from_urdf}, "
            f"unrecorded_urdf={unrecorded_urdf}"
        )


def _capture_env_layout(
    manifest: dict[str, Any], channels: Mapping[str, np.ndarray]
) -> tuple[bool, int]:
    if not manifest["robots"]:
        raise ValueError("capture manifest has no robots")
    robot = manifest["robots"][0]
    values = _required_channel(channels, f"joint_pos_{robot['name']}")
    if values.ndim == 2:
        return False, 1
    if values.ndim != 3:
        raise ValueError(
            f"robot {robot['name']!r} joint channel has {values.ndim} dimensions; "
            "expected frame+joint or frame+env+joint"
        )
    captured_envs = values.shape[1]
    if captured_envs < 1 or captured_envs > len(DISPLAY_GRID):
        raise ValueError(
            f"multi-env capture has {captured_envs} envs; the display grid "
            f"supports 1 through {len(DISPLAY_GRID)}"
        )
    declared = manifest.get("extras_meta", {}).get("captured_envs")
    if declared != str(captured_envs):
        raise ValueError(
            f"multi-env capture declares captured_envs={declared!r}; "
            f"channel shape contains {captured_envs}"
        )
    return True, captured_envs


def _display_poses(captured_envs: int) -> tuple[np.ndarray, ...]:
    poses: list[np.ndarray] = []
    for x, y in DISPLAY_GRID[:captured_envs]:
        pose = np.eye(4, dtype=np.float64)
        pose[:2, 3] = (x, y)
        poses.append(pose)
    return tuple(poses)


def _object_env_index(name: str) -> int | None:
    match = re.fullmatch(r"object_env(\d+)", name)
    return None if match is None else int(match.group(1))


def _camera_settings(
    manifest: dict[str, Any],
    *,
    env_axis: bool,
    captured_envs: int,
    camera_eye: Sequence[float] | None,
    camera_target: Sequence[float] | None,
    camera_fov: float | None,
) -> tuple[Sequence[float], Sequence[float], float]:
    if manifest["benchmark"] not in CAMERA_PRESETS:
        if camera_eye is None or camera_target is None or camera_fov is None:
            raise ValueError(
                f"no camera preset for benchmark {manifest['benchmark']!r}; "
                "provide camera_eye, camera_target, and camera_fov"
            )
        preset: Mapping[str, object] = {}
    else:
        preset = CAMERA_PRESETS[manifest["benchmark"]]
    if manifest["benchmark"] == "simtoolreal" and env_axis and captured_envs > 1:
        grid = np.asarray(DISPLAY_GRID[:captured_envs], dtype=np.float64)
        grid_center = 0.5 * (grid.min(axis=0) + grid.max(axis=0))
        grid_extent = float(np.linalg.norm(grid.max(axis=0) - grid.min(axis=0)))
        preset_target = np.asarray(preset["target"], dtype=np.float64)
        preset_eye = np.asarray(preset["eye"], dtype=np.float64)
        view = preset_eye - preset_target
        base_distance = float(np.linalg.norm(view))
        base_fov = math.radians(float(preset["fov"]))
        distance = (
            base_distance + 0.2 + 0.58 * grid_extent / (2.0 * math.tan(base_fov / 2.0))
        )
        target_with_grid = preset_target.copy()
        target_with_grid[:2] += grid_center
        target_with_grid[2] -= 0.32
        grid_eye = target_with_grid + view / base_distance * distance
        grid_eye[2] += 0.1
        preset = {
            **preset,
            "eye": tuple(grid_eye),
            "target": tuple(target_with_grid),
        }
    eye = camera_eye if camera_eye is not None else preset["eye"]
    target = camera_target if camera_target is not None else preset["target"]
    fov = camera_fov if camera_fov is not None else float(preset["fov"])
    if not math.isfinite(fov) or not 0.0 < fov < 180.0:
        raise ValueError("camera field of view must be between 0 and 180 degrees")
    return eye, target, fov


def _default_stride(control_dt: float) -> int:
    ratio = 1.0 / (30.0 * control_dt)
    nearest = round(ratio)
    if math.isclose(ratio, nearest, rel_tol=1.0e-9, abs_tol=1.0e-9):
        ratio = float(nearest)
    return max(1, math.ceil(ratio))


def check_render_dependencies():
    """Verify CPU OpenGL and FFmpeg before starting a recorded training run."""
    if (
        "pyrender" in sys.modules or "OpenGL.platform" in sys.modules
    ) and os.environ.get("PYOPENGL_PLATFORM") != "osmesa":
        raise RuntimeError("OpenGL was already imported with a non-OSMesa backend")
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import pyrender
    import yourdfpy

    imageio_ffmpeg.get_ffmpeg_exe()
    _stabilize_opengl_context_keys()
    renderer = pyrender.OffscreenRenderer(8, 8)
    renderer.delete()
    return imageio, pyrender, yourdfpy


@dataclass
class _Runtime:
    rendered: _RenderedAsset
    base_pose: np.ndarray
    joint_names: tuple[str, ...] = ()
    joints: np.ndarray | None = None
    poses: np.ndarray | None = None
    scales: np.ndarray | None = None
    goal_position: Sequence[float] | None = None

    def update(self, scene, frame):
        if self.joints is not None:
            self.rendered.urdf.update_cfg(
                dict(zip(self.joint_names, self.joints[frame]))
            )
        pose = self.base_pose
        if self.poses is not None:
            local = _require_finite_pose(self.poses[frame], label="recorded pose")
            if self.goal_position is not None:
                local[:3, 3] = self.goal_position
            pose = pose @ local
        _set_asset_pose(
            scene,
            self.rendered,
            pose,
            None if self.scales is None else self.scales[frame],
        )


def _float_channel(channels, name, shape):
    values = _required_channel(channels, name)
    if (
        values.shape != shape
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{name} must be finite float32 with shape {shape}")
    return values


def _build_runtimes(
    *,
    manifest,
    channels,
    roots,
    render_scene,
    ghost_goal,
    env_axis,
    captured_envs,
    yourdfpy,
    pyrender,
):
    frames = manifest["num_frames"]
    prefix_shape = (frames, captured_envs) if env_axis else (frames,)
    runtimes = []
    for group in ("robots", "objects"):
        names = [entry["name"] for entry in manifest[group]]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate {group} names in capture")
    indexed = {_object_env_index(entry["name"]) for entry in manifest["objects"]}
    indexed.discard(None)
    if env_axis and indexed != set(range(captured_envs)):
        raise ValueError(
            "multi-env objects must have one object_env<i> entry per environment"
        )
    if not env_axis and indexed:
        raise ValueError("single-env capture cannot contain object_env<i> entries")

    def scale_channel(name):
        if name not in channels:
            return None
        values = _float_channel(channels, name, (frames, 3))
        if not (values > 0).all():
            raise ValueError(f"{name} must be positive")
        return values

    def geometry(entry, urdf, name, opacity=None):
        return _add_urdf_geometry(
            render_scene=render_scene,
            urdf=urdf,
            prefix=name,
            scale=entry.get("scale"),
            opacity=opacity,
            pyrender=pyrender,
        )

    for env_index, display in enumerate(_display_poses(captured_envs)):

        def select(values):
            return values[:, env_index] if env_axis else values

        for robot in manifest["robots"]:
            name = robot["name"]
            names = tuple(robot["joint_names"])
            urdf = _load_urdf(robot["asset"], roots, yourdfpy, joint_names=names)
            _validate_joint_names(name=name, recorded=names, urdf=urdf)
            joints = select(
                _float_channel(
                    channels, f"joint_pos_{name}", (*prefix_shape, len(names))
                )
            )
            base = (
                np.eye(4)
                if robot.get("base_pose") is None
                else _require_finite_pose(robot["base_pose"], label=name)
            )
            runtimes.append(
                _Runtime(
                    geometry(robot, urdf, f"robot:{name}:{env_index}"),
                    display @ base,
                    names,
                    joints,
                    scales=scale_channel(f"robot_scale_{name}"),
                )
            )

        entries = [
            entry
            for entry in manifest["objects"]
            if _object_env_index(entry["name"]) in (None, env_index)
        ]
        primary = [
            entry
            for entry in entries
            if _object_env_index(entry["name"]) == env_index
            or entry["name"] == "object"
            or (
                not env_axis
                and entry.get("base_pose") is None
                and f"{entry['name']}_pose" not in channels
            )
        ]
        if len(primary) > 1:
            raise ValueError("object_pose maps to multiple objects")
        primary = primary[0] if primary else None
        for entry in entries:
            name = entry["name"]
            urdf = _load_urdf(entry["asset"], roots, yourdfpy)
            pose_name = f"{name}_pose"
            if pose_name not in channels and entry is primary:
                pose_name = "object_pose"
            poses = None
            base = display
            if pose_name in channels:
                poses = select(_float_channel(channels, pose_name, (*prefix_shape, 7)))
                _validate_pose_channel(poses, label=pose_name, num_frames=frames)
            elif entry.get("base_pose") is not None:
                base = display @ _require_finite_pose(entry["base_pose"], label=name)
            else:
                raise ValueError(f"{name} has no recorded or static pose")
            names = tuple(urdf.actuated_joint_names)
            joints = None
            if entry["articulated"]:
                if not names or poses is None:
                    raise ValueError(
                        "articulated object requires URDF joints and recorded poses"
                    )
                joints = select(
                    _float_channel(channels, "object_dof", (*prefix_shape, len(names)))
                )
            elif names:
                raise ValueError("non-articulated object URDF contains actuated joints")
            scales = scale_channel("object_scale") if entry is primary else None
            runtimes.append(
                _Runtime(
                    geometry(entry, urdf, f"object:{name}:{env_index}"),
                    base,
                    names,
                    joints,
                    poses,
                    scales,
                )
            )
            if entry is primary and ghost_goal and "goal_pose" in channels:
                goals = select(
                    _float_channel(channels, "goal_pose", (*prefix_shape, 7))
                )
                _validate_pose_channel(goals, label="goal_pose", num_frames=frames)
                settings = GOAL_DISPLAY_OVERRIDES.get(manifest["benchmark"], {})
                runtimes.append(
                    _Runtime(
                        geometry(
                            entry,
                            urdf,
                            f"goal:{name}:{env_index}",
                            settings.get("opacity", 0.3),
                        ),
                        display,
                        names,
                        joints,
                        goals,
                        scales,
                        settings.get("static_position"),
                    )
                )
        if "goal_pose" in channels and primary is None:
            raise ValueError("goal_pose has no corresponding object")
    return runtimes


def render_capture(
    capture_dir: str | Path,
    output: str | Path,
    *,
    asset_roots: Mapping[str, str | Path],
    width: int = 1280,
    height: int = 720,
    stride: int | None = None,
    camera_eye=None,
    camera_target=None,
    camera_fov=None,
    ghost_goal=True,
) -> Path:
    """Render recorded geometry and motion using CPU OpenGL, publishing one MP4."""
    output_path = Path(output).expanduser().resolve()
    if os.path.lexists(output_path):
        raise FileExistsError(output_path)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("render output must have an .mp4 suffix")
    if any(
        type(value) is not int or value <= 0 or value % 2 for value in (width, height)
    ):
        raise ValueError("MP4 width and height must be positive even integers")
    manifest, channels = read_capture(capture_dir)
    stride = _default_stride(manifest["control_dt"]) if stride is None else stride
    if type(stride) is not int or stride < 1:
        raise ValueError("render stride must be a positive integer")
    fps = 1 / (manifest["control_dt"] * stride)
    env_axis, captured_envs = _capture_env_layout(manifest, channels)
    shape = (
        (manifest["num_frames"], captured_envs)
        if env_axis
        else (manifest["num_frames"],)
    )
    resets = _required_channel(channels, "reset")
    if resets.shape != shape or resets.dtype != np.bool_:
        raise ValueError(f"reset channel must be bool with shape {shape}")
    roots = _resolve_asset_roots(manifest, asset_roots)
    eye, target, fov = _camera_settings(
        manifest,
        env_axis=env_axis,
        captured_envs=captured_envs,
        camera_eye=camera_eye,
        camera_target=camera_target,
        camera_fov=camera_fov,
    )
    imageio, pyrender, yourdfpy = check_render_dependencies()
    scene = pyrender.Scene(
        bg_color=(0.82, 0.84, 0.87, 1), ambient_light=(0.55, 0.55, 0.55)
    )
    runtimes = _build_runtimes(
        manifest=manifest,
        channels=channels,
        roots=roots,
        render_scene=scene,
        ghost_goal=ghost_goal,
        env_axis=env_axis,
        captured_envs=captured_envs,
        yourdfpy=yourdfpy,
        pyrender=pyrender,
    )
    camera = _camera_pose(eye, target)
    scene.add(
        pyrender.PerspectiveCamera(
            yfov=math.radians(fov), aspectRatio=width / height, znear=0.005, zfar=50.0
        ),
        pose=camera,
    )
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=camera)
    fill = _camera_pose((-eye[0], -eye[1], max(float(eye[2]), 0.5) + 0.6), target)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=fill)
    temporary = output_path.with_name(f".{output_path.name}.tmp.mp4")
    if os.path.lexists(temporary):
        raise FileExistsError(temporary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(width, height)
    flash_frames = max(1, round(0.25 * fps))
    remaining = 0
    previous = -1
    try:
        with imageio.get_writer(
            temporary,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,
            ffmpeg_log_level="error",
        ) as writer:
            for frame in range(0, manifest["num_frames"], stride):
                for runtime in runtimes:
                    runtime.update(scene, frame)
                color, _ = renderer.render(
                    scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL
                )
                if np.any(resets[previous + 1 : frame + 1]):
                    remaining = flash_frames
                writer.append_data(apply_reset_emphasis(color, active=remaining > 0))
                remaining = max(0, remaining - 1)
                previous = frame
        if not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError("renderer produced an empty MP4")
        if os.path.lexists(output_path):
            raise FileExistsError(output_path)
        os.replace(temporary, output_path)
    finally:
        renderer.delete()
        temporary.unlink(missing_ok=True)
    return output_path
