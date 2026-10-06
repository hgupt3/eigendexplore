"""MuJoCo task structure, effort limits, and cold-start coordinate binding."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from .adapter import SpiderEigenProposalAdapter


def control_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names: list[str] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            raise ValueError(f"actuator {actuator_id} has no joint transmission")
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise ValueError(f"actuator {actuator_id} transmits an unnamed joint")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("scene actuator joint names contain duplicates")
    return tuple(names)


def bind_adapter(
    *,
    artifact_path: Path,
    model: mujoco.MjModel,
    binding: dict[str, Any],
    k: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> SpiderEigenProposalAdapter:
    layout = binding["layout_object"]
    adapter = SpiderEigenProposalAdapter.from_artifact(
        artifact_path,
        control_joint_names=control_joint_names(model),
        host_name_map=binding["host_name_map"],
        k=k,
        device=device,
        dtype=dtype,
    )
    expected = list(range(6, model.nu))
    observed = sorted(adapter.finger_control_indices.detach().cpu().tolist())
    if observed != expected or adapter.joint_dim != layout.joint_count:
        raise ValueError(
            "representation does not cover every host finger actuator exactly once"
        )
    return adapter


def cold_mean_binding(
    *,
    artifact_path: Path,
    scene_path: Path,
    binding: dict[str, Any],
) -> dict[str, Any]:
    layout = binding["layout_object"]
    posture = torch.tensor(binding["cold_start_posture"], dtype=torch.float64)
    if posture.shape != (layout.joint_count,):
        raise ValueError("cold-start posture differs from the hand joint count")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    adapter = bind_adapter(
        artifact_path=artifact_path,
        model=model,
        binding=binding,
        k=layout.joint_count,
    )
    canonical = posture.view(1, 1, layout.joint_count)
    expanded = adapter.expand_canonical_hand_values(canonical)[0, 0].cpu().numpy()
    controls = adapter.finger_control_indices.cpu().tolist()
    qpos: list[int] = []
    qvel: list[int] = []
    for control_index in controls:
        joint_id = int(model.actuator_trnid[control_index, 0])
        qpos.append(int(model.jnt_qposadr[joint_id]))
        qvel.append(int(model.jnt_dofadr[joint_id]))
    low = model.actuator_ctrlrange[controls, 0]
    high = model.actuator_ctrlrange[controls, 1]
    limited = model.actuator_ctrllimited[controls].astype(bool)
    deployed = np.where(limited, np.clip(expanded, low, high), expanded)
    if deployed.shape != (adapter.host_joint_dim,) or not np.isfinite(deployed).all():
        raise ValueError("expanded cold mean is malformed")
    return {
        "joint_names": list(adapter.host_joint_names[0]),
        "control_indices": controls,
        "qpos_indices": qpos,
        "qvel_indices": qvel,
        "mean": deployed,
        "unclamped_mean": expanded,
        "clamped_coordinate_count": int(np.count_nonzero(deployed != expanded)),
    }


def apply_effort_profile(
    model: mujoco.MjModel,
    *,
    adapter: SpiderEigenProposalAdapter,
    efforts: tuple[float, ...],
    enabled: bool,
) -> dict[str, Any]:
    indices = adapter.finger_control_indices.detach().cpu().tolist()
    if len(efforts) != len(indices):
        raise ValueError("effort vector differs from bound host finger width")
    joint_ids = np.asarray(model.actuator_trnid[indices, 0], dtype=np.int64)
    applied: dict[str, float] = {}
    if enabled:
        for name, index, effort in zip(adapter.host_joint_names[0], indices, efforts):
            model.actuator_forcelimited[index] = True
            model.actuator_forcerange[index] = (-effort, effort)
            applied[name] = effort
    joint_force_limits = {
        name: [
            float(model.jnt_actfrcrange[joint_id, 0]),
            float(model.jnt_actfrcrange[joint_id, 1]),
        ]
        for name, joint_id in zip(adapter.host_joint_names[0], joint_ids)
        if model.jnt_actfrclimited[joint_id]
    }
    return {
        "name": "urdf" if enabled else "stock",
        "ordered_host_finger_joint_names": list(adapter.host_joint_names[0]),
        "limited_actuator_count": len(applied),
        "joint_effort_limits": applied,
        "limited_joint_count": len(joint_force_limits),
        "joint_force_limits": joint_force_limits,
    }


def structural_contract(
    *,
    dataset_dir: Path,
    hand: str,
    task: str,
    artifact_path: Path,
    binding: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    root = dataset_dir / "processed" / "oakinkv2" / hand / "right" / task
    scene = root / "scene.xml"
    trajectory = root / "0" / "trajectory_kinematic.npz"
    if not scene.is_file() or not trajectory.is_file():
        raise FileNotFoundError(root)
    model = mujoco.MjModel.from_xml_path(str(scene))
    adapter = bind_adapter(
        artifact_path=artifact_path,
        model=model,
        binding=binding,
        k=k,
    )
    if hand == "allegro":
        from importlib.resources import files

        import yaml

        table = yaml.safe_load(
            files(__package__).joinpath("configs/allegro_ranges.yaml").read_text()
        )
        names = control_joint_names(model)
        for row in table["ranges"]:
            index = names.index(row["host_joint"])
            expected = [row["lower"], row["upper"]]
            joint = int(model.actuator_trnid[index, 0])
            if not np.allclose(
                model.actuator_ctrlrange[index], expected, rtol=0, atol=1e-6
            ) or not np.allclose(model.jnt_range[joint], expected, rtol=0, atol=1e-6):
                raise ValueError(
                    "Allegro scene lacks the recorded joint-range correction"
                )
    with np.load(trajectory, allow_pickle=False) as values:
        qpos = np.asarray(values["qpos"])
        frequency = float(values["frequency"])
    if (
        qpos.shape != (200, model.nq)
        or not math.isclose(frequency, 50.0, abs_tol=1.0e-9)
        or model.npair < 1
    ):
        raise ValueError("processed SPIDER episode failed its structural contract")
    return {
        "task_root": str(root),
        "scene_nq": model.nq,
        "scene_nu": model.nu,
        "frames": 200,
        "frequency_hz": frequency,
        "artifact_joint_count": adapter.joint_dim,
        "host_finger_actuator_count": adapter.host_joint_dim,
        "finger_control_indices": adapter.finger_control_indices.cpu().tolist(),
        "host_finger_joint_names": list(adapter.host_joint_names[0]),
    }


def apply_integration_profile(model: mujoco.MjModel, profile: str) -> dict[str, Any]:
    """Keep CPU initialization and Warp rollout on the same integration path.

    MuJoCo 3.7's free-body midpoint update uses body inertia without joint
    armature. Contact forces are solved with armature, so that update can
    amplify an initial contact impulse. Warp uses the ordinary ImplicitFast
    mass-matrix update. INVDISCRETE opts the CPU out of midpoint integration
    without changing model masses, armature, contacts, or actuator parameters.
    """
    if profile != "implicitfast_no_midpoint_v1":
        raise ValueError(f"unsupported integration profile {profile!r}")
    if mujoco.__version__ != "3.7.0":
        raise ValueError("Spider integration profile requires MuJoCo 3.7.0")
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise ValueError("Spider integration profile requires ImplicitFast")
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
    return {
        "name": profile,
        "mujoco_version": mujoco.__version__,
        "integrator": int(model.opt.integrator),
        "enableflags": int(model.opt.enableflags),
    }


def _collision_pair_kind(model: mujoco.MjModel, pair_id: int) -> str | None:
    names = []
    for geom_id in (model.pair_geom1[pair_id], model.pair_geom2[pair_id]):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        if name is None:
            raise ValueError(f"contact pair {pair_id} contains an unnamed geom")
        names.append(name)
    first, second = names
    first_hand = first.startswith("collision_hand_")
    second_hand = second.startswith("collision_hand_")
    first_object = first.startswith(("right_object_", "left_object_"))
    second_object = second.startswith(("right_object_", "left_object_"))
    if first_hand and second_hand:
        return "hand_self"
    if (first_hand and second_object) or (second_hand and first_object):
        return "hand_object"
    return None


def _apply_contact_profile(model: mujoco.MjModel, profile: str) -> dict[str, object]:
    pair_ids = [
        pair_id
        for pair_id in range(model.npair)
        if _collision_pair_kind(model, pair_id) is not None
    ]
    solimp_by_profile = {"collision_firm": (0.99, 0.999, 0.0005, 0.5, 2.0)}
    if profile in solimp_by_profile:
        contact_solref = (2.0 * float(model.opt.timestep), 1.0)
        for pair_id in pair_ids:
            model.pair_solref[pair_id] = contact_solref
            model.pair_solimp[pair_id] = solimp_by_profile[profile]
    elif profile != "stock":
        raise ValueError(f"unsupported contact profile {profile!r}")
    counts = {
        kind: sum(_collision_pair_kind(model, pair_id) == kind for pair_id in pair_ids)
        for kind in ("hand_self", "hand_object")
    }
    return {
        "name": profile,
        "pair_counts": counts,
        "total_modified_pairs": (len(pair_ids) if profile in solimp_by_profile else 0),
        "solref": (list(contact_solref) if profile in solimp_by_profile else None),
        "solimp": (
            list(solimp_by_profile[profile]) if profile in solimp_by_profile else None
        ),
        "timestep_s": float(model.opt.timestep),
        "iterations": int(model.opt.iterations),
        "line_search_iterations": int(model.opt.ls_iterations),
        "integrator": int(model.opt.integrator),
    }
