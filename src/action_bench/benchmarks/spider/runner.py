"""Run one resolved cold-start episode in the dedicated Spider environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from . import tasks
from .protocol import Job, defaults, hand_binding, spider_basis
from .sampling import (
    PersistentAugmentedCenter,
    install_cold_start,
    normalize_basis,
    random_orthonormal_rows,
)


def load_host(root):
    root = Path(root).resolve()
    # The entrypoint and every native module must come from the same selected
    # checkout, including when run_episode is called directly in an interpreter.
    sys.path.insert(0, str(root))
    path = root / "examples/run_mjwp.py"
    spec = importlib.util.spec_from_file_location("action_bench_spider_host", path)
    host = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = host
    spec.loader.exec_module(host)
    for name, module in tuple(sys.modules.items()):
        if name == "spider" or name.startswith("spider."):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(root / "spider"):
                raise ValueError(
                    f"native Spider module comes from a different checkout: {name}"
                )
    return host


def overrides(job, dataset, output, *, record=False):
    values = [
        "+override=oakinkv2",
        f"dataset_dir={dataset}",
        f"task={job.task}",
        f"robot_type={job.hand}",
        "embodiment_type=right",
        "data_id=0",
        "device=cuda:0",
        f"seed={job.seed}",
        "sim_dt=0.01",
        "num_samples=1024",
        "max_num_iterations=16",
        "terminate_resample=false",
        "joint_noise_scale=0.1",
        "pos_noise_scale=0.01",
        "rot_noise_scale=0.01",
        "first_ctrl_noise_scale=2.0",
        "last_ctrl_noise_scale=4.0",
        "final_noise_scale=0.01",
        "joint_rew_scale=0.003",
        "+contact_rew_scale=0.0",
        "nconmax_per_env=256",
        "njmax_per_env=768",
        "max_sim_steps=-1",
        "show_viewer=false",
        "viewer=mujoco",
        "+wait_on_finish=false",
        f"save_video={str(record).lower()}",
        "save_info=true",
        "save_config=true",
        "save_rerun=false",
        "save_viser=false",
        "num_trace_uniform_samples=0",
        "num_trace_topk_samples=0",
        "+sanity_check_seconds=0",
        f"+run_output_dir={output}",
        "hydra.run.dir=.",
    ]
    return values


def install_physics(host, binding, artifact_path, audit):
    native = host.setup_mj_model

    def setup(config):
        model = native(config)
        adapter = tasks.bind_adapter(
            artifact_path=artifact_path,
            model=model,
            binding=binding,
            k=binding["layout_object"].joint_count,
        )
        current = {
            "integration": tasks.apply_integration_profile(
                model, defaults()["integration_profile"]
            ),
            "contact": tasks._apply_contact_profile(model, "collision_firm"),
            "effort": tasks.apply_effort_profile(
                model,
                adapter=adapter,
                efforts=tuple(binding["effort_limits"]),
                enabled=True,
            ),
        }
        if audit and audit != current:
            raise ValueError("physics changed between model constructions")
        audit.update(current)
        return model

    if host.setup_env.__globals__["setup_mj_model"] is not native:
        raise ValueError("host setup hook differs from the pinned interface")
    host.setup_mj_model = setup
    host.setup_env.__globals__["setup_mj_model"] = setup


def run_episode(job, *, host_root, dataset, output, record=False):
    binding = hand_binding(job.hand)
    artifact = spider_basis(job.hand)
    structure = tasks.structural_contract(
        dataset_dir=dataset,
        hand=job.hand,
        task=job.task,
        artifact_path=artifact.path,
        binding=binding,
        k=job.k or binding["layout_object"].joint_count,
    )
    scene = dataset / f"processed/oakinkv2/{job.hand}/right/{job.task}/scene.xml"
    cold_binding = tasks.cold_mean_binding(
        artifact_path=artifact.path, scene_path=scene, binding=binding
    )
    host = load_host(host_root)
    physics, cold_audit = {}, {}
    install_physics(host, binding, artifact.path, physics)
    install_cold_start(host, binding=cold_binding, audit=cold_audit)
    tracker = None
    noise_audit = None
    if job.method == "eigendexplore":
        from .proposal import EigenProposal

        def build(config, model):
            nonlocal tracker, noise_audit
            adapter = tasks.bind_adapter(
                artifact_path=artifact.path,
                model=model,
                binding=binding,
                k=job.k,
                device=config.device,
                dtype=config.noise_scale.dtype,
            )
            if job.basis == "random":
                adapter.components.copy_(
                    random_orthonormal_rows(adapter.joint_dim, job.k)
                )
            noise_audit = normalize_basis(
                adapter,
                artifact_path=artifact.path,
                target_rms=binding["eigen_noise_rms"],
                retained_variance=job.retained_variance,
                artifact_layout=binding["layout_object"],
            )
            native = EigenProposal(
                adapter=adapter, device=config.device, seed=config.seed
            )
            tracker = PersistentAugmentedCenter(
                native,
                updates_per_commit=16,
                iid_authority=job.iid_scale,
                eigen_authority=job.eigen_scale,
            )
            return tracker

        host.build_control_proposal_augmenter = build
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "completed": False,
        "job": job.model_dump(),
        "basis_id": artifact.artifact_id,
        "structure": structure,
        "particles": 1024,
        "updates_per_commit": 16,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    # This flag changes CUDA-graph execution only, for the installed driver capability.
    import mujoco_warp as mjwarp

    native_put = mjwarp.put_model

    def put_model(model):
        result = native_put(model)
        result.opt.graph_conditional = False
        return result

    mjwarp.put_model = put_model
    with initialize_config_dir(
        version_base=None, config_dir=str(host_root / "examples/config")
    ):
        cfg = compose(
            config_name="default",
            overrides=overrides(job, dataset, output, record=record),
        )
    host.run_main.__wrapped__(cfg)
    summary = json.loads((output / "run_summary.json").read_text())
    iterations = summary.get("optimizer_iterations_per_commit", [])
    if (
        not summary.get("completed")
        or not iterations
        or set(iterations) != {16}
        or summary.get("optimizer_iterations") != 16 * len(iterations)
        or any(
            summary.get(key) != 400
            for key in (
                "sim_step",
                "generated_frames",
                "final_full_physics_cost_step_count",
                "max_sim_steps",
            )
        )
        or summary.get("terminal_rebase_performed") is not False
        or summary.get("proposal_center_max_abs_error") != 0.0
    ):
        raise ValueError("episode failed the recorded cold-start execution contract")
    if tracker is not None:
        if len(tracker.history) != summary["optimizer_iterations"]:
            raise ValueError("incomplete Eigen center trace")
        tracker.save(output)
    if (
        cold_audit.get("control_center_all_frames_max_abs_error", float("inf")) > 1e-7
        or cold_audit.get("initial_finger_qpos_max_abs_error", float("inf")) > 1e-7
        or any(
            cold_audit.get(k) != 0
            for k in (
                "nonfinger_control_max_abs_change",
                "initial_nonfinger_qpos_max_abs_change",
                "initial_finger_qvel_max_abs",
            )
        )
    ):
        raise ValueError("cold initialization differs from the cold-start posture")
    result.update(
        completed=True,
        cold_start=cold_audit,
        physics=physics,
        noise=noise_audit,
        final_full_physics_cost=summary["final_full_physics_cost"],
        wall_seconds=summary["wall_seconds"],
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    job = Job.model_validate(plan["jobs"][args.job])
    run_episode(
        job,
        host_root=Path(plan["host"]),
        dataset=Path(plan["dataset"]),
        output=Path(plan["output"]) / job.id,
        record=plan["record"],
    )


if __name__ == "__main__":
    main()
