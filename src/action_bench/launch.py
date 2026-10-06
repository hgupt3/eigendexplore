"""Resolve and record portable host commands before launching a simulator."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import yaml

from .catalog import basis_entry
from .exploration import policy_noise
from .settings import benchmark_defaults


def prepare_run(study, args):
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        raise ValueError("seed must be an integer from 0 through 4294967295")
    installation = json.loads(args.installation.read_text())
    if installation["benchmark"] != study.benchmark:
        raise ValueError("installation and study benchmarks differ")
    host, python = Path(installation["host"]), Path(installation["python"])
    if not host.is_dir() or not python.is_file():
        raise FileNotFoundError("installation host or interpreter is missing")
    head = subprocess.check_output(
        ["git", "-C", str(host), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != installation["host_commit"]:
        raise ValueError("host revision differs from installation record")
    root = args.output.expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"run directory already exists: {root}")
    defaults = benchmark_defaults(study.benchmark)
    if args.max_epochs is not None and args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if args.num_envs is not None and args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    count = args.num_envs or {"dextreme": 16384, "simtoolreal": 12288}[study.benchmark]
    if study.benchmark == "dextreme" and count % 1024:
        raise ValueError("DeXtreme PPO num-envs must be a multiple of 1024")
    if study.benchmark == "simtoolreal" and count % 12288:
        raise ValueError("SimToolReal SAPG num-envs must be a multiple of 12288")
    record = getattr(args, "record", False)
    record_every = getattr(args, "record_every_epochs", None)
    if record_every is not None and (not record or record_every < 1):
        raise ValueError(
            "--record-every-epochs requires --record and a positive integer"
        )
    identity = {
        "study": study.model_dump(mode="json"),
        "host_commit": head,
        "release": "0.3.0",
        "seed": args.seed,
        "num_envs": count,
        "basis_id": basis_entry(study.hand).artifact_id
        if study.k is not None
        else None,
    }
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        # Resume only an identified release run with the exact resolved study.
        origins = [
            p / "launch.json"
            for p in checkpoint.parents
            if (p / "launch.json").is_file()
        ]
        if len(origins) != 1:
            raise ValueError("checkpoint must belong to one recorded release run")
        previous = json.loads(origins[0].read_text())
        if previous.get("identity") != identity:
            raise ValueError(
                "checkpoint method, basis, geometry, seed or software version differs"
            )
    else:
        checkpoint = None
    root.mkdir(parents=True)
    (root / "installation.json").write_text(json.dumps(installation, indent=2) + "\n")
    config_path = root / "study.yaml"
    config_path.write_text(
        yaml.safe_dump(
            study.model_dump(mode="json", exclude_none=True), sort_keys=False
        )
    )
    env = {
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTHONUNBUFFERED": "1",
        "PATH": os.pathsep.join((str(python.parent), os.environ["PATH"])),
        "PYTHONPATH": str(host),
        "ACTION_BENCH_STUDY": str(config_path),
        "ACTION_BENCH_RUN_DIR": str(root),
    }
    if record:
        env["ACTION_BENCH_RECORD_DIR"] = str(root / "recordings")
        env["ACTION_BENCH_RECORDING_MODULES"] = str(Path(__file__).parent / "recording")
        if record_every is not None:
            env["ACTION_BENCH_RECORD_EVERY_EPOCHS"] = str(record_every)
    command = [str(python), "-u"]
    name = f"{study.hand}-{study.method}-s{args.seed}"
    if study.benchmark == "dextreme":
        # Preview 4 dynamically loads libpython from the selected environment.
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(
                None,
                (str(python.parent.parent / "lib"), os.environ.get("LD_LIBRARY_PATH")),
            )
        )
        from .benchmarks.dextreme import export_dextreme_bundle

        manifest = export_dextreme_bundle(study, root / "bundle", num_envs=count)
        env["ACTION_BENCH_BUNDLE"] = str(root / "bundle")
        if study.method == "eigendexplore":
            settings = root / "eigendexplore.json"
            settings.write_text(json.dumps(policy_noise(study), indent=2) + "\n")
            env["EIGENDEXPLORE_SETTINGS"] = str(settings)
        command += [
            "-m",
            "isaacgymenvs.train",
            "task=AllegroHandDextremeADR",
            "multi_gpu=False",
            "headless=True",
            "force_render=False",
            f"task.env.numEnvs={count}",
            f"task.env.numActions={manifest['action_dim']}",
            f"task.env.numLatentObs={manifest['latent_obs_dim']}",
            f"seed={args.seed}",
            f"experiment={name}",
            f"+full_experiment_name={name}",
            f"+train.params.config.train_dir={root}",
            "task.env.asset.assetFileName=urdf/kuka_allegro_description/allegro_v4_caps.urdf",
            "task.env.random_network_adversary.enable=False",
            "task.env.actionPenaltyScale=0",
            "task.env.actionDeltaPenaltyScale=0",
            "task.task.adr.params.action_delay_prob.init_range=[0,0]",
            "task.task.adr.params.action_delay_prob.limits=[0,0]",
            "task.task.adr.params.action_delay_prob.delta=0.0",
        ]
        if args.max_epochs is not None:
            command.append(f"max_iterations={args.max_epochs}")
        if checkpoint is not None:
            command += [
                f"checkpoint={checkpoint}",
                "task.task.adr.adr_load_from_checkpoint=True",
            ]
    else:  # simtoolreal
        command += [
            str(host / "isaacsimenvs/train.py"),
            "--task",
            "Isaacsimenvs-SimToolReal-Direct-v0",
            "--agent",
            "rl_games_sapg_cfg_entry_point",
            "--headless",
            f"env.scene.num_envs={count}",
            "env.action.hand_moving_average=0.1",
            f"env.termination.success_tolerance={defaults['success_tolerance']}",
            f"env.termination.tolerance_curriculum_start_frames={defaults['curriculum_start_frames']}",
            f"agent.params.seed={args.seed}",
            f"agent.params.config.name=0_{name}",
            f"hydra.run.dir={root}",
        ]
        if args.max_epochs is not None:
            command.append(f"agent.params.config.max_epochs={args.max_epochs}")
        if checkpoint is not None:
            command += ["--checkpoint", str(checkpoint)]
    plan = {
        "recording": {"enabled": record, "every_epochs": record_every},
        "identity": identity,
        "study": study.model_dump(mode="json"),
        "seed": args.seed,
        "num_envs": count,
        "host_commit": head,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "command": command,
        "environment": env,
        "cwd": str(host),
        "output": str(root),
    }
    (root / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def execute_run(plan):
    env = os.environ.copy()
    # The selected host and its environment define import order, not an inherited development checkout.
    for name in (
        "ACTION_BENCH_RECORD_DIR",
        "ACTION_BENCH_RECORD_EVERY_EPOCHS",
        "ACTION_BENCH_RECORDING_MODULES",
    ):
        env.pop(name, None)
    env.update(plan["environment"])
    videos = None
    if plan["recording"]["enabled"]:
        from .recording.renderer import check_render_dependencies
        from .recording.videos import VideoWorker, asset_roots

        check_render_dependencies()
        videos = VideoWorker(
            Path(plan["output"]) / "recordings",
            asset_roots({"benchmark": plan["study"]["benchmark"], "host": plan["cwd"]}),
        )
    try:
        with open(Path(plan["output"]) / "train.log", "a") as log:
            process = subprocess.Popen(
                plan["command"],
                cwd=plan["cwd"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                    log.flush()
                result = process.wait()
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT)
                process.wait()
                raise
    finally:
        if videos is not None:
            videos.close()
    if result:
        raise subprocess.CalledProcessError(result, plan["command"])
