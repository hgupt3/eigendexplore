"""Spider command line: run paired IID/EigenDExplore episodes for one hand, then report."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .protocol import (
    PACK_ID,
    SETTING_NAMES,
    Job,
    defaults,
    make_jobs,
    settings_grid,
    spider_basis,
)


def add_parser(commands):
    parser = commands.add_parser(
        "spider", help="Cold-start trajectory optimization: IID vs EigenDExplore"
    )
    sub = parser.add_subparsers(dest="spider_command", required=True)
    report = sub.add_parser("report", help="Summarize a finished run")
    report.add_argument("run", type=Path)
    run = sub.add_parser("run", help="Paired IID/EigenDExplore episodes for one hand")
    run.add_argument("--hand", default=defaults()["hand"], choices=defaults()["hands"])
    run.add_argument("--installation", type=Path, required=True)
    run.add_argument("--assets", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=V1[,V2,...]",
        help=f"One of {', '.join(SETTING_NAMES)}; several values sweep every combination",
    )
    run.add_argument("--task", action="append", help="Repeat to run a subset of tasks")
    run.add_argument(
        "--seed", type=int, action="append", help="Repeat to run a subset of seeds"
    )
    run.add_argument("--gpu", default="0")
    run.add_argument("--prepare-only", action="store_true")
    run.add_argument("--record", action="store_true", help="Save each episode video")
    run.add_argument("--resume", action="store_true", help="Skip completed episodes")
    parser.set_defaults(handler=main)


def parse_settings(items):
    values = {}
    for item in items:
        name, _, raw = item.partition("=")
        if name in values or not raw:
            raise ValueError(f"--set {item!r}: give each name once as NAME=V1[,V2,...]")
        items = raw.split(",")
        values[name] = items if name == "basis" else [float(v) for v in items]
    return settings_grid(values)


def prepare(args):
    installation = json.loads(args.installation.read_text())
    if installation["benchmark"] != "spider":
        raise ValueError("installation is not Spider")
    host = Path(installation["host"])
    head = subprocess.check_output(
        ["git", "-C", str(host), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != installation["host_commit"]:
        raise ValueError("host revision differs from installation")
    settings = parse_settings(args.set)
    jobs = make_jobs(args.hand, tasks=args.task, seeds=args.seed, settings=settings)
    dataset = args.assets.resolve()
    manifest = json.loads((dataset / "taskpack.json").read_text())
    if manifest.get("id") != PACK_ID or manifest.get("schema_version") != 1:
        raise ValueError(f"--assets is not the extracted {PACK_ID} task pack")
    if args.hand not in manifest["hands"] or not {j.task for j in jobs} <= set(
        manifest["tasks"]
    ):
        raise ValueError("task pack does not contain every requested task")
    root = args.output.resolve()
    plan = {
        "schema_version": 2,
        "benchmark": "spider",
        "release": "0.3.0",
        "hand": args.hand,
        "settings": [s.model_dump() for s in settings],
        "host": str(host),
        "host_commit": head,
        "python": installation["python"],
        "dataset": str(dataset),
        "asset_pack_id": manifest["id"],
        "output": str(root),
        "record": args.record,
        "protocol": defaults(),
        "basis_id": spider_basis(args.hand).artifact_id,
        "jobs": [j.model_dump() for j in jobs],
    }
    if root.exists():
        if not args.resume:
            raise FileExistsError(root)
        previous = json.loads((root / "spider-plan.json").read_text())
        if previous != plan:
            raise ValueError("--resume needs the same settings, assets, and software")
    else:
        root.mkdir(parents=True)
        (root / "installation.json").write_text(
            json.dumps(installation, indent=2) + "\n"
        )
        (root / "spider-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def validate_result(result, job, plan):
    if result.get("job") != job.model_dump():
        raise ValueError("completed result has a different job identity")
    if result.get("basis_id") != plan["basis_id"]:
        raise ValueError("completed result has a different basis identity")


def execute(plan, gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", PYTHONPATH=plan["host"])
    if plan["record"]:
        # Headless MuJoCo video needs EGL on the selected physical GPU.
        env.update(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=gpu)
    root = Path(plan["output"])
    for index, raw in enumerate(plan["jobs"]):
        job = Job.model_validate(raw)
        output = root / job.id
        if output.exists():
            result_path = output / "result.json"
            result = (
                json.loads(result_path.read_text()) if result_path.is_file() else {}
            )
            if result.get("completed"):
                validate_result(result, job, plan)
                continue
            raise ValueError(
                f"partial job preserved at {output}; move it aside explicitly before retrying"
            )
        logpath = root / "logs" / f"{index:04d}.log"
        logpath.parent.mkdir(exist_ok=True)
        command = [
            plan["python"],
            "-u",
            "-m",
            "action_bench.benchmarks.spider.runner",
            "--plan",
            str(root / "spider-plan.json"),
            "--job",
            str(index),
        ]
        print(f"[{index + 1}/{len(plan['jobs'])}] {job.id}", flush=True)
        with logpath.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=plan["host"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)
    return report(root)


def report(root):
    root = Path(root)
    plan = json.loads((root / "spider-plan.json").read_text())
    results = {}
    missing = []
    for raw in plan["jobs"]:
        job = Job.model_validate(raw)
        path = root / job.id / "result.json"
        result = json.loads(path.read_text()) if path.is_file() else {}
        if not result.get("completed") or result.get("job") != raw:
            missing.append(job.id)
            continue
        validate_result(result, job, plan)
        cost = float(result["final_full_physics_cost"])
        if not np.isfinite(cost):
            raise ValueError(f"nonfinite episode cost: {job.id}")
        results[(job.hand, job.task, job.seed, job.setting_id)] = cost
    # Partial studies are visible, but never summarized as a completed comparison.
    if missing:
        status = {
            "complete": False,
            "completed": len(results),
            "expected": len(plan["jobs"]),
            "missing": missing,
        }
        (root / "report-status.json").write_text(json.dumps(status, indent=2) + "\n")
        return status
    groups = {}
    for raw in plan["jobs"]:
        job = Job.model_validate(raw)
        if job.method == "iid":
            continue
        key = (job.hand, job.setting_id)
        cost = results[(job.hand, job.task, job.seed, job.setting_id)]
        iid = results[(job.hand, job.task, job.seed, "iid")]
        groups.setdefault(key, []).append((job.task, job.seed, cost, iid))
    rows = []
    for (hand, setting), values in sorted(groups.items()):
        costs = np.array([v[2] for v in values])
        iid = np.array([v[3] for v in values])
        delta = costs - iid
        tasks = sorted({v[0] for v in values})
        seeds = sorted({v[1] for v in values})
        grid = np.array(
            [
                [
                    next(v[2] - v[3] for v in values if v[:2] == (task, seed))
                    for seed in seeds
                ]
                for task in tasks
            ]
        )
        rng = np.random.default_rng(0)
        ti = rng.integers(len(tasks), size=(5000, len(tasks)))
        si = rng.integers(len(seeds), size=(5000, len(seeds)))
        draws = grid[ti[:, :, None], si[:, None, :]].mean(axis=(1, 2))
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "hand": hand,
                "setting": setting,
                "pairs": len(values),
                "mean_cost": float(costs.mean()),
                "iid_mean_cost": float(iid.mean()),
                "paired_cost_delta": float(delta.mean()),
                "delta_crossed_95ci_low": float(low),
                "delta_crossed_95ci_high": float(high),
                "wins": int((delta < 0).sum()),
                "losses": int((delta > 0).sum()),
            }
        )
    with (root / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status = {
        "complete": True,
        "completed": len(results),
        "primary_metric": "final_full_physics_cost",
        "comparison": "EigenDExplore minus paired IID; negative favors EigenDExplore",
        "interval": "crossed task/seed bootstrap, 5000 draws, seed 0",
        "rows": rows,
    }
    (root / "comparison.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main(args):
    if args.spider_command == "report":
        print(json.dumps(report(args.run), indent=2))
        return
    plan = prepare(args)
    print(
        f"Prepared {len(plan['jobs'])} episodes: {Path(plan['output']) / 'spider-plan.json'}"
    )
    if not args.prepare_only:
        execute(plan, args.gpu)
