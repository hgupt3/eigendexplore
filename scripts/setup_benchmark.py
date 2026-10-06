#!/usr/bin/env python3
"""Prepare pinned sources, then install them into a dedicated simulator environment."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml


def installation_env():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def run(*args, cwd=None):
    subprocess.run([str(x) for x in args], cwd=cwd, env=installation_env(), check=True)


def checkout(spec, path):
    run("git", "clone", "--no-checkout", spec["repository"], path)
    run("git", "checkout", "--detach", spec["commit"], cwd=path)


def install_task_supplement(repo, host, supplement):
    manifest = json.loads((repo / "taskpacks" / (supplement + ".json")).read_text())
    archive = repo / "taskpacks" / (supplement + ".tar.gz")
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
            raise ValueError("task supplement differs from its file inventory")
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("task supplement must contain relative regular files")
            target = host / path
            if not target.resolve().is_relative_to(host.resolve()):
                raise ValueError("task supplement escapes the host directory")
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"task supplement would overwrite {target}")
        for member in members:
            target = host / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def prepare_sources(repo, root, benchmark, spec):
    record = {"schema_version": 1, "benchmark": benchmark, "spec": spec}
    supplement_id = spec.get("task_supplement")
    if supplement_id:
        supplement = json.loads(
            (repo / "taskpacks" / (supplement_id + ".json")).read_text()
        )
        if supplement["host_commit"] != spec["commit"]:
            raise ValueError("task supplement is pinned to a different host revision")
        record["task_supplement"] = supplement["id"]
    if root.exists():
        plan = root / "sources.json"
        if (root / "installation.json").exists() or not plan.is_file():
            raise FileExistsError(f"refusing to modify existing installation: {root}")
        if json.loads(plan.read_text()) != record:
            raise ValueError("prepared source identity differs from the lockfile")
        paths = [(root / "host", spec)]
        for path, source in paths:
            head = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            if head != source["commit"]:
                raise ValueError(f"prepared source revision changed: {path}")
        run(
            "git",
            "apply",
            "--reverse",
            "--check",
            repo / "third_party" / spec["patch"],
            cwd=root / "host",
        )
        if benchmark == "simtoolreal":
            # Compare against the pinned subtree plus its declared patch, using
            # a temporary index so both staged and unstaged drift are rejected.
            with tempfile.TemporaryDirectory() as temporary:
                env = installation_env()
                env["GIT_INDEX_FILE"] = str(Path(temporary) / "index")
                commands = [
                    ["git", "read-tree", spec["rl_games"]["commit"]],
                    [
                        "git",
                        "apply",
                        "--cached",
                        str(repo / "third_party" / spec["rl_games"]["patch"]),
                    ],
                    ["git", "diff", "--exit-code", "--", spec["rl_games"]["vendored"]],
                ]
                for command in commands:
                    subprocess.run(command, cwd=root / "host", env=env, check=True)
            untracked = subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "--",
                    spec["rl_games"]["vendored"],
                ],
                cwd=root / "host",
                text=True,
            ).strip()
            if untracked:
                raise ValueError("prepared rl_games subtree contains untracked files")
        if supplement_id:
            with tarfile.open(repo / "taskpacks" / (supplement_id + ".tar.gz")) as tar:
                for member in tar.getmembers():
                    if (root / "host" / member.name).read_bytes() != tar.extractfile(
                        member
                    ).read():
                        raise ValueError("prepared task supplement changed")
        return
    root.mkdir(parents=True)
    host = root / "host"
    checkout(spec, host)
    if benchmark == "simtoolreal":
        run(
            "git",
            "restore",
            "--source",
            spec["rl_games"]["commit"],
            "--staged",
            "--worktree",
            "--",
            spec["rl_games"]["vendored"],
            cwd=host,
        )
    run("git", "apply", "--check", repo / "third_party" / spec["patch"], cwd=host)
    run("git", "apply", repo / "third_party" / spec["patch"], cwd=host)
    if benchmark == "simtoolreal":
        patch = repo / "third_party" / spec["rl_games"]["patch"]
        run("git", "apply", "--check", patch, cwd=host)
        run("git", "apply", patch, cwd=host)
    if supplement_id:
        install_task_supplement(repo, host, supplement_id)
    (root / "sources.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["dextreme", "simtoolreal", "spider"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--python", type=Path, help="Dedicated simulator environment Python"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Fetch and patch sources/task inputs; do not install packages",
    )
    args = parser.parse_args()
    if not args.prepare_only and args.python is None:
        parser.error("--python is required unless --prepare-only is selected")
    repo = Path(__file__).resolve().parents[1]
    spec = yaml.safe_load((repo / "third_party/benchmarks.lock.yaml").read_text())[
        "benchmarks"
    ][args.benchmark]
    root = args.root.expanduser().resolve()
    python = args.python.expanduser().absolute() if args.python else None
    if not args.prepare_only:
        # Preserve venv interpreter symlinks: resolving them selects the base environment.
        version = subprocess.check_output(
            [str(python), "-c", 'import sys;print("%d.%d" % sys.version_info[:2])'],
            text=True,
            env=installation_env(),
        ).strip()
        if version != spec["python"]:
            raise ValueError(
                f"{args.benchmark} requires Python {spec['python']}, found {version}"
            )
    prepare_sources(repo, root, args.benchmark, spec)
    if args.prepare_only:
        print(
            f"Sources ready at {root}. Follow docs/{args.benchmark}.md to install dependencies, then rerun with --python."
        )
        return
    host = root / "host"
    if args.benchmark == "simtoolreal":
        run(
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-e",
            host / spec["rl_games"]["vendored"],
        )
    elif args.benchmark == "dextreme":
        run(
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            "rl-games==" + spec["rl_games"]["version"],
        )
        package_root = Path(
            subprocess.check_output(
                [
                    str(python),
                    "-c",
                    "import pathlib, rl_games; print(pathlib.Path(rl_games.__file__).resolve().parent.parent)",
                ],
                text=True,
                env=installation_env(),
            ).strip()
        )
        patch = repo / "third_party" / spec["rl_games"]["patch"]
        run("git", "apply", "--check", patch, cwd=package_root)
        run("git", "apply", patch, cwd=package_root)
    run(python, "-m", "pip", "install", "--no-deps", "-e", host)
    run(
        python,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-e",
        repo / "packages/eigendexplore",
    )
    if args.benchmark != "dextreme":
        run(python, "-m", "pip", "install", "--no-deps", "-e", repo)
    if args.benchmark == "spider":
        probe = (
            "import importlib.metadata as m; "
            f"expected={spec['runtime']!r}; "
            "assert all(m.version(k)==v for k,v in expected.items()), 'Spider dependency versions differ from lock'; "
            "from pathlib import Path; "
            "from action_bench.benchmarks.spider.runner import load_host; "
            "load_host(Path('host')); print('Spider host imports ready')"
        )
    else:
        probe = 'from eigendexplore.rl_games import register; register(); print("EigenDExplore registered")'
    run(python, "-c", probe, cwd=root)
    (root / "installation.json").write_text(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "host": str(host),
                "python": str(python),
                "host_commit": spec["commit"],
                "source": str(repo),
                "dependencies": spec,
                "packages": subprocess.check_output(
                    [str(python), "-m", "pip", "freeze"],
                    text=True,
                    env=installation_env(),
                ).splitlines(),
            },
            indent=2,
        )
        + "\n"
    )
    print(root / "installation.json")


if __name__ == "__main__":
    main()
