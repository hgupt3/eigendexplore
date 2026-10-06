"""Small public CLI: list bases, resolve a method, and prepare or run a benchmark."""

import argparse
import json
from pathlib import Path

import yaml

from .catalog import catalog
from .settings import Study, load_study


def main(argv=None):
    parser = argparse.ArgumentParser(prog="action-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List packaged hands and ranks")
    resolve = commands.add_parser("config", help="Print a fully resolved method config")
    resolve.add_argument("--benchmark", required=True)
    resolve.add_argument("--hand", required=True)
    resolve.add_argument("--method", required=True)
    resolve.add_argument("--k", type=int)
    resolve.add_argument(
        "--set", action="append", default=[], metavar="PARAMETER=VALUE"
    )
    run = commands.add_parser(
        "run", help="Run a study in its separate benchmark environment"
    )
    run.add_argument("study", type=Path)
    run.add_argument("--installation", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Trial seed, 0 through 4294967295 (default: 101)",
    )
    run.add_argument("--num-envs", type=int)
    run.add_argument("--max-epochs", type=int)
    run.add_argument("--gpu", default="0")
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--prepare-only", action="store_true")
    run.add_argument(
        "--record",
        action="store_true",
        help="Save scheduled replayable state captures and render local MP4s",
    )
    run.add_argument(
        "--record-every-epochs",
        type=int,
        help="With --record, override the benchmark capture cadence",
    )
    render = commands.add_parser(
        "render", help="Render a saved state capture to MP4 on CPU"
    )
    render.add_argument("capture", type=Path)
    render.add_argument("--installation", required=True, type=Path)
    render.add_argument("--output", required=True, type=Path)
    from .benchmarks.spider.cli import add_parser

    add_parser(commands)
    args = parser.parse_args(argv)
    if args.command == "render":
        from .recording.renderer import render_capture
        from .recording.videos import asset_roots

        installation = json.loads(args.installation.read_text())
        print(
            render_capture(
                args.capture, args.output, asset_roots=asset_roots(installation)
            )
        )
        return
    if args.command == "spider":
        return args.handler(args)
    if args.command == "list":
        print(yaml.safe_dump(catalog(), sort_keys=False))
    elif args.command == "config":
        parameters = {}
        for item in args.set:
            name, value = item.split("=", 1)
            if name in parameters:
                raise ValueError(f"duplicate parameter {name}")
            parameters[name] = float(value)
        study = Study(
            benchmark=args.benchmark,
            hand=args.hand,
            method=args.method,
            k=args.k,
            parameters=parameters,
        )
        print(
            yaml.safe_dump(
                study.model_dump(mode="json", exclude_none=True), sort_keys=False
            ),
            end="",
        )
    else:
        from .launch import execute_run, prepare_run

        plan = prepare_run(load_study(args.study), args)
        print(json.dumps(plan, indent=2))
        if not args.prepare_only:
            execute_run(plan)


if __name__ == "__main__":
    main()
