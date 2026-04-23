#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable


def expand(p: str) -> str:
    return str(Path(os.path.expandvars(os.path.expanduser(p))).resolve())


DEFAULT_CONFIGS = [
    "swebench_backticks.yaml",
    expand("~/agent-stack/configs/qwen_vllm_textbased_overlay.yaml"),
]


def normalize_configs(configs: list[str] | None) -> list[str]:
    if not configs:
        return DEFAULT_CONFIGS.copy()
    return [cfg for cfg in configs if cfg]


def has_agent_mode_override(configs: Iterable[str]) -> bool:
    return any(cfg.strip().startswith("agent.mode=") for cfg in configs)


def build_common_args(args: argparse.Namespace) -> list[str]:
    cli_args: list[str] = []
    for cfg in normalize_configs(args.config):
        cli_args += ["-c", cfg]
    cli_args += [
        "--subset", expand(args.subset),
        "--split", args.split,
        "--model", args.model,
        "--environment-class", args.environment_class,
    ]
    return cli_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mini-swe-agent on SWE-bench with Phoenix tracing."
    )
    parser.add_argument("mode", choices=["single", "all"], help="Run one SWE-bench instance or the whole split")
    parser.add_argument("--instance", default="sympy__sympy-15599", help="Instance ID for single mode")
    parser.add_argument("--subset", default=expand("~/agent-stack/benchmarks/SWE-bench_Verified"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="hosted_vllm/qwen-coder")
    parser.add_argument("--environment-class", default="docker")
    parser.add_argument("--workers", type=int, default=1, help="Workers for batch mode")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Config file(s) or inline overrides like agent.mode=yolo. Can be repeated.",
    )
    parser.add_argument("--output", default=None, help="Output file (single) or directory (all)")
    parser.add_argument(
        "--exit-immediately",
        action="store_true",
        default=False,
        help="Only used in single mode.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        default=False,
        help="Only used in single mode. Equivalent to CLI -y for swebench-single.",
    )
    parser.add_argument(
        "--batch-agent-mode",
        choices=["yolo", "confirm", "human"],
        default="yolo",
        help="Agent mode injected for batch runs unless already overridden in --config.",
    )
    args = parser.parse_args()

    from phoenix.otel import register

    tracer_provider = register(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME", "mini-swe-agent"),
        auto_instrument=True,
        endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces"),
        protocol="http/protobuf",
        batch=False,
    )

    try:
        configs = normalize_configs(args.config)
        common_args = build_common_args(args)

        if args.mode == "single":
            from minisweagent.run.benchmarks.swebench_single import app as swebench_single_app

            output = args.output or expand(f"~/agent-stack/runs/debug/{args.instance}.traj.json")
            Path(output).parent.mkdir(parents=True, exist_ok=True)

            cli_args = common_args + [
                "-i", args.instance,
                "-o", output,
            ]
            if args.yolo:
                cli_args.append("-y")
            if args.exit_immediately:
                cli_args.append("--exit-immediately")

            swebench_single_app(cli_args, prog_name="mini-extra swebench-single")
        else:
            from minisweagent.run.benchmarks.swebench import app as swebench_app

            default_dir = expand(f"~/agent-stack/runs/batch/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            output = args.output or default_dir
            Path(output).mkdir(parents=True, exist_ok=True)

            cli_args = common_args.copy()
            if not has_agent_mode_override(configs):
                cli_args += ["-c", f"agent.mode={args.batch_agent_mode}"]
            cli_args += [
                "--workers", str(args.workers),
                "-o", output,
            ]

            swebench_app(cli_args, prog_name="mini-extra swebench")
    finally:
        tracer_provider.shutdown()


if __name__ == "__main__":
    main()
