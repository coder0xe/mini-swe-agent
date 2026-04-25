#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

from __future__ import annotations

import concurrent.futures
import json
import os
import random
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, TypeVar

import typer
from jinja2 import StrictUndefined, Template
from opentelemetry import context, trace
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge
from minisweagent.utils.tracing import mark_span_error, mark_span_ok, preview_json, set_openinference_kind, set_span_attributes

tracer = trace.get_tracer(__name__)

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.max_iterations=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "rebench": "nebius/SWE-rebench",
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()

_R = TypeVar("_R")


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None) or instance.get("docker_image", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    with tracer.start_as_current_span("benchmark.get_environment") as span:
        set_openinference_kind(span, "CHAIN")
        set_span_attributes(
            span,
            {
                "mini.instance_id": instance.get("instance_id"),
                "metadata": preview_json({"instance": {"instance_id": instance.get("instance_id")}}),
            },
        )
        try:
            # Important:
            # Do not mutate the shared config["environment"] in-place.
            # Multiple instances may share the same config object. Even when workers=1,
            # copying here prevents accidental image leakage across instances.
            env_config = dict(config.get("environment", {}))
            env_config["environment_class"] = env_config.get("environment_class", "docker")

            image_name = get_swebench_docker_image_name(instance)
            if env_config["environment_class"] in ["docker", "swerex_modal"]:
                env_config["image"] = image_name
            elif env_config["environment_class"] in ["singularity", "contree"]:
                env_config["image"] = "docker://" + image_name

            set_span_attributes(
                span,
                {
                    "mini.environment_class": env_config.get("environment_class"),
                    "mini.environment_image": env_config.get("image"),
                },
            )

            env = get_environment(env_config)

            if startup_command := config.get("run", {}).get("env_startup_command"):
                startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
                out = env.execute({
                    "command": startup_command,
                    "_trace_meta": {"step_index": 0, "action_index": -1, "agent_class": "EnvironmentStartup"},
                })
                if out["returncode"] != 0:
                    raise RuntimeError(f"Error executing startup command: {out}")

            mark_span_ok(span)
            return env

        except Exception as e:
            mark_span_error(span, e)
            raise


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance.

    Tracing design:
    - Every operation for one SWE-bench instance is kept under one
      benchmark.instance_run span.
    - Docker/environment startup failures must not let later get_environment,
      agent.run, agent.step, save, or cleanup spans become root-level spans.
    - The instance span context is explicitly attached while we run the
      instance. This is intentionally redundant with start_as_current_span(),
      because worker threads and nested instrumentation can otherwise lose the
      current OTel context in some setups.
    """
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id

    agent = None
    env = None
    model = None
    exit_status = None
    result = ""
    extra_info = {}
    progress_started = False
    instance_error: Exception | None = None
    cleanup_error: Exception | None = None

    with tracer.start_as_current_span("benchmark.instance_run") as span:
        # Pin the current OpenTelemetry context to this instance span for the
        # whole lifetime of the instance. This keeps child spans generated by
        # get_environment(), agent.run(), agent.step(), and env.cleanup() under
        # benchmark.instance_run even after exceptions/retries.
        instance_context_token = context.attach(trace.set_span_in_context(span))

        set_openinference_kind(span, "AGENT")
        set_span_attributes(
            span,
            {
                "mini.instance_id": instance_id,
                "mini.run.subset": config.get("run", {}).get("subset", ""),
                "mini.run.split": config.get("run", {}).get("split", ""),
                "mini.output_dir": str(output_dir),
                "metadata": preview_json(
                    {
                        "instance_id": instance_id,
                        "subset": config.get("run", {}).get("subset", ""),
                        "split": config.get("run", {}).get("split", ""),
                    }
                ),
            },
        )

        try:
            progress_manager.on_instance_start(instance_id)
            progress_started = True

            # Keep preparation inside benchmark.instance_run. Otherwise a
            # timeout during Docker startup can close the instance span, and any
            # later retry/save/cleanup/agent spans may leak out as root spans.
            with tracer.start_as_current_span("benchmark.prepare_instance") as prep_span:
                set_openinference_kind(prep_span, "CHAIN")
                set_span_attributes(prep_span, {"mini.instance_id": instance_id})

                instance_dir.mkdir(parents=True, exist_ok=True)
                remove_from_preds_file(output_dir / "preds.json", instance_id)
                (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

                mark_span_ok(prep_span)

            task = instance["problem_statement"]

            with tracer.start_as_current_span("benchmark.get_model") as model_span:
                set_openinference_kind(model_span, "CHAIN")
                set_span_attributes(model_span, {"mini.instance_id": instance_id})
                model = get_model(config=config.get("model", {}))
                set_span_attributes(
                    model_span,
                    {
                        "mini.model_name": getattr(model.config, "model_name", ""),
                    },
                )
                mark_span_ok(model_span)

            set_span_attributes(
                span,
                {
                    "input.value": task,
                    "mini.model_name": getattr(model.config, "model_name", ""),
                },
            )

            progress_manager.update_instance_status(instance_id, "Pulling/starting environment")
            env = get_sb_environment(config, instance)

            agent = ProgressTrackingAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                **config.get("agent", {}),
            )

            with tracer.start_as_current_span("agent.run") as agent_span:
                set_openinference_kind(agent_span, "AGENT")
                set_span_attributes(
                    agent_span,
                    {
                        "mini.instance_id": instance_id,
                        "mini.run.subset": config.get("run", {}).get("subset", ""),
                        "mini.run.split": config.get("run", {}).get("split", ""),
                        "input.value": task,
                    },
                )

                info = agent.run(
                    task,
                    instance_id=instance_id,
                    benchmark_subset=config.get("run", {}).get("subset", ""),
                    benchmark_split=config.get("run", {}).get("split", ""),
                ) or {}

                exit_status = info.get("exit_status")
                result = info.get("submission") or ""

                set_span_attributes(
                    agent_span,
                    {
                        "mini.agent_exit_status": exit_status,
                        "output.value": preview_json(info),
                    },
                )
                mark_span_ok(agent_span)

            set_span_attributes(
                span,
                {
                    "mini.agent_exit_status": exit_status,
                    "output.value": preview_json(
                        {
                            "exit_status": exit_status,
                            "submission_length": len(result),
                        }
                    ),
                },
            )

        except Exception as e:
            # Do not re-raise here. A failed instance should be recorded as a
            # failed instance and the batch should move on to the next one.
            # Most importantly, agent.run must not execute after environment
            # startup has failed.
            instance_error = e
            logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
            exit_status, result = type(e).__name__, ""
            extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
            mark_span_error(span, e)

        finally:
            try:
                if agent is not None:
                    with tracer.start_as_current_span("benchmark.save_trajectory") as save_span:
                        set_openinference_kind(save_span, "CHAIN")
                        traj_path = instance_dir / f"{instance_id}.traj.json"
                        set_span_attributes(
                            save_span,
                            {
                                "mini.instance_id": instance_id,
                                "mini.traj_path": str(traj_path),
                                "mini.exit_status": exit_status,
                            },
                        )
                        try:
                            agent.save(
                                traj_path,
                                {
                                    "info": {
                                        "exit_status": exit_status,
                                        "submission": result,
                                        **extra_info,
                                    },
                                    "instance_id": instance_id,
                                },
                            )
                            logger.info(f"Saved trajectory to '{traj_path}'")
                            mark_span_ok(save_span)
                        except Exception as e:
                            instance_error = instance_error or e
                            mark_span_error(save_span, e)
                            logger.error(f"Error saving trajectory for instance {instance_id}: {e}", exc_info=True)

                if env is not None and hasattr(env, "cleanup"):
                    # Use a wrapper span with a different name so an environment
                    # implementation that already emits sandbox.cleanup will
                    # appear underneath it rather than creating duplicate sibling
                    # spans. This should remain inside benchmark.instance_run.
                    with tracer.start_as_current_span("benchmark.cleanup_environment") as cleanup_span:
                        set_openinference_kind(cleanup_span, "CHAIN")
                        set_span_attributes(cleanup_span, {"mini.instance_id": instance_id})
                        try:
                            progress_manager.update_instance_status(instance_id, "Cleaning up environment")
                            env.cleanup()
                            mark_span_ok(cleanup_span)
                        except Exception as e:
                            cleanup_error = e
                            mark_span_error(cleanup_span, e)
                            logger.error(f"Error cleaning up instance {instance_id}: {e}", exc_info=True)

                model_name = ""
                if model is not None:
                    model_name = getattr(model.config, "model_name", "")
                if not model_name:
                    model_name = str(config.get("model", {}).get("model_name", ""))

                with tracer.start_as_current_span("benchmark.save_prediction") as pred_span:
                    set_openinference_kind(pred_span, "CHAIN")
                    set_span_attributes(
                        pred_span,
                        {
                            "mini.instance_id": instance_id,
                            "mini.model_name": model_name,
                            "mini.result_length": len(result),
                            "mini.exit_status": exit_status,
                        },
                    )
                    try:
                        update_preds_file(output_dir / "preds.json", instance_id, model_name, result)
                        mark_span_ok(pred_span)
                    except Exception as e:
                        instance_error = instance_error or e
                        mark_span_error(pred_span, e)
                        logger.error(f"Error writing prediction for instance {instance_id}: {e}", exc_info=True)

                if progress_started:
                    progress_manager.on_instance_end(instance_id, exit_status)

                if instance_error is None and cleanup_error is None:
                    mark_span_ok(span)
                else:
                    mark_span_error(span, instance_error or cleanup_error)

            finally:
                context.detach(instance_context_token)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


def run_with_current_context(fn: Callable[[], _R]) -> Callable[[], _R]:
    """Wrap a callable so it runs in the OpenTelemetry context captured here.

    ThreadPoolExecutor does not reliably preserve the current OTel context.
    Without this wrapper, benchmark.instance_run spans created inside worker
    threads may become independent root traces instead of children of the
    agent-level workflow span.
    """
    current_ctx = context.get_current()

    def _wrapped() -> _R:
        token = context.attach(current_ctx)
        try:
            return fn()
        finally:
            context.detach(token)

    return _wrapped


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
        "run": {"subset": subset, "split": split},
    })
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    run_id = os.getenv("MSA_EXPERIMENT_RUN_ID", "")
    agent_id = os.getenv("MSA_AGENT_ID", "")
    agent_rank = os.getenv("MSA_AGENT_RANK", "")
    agent_process_count = os.getenv("MSA_AGENT_PROCESS_COUNT", "")
    workflow_name = os.getenv("MSA_WORKFLOW_NAME", "")
    service_name = os.getenv("OTEL_SERVICE_NAME", "")

    # Keep your current naming logic unchanged.
    # If you later want the root span name fixed as "agent.process",
    # replace this line with: top_span_name = "agent.process"
    top_span_name = workflow_name or agent_id or "agent.process"

    with tracer.start_as_current_span(top_span_name) as workflow_span:
        set_openinference_kind(workflow_span, "AGENT")
        set_span_attributes(
            workflow_span,
            {
                "mini.workflow_name": workflow_name or top_span_name,
                "mini.run_id": run_id,
                "mini.agent_id": agent_id,
                "mini.agent_rank": agent_rank,
                "mini.agent_process_count": agent_process_count,
                "mini.run.subset": subset,
                "mini.run.split": split,
                "mini.output_dir": str(output_path),
                "mini.instance_count": len(instances),
                "mini.workers": workers,
                "service.name": service_name,
                "metadata": preview_json(
                    {
                        "workflow_name": workflow_name or top_span_name,
                        "run_id": run_id,
                        "agent_id": agent_id,
                        "agent_rank": agent_rank,
                        "agent_process_count": agent_process_count,
                        "subset": subset,
                        "split": split,
                        "output_path": str(output_path),
                        "instance_count": len(instances),
                        "workers": workers,
                        "service_name": service_name,
                    }
                ),
            },
        )

        try:
            with Live(progress_manager.render_group, refresh_per_second=4):
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            run_with_current_context(
                                lambda instance=instance: process_instance(
                                    instance,
                                    output_path,
                                    config,
                                    progress_manager,
                                )
                            )
                        ): instance["instance_id"]
                        for instance in instances
                    }
                    try:
                        process_futures(futures)
                    except KeyboardInterrupt:
                        logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                        for future in futures:
                            if not future.running() and not future.done():
                                future.cancel()
                        process_futures(futures)

            set_span_attributes(
                workflow_span,
                {
                    "mini.completed_instance_count": len(instances),
                    "output.value": preview_json(
                        {
                            "run_id": run_id,
                            "agent_id": agent_id,
                            "workflow_name": workflow_name or top_span_name,
                            "instance_count": len(instances),
                            "output_path": str(output_path),
                        }
                    ),
                },
            )
            mark_span_ok(workflow_span)

        except Exception as e:
            mark_span_error(workflow_span, e)
            raise


if __name__ == "__main__":
    app()