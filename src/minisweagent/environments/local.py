from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

from opentelemetry import trace
from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge
from minisweagent.utils.tracing import (
    get_action_command,
    get_trace_meta,
    mark_span_error,
    mark_span_ok,
    normalize_action,
    preview_json,
    preview_text,
    set_openinference_kind,
    set_span_attributes,
)

tracer = trace.get_tracer(__name__)


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)

    def execute(self, action: dict | str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the local environment and return the result as a dict."""
        action_dict = normalize_action(action)
        command = get_action_command(action_dict)
        trace_meta = get_trace_meta(action_dict)
        cwd = cwd or self.config.cwd or os.getcwd()
        with tracer.start_as_current_span("tool.exec") as span:
            set_openinference_kind(span, "TOOL")
            set_span_attributes(
                span,
                {
                    "input.value": preview_text(command),
                    "tool.name": "bash",
                    "tool.parameters": preview_json({"cwd": cwd, "timeout": timeout or self.config.timeout}),
                    "mini.tool.backend": "local",
                    "mini.tool.command": command,
                    "mini.tool.cwd": cwd,
                    "mini.step_index": trace_meta.get("step_index"),
                    "mini.action_index": trace_meta.get("action_index"),
                    "mini.agent_class": trace_meta.get("agent_class"),
                },
            )
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    text=True,
                    cwd=cwd,
                    env=os.environ | self.config.env,
                    timeout=timeout or self.config.timeout,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
                set_span_attributes(
                    span,
                    {
                        "output.value": preview_text(result.stdout),
                        "mini.tool.returncode": result.returncode,
                    },
                )
                self._check_finished(output)
                mark_span_ok(span)
                return output
            except Submitted as e:
                set_span_attributes(
                    span,
                    {
                        "mini.tool.returncode": 0,
                        "output.value": preview_text(e.args[0].get("content", "") if e.args else ""),
                        "mini.tool.submitted": True,
                    },
                )
                mark_span_ok(span)
                raise
            except Exception as e:
                raw_output = getattr(e, "output", None)
                raw_output = (
                    raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
                )
                output = {
                    "output": raw_output,
                    "returncode": -1,
                    "exception_info": f"An error occurred while executing the command: {e}",
                    "extra": {"exception_type": type(e).__name__, "exception": str(e)},
                }
                set_span_attributes(
                    span,
                    {
                        "output.value": preview_text(raw_output),
                        "mini.tool.returncode": -1,
                    },
                )
                mark_span_error(span, e)
                return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), os.environ, kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
