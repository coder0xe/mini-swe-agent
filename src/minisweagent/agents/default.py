"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from opentelemetry import trace
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded
from minisweagent.utils.serialize import recursive_merge
from minisweagent.utils.tracing import (
    get_action_command,
    mark_span_error,
    mark_span_ok,
    normalize_action,
    preview_json,
    preview_text,
    set_openinference_kind,
    set_span_attributes,
)

tracer = trace.get_tracer(__name__)


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def _run_span_attributes(self, task: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        model_name = getattr(getattr(self.model, "config", None), "model_name", None)
        return {
            "input.value": preview_text(task, 4000),
            "agent.class": self.__class__.__name__,
            "agent.module": self.__class__.__module__,
            "agent.step_limit": self.config.step_limit,
            "agent.cost_limit": self.config.cost_limit,
            "environment.class": self.env.__class__.__name__,
            "environment.module": self.env.__class__.__module__,
            "llm.model_name": model_name,
            "metadata": preview_json(kwargs),
            "mini.instance_id": kwargs.get("instance_id"),
            "mini.subset": kwargs.get("subset") or kwargs.get("benchmark_subset"),
            "mini.split": kwargs.get("split") or kwargs.get("benchmark_split"),
        }

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        with tracer.start_as_current_span("agent.run") as span:
            set_openinference_kind(span, "AGENT")
            set_span_attributes(span, self._run_span_attributes(task, kwargs))
            try:
                self.extra_template_vars |= {"task": task, **kwargs}
                self.messages = []
                self.add_messages(
                    self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
                    self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
                )
                while True:
                    try:
                        self.step()
                    except InterruptAgentFlow as e:
                        self.add_messages(*e.messages)
                    except Exception as e:
                        self.handle_uncaught_exception(e)
                        raise
                    finally:
                        self.save(self.config.output_path)
                    if self.messages[-1].get("role") == "exit":
                        break
                result = self.messages[-1].get("extra", {})
                set_span_attributes(
                    span,
                    {
                        "output.value": preview_json(result),
                        "mini.exit_status": result.get("exit_status"),
                        "mini.submission_preview": preview_text(result.get("submission", ""), 1000),
                        "mini.api_calls": self.n_calls,
                        "llm.cost.total": self.cost,
                    },
                )
                mark_span_ok(span)
                return result
            except Exception as e:
                mark_span_error(span, e)
                raise

    def _step_impl(self) -> list[dict]:
        return self.execute_actions(self.query())

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        step_index = self.n_calls + 1
        with tracer.start_as_current_span("agent.step") as span:
            set_openinference_kind(span, "CHAIN")
            set_span_attributes(
                span,
                {
                    "mini.step_index": step_index,
                    "mini.messages_before": len(self.messages),
                    "llm.cost.total": self.cost,
                },
            )
            try:
                result = self._step_impl()
                set_span_attributes(span, {"mini.messages_after": len(self.messages)})
                mark_span_ok(span)
                return result
            except Exception as e:
                mark_span_error(span, e)
                raise

    def _query_impl(self) -> dict:
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        step_index = self.n_calls + 1
        with tracer.start_as_current_span("agent.query") as span:
            set_openinference_kind(span, "CHAIN")
            set_span_attributes(
                span,
                {
                    "mini.step_index": step_index,
                    "input.value": preview_json(self.messages[-3:] if self.messages else []),
                    "mini.message_count": len(self.messages),
                },
            )
            try:
                message = self._query_impl()
                actions = message.get("extra", {}).get("actions", [])
                set_span_attributes(
                    span,
                    {
                        "output.value": preview_json(message),
                        "mini.action_count": len(actions),
                        "llm.cost.total": self.cost,
                    },
                )
                mark_span_ok(span)
                return message
            except Exception as e:
                mark_span_error(span, e)
                raise

    def _annotate_actions(self, actions: list[Any], *, step_index: int) -> list[dict[str, Any]]:
        annotated: list[dict[str, Any]] = []
        for action_index, raw_action in enumerate(actions):
            action = normalize_action(raw_action)
            trace_meta = dict(action.get("_trace_meta", {})) if isinstance(action.get("_trace_meta"), dict) else {}
            trace_meta |= {
                "step_index": step_index,
                "action_index": action_index,
                "agent_class": self.__class__.__name__,
            }
            action["_trace_meta"] = trace_meta
            annotated.append(action)
        return annotated

    def _execute_actions_impl(self, message: dict, *, step_index: int) -> list[dict]:
        actions = self._annotate_actions(message.get("extra", {}).get("actions", []), step_index=step_index)
        outputs = [self.env.execute(action) for action in actions]
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        step_index = self.n_calls
        actions = self._annotate_actions(message.get("extra", {}).get("actions", []), step_index=step_index)
        with tracer.start_as_current_span("agent.execute_actions") as span:
            set_openinference_kind(span, "CHAIN")
            set_span_attributes(
                span,
                {
                    "mini.step_index": step_index,
                    "mini.action_count": len(actions),
                    "input.value": preview_json([get_action_command(action) for action in actions]),
                },
            )
            try:
                outputs = [self.env.execute(action) for action in actions]
                result = self.add_messages(
                    *self.model.format_observation_messages(message, outputs, self.get_template_vars())
                )
                set_span_attributes(span, {"output.value": preview_json(outputs)})
                mark_span_ok(span)
                return result
            except Exception as e:
                mark_span_error(span, e)
                raise

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
