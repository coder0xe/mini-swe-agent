"""Helpers for OpenInference/OpenTelemetry span annotations used inside mini-swe-agent."""

from __future__ import annotations

import json
from typing import Any

from opentelemetry.trace import Span, Status, StatusCode


MAX_TEXT_PREVIEW = 4000
MAX_JSON_PREVIEW = 8000


def preview_text(value: Any, limit: int = MAX_TEXT_PREVIEW) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def preview_json(value: Any, limit: int = MAX_JSON_PREVIEW) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return preview_text(text, limit=limit)


def set_span_attributes(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, preview_json(value))


def set_openinference_kind(span: Span, kind: str) -> None:
    span.set_attribute("openinference.span.kind", kind)


def mark_span_ok(span: Span) -> None:
    span.set_status(Status(StatusCode.OK))


def mark_span_error(span: Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.set_attribute("exception.type", type(exc).__name__)
    span.set_attribute("exception.message", str(exc))


def normalize_action(action: dict | str | None) -> dict[str, Any]:
    if isinstance(action, dict):
        return dict(action)
    if action is None:
        return {"command": ""}
    return {"command": str(action)}


def get_action_command(action: dict | str | None) -> str:
    normalized = normalize_action(action)
    command = normalized.get("command", "")
    return "" if command is None else str(command)


def get_trace_meta(action: dict | str | None) -> dict[str, Any]:
    normalized = normalize_action(action)
    meta = normalized.get("_trace_meta", {})
    return dict(meta) if isinstance(meta, dict) else {}
