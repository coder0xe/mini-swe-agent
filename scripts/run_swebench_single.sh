#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-sympy__sympy-15599}"

source ~/agent-stack/scripts/load_env.sh
source ~/agent-stack/.venv-agent/bin/activate

export OTEL_SERVICE_NAME=mini-swe-agent
export PHOENIX_PROJECT_NAME=mini-swe-agent
export PHOENIX_COLLECTOR_ENDPOINT="${PHOENIX_COLLECTOR_ENDPOINT:-http://127.0.0.1:6006/v1/traces}"

mkdir -p ~/agent-stack/runs/debug

exec python ~/agent-stack/scripts/swebench_trace_runner.py \
  single \
  --instance "$INSTANCE_ID" \
  --yolo \
  --exit-immediately
