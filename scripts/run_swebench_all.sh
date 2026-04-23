#!/usr/bin/env bash
set -euo pipefail

WORKERS="${1:-1}"

source ~/agent-stack/scripts/load_env.sh
source ~/agent-stack/.venv-agent/bin/activate

export OTEL_SERVICE_NAME=mini-swe-agent
export PHOENIX_PROJECT_NAME=mini-swe-agent
export PHOENIX_COLLECTOR_ENDPOINT="${PHOENIX_COLLECTOR_ENDPOINT:-http://127.0.0.1:6006/v1/traces}"

mkdir -p ~/agent-stack/runs/batch

exec python ~/agent-stack/scripts/swebench_trace_runner.py \
  all \
  --workers "$WORKERS"
