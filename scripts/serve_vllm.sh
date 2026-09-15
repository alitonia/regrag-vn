#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:?Usage: $0 MODEL_ID [additional vLLM arguments...]}"
shift

if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM is not installed. Install it with: pip install vllm" >&2
  exit 127
fi

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
SERVED_NAME="${VLLM_SERVED_MODEL_NAME:-$MODEL_ID}"

echo "Starting $SERVED_NAME on http://$HOST:$PORT/v1"
exec vllm serve "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  "$@"
