#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER=/playpen/hisham26/llama.cpp/build/bin/llama-server
MODEL=/playpen/hisham26/models/qwen3.8-27b/Qwen3.8-27B-Q8_0.gguf
HOST="${QWEN_HOST:-127.0.0.1}"
PORT="${QWEN_PORT:-8080}"
RESTART=false

if [[ "${1:-}" == "--restart" ]]; then
  RESTART=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--restart]" >&2
  exit 2
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server is not built at $LLAMA_SERVER" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Qwen model is not downloaded at $MODEL" >&2
  exit 1
fi

# A previous llama-server is easy to forget because it commonly runs detached.
# Treat an already healthy instance as success, or replace it explicitly with
# --restart when launcher settings have changed.
existing_pid="$(
  pgrep -u "$(id -u)" -f "^${LLAMA_SERVER//./\\.} .*--port ${PORT}([[:space:]]|$)" |
    head -n 1 || true
)"

healthy=false
if curl --silent --fail --max-time 1 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  healthy=true
fi

if [[ "$healthy" == true && "$RESTART" == false ]]; then
  echo "Qwen is already running${existing_pid:+ (PID $existing_pid)}."
  echo "Chat UI: http://${HOST}:${PORT}"
  echo "Use '$0 --restart' to apply changed launcher settings."
  exit 0
fi

if [[ "$RESTART" == true ]]; then
  if [[ -z "$existing_pid" ]]; then
    if [[ "$healthy" == true ]]; then
      echo "Port $PORT has a healthy HTTP service, but it was not started by $LLAMA_SERVER." >&2
      echo "Refusing to stop it; choose another port with QWEN_PORT=<port>." >&2
      exit 1
    fi
  else
    echo "Stopping existing Qwen server (PID $existing_pid)..."
    kill "$existing_pid"
    for _ in {1..50}; do
      kill -0 "$existing_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$existing_pid" 2>/dev/null; then
      echo "Qwen server did not stop cleanly." >&2
      exit 1
    fi
  fi
elif [[ -n "$existing_pid" ]]; then
  echo "A llama-server process (PID $existing_pid) is starting or unhealthy on port $PORT." >&2
  echo "Wait for it, or run '$0 --restart'." >&2
  exit 1
elif ss -ltnH "sport = :$PORT" | grep -q .; then
  echo "Port $PORT is already occupied by another service." >&2
  echo "Choose another port with QWEN_PORT=<port> $0" >&2
  exit 1
fi

# GPU 0 is occupied by the lens-design workload; expose only GPU 1 to llama.cpp.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias qwen3.8-27b \
  --n-gpu-layers 999 \
  --ctx-size 32768 \
  --flash-attn on \
  --jinja \
  --reasoning on \
  --reasoning-budget -1 \
  --reasoning-format deepseek \
  --reasoning-preserve \
  --ui-config '{"showThoughtInProgress":true,"renderThinkingAsMarkdown":true}' \
  --cors-origins localhost \
  --host "$HOST" \
  --port "$PORT"
