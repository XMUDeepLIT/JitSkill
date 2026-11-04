#!/usr/bin/env bash
set -euo pipefail
: "${EXECUTOR_DIR:?frozen 27B executor directory}"
: "${NAV_BASE:?encoder base directory}"
: "${NAV_CKPT:?navigator checkpoint}"
: "${ENC_DIR:?encoder LoRA directory}"
: "${PROJECTOR:?trained state projector checkpoint}"
: "${SKILLS:?skill library JSONL}"
ENGINE_PORT=${ENGINE_PORT:-8000}
PROXY_PORT=${PROXY_PORT:-8100}
SERVE_HOST=${SERVE_HOST:-127.0.0.1}
MODEL_NAME=${MODEL_NAME:-executor}
ALPHA=${ALPHA:-0.75}
INJECTION_ROLE=${INJECTION_ROLE:-user}
NAV_DEVICE=${NAV_DEVICE:-cuda:0}
python -m vllm.entrypoints.openai.api_server \
  --model "$EXECUTOR_DIR" --served-model-name "$MODEL_NAME" --enable-prompt-embeds \
  --host "$SERVE_HOST" --port "$ENGINE_PORT" &
ENGINE_PID=$!
trap 'kill "$ENGINE_PID" 2>/dev/null || true' EXIT
# Wait for the executor to be ready; fail if it exited.
for attempt in $(seq 1 120); do
  kill -0 "$ENGINE_PID" 2>/dev/null || { echo 'executor exited' >&2; exit 1; }
  if curl --fail --silent "http://$SERVE_HOST:$ENGINE_PORT/health" >/dev/null; then break; fi
  sleep 2
done
curl --fail --silent "http://$SERVE_HOST:$ENGINE_PORT/health" >/dev/null
python -m serving.inject_proxy \
  --host "$SERVE_HOST" --port "$PROXY_PORT" \
  --upstream "http://$SERVE_HOST:$ENGINE_PORT" --upstream_model "$MODEL_NAME" \
  --tokenizer "$EXECUTOR_DIR" --skills "$SKILLS" \
  --nav_base "$NAV_BASE" --nav_ckpt "$NAV_CKPT" --enc_dir "$ENC_DIR" \
  --nav_device "$NAV_DEVICE" --projector "$PROJECTOR" \
  --alpha "$ALPHA" --lex step --inject_record 1 --exec_latent 1 \
  --injection_role "$INJECTION_ROLE"
