#!/usr/bin/env bash
set -euo pipefail
: "${EXECUTOR_DIR:?}" "${SFT_ROWS:?}" "${LATENT_ROWS:?}" "${WORK_DIR:?}"
mkdir -p "$WORK_DIR"
python -m jitskill.executor.train_projection --sft "$SFT_ROWS" --latents "$LATENT_ROWS" \
  --model "$EXECUTOR_DIR" --out "$WORK_DIR/projector.pt" --epochs 3 --holdout 5
