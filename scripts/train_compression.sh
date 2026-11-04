#!/usr/bin/env bash
set -euo pipefail
: "${ENCODER_BASE:?}" "${TRAIN_TRAJECTORIES:?}" "${WORK_DIR:?}"
mkdir -p "$WORK_DIR"
python -m jitskill.train_compression --traj "$TRAIN_TRAJECTORIES" \
  --model "$ENCODER_BASE" --out "$WORK_DIR/compression.pt" --epochs "${COMP_EPOCHS:-3}"
