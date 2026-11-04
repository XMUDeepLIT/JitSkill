#!/usr/bin/env bash
set -euo pipefail
: "${ENCODER_BASE:?}" "${EMBEDDER:?}" "${TRAIN_TRAJECTORIES:?}" "${SKILLS:?}" "${WORK_DIR:?}" "${COMPRESSOR_CKPT:?}"
mkdir -p "$WORK_DIR"
python -m jitskill.train_navigator --traj "$TRAIN_TRAJECTORIES" --skills "$SKILLS" \
  --model "$ENCODER_BASE" --emb_model "$EMBEDDER" --out "$WORK_DIR/nav" \
  --head "${HEAD:-cross}" --epochs 3 --val_scenarios 5 --seed 0 \
  --init_encoder "$COMPRESSOR_CKPT"
