#!/usr/bin/env bash
set -euo pipefail

resume_args=()
if [[ -f results/share_guard_v2_optimized_five_seed/results.json ]]; then
  resume_args=(--resume)
fi

python scripts/run_share_guard_experiments.py \
  --episodes datasets/share_synthetic_augmented_guard_development/episodes.jsonl \
  --out results/share_guard_v2_optimized_five_seed \
  --seeds 17,29,43,71,101 \
  --candidate-duals 0:4 \
  --selection-human-target 0.20 \
  --selection-unsafe-target 0.05 \
  --unsafe-budget 0.05 \
  --support-mode joint_calibrated \
  --joint-human-budget 0.18 \
  --joint-unsafe-budget 0.01 \
  --joint-max-stopped-rate 0.50 \
  --joint-stop-penalty 0.50 \
  --joint-grid-size 17 \
  --joint-human-reserve-rate 0.06 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 17 \
  "${resume_args[@]}"

python scripts/summarize_share_guard_selection.py \
  --results results/share_guard_v2_optimized_five_seed/results.json \
  --out results/share_guard_v2_optimized_five_seed/selection_summary.json
