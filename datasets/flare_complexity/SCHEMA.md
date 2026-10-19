# FLARE Complexity Benchmark Schema

This benchmark is split from `datasets/flare_domain` by trace complexity.

- `simple`: traces with at most 2 planned service-composition steps.
- `complex`: traces with more than 2 planned service-composition steps.

Each row remains compatible with `scripts/run_flare_experiments.py` and preserves the source domain in `metadata.source_dataset`.
