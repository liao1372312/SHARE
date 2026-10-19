# FLARE Domain Benchmark Schema

This benchmark is converted from existing domain service-composition task files under `dataset/`.
The user requests and task decompositions come from the existing domain datasets; execution traces and injected primary failures are generated in the FLARE schema for controlled localization and repair experiments.

Domains: finance, medical, technology, tourism, cross_domain.

Each JSONL row is compatible with `scripts/run_flare_experiments.py` and contains `events`, `repair_label`, and source metadata.
