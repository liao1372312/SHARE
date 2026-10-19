# SHARE

Sequential Human–AI Authority Allocation for Safety-Oriented Repair of Agentic Software Workflows.

SHARE constructs a shared repair context from a workflow execution graph, learns downstream AI/Human/Stop action values with paired Double DQN, and enforces a shared intervention quota with an online ledger. Human and Stop terminate the modeled episode; AI can continue to a later repair state.

## Repository status

This checkout is being migrated to the protocol described in `SHARE__Repair_Grounded.pdf`. Core implementations and the paper's synthetic repair-backbone corpus are retained. Updated observed authority branches, their execution evidence, and the corresponding trained artifacts still need to be supplied. This checkout does **not** currently establish reproduction of the paper's expert-repair experiment.

The actionable handover document is [PAPER_COMPLETION_CHECKLIST.md](PAPER_COMPLETION_CHECKLIST.md). It records required inputs, remaining implementation differences, and acceptance criteria. Historical controller simulations retain their original provenance in a verified private backup. Broad cleanup of this source directory is awaiting confirmation of its exact scope; use the curated `release/SHARE/` directory for the proposed public release, rather than uploading this entire source directory.

## Installation

Python 3.12 is the baseline for the checked environment.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Use `requirements-paper.txt` for optional statistical analysis and plotting. A validated dependency lock and the final software/data licenses are pending in the checklist.

## Layout

| Path | Purpose |
| --- | --- |
| `flare/` | Execution graphs, gap detection, localization, slicing, and repair operators |
| `share/` | States, rewards, paired replay, policies, Double DQN, and the intervention ledger |
| `configs/paper_protocol.json` | Paper settings and expected sample counts; these are specifications, not measured results |
| `configs/paper_artifacts.template.json` | Template for supplied frozen models and their hashes |
| `datasets/flare_complexity/` | Synthetic backbone workflows explicitly described in paper Section 4.1 |
| `datasets/paper/` | Destination for updated observed authority branches and import reports |
| `artifacts/paper/` | Destination for updated encoders, checkpoints, and validation calibration |
| `scripts/evaluate_share_paper.py` | Parameterized, read-only input evaluation with a separate output directory |
| `scripts/check_paper_artifacts.py` | Availability check; reports missing inputs without inventing results |
| `tests/` | Core regression tests and clearly marked test fixtures |

Older experiment helpers needed by the retained tests remain in `scripts/`. Their historical default paths are not the current paper reproduction interface. In particular, `build_share_episodes.py` creates test/development simulations, not observed Human branches.

## Supply observed branches

Use the existing importer to bind each branch outcome to its supplied execution artifact and ledger entry:

```bash
python scripts/import_share_real_paired_episodes.py --input /path/to/observed_episodes.jsonl --artifact-ledger /path/to/paired_artifact_ledger.jsonl --artifact-root /path/to/branch_artifacts --out datasets/paper/primary/episodes.jsonl --report datasets/paper/primary/import_report.json
```

Repeat for development, mechanism, and retrospective-budget data. The importer verifies internal consistency of supplied files; source authenticity and expert participation must be documented separately. The branch schema's `synthetic` flag concerns generated authority outcomes, while the underlying workflow can be synthetic as specified by the paper.

## Evaluate a supplied frozen SHARE ensemble

Fill `configs/paper_artifacts.template.json` using actual artifact paths, SHA-256 hashes, training-data bindings, and validation-only calibration. Save it as `artifacts/paper/artifacts.json`.

```bash
python scripts/check_paper_artifacts.py
python scripts/evaluate_share_paper.py --episodes datasets/paper/primary/episodes.jsonl --import-report datasets/paper/primary/import_report.json --artifacts artifacts/paper/artifacts.json --out results/paper/primary --device cpu
```

The evaluator preserves input order, uses the paper's common evaluation reward, averages five raw Q vectors, applies the supplied Human penalty and training-only support rule, and enforces `floor(B*N)` interventions. It writes metrics, per-workflow records, the decision ledger, and input hashes. Replays use a new `--out` directory each time. Missing inputs fail explicitly.

The paper reports Repair.SR 0.932, HIR 0.200, UAR approximately 0.069, and Stop 0.044 on the primary holdout. These are reference values from the manuscript. This cleaned checkout does not ship a new observed-data run establishing those values.

## Publication

Before publishing, complete the checklist, add the agreed software license and data permissions, and verify the clean environment. Raw participant records, credentials, logs, and local backups must remain outside the public release.
