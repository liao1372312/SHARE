# SHARE

**Sequential Human–AI Authority Allocation for Safety-Oriented Repair of Agentic Software Workflows**

SHARE is a framework for sequential Human–AI authority allocation during the repair of failed agentic software workflows. It constructs a shared repair context from workflow execution evidence, learns downstream **AI / Human / Stop** action values with paired Double DQN, and enforces a finite human-intervention budget through an online ledger. Human intervention and Stop terminate the modeled repair episode, while AI repair can continue to later repair states.

## Overview

The repository is organized around three components:

1. **Repair-grounded state construction** — execution graphs, requirement-gap detection, failure localization, backward slicing, and bounded repair generation.
2. **Sequential authority learning** — paired authority branches, replay construction, value learning, and Double-DQN-based AI / Human / Stop decisions.
3. **Budget-faithful deployment** — validation-calibrated intervention preferences together with an online ledger that enforces the deployment quota.

The implementation follows the experimental protocol described in the paper and keeps evaluation inputs, model artifacts, and generated outputs separated so that reported results can be audited without silently modifying source data.

## Open Science and Reproducibility

To support transparent peer review, we publicly provide substantial experimental material during the review stage. The current release includes:

- the **experimental procedure and evaluation protocol** used in the paper;
- **most of the releasable experimental data and processed results**;
- selected **implementation code**, configuration files, evaluation scripts, and regression tests;
- the synthetic repair-backbone corpus used by the paper;
- schemas and import/evaluation interfaces for authority-branch experiments; and
- documentation needed to inspect how experimental inputs are bound to execution evidence and evaluation outputs.

Some artifacts are intentionally not included in the review-stage release, including selected observed authority branches, trained artifacts, and materials that may contain identifying, restricted, or non-redistributable information. Therefore, the current repository should be viewed as a **review-stage reproducibility package**, rather than a claim that every experiment involving expert repair can already be reproduced end to end from the public checkout alone.

**Upon acceptance, we will release the complete releasable research package**, including the remaining source code, full releasable experimental datasets, trained checkpoints/artifacts, finalized configuration files, experiment scripts, and additional reproduction documentation. Any participant-related, credential-bearing, proprietary, or otherwise restricted material will be excluded or appropriately anonymized in accordance with applicable data and privacy requirements.

## Installation

Python 3.12 is the baseline environment used by the checked implementation.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Optional statistical analysis and plotting dependencies are listed in:

```bash
requirements-paper.txt
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `flare/` | Execution graphs, gap detection, localization, slicing, and repair operators |
| `share/` | States, rewards, paired replay, policies, Double DQN, and intervention ledger |
| `configs/paper_protocol.json` | Paper-level protocol settings and expected sample counts; specifications rather than measured results |
| `configs/paper_artifacts.template.json` | Template for registering frozen models, hashes, and data bindings |
| `datasets/flare_complexity/` | Synthetic repair-backbone workflows described in the paper |
| `datasets/paper/` | Location for paper evaluation datasets and import reports |
| `artifacts/paper/` | Location for encoders, checkpoints, and validation-calibration artifacts |
| `scripts/evaluate_share_paper.py` | Parameterized evaluation entry point with read-only inputs and separate outputs |
| `scripts/check_paper_artifacts.py` | Artifact-availability and consistency checks |
| `tests/` | Core regression tests and explicitly marked test fixtures |

Older experiment helpers required by retained tests remain under `scripts/`. They are preserved for compatibility but are not the primary paper-reproduction interface. In particular, `build_share_episodes.py` creates test/development simulations and should not be interpreted as generating observed Human branches.

## Preparing Observed Authority Branches

Observed authority outcomes can be imported by binding each branch to its supplied execution artifact and ledger entry:

```bash
python scripts/import_share_real_paired_episodes.py \
  --input /path/to/observed_episodes.jsonl \
  --artifact-ledger /path/to/paired_artifact_ledger.jsonl \
  --artifact-root /path/to/branch_artifacts \
  --out datasets/paper/primary/episodes.jsonl \
  --report datasets/paper/primary/import_report.json
```

The same procedure can be applied to development, mechanism-analysis, and retrospective-budget data. The importer checks internal consistency among supplied files. Provenance, expert participation, and source authenticity should be documented independently of these structural checks.

The branch-level `synthetic` flag refers to whether an authority outcome was generated synthetically. It does not necessarily imply that the underlying workflow itself is synthetic.

## Evaluating a Frozen SHARE Ensemble

Populate `configs/paper_artifacts.template.json` with the actual artifact paths, SHA-256 hashes, training-data bindings, and validation-only calibration information, then save it as:

```text
artifacts/paper/artifacts.json
```

Run:

```bash
python scripts/check_paper_artifacts.py

python scripts/evaluate_share_paper.py \
  --episodes datasets/paper/primary/episodes.jsonl \
  --import-report datasets/paper/primary/import_report.json \
  --artifacts artifacts/paper/artifacts.json \
  --out results/paper/primary \
  --device cpu
```

The evaluator preserves input order, uses the paper's common evaluation reward, averages five raw Q-value vectors, applies the supplied Human-intervention penalty and training-only support rule, and enforces `floor(B*N)` Human interventions. Each run writes metrics, per-workflow records, the decision ledger, and input hashes to a new output directory. Missing required inputs fail explicitly rather than being silently substituted.

## Reference Results

For the primary holdout, the manuscript reports:

| Metric | Value |
| --- | ---: |
| Repair.SR | 0.932 |
| HIR | 0.200 |
| UAR | ~0.069 |
| Stop | 0.044 |

These values are provided as manuscript reference results. Reproducing experiments that depend on currently withheld observed branches or trained artifacts requires the corresponding files described above.

## Release Plan

| Material | Review-stage release | Full release after acceptance |
| --- | :---: | :---: |
| Experimental protocol and procedure | ✓ | ✓ |
| Evaluation scripts and configuration interfaces | ✓ | ✓ |
| Synthetic repair-backbone corpus | ✓ | ✓ |
| Most releasable experimental data / processed results | ✓ | ✓ |
| Selected implementation code | ✓ | ✓ |
| Remaining releasable source code | Partial | ✓ |
| Trained checkpoints and final artifacts | Partial | ✓ |
| Full releasable reproduction package | Partial | ✓ |

The post-acceptance release will prioritize reproducibility while respecting privacy, participant protection, third-party licensing, and institutional data-sharing constraints.

## Responsible Release

Before the final public release, we will verify the clean environment, finalize software/data licenses, validate artifact hashes and data bindings, and remove credentials, private logs, local backups, and non-redistributable records.

For the authoritative experiment specification, refer to the paper and the repository configuration files associated with the corresponding experiment.
