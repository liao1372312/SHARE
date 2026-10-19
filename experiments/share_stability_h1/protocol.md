# H1 Protocol: Robust Checkpoint Selection

Status: exploratory development protocol, locked before running.

## Hypothesis

Selecting checkpoints by a robust multi-fold utility rather than a single
validation aggregate will reduce cross-seed variance in Repair.SR, UAR, and
Stop because the current selector can choose different conservative regimes
from small validation fluctuations.

## Fixed boundaries

- Use development/training data only.
- Do not read any opened confirmation/final outcomes for candidate selection.
- Keep the online 20% hard cap and common evaluator unchanged.
- Use seeds 17, 29, 43, 71, and 101 for comparability.

## Primary development score

Candidates are ranked lexicographically by:

1. all-fold hard-cap feasibility;
2. worst-fold normalized return;
3. lower across-fold Stop SD;
4. lower across-fold UAR SD;
5. higher mean Repair.SR.

## Prediction

Relative to the current V2 development baseline, robust selection will reduce
Repair.SR and Stop SD without materially increasing mean UAR.

