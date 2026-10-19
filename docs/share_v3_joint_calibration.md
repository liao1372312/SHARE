# SHARE-v3 risk-directed joint calibration

## Purpose

SHARE-v3 chooses among AI, Human, and Stop under three fitting-only operating
constraints: episode-level Human intervention rate, unsafe autonomous cost per
episode, and episode Stop rate. It adds risk-directed Human reserve and
group-robust calibration to the frozen-replay SHARE-Guard policy. Selection and
locked-holdout records are not inputs to calibration.

## Inputs and outputs

Inputs are a frozen learned critic, fitting episodes with paired action outcomes,
AI support counts, three operating budgets, a non-negative penalty grid size, a
maximum risk-reserve rate, a reserve-rate grid size, and deterministic seeds.
The risk model uses the same fitted feature encoder as the critic. Its positive
label is an unsafe AI result or an unsuccessful terminal AI result.

The output is a deterministic policy plus its selected AI penalty, Human
penalty, risk-reserve rate and threshold, fold-wise worst metrics, and a flag
showing whether every fitting fold met every budget.

## Algorithm

1. Fit a class-balanced logistic risk model on fitting states and paired AI
   outcomes. Compute a maximum risk score per fitting episode.
2. Batch-cache the critic's three action values for all fitting states.
3. Build AI and Human penalty candidates from quantiles of every pairwise
   non-negative action-value gap. Build evenly spaced reserve-rate candidates
   from zero to the configured cap.
4. Assign whole workflow groups deterministically to calibration folds.
5. For every reserve rate and penalty pair, replay the fitting episodes. AI is
   removed when its support count is below the declared minimum; a state above
   the fitted risk threshold chooses Human when Human is feasible; otherwise the
   penalized critic chooses among feasible actions with the fixed tie order
   Stop, Human, AI.
6. Retain candidates only if every group fold meets all three budgets. Among
   feasible candidates, maximize normalized return minus the declared Stop
   penalty. If none is feasible, choose the smallest maximum fold violation and
   mark the calibration infeasible.
7. Freeze all learned parameters and evaluate once on the selection partition.
   A final claim requires a separately locked, previously unseen holdout.

## Complexity and convergence

For `N` fitting decision states, `E` episodes, `F` folds, `R` reserve rates,
and `P_A P_H` penalty pairs, critic caching is `O(N)` batched inference and the
finite calibration search is `O(R P_A P_H F E L)`, where `L` is mean executed
episode length. Memory is `O(N + E)`. The search terminates after its finite
grid and therefore returns the exact optimum of the declared grid objective;
it does not guarantee a continuous-parameter optimum or out-of-sample budget
satisfaction.

## Implementation and reproducibility

The implementation is in `share/support.py`; scalar and vector critics support
CPU or CUDA through `share/dqn.py` and `share/vector_dqn.py`. CUDA is useful when
the device is not compute-saturated, but device choice does not alter the
declared protocol. Seeds control replay, network initialization, risk fitting,
and workflow-fold assignment. Result files record the full protocol signature,
selected calibration values, and data hashes so interrupted runs can only resume
under a compatible configuration.
