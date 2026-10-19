# R002b verified patch refinement

## Status and purpose

R002b is a separately named exploratory extension of immutable R002. It tests
whether a uniform second-stage audit of an already generated patch improves the
11-task resolution count. It never replaces, deletes, or relabels an R002
fixed-attempt outcome. Because R002 outcomes were inspected while designing this
extension, R002b on the same tasks is development evidence, not a new
confirmatory backbone comparison.

## Input and output specification

Inputs are the 11 frozen R002 task/image/base-commit bindings, the primary
model's frozen first-stage patch for every task (including successful tasks), a
single frozen refiner profile, a turn/token budget, the refiner source and prompt
hashes, and a deterministic decoding configuration. The model receives the
public issue text, repository state, and first-stage patch. It does not receive
R002 evaluation summaries, FAIL_TO_PASS/PASS_TO_PASS labels, gold patches, or
test patches.

For each task, the output is a complete refined diff against the original base
commit, an interaction transcript, model usage, container/image bindings,
source and prompt hashes, and a validation record. A separate standard
SWE-bench execution produces the resolved outcome. All 11 outcomes remain in
the denominator, including generation errors and empty or non-applicable
patches.

## Algorithm

```text
ALGORITHM: Uniform verified second-stage refinement
INPUT: frozen task set T, first-stage patches P, refiner M, turn cap K
OUTPUT: refined patches P', audit artifacts A

1. For every task t in T:
   1.1 Start the bound image offline at the bound base commit.
   1.2 Verify a clean tracked baseline and apply P[t].
   1.3 Give M the public issue and state that the current diff is an untrusted
       first draft requiring inspection and repository-test verification.
   1.4 For at most K turns, execute exactly one validated shell, exact replace,
       or final action and return its bounded output to M.
   1.5 After each mutation, invalidate earlier verification evidence.
   1.6 Accept final only after a successful test-like command and git-diff
       inspection occurred after the latest mutation; otherwise ask M to
       continue while turns remain.
   1.7 Validate git diff and Python syntax, emit the complete base-relative
       patch and immutable audit artifact, and remove the container.
2. Evaluate every emitted patch once with the standard frozen SWE-bench runner.
3. Count every task exactly once and report resolution, failures, usage, and
   Wilson interval. Keep the original R002 table unchanged.
```

## Complexity and termination

For `n=11` tasks, at most `K` model turns, and bounded command time `C`, the
orchestrator performs `O(nK)` model/tool interactions and has worst-case wall
time `O(nK(C + L))`, where `L` is one model-call latency. Per-task working space
is the repository image plus `O(KB)` transcript bytes for bounded observation
size `B`; tasks are processed sequentially. The algorithm terminates finitely
after `K` turns or the first admissible final action. There is no guarantee that
refinement improves any patch or exceeds 7/11.

## Interpretation boundary

If R002b exceeds 7/11, it supports an exploratory claim about this two-stage
system on these 11 tasks. It does not show that the primary backbone alone is
better than `gpt-4.1-mini`, because it uses an additional model-call/test budget.
A confirmatory system comparison requires a newly frozen task set and
compute-matched baselines.
