# SHARE Observed Deployment Event Contract

`run_share_longitudinal.py` does not accept `RepairEpisode` records. Paired
`ai`/`human`/`stop` outcomes are counterfactual evaluation data and cannot be
presented as evidence of what a deployed controller actually did.

The input `observed_events.jsonl` must contain exactly one action and one
observed outcome per decision. Every row uses schema version 2 and must include:

- `event_id`, `episode_id`, `group_id`, and a `test` split with no group crossing
  another split;
- `decision_at`, `executed_at`, and `outcome_observed_at` as timezone-aware
  timestamps in that order;
- an immutable `execution_id`, `execution_artifact_sha256`, selected action,
  observed outcome, `outcome_sha256`, and `event_digest`;
- a policy arm (`frozen` or `adaptive`), its `policy_state_sha256`, and the
  evidence-ledger snapshot hash used at decision time. Both hashes must remain
  unchanged for every decision in one episode;
- a next state only for a nonterminal AI action. Human and Stop actions must be
  terminal. Each multi-step episode must be contiguous and may not decide its
  next action before the previous outcome is observed.

`validated_corrections.jsonl` can be empty during the initial deployment window.
Each nonempty correction must have a canonical `correction_digest`, match the
selected action, `event_id`, and `observed_outcome_sha256` in an observed event,
carry a `validation_artifact_sha256`, and be timestamped strictly after the
entire referenced episode terminates. It therefore cannot feed back into the
episode that generated it. The runner only admits verified corrections from the
Adaptive arm to its local replay audit; it never replays a policy to synthesize a
historical action or outcome.

Before running the audit, create `manifest.json` with
`create_share_longitudinal_manifest.py`. The manifest binds the event-file hash,
identifiers, policy-state allowlists, frozen time, budgets, reward configuration,
feature-encoder hash, code revision, and external references for provenance,
prior-unseen status, and the arm assignment protocol. The code verifies internal hashes but cannot itself
verify an external reference, signature authority, reality of an execution, or a
causal Frozen-versus-Adaptive effect. Longitudinal output is consequently marked
as observational and protocol-only until the study design and attestations are
independently reviewed.

`validated_outcome` is a complete canonical outcome record, whose
`validated_outcome_sha256` and duplicated `validated_success`/`validated_unsafe`
must agree. It may revise the preliminary outcome stored in the deployment
event, but it cannot change terminal status or `next_step`. The original
observed-event summaries are never rewritten. The corrected values are used only
by the later local Adaptive replay audit and remain bound to the original
event/outcome digest plus an external validation-artifact digest.
Deployment windows must be non-overlapping: all outcomes in one window must be
observed before any decision in the next window.

`share_observed_deployment_event_template.jsonl` and
`share_validated_corrections_template.jsonl` are internally digest-consistent
examples only. After replacing any field, regenerate `outcome_sha256`,
`event_digest`, or `correction_digest` with the canonical JSON functions in
`share.deployment` and `share.evidence_store`; do not hand-edit a digest.
