# SHARE Paired Artifact Ledger Contract

The real paired-outcome importer requires a read-only JSONL ledger and a local
artifact root. It recomputes every listed artifact's SHA-256 before accepting a
`RepairEpisode` as real paired evidence.

Each ledger row has schema version 1 and these fields:

- `kind`: `share_paired_outcome_artifact_ledger_entry`.
- `collection_id`, `episode_id`, `step`, and `branch`, where `branch` is `ai`,
  `human`, or `stop`.
- `execution_artifact_file`: a relative path under `--artifact-root`.
- `execution_artifact_sha256`: SHA-256 of that file.
- `outcome_sha256`: SHA-256 of the canonical outcome object at this exact step.
- `model_id`: required only for the `ai` branch.

Each referenced artifact must be a JSON object with
`kind=share_paired_outcome_branch_artifact` and matching `collection_id`,
`episode_id`, `step`, and `branch`. It must embed the full canonical `outcome`
object and its matching `outcome_sha256`; AI artifacts must also contain the same
`model_id`. The importer compares every outcome field in the `RepairEpisode`
against the ledger entry at the same step and branch.

The episode's `metadata.paired_outcome_provenance` repeats the `collection_id`.
For each of `ai`, `human`, and `stop`, it must contain an
`execution_artifact_sha256_by_step` object whose zero-based string keys cover
every episode step and match the corresponding ledger digest. The `ai` object
also contains `model_id`.

`human_beneficial` is not accepted as a free annotation. Every episode declares
`human_benefit_label_rule=share_label_human_benefit_v1`; the importer recomputes
the label from the bound outcomes using the fixed default `RewardConfig`, margin
`0.1`, and discount `0.95`, then rejects any mismatch.
The artifact payload should preserve the externally maintained execution record
or ledger reference. The importer verifies bytes and internal bindings only; it
does not establish that an external attestation, tool execution, or human action
is genuine.
