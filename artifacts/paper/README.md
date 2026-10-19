# Updated frozen artifacts

Provide the encoder, five SHARE checkpoints, development import report, and a validation-only ensemble-calibration record. Copy and complete `configs/paper_artifacts.template.json` as `artifacts.json`. Artifact paths in that manifest are relative to this directory unless absolute.

The calibration record must include `partition: "validation"`, `selection_data_sha256` bound to the development input, `human_calibration_penalty`, `validation_episode_ids_sha256`, and `support_episode_ids_sha256`. Both ID digests use `share.paper.identifier_digest` on the corresponding partition's episode IDs. These values must come from the supplied run.

Historical simulation-trained checkpoints are not part of this release. Updated checkpoints and comparator/ablation artifacts remain pending.
