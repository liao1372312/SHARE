from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from share.data import label_human_benefit, load_episodes, write_episodes
from share.types import RewardConfig


HUMAN_BENEFIT_LABEL_RULE = "share_label_human_benefit_v1"
HUMAN_BENEFIT_MARGIN = 0.1
HUMAN_BENEFIT_GAMMA = 0.95


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def outcome_sha256(outcome: object) -> str:
    return sha256_text(canonical_json(asdict(outcome)))


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def artifact_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifact ledger has an invalid execution_artifact_file")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("artifact ledger path escapes --artifact-root") from error
    if not candidate.is_file():
        raise ValueError("artifact ledger references a missing artifact")
    return candidate


def load_artifact_ledger(path: Path, root: Path) -> dict[tuple[str, str, int, str], Mapping]:
    if not path.is_file():
        raise ValueError("--artifact-ledger must be one existing JSONL file")
    if not root.is_dir():
        raise ValueError("--artifact-root must be one existing directory")
    entries: dict[tuple[str, str, int, str], Mapping] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"artifact ledger line {line_number} is not JSON") from error
            if not isinstance(entry, Mapping):
                raise ValueError(f"artifact ledger line {line_number} must be a JSON object")
            if entry.get("schema_version") != 1 or entry.get("kind") != "share_paired_outcome_artifact_ledger_entry":
                raise ValueError(f"artifact ledger line {line_number} has an unsupported schema")
            collection_id = entry.get("collection_id")
            episode_id = entry.get("episode_id")
            step = entry.get("step")
            branch = entry.get("branch")
            if not all(isinstance(value, str) and value.strip() for value in (collection_id, episode_id)) or not isinstance(step, int) or step < 0 or branch not in {"ai", "human", "stop"}:
                raise ValueError(f"artifact ledger line {line_number} has invalid collection, episode, step, or branch fields")
            digest = entry.get("execution_artifact_sha256")
            if not is_sha256(digest):
                raise ValueError(f"artifact ledger line {line_number} has an invalid execution artifact digest")
            recorded_outcome_sha256 = entry.get("outcome_sha256")
            if not is_sha256(recorded_outcome_sha256):
                raise ValueError(f"artifact ledger line {line_number} has an invalid outcome digest")
            if branch == "ai" and (not isinstance(entry.get("model_id"), str) or not entry["model_id"].strip()):
                raise ValueError(f"artifact ledger line {line_number} lacks AI model_id")
            file_path = artifact_path(root, entry.get("execution_artifact_file"))
            if sha256_file(file_path) != digest:
                raise ValueError(f"artifact ledger line {line_number} digest does not match its artifact")
            try:
                artifact = json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"artifact ledger line {line_number} artifact is not JSON") from error
            if not isinstance(artifact, Mapping) or artifact.get("kind") != "share_paired_outcome_branch_artifact":
                raise ValueError(f"artifact ledger line {line_number} artifact has an unsupported schema")
            if (
                artifact.get("collection_id") != collection_id
                or artifact.get("episode_id") != episode_id
                or artifact.get("step") != step
                or artifact.get("branch") != branch
                or (branch == "ai" and artifact.get("model_id") != entry.get("model_id"))
            ):
                raise ValueError(f"artifact ledger line {line_number} artifact metadata does not match its ledger entry")
            artifact_outcome = artifact.get("outcome")
            if not isinstance(artifact_outcome, Mapping) or artifact.get("outcome_sha256") != recorded_outcome_sha256 or sha256_text(canonical_json(artifact_outcome)) != recorded_outcome_sha256:
                raise ValueError(f"artifact ledger line {line_number} artifact outcome does not match its outcome digest")
            key = (collection_id, episode_id, step, branch)
            if key in entries:
                raise ValueError(f"artifact ledger has a duplicate entry for {episode_id}/{step}/{branch}")
            entries[key] = entry
    if not entries:
        raise ValueError("--artifact-ledger contains no entries")
    return entries


def validate_real_paired_episodes(episodes: Sequence, artifact_ledger: Mapping[tuple[str, str, int, str], Mapping]) -> Counter:
    """Reject data that cannot support an auditable real paired-outcome claim."""
    models: Counter = Counter()
    for episode in episodes:
        if episode.synthetic:
            raise ValueError(f"real paired import rejects synthetic episode: {episode.episode_id}")
        provenance = episode.metadata.get("paired_outcome_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"episode {episode.episode_id} lacks paired_outcome_provenance")
        if not isinstance(provenance.get("collection_id"), str) or not provenance["collection_id"].strip():
            raise ValueError(f"episode {episode.episode_id} lacks provenance collection_id")
        if provenance.get("human_benefit_label_rule") != HUMAN_BENEFIT_LABEL_RULE:
            raise ValueError(f"episode {episode.episode_id} lacks the required human-benefit label rule")
        for branch in ("ai", "human", "stop"):
            item = provenance.get(branch)
            step_digests = item.get("execution_artifact_sha256_by_step") if isinstance(item, Mapping) else None
            if not isinstance(step_digests, Mapping):
                raise ValueError(f"episode {episode.episode_id} lacks {branch} execution_artifact_sha256_by_step")
            if branch == "ai":
                model_id = item.get("model_id")
                if not isinstance(model_id, str) or not model_id.strip():
                    raise ValueError(f"episode {episode.episode_id} lacks AI model_id")
                models[model_id.strip()] += 1
            for step_index, step in enumerate(episode.steps):
                step_digest = step_digests.get(str(step_index))
                if not is_sha256(step_digest):
                    raise ValueError(f"episode {episode.episode_id} lacks a valid {branch} artifact digest for step {step_index}")
                ledger_entry = artifact_ledger.get((provenance["collection_id"].strip(), episode.episode_id, step_index, branch))
                if not isinstance(ledger_entry, Mapping):
                    raise ValueError(f"episode {episode.episode_id} lacks a verified {branch} artifact ledger entry for step {step_index}")
                if ledger_entry.get("execution_artifact_sha256") != step_digest:
                    raise ValueError(f"episode {episode.episode_id} {branch} artifact digest does not match the ledger at step {step_index}")
                if ledger_entry.get("outcome_sha256") != outcome_sha256(step.outcomes[branch]):
                    raise ValueError(f"episode {episode.episode_id} {branch} outcome does not match its artifact at step {step_index}")
                if branch == "ai" and ledger_entry.get("model_id") != model_id.strip():
                    raise ValueError(f"episode {episode.episode_id} AI model_id does not match the ledger")
        recomputed_steps = label_human_benefit(
            episode.steps,
            RewardConfig(),
            margin=HUMAN_BENEFIT_MARGIN,
            gamma=HUMAN_BENEFIT_GAMMA,
        )
        for step_index, (observed, expected) in enumerate(zip(episode.steps, recomputed_steps)):
            if observed.human_beneficial != expected.human_beneficial:
                raise ValueError(f"episode {episode.episode_id} human_beneficial does not match the bound outcomes at step {step_index}")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and import externally collected real paired SHARE outcomes.")
    parser.add_argument("--input", type=Path, required=True, help="One JSONL file in the RepairEpisode schema with synthetic=false.")
    parser.add_argument("--out", type=Path, required=True, help="Validated canonical RepairEpisode JSONL.")
    parser.add_argument("--report", type=Path, required=True, help="JSON import report with source binding and model counts.")
    parser.add_argument("--artifact-ledger", type=Path, required=True, help="Read-only JSONL ledger binding every paired branch to a local artifact file.")
    parser.add_argument("--artifact-root", type=Path, required=True, help="Directory containing the immutable paired branch artifact files.")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit("--input must be one existing JSONL file")
    if args.input.resolve() == args.out.resolve():
        raise SystemExit("--input and --out must differ so the source digest remains auditable")
    protected_paths = (args.input.resolve(), args.artifact_ledger.resolve())
    write_paths = (args.out.resolve(), args.report.resolve())
    if args.out.resolve() == args.report.resolve() or any(write_path == protected for write_path in write_paths for protected in protected_paths):
        raise SystemExit("--out and --report must not overwrite the input or artifact ledger")
    artifact_root = args.artifact_root.resolve()
    try:
        if any(write_path.is_relative_to(artifact_root) for write_path in write_paths):
            raise SystemExit("--out and --report must not be written inside --artifact-root")
    except AttributeError:
        if any(str(write_path).startswith(str(artifact_root) + "/") for write_path in write_paths):
            raise SystemExit("--out and --report must not be written inside --artifact-root")
    episodes = load_episodes(args.input)
    try:
        artifact_ledger = load_artifact_ledger(args.artifact_ledger, artifact_root)
        models = validate_real_paired_episodes(episodes, artifact_ledger)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    write_episodes(args.out, episodes)
    report = {
        "schema_version": 1,
        "evidence_kind": "externally_collected_real_paired_outcomes",
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "output_path": str(args.out.resolve()),
        "output_sha256": sha256_file(args.out),
        "artifact_ledger_path": str(args.artifact_ledger.resolve()),
        "artifact_ledger_sha256": sha256_file(args.artifact_ledger),
        "artifact_root": str(args.artifact_root.resolve()),
        "episodes": len(episodes),
        "workflow_groups": len({episode.group_id for episode in episodes}),
        "splits": dict(sorted(Counter(episode.split for episode in episodes).items())),
        "ai_model_episode_counts": dict(sorted(models.items())),
        "human_benefit_label_rule": HUMAN_BENEFIT_LABEL_RULE,
        "human_benefit_margin": HUMAN_BENEFIT_MARGIN,
        "human_benefit_gamma": HUMAN_BENEFIT_GAMMA,
        "evidence_warning": "Artifact bytes and metadata are internally bound to the supplied ledger. This report cannot independently authenticate the external ledger authority, execution reality, or provenance claims.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "report": str(args.report), "episodes": len(episodes)}, indent=2))


if __name__ == "__main__":
    main()
