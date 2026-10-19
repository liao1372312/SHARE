"""Regression fixtures below are not participant or experimental records."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from scripts.evaluate_share_paper import evaluate
from scripts.check_paper_artifacts import check
from share.data import write_episodes
from share.dqn import DQNConfig, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.paper import identifier_digest, paper_metrics, require_disjoint, sha256_file
from share.policies import AutonomousPolicy
from share.types import BranchOutcome, RewardConfig
from test_share import bind_paired_artifacts, sample_episode


def test_loading_cuda_checkpoint_on_cpu_preserves_parameters_and_source(tmp_path, monkeypatch):
    episode = sample_episode()
    encoder = FeatureEncoder.fit(s.state for s in episode.steps)
    policy = DQNPolicy(encoder, DQNConfig(seed=17))
    path = tmp_path / "checkpoint.pt"
    policy.save(str(path))
    payload = torch.load(path, weights_only=True)
    payload["config"]["device"] = "cuda"
    torch.save(payload, path)
    original = path.read_bytes()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loaded = DQNPolicy.load(str(path), encoder)
    assert str(loaded.device) == "cpu"
    assert all(torch.equal(a, b) for a, b in zip(policy.online.parameters(), loaded.online.parameters()))
    assert path.read_bytes() == original


def test_original_unsafe_event_survives_success_and_ordinary_failure_is_separate():
    episode = sample_episode()
    flagged = replace(episode, metadata={"initial_unsafe": True})
    result = paper_metrics(evaluate_policy(AutonomousPolicy(), [flagged]))
    assert result["repair_sr"] == 1.0
    assert result["uar"] == 1.0
    assert result["flagged_episode_rate"] == 1.0
    failed_step = replace(episode.steps[0], outcomes={**episode.steps[0].outcomes, "ai": BranchOutcome(False, False, 0, 0, 0)})
    failed = replace(episode, steps=[failed_step])
    result = paper_metrics(evaluate_policy(AutonomousPolicy(), [failed]))
    assert result["uar"] == 1.0
    assert result["flagged_episode_rate"] == 0.0
    assert result["autonomous_terminal_failure_rate"] == 1.0


def test_source_overlap_and_missing_inputs_are_reported(tmp_path):
    episode = sample_episode()
    with pytest.raises(ValueError, match="overlap"):
        require_disjoint([episode], [episode])
    report = check(tmp_path)
    assert report["status"] == "pending_inputs"
    assert not any(row["present"] for row in report["files"])


def write_observed_fixture(root, episodes):
    root.mkdir()
    episodes = [replace(e, synthetic=False, source="unit-test-fixture", metadata={
        "initial_unsafe": False,
        "source_workflow_id": "workflow-" + e.episode_id,
        "source_document_sha256": hashlib.sha256(("document-" + e.episode_id).encode()).hexdigest(),
        "normalized_source_fingerprint": hashlib.sha256(("fingerprint-" + e.episode_id).encode()).hexdigest(),
    }) for e in episodes]
    originals = episodes
    episodes, ledger, artifacts = bind_paired_artifacts(root, originals)
    episodes = [replace(bound, metadata={**original.metadata, **bound.metadata}) for original, bound in zip(originals, episodes)]
    path = root / "episodes.jsonl"
    write_episodes(path, episodes)
    report = root / "import_report.json"
    report.write_text(json.dumps({
        "evidence_kind": "externally_collected_real_paired_outcomes",
        "output_path": str(path), "output_sha256": sha256_file(path),
        "artifact_ledger_path": str(ledger), "artifact_ledger_sha256": sha256_file(ledger),
        "artifact_root": str(artifacts),
    }), encoding="utf-8")
    return episodes, path, report


def test_paper_replay_binds_inputs_applies_support_and_preserves_outputs(tmp_path):
    development, dev_path, dev_report = write_observed_fixture(tmp_path / "development", [
        sample_episode(split=split, group=split, episode_id=split)
        for split in ("train", "validation", "test")
    ])
    _, holdout_path, holdout_report = write_observed_fixture(tmp_path / "holdout", [
        sample_episode(split="test", group="holdout", episode_id="holdout")
    ])
    encoder = FeatureEncoder.fit(s.state for e in development if e.split == "train" for s in e.steps)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    encoder_path = model_dir / "encoder.json"
    encoder.save(encoder_path)
    def entry(p):
        return {"path": str(p), "sha256": sha256_file(p)}
    seeds = [17, 29, 43, 71, 101]
    checkpoints = []
    for seed in seeds:
        policy = DQNPolicy(encoder, DQNConfig(seed=seed, paired_replay=True, unsafe_constraint=False), RewardConfig(rollback_penalty=.60))
        # Select AI so the training-only support fallback is exercised.
        with torch.no_grad():
            for parameter in policy.online.parameters():
                parameter.zero_()
            policy.online.layers[-1].bias[0] = 10
        checkpoint = model_dir / f"{seed}.pt"
        policy.save(str(checkpoint))
        checkpoints.append({"seed": seed, **entry(checkpoint)})
    calibration = model_dir / "calibration.json"
    calibration.write_text(json.dumps({
        "partition": "validation", "selection_data_sha256": sha256_file(dev_path),
        "validation_episode_ids_sha256": identifier_digest(["validation"]),
        "support_episode_ids_sha256": identifier_digest(["train"]),
        "human_calibration_penalty": 0,
    }), encoding="utf-8")
    artifacts = model_dir / "artifacts.json"
    artifacts.write_text(json.dumps({
        "status": "ready", "protocol_id": "unit-test-only", "encoder": entry(encoder_path),
        "training_import_report": entry(dev_report), "validation_calibration": entry(calibration),
        "human_calibration_penalty": 0, "checkpoints": checkpoints,
    }), encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "protocol_id": "unit-test-only", "seeds": seeds, "state_dimension": encoder.dimension,
        "training": {"training_stop_coefficient": .60}, "human_budget": .2, "minimum_ai_support": 3,
        "datasets": {"primary": {"episodes": 1, "states": 2}, "development": {"splits": {"train": 1, "validation": 1, "test": 1}}},
    }), encoding="utf-8")
    args = Namespace(protocol=protocol, artifacts=artifacts, episodes=holdout_path,
                     import_report=holdout_report, out=tmp_path / "output", device="cpu", budget=None, dataset_role="primary")
    result = evaluate(args)
    assert result["cap_respected"] and result["human_interventions"] == 0
    assert result["stopped_rate"] == 1 and result["uar"] is None
    assert {p.name for p in args.out.iterdir()} == {"metrics.json", "manifest.json", "decision_ledger.json", "records.jsonl"}
    original = (args.out / "metrics.json").read_bytes()
    with pytest.raises(ValueError, match="new --out"):
        evaluate(args)
    assert (args.out / "metrics.json").read_bytes() == original
    # A new output directory is not created if a bound input has been altered.
    args.out = tmp_path / "tampered-output"
    encoder_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate(args)
    assert not args.out.exists()
