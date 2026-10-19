"""Calibrate and gate a source-adapted SHARE ensemble on frozen replay groups.

The historical V1/V2/V3 groups have been reused during model development, so
all results from this script are exploratory. The 232-group diagnostic is run
only when the prewritten V3 gate passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import CHECKPOINTS, SEEDS, sha256_file
from scripts.run_share_component_e2_232 import choose_threshold, state_values
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNEnsemblePolicy, DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RewardConfig


DATA = ROOT / "datasets/share_source_adaptation_20260930"
OUT = ROOT / "results/share_source_adaptation_20260930"
OLD_ENCODER = ROOT / "results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json"
NEW_ENCODER = DATA / "feature_encoder.json"
VALIDATION = DATA / "validation.jsonl"
GATE = ROOT / "datasets/share_v3_final_holdout_20260827/episodes.jsonl"
DIAGNOSTIC = ROOT / "datasets/share_component_confirmation_232_20260930/episodes.jsonl"
EXPECTED = {
    VALIDATION: "9498f326340fa6b526d20ef09111942f2a2b2f214f89bd61b3c9ccaab0d7eb5d",
    GATE: "104a0671ad895327f27dd261b9cb6133f8a1e381917f6db02bb7b267d4c4cbee",
    DIAGNOSTIC: "52e528dc385368af0329a0d9711c43fba0e9ee4827862341dd4a19f301631989",
}
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")


def new_checkpoint(seed: int) -> Path:
    return OUT / "checkpoints" / f"source_adapted_seed_{seed}.pt"


def checked_inputs() -> dict:
    protocol = json.loads((OUT / "protocol.json").read_text())
    if protocol["protocol"] != "share-public-api-source-adaptation-v1":
        raise SystemExit("unexpected protocol")
    manifest = json.loads((DATA / "manifest.json").read_text())
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise SystemExit(f"changed dataset: {path}")
    if sha256_file(NEW_ENCODER) != manifest["encoder_sha256"]:
        raise SystemExit("changed source encoder")
    checkpoint_hashes = {}
    for seed in SEEDS:
        audit_path = OUT / "training_audit" / f"seed_{seed}.json"
        audit = json.loads(audit_path.read_text())
        checkpoint = new_checkpoint(seed)
        if audit["gradient_steps"] != 30_000 or sha256_file(checkpoint) != audit["checkpoint_sha256"]:
            raise SystemExit(f"invalid new training audit for seed {seed}")
        checkpoint_hashes[f"old_{seed}"] = sha256_file(CHECKPOINTS[seed])
        checkpoint_hashes[f"new_{seed}"] = audit["checkpoint_sha256"]
    return checkpoint_hashes


def ensembles() -> dict[str, DQNEnsemblePolicy]:
    training_reward = RewardConfig(rollback_penalty=0.60)
    output = {}
    for name, encoder_path, checkpoints in (
        ("old", OLD_ENCODER, CHECKPOINTS),
        ("source_adapted", NEW_ENCODER, {seed: new_checkpoint(seed) for seed in SEEDS}),
    ):
        encoder = FeatureEncoder.load(encoder_path)
        members = []
        for seed in SEEDS:
            model = DQNPolicy.load(str(checkpoints[seed]), encoder, training_reward, require_reward_config=True)
            if model.config.training_epochs != 30 or model.config.gamma != 0.95 or not model.config.paired_replay:
                raise SystemExit(f"incompatible SHARE model: {name}/{seed}")
            model.human_calibration_penalty = 0.0
            model.ai_calibration_penalty = 0.0
            members.append(model)
        policy = DQNEnsemblePolicy(members, encoder, members[0].config, training_reward)
        policy.name = name
        output[name] = policy
    return output


def calibrate() -> None:
    hashes = checked_inputs()
    path = OUT / "calibration.json"
    if path.exists():
        raise SystemExit("refusing to overwrite calibration")
    validation = load_episodes(VALIDATION)
    if len(validation) != 400 or len({ep.group_id for ep in validation}) != 400:
        raise SystemExit("invalid validation groups")
    states = [step.state for ep in validation for step in ep.steps]
    calibrations = {}
    for name, policy in ensembles().items():
        raw = policy._raw_values_batch(states)
        threshold = choose_threshold(validation, state_values(validation, raw), "neural")
        if not 0.18 <= threshold["validation_hir"] <= 0.22:
            raise SystemExit(f"{name} validation HIR out of band")
        calibrations[name] = threshold
    result = {
        "protocol": "share-public-api-source-adaptation-v1",
        "selection_data": "historical V2 public-API validation groups",
        "selection_metric": "episode HIR only",
        "checkpoint_sha256": hashes,
        "old_encoder_sha256": sha256_file(OLD_ENCODER),
        "new_encoder_sha256": sha256_file(NEW_ENCODER),
        "validation_sha256": EXPECTED[VALIDATION],
        "calibrations": calibrations,
        "gate_or_diagnostic_outcomes_read": False,
    }
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"calibration": str(path.relative_to(ROOT)), "calibrations": calibrations}, indent=2))


def load_calibrated() -> tuple[dict[str, DQNEnsemblePolicy], dict]:
    hashes = checked_inputs()
    path = OUT / "calibration.json"
    calibration = json.loads(path.read_text())
    if hashes != calibration["checkpoint_sha256"]:
        raise SystemExit("checkpoint changed since calibration")
    if sha256_file(OLD_ENCODER) != calibration["old_encoder_sha256"] or sha256_file(NEW_ENCODER) != calibration["new_encoder_sha256"]:
        raise SystemExit("encoder changed since calibration")
    policies = ensembles()
    for name, policy in policies.items():
        policy.human_calibration_penalty = calibration["calibrations"][name]["threshold"]
    return policies, calibration


def evaluate_set(path: Path, cap: int) -> dict:
    policies, calibration = load_calibrated()
    episodes = load_episodes(path)
    if len(episodes) != (400 if path == GATE else 232) or len({ep.group_id for ep in episodes}) != len(episodes):
        raise SystemExit("unexpected evaluation group count")
    results = {}
    records = {}
    for name, policy in policies.items():
        wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
        raw = evaluate_policy(wrapper, episodes, RewardConfig(), cap / len(episodes), None)
        if wrapper.human_interventions > cap:
            raise RuntimeError("Human cap violated")
        result = {metric: raw[metric] for metric in METRICS}
        result.update({
            "episodes": len(episodes), "human_count": wrapper.human_interventions,
            "success_count": sum(bool(row["success"]) for row in raw["records"]),
            "stop_count": sum(int(row["actions"][-1]) == int(AuthorityAction.STOP) for row in raw["records"]),
            "decision_states": raw["decision_states"],
        })
        results[name] = result
        records[name] = raw["records"]
    return {
        "protocol": "share-public-api-source-adaptation-v1",
        "evidence_class": "exploratory deterministic replay; historical groups reused in development",
        "dataset_sha256": EXPECTED[path], "calibration_sha256": sha256_file(OUT / "calibration.json"),
        "human_cap": cap, "metrics": results, "records": records,
    }


def gate() -> None:
    path = OUT / "development_gate.json"
    if path.exists():
        raise SystemExit("refusing to overwrite gate")
    result = evaluate_set(GATE, 80)
    old, new = result["metrics"]["old"], result["metrics"]["source_adapted"]
    noninferior = (new["success_count"] >= old["success_count"] and
                   new["uar"] <= old["uar"] + 1e-12 and
                   new["stop_count"] <= old["stop_count"])
    improvement = (new["success_count"] > old["success_count"] or
                   new["uar"] < old["uar"] - 1e-12 or
                   new["stop_count"] < old["stop_count"])
    result["gate_passed"] = bool(noninferior and improvement)
    result["gate_rule"] = "new SR >= old; UAR <= old; Stop <= old; one strict gain"
    del result["records"]
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


def diagnostic() -> None:
    gate_path = OUT / "development_gate.json"
    gate_result = json.loads(gate_path.read_text())
    if not gate_result["gate_passed"]:
        raise SystemExit("prewritten V3 gate failed; 232 diagnostic prohibited")
    path = OUT / "diagnostic_232.json"
    records_path = OUT / "diagnostic_232_records.jsonl"
    if path.exists() or records_path.exists():
        raise SystemExit("refusing to overwrite 232 diagnostic")
    result = evaluate_set(DIAGNOSTIC, 46)
    records = result.pop("records")
    records_path.write_text("".join(json.dumps({"model": name, **row}) + "\n" for name in records for row in records[name]))
    result["records_sha256"] = sha256_file(records_path)
    result["conditional_on_gate_pass"] = True
    result["independent_confirmation"] = False
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "gate", "diagnostic"))
    args = parser.parse_args()
    {"calibrate": calibrate, "gate": gate, "diagnostic": diagnostic}[args.phase]()
