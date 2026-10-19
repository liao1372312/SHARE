"""Evaluate a training-derived repair root-type schema alignment.

The mapping is frozen from the original training split alone. Public-API
holdouts have already been inspected, so every policy comparison here is a
retrospective diagnostic rather than new independent confirmation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_share_v2_validation_hard_cap import sha256_file
from scripts.run_share_component_e2_232 import DEVELOPMENT, ENCODER, EXPECTED_DEV, EXPECTED_HOLDOUT, HOLDOUT, OUT, load_policies
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.policies import Policy
from share.types import RepairState, RewardConfig


MAPPING_PATH = OUT / "root_schema_alignment_training_only.json"
RESULT_PATH = OUT / "exploratory_root_schema_alignment.json"
DATASETS = {
    "component_232": HOLDOUT,
    "prior_v4_250": ROOT / "datasets/share_v4_ensemble_holdout_20260829/episodes.jsonl",
    "prior_v5_150": ROOT / "datasets/share_v5_expected_voh_holdout_20260829/episodes.jsonl",
}


class RootSchemaAlignedPolicy(Policy):
    name = "full_share_root_schema_aligned"

    def __init__(self, base: Policy, mapping: dict[str, str], known: set[str]):
        self.base = base
        self.mapping = mapping
        self.known = known

    def _state(self, state: RepairState) -> RepairState:
        if state.root_type in self.known:
            return state
        mapped = self.mapping.get(state.gap_type)
        return replace(state, root_type=mapped) if mapped in self.known else state

    def select(self, state: RepairState, feasible):
        return self.base.select(self._state(state), feasible)

    def escalation_score(self, state: RepairState) -> float:
        return self.base.escalation_score(self._state(state))


def freeze() -> None:
    if MAPPING_PATH.exists():
        raise SystemExit("refusing to overwrite training-only schema mapping")
    if sha256_file(DEVELOPMENT) != EXPECTED_DEV or sha256_file(HOLDOUT) != EXPECTED_HOLDOUT:
        raise SystemExit("dataset hash changed")
    train = [ep for ep in load_episodes(DEVELOPMENT) if ep.split == "train"]
    encoder = FeatureEncoder.load(ENCODER)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for ep in train:
        for step in ep.steps:
            state = step.state
            counts[state.gap_type][state.root_type] += 1
    mapping = {gap: count.most_common(1)[0][0] for gap, count in sorted(counts.items())}
    artifact = {
        "status": "training_only_mapping_frozen_before_this_evaluation",
        "interpretation": "subsequent tests on previously inspected public-API holdouts are exploratory",
        "development_sha256": EXPECTED_DEV,
        "encoder_sha256": sha256_file(ENCODER),
        "holdout_sha256_for_identity_only": EXPECTED_HOLDOUT,
        "train_episodes": len(train),
        "train_states": sum(len(ep.steps) for ep in train),
        "known_encoder_root_types": encoder.root_types,
        "counts": {gap: dict(count) for gap, count in sorted(counts.items())},
        "mapping": mapping,
        "rule": "if deployment root_type is unseen in the training-fitted encoder, replace it with the training-split modal root_type for its gap_type",
        "test_outcomes_read": False,
    }
    MAPPING_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


def run_policy(policy: Policy, episodes) -> tuple[dict, list[dict]]:
    cap = math.floor(0.2 * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, False)
    result = evaluate_policy(wrapper, episodes, RewardConfig(), cap / len(episodes), None)
    values = {key: result[key] for key in ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")}
    values.update({"success_count": sum(bool(row["success"]) for row in result["records"]), "human_count": wrapper.human_interventions, "stop_count": sum(row["actions"][-1] == 2 for row in result["records"])})
    return values, result["records"]


def paired_intervals(before: list[dict], after: list[dict]) -> dict:
    if [r["episode_id"] for r in before] != [r["episode_id"] for r in after]:
        raise ValueError("unpaired workflow records")
    rng = np.random.default_rng(2026100201)
    draw = rng.integers(0, len(before), size=(20_000, len(before)))
    def arrays(rows):
        success = np.asarray([int(r["success"]) for r in rows]); human = np.asarray([int(r["human_used"]) for r in rows]); stop = np.asarray([int(r["actions"][-1] == 2) for r in rows]); autonomous = 1 - human - stop; adverse = np.asarray([int(r["unsafe"] or not r["success"]) for r in rows]) * autonomous
        return success, stop, autonomous, adverse
    b, a = arrays(before), arrays(after)
    result = {}
    for name, ix in (("repair_sr", 0), ("stopped_rate", 1)):
        difference = a[ix][draw].mean(axis=1) - b[ix][draw].mean(axis=1)
        result[name] = [float(np.percentile(difference, 2.5)), float(np.percentile(difference, 97.5))]
    b_uar = b[3][draw].sum(axis=1) / b[2][draw].sum(axis=1)
    a_uar = a[3][draw].sum(axis=1) / a[2][draw].sum(axis=1)
    result["uar"] = [float(np.percentile(a_uar - b_uar, 2.5)), float(np.percentile(a_uar - b_uar, 97.5))]
    return result


def evaluate() -> None:
    if not MAPPING_PATH.is_file() or RESULT_PATH.exists():
        raise SystemExit("missing frozen mapping or refusing to overwrite result")
    artifact = json.loads(MAPPING_PATH.read_text())
    if sha256_file(DEVELOPMENT) != artifact["development_sha256"] or sha256_file(ENCODER) != artifact["encoder_sha256"]:
        raise SystemExit("training inputs changed")
    encoder = FeatureEncoder.load(ENCODER)
    base = load_policies(encoder, json.loads((OUT / "calibration.json").read_text()))["Full SHARE"][-1]
    aligned = RootSchemaAlignedPolicy(base, artifact["mapping"], set(encoder.root_types))
    results = {}
    for name, path in DATASETS.items():
        episodes = load_episodes(path)
        before, before_records = run_policy(base, episodes)
        after, after_records = run_policy(aligned, episodes)
        unseen = sum(step.state.root_type not in encoder.root_types for ep in episodes for step in ep.steps)
        results[name] = {
            "episodes": len(episodes), "states_with_unseen_root_type": unseen,
            "dataset_sha256": sha256_file(path), "baseline": before, "aligned": after,
            "delta_aligned_minus_baseline": {key: after[key] - before[key] for key in ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")},
            "paired_fixed_trajectory_ci95": paired_intervals(before_records, after_records),
        }
    result = {"status": "retrospective_diagnostic_on_previously_inspected_replay_sets", "mapping_sha256": sha256_file(MAPPING_PATH), "results": results}
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "evaluate"))
    args = parser.parse_args()
    freeze() if args.phase == "freeze" else evaluate()
