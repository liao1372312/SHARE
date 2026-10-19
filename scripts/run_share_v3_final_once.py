"""Seal, then execute, the SHARE-V3 third-holdout evaluation exactly once."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_online_hard_cap import BINARY_POLICIES, OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RepairEpisode, RewardConfig

BUDGETS = (0.10, 0.20, 0.30)
SEEDS = (17, 29, 43, 71, 101)
METHODS = ("share_conf", "share_static", "share_bandit", "share_ddqn", "share_v3")
METRICS = ("repair_sr", "hir", "uar", "stopped_rate", "normalized_return", "collaboration_regret")
DATASET = ROOT / "datasets/share_v3_final_holdout_20260827/episodes.jsonl"
DATASET_MANIFEST = ROOT / "datasets/share_v3_final_holdout_20260827/manifest.json"
PROTOCOL_ROOT = ROOT / "results/share_v3_final_evaluation_protocol_20260827"
RESULT_ROOT = ROOT / "results/share_v3_final_evaluation_20260827"
V3_ROOTS = {
    0.10: ROOT / "results/share_v3_budget_0p1_training_20260827",
    0.20: ROOT / "results/share_v3_budget_0p2_training_20260827",
    0.30: ROOT / "results/share_v3_budget_0p3_training_20260827",
}
COMPARATOR_ROOTS = {
    0.10: ROOT / "results/share_v3_budget_0p1_fair_comparators_20260827",
    0.20: ROOT / "results/share_v3_budget_0p2_fair_comparators_20260827",
    0.30: ROOT / "results/share_v3_budget_0p3_fair_comparators_20260827",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def artifact_paths() -> list[Path]:
    paths = [DATASET, DATASET_MANIFEST, ROOT / "scripts/run_share_online_hard_cap.py", Path(__file__)]
    for budget in BUDGETS:
        paths.extend([V3_ROOTS[budget] / "manifest.json", V3_ROOTS[budget] / "feature_encoder.json", COMPARATOR_ROOTS[budget] / "manifest.json"])
        for seed in SEEDS:
            paths.extend([
                V3_ROOTS[budget] / f"share_v3_budget_{budget:.1f}_seed_{seed}.pt",
                COMPARATOR_ROOTS[budget] / f"comparators_seed_{seed}.joblib",
                COMPARATOR_ROOTS[budget] / f"share_ddqn_seed_{seed}.pt",
            ])
    return paths


def seal() -> None:
    if PROTOCOL_ROOT.exists() and any(PROTOCOL_ROOT.iterdir()):
        raise SystemExit("refusing to overwrite V3 final evaluation protocol")
    dataset_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if dataset_manifest["audit_result"] != "PASS" or dataset_manifest["evaluator_bound"]:
        raise SystemExit("third holdout is not audited and unbound")
    files = []
    for path in artifact_paths():
        if not path.is_file():
            raise SystemExit(f"missing frozen artifact: {path}")
        files.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    protocol = {
        "protocol": "share-v3-third-holdout-once-v1",
        "sealed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "evidence_class": "new workflow-group-disjoint real-source replayable synthetic final test",
        "dataset": {"path": str(DATASET), "sha256": sha256(DATASET), "episodes": 400, "synthetic": True},
        "methods": list(METHODS),
        "budgets": list(BUDGETS),
        "seeds": list(SEEDS),
        "metrics": list(METRICS),
        "hard_cap": "floor(nominal_budget * 400) Human-intervened episodes; Human masked after exhaustion",
        "episode_order": "lexicographic episode_id, identical for all runs",
        "analysis": {
            "aggregate": "mean and sample SD over five paired seeds",
            "paired_differences": "SHARE-V3 minus each comparator with exact two-sided seed sign-flip p-values",
            "pareto": "maximize Repair.SR and Return; minimize UAR, Stop, and Regret; HIR treated as a cap",
        },
        "prohibitions": ["no retraining", "no recalibration", "no threshold changes", "no rerun", "no post-outcome method selection"],
        "files": files,
    }
    PROTOCOL_ROOT.mkdir(parents=True)
    manifest_path = PROTOCOL_ROOT / "pre_evaluation_manifest.json"
    atomic_json(manifest_path, protocol)
    seal_payload = {"manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path), "status": "SEALED_BEFORE_EVALUATION"}
    atomic_json(PROTOCOL_ROOT / "pre_evaluation_seal.json", seal_payload)
    print(json.dumps({"out": str(PROTOCOL_ROOT), "files": len(files), "status": seal_payload["status"]}, indent=2))


def verify_seal() -> Mapping[str, Any]:
    manifest_path = PROTOCOL_ROOT / "pre_evaluation_manifest.json"
    seal_path = PROTOCOL_ROOT / "pre_evaluation_seal.json"
    manifest, binding = json.loads(manifest_path.read_text()), json.loads(seal_path.read_text())
    if sha256(manifest_path) != binding["manifest_sha256"]:
        raise SystemExit("V3 pre-evaluation seal mismatch")
    for item in manifest["files"]:
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise SystemExit(f"post-seal artifact change: {path}")
    return manifest


def run_policy(policy: Any, episodes: Sequence[RepairEpisode], budget: float, seed: int) -> dict[str, Any]:
    cap = math.floor(budget * len(episodes))
    wrapper = OnlineHardHumanCapPolicy(policy, cap, policy.name in BINARY_POLICIES)
    raw = evaluate_policy(wrapper, episodes, RewardConfig(), budget, None)
    if wrapper.human_interventions > cap:
        raise RuntimeError("hard-cap violation")
    raw.update(
        nominal_budget=budget,
        seed=seed,
        hard_human_cap=cap,
        human_interventions=wrapper.human_interventions,
        cap_respected=True,
    )
    return raw


def aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for budget in BUDGETS:
        for method in METHODS:
            selected = [row for row in rows if row["nominal_budget"] == budget and row["policy"] == method]
            item: dict[str, Any] = {"budget": budget, "policy": method, "seeds": len(selected)}
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = stdev(values)
            item["all_caps_respected"] = all(row["cap_respected"] for row in selected)
            output.append(item)
    return output


def sign_flip(values: Sequence[float]) -> float:
    observed = abs(mean(values))
    return sum(abs(mean(sign * value for sign, value in zip(signs, values))) >= observed - 1e-15 for signs in itertools.product((-1.0, 1.0), repeat=len(values))) / (2 ** len(values))


def paired(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by = {(float(row["nominal_budget"]), int(row["seed"]), row["policy"]): row for row in rows}
    output = []
    for budget in BUDGETS:
        for comparator in METHODS[:-1]:
            for metric in METRICS:
                differences = [float(by[(budget, seed, "share_v3")][metric]) - float(by[(budget, seed, comparator)][metric]) for seed in SEEDS]
                output.append({"budget": budget, "comparator": comparator, "metric": metric, "direction": "SHARE-V3 minus comparator", "mean_difference": mean(differences), "std_difference": stdev(differences), "per_seed_differences": differences, "exact_two_sided_sign_flip_p": sign_flip(differences)})
    return output


def pareto(aggregate_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for budget in BUDGETS:
        rows = [row for row in aggregate_rows if row["budget"] == budget]
        for row in rows:
            dominated_by = []
            for other in rows:
                if other is row:
                    continue
                weak = other["repair_sr_mean"] >= row["repair_sr_mean"] and other["normalized_return_mean"] >= row["normalized_return_mean"] and other["uar_mean"] <= row["uar_mean"] and other["stopped_rate_mean"] <= row["stopped_rate_mean"] and other["collaboration_regret_mean"] <= row["collaboration_regret_mean"]
                strict = any((other[key] != row[key]) for key in ("repair_sr_mean", "normalized_return_mean", "uar_mean", "stopped_rate_mean", "collaboration_regret_mean"))
                if weak and strict:
                    dominated_by.append(other["policy"])
            result.append({"budget": budget, "policy": row["policy"], "nondominated": not dominated_by, "dominated_by": dominated_by})
    return result


def evaluate() -> None:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()):
        raise SystemExit("V3 final evaluation already exists; rerun prohibited")
    protocol = verify_seal()
    episodes = sorted(load_episodes(DATASET), key=lambda episode: episode.episode_id)
    RESULT_ROOT.mkdir(parents=True)
    rows = []
    for budget in BUDGETS:
        encoder = FeatureEncoder.load(V3_ROOTS[budget] / "feature_encoder.json")
        for seed in SEEDS:
            pack = joblib.load(COMPARATOR_ROOTS[budget] / f"comparators_seed_{seed}.joblib")
            for name in METHODS[:3]:
                rows.append(run_policy(pack[name], episodes, budget, seed))
            ddqn = DQNPolicy.load(str(COMPARATOR_ROOTS[budget] / f"share_ddqn_seed_{seed}.pt"), encoder, RewardConfig(), require_reward_config=True)
            ddqn.name = "share_ddqn"
            rows.append(run_policy(ddqn, episodes, budget, seed))
            v3 = DQNPolicy.load(str(V3_ROOTS[budget] / f"share_v3_budget_{budget:.1f}_seed_{seed}.pt"), encoder, RewardConfig(rollback_penalty=0.60), require_reward_config=True)
            v3.name = "share_v3"
            rows.append(run_policy(v3, episodes, budget, seed))
    summaries = [{key: value for key, value in row.items() if key != "records"} for row in rows]
    aggregate_rows = aggregate(summaries)
    final = {
        "protocol_sha256": sha256(PROTOCOL_ROOT / "pre_evaluation_manifest.json"),
        "evidence_class": protocol["evidence_class"],
        "aggregate": aggregate_rows,
        "per_seed": summaries,
        "paired_differences": paired(summaries),
        "pareto_status": pareto(aggregate_rows),
        "all_caps_respected": all(row["cap_respected"] for row in summaries),
        "runs": len(summaries),
    }
    atomic_json(RESULT_ROOT / "FINAL_RESULTS.json", final)
    atomic_json(RESULT_ROOT / "raw_records.json", [{"budget": row["nominal_budget"], "seed": row["seed"], "policy": row["policy"], "records": row["records"]} for row in rows])
    audit = {
        "status": "COMPLETE_DO_NOT_RERUN",
        "completed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "runs": len(summaries),
        "hard_cap_violations": sum(not row["cap_respected"] for row in summaries),
        "post_seal_changes": 0,
        "final_results_sha256": sha256(RESULT_ROOT / "FINAL_RESULTS.json"),
        "raw_records_sha256": sha256(RESULT_ROOT / "raw_records.json"),
    }
    atomic_json(RESULT_ROOT / "final_audit.json", audit)
    paths = sorted(path for path in RESULT_ROOT.iterdir() if path.name != "SHA256SUMS")
    (RESULT_ROOT / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in paths), encoding="utf-8")
    print(json.dumps({"out": str(RESULT_ROOT), "runs": len(summaries), "all_caps_respected": final["all_caps_respected"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "evaluate"))
    args = parser.parse_args()
    seal() if args.action == "seal" else evaluate()


if __name__ == "__main__":
    main()
