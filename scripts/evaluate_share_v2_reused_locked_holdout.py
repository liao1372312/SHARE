from __future__ import annotations

"""Retrospective SHARE-V2 evaluation on the previously opened 400 holdout.

This is explicitly not a new locked or confirmatory test.  It measures whether
the post-locked V2 redesign repairs the known shift on the old replay corpus.
No fitting, calibration, threshold selection, or parameter change occurs here.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction, RewardConfig


PROTOCOL_VERSION = "share-v2-reused-open-holdout-retrospective-v1"
DATASET = ROOT / "datasets" / "share_locked_holdout_replayable_20260826" / "episodes.jsonl"
DATASET_SHA256 = "a7e66906538a419d1be18e78120f998e3a70a8776d84897b34ffa7cdd301b74f"
ENCODER = ROOT / "results" / "share_v2_stop_penalty_tuning_20260827" / "p060" / "feature_encoder.json"
SEEDS = (17, 29, 43, 71, 101)
CHECKPOINTS = {
    17: ROOT / "results" / "share_v2_stop_penalty_tuning_20260827" / "p060" / "share_v2_seed_17.pt",
    **{
        seed: ROOT / "results" / "share_v2_stop_penalty_tuning_20260827" / "p060_remaining" / f"share_v2_seed_{seed}.pt"
        for seed in (29, 43, 71, 101)
    },
}
METRICS = (
    "repair_sr",
    "hir",
    "uar",
    "unsafe_autonomy_cost_rate",
    "stopped_rate",
    "normalized_return",
    "collaboration_regret",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def protocol(out: Path) -> dict[str, Any]:
    if sha256_file(DATASET) != DATASET_SHA256:
        raise SystemExit("reused holdout hash mismatch")
    if sha256_file(ENCODER) != "7320e0e177fdbcfec5ff1308d157e1152d520bbfb25e2ae853fdbcc0e4a8c7b7":
        raise SystemExit("V2 encoder hash mismatch")
    episodes = load_episodes(DATASET)
    if len(episodes) != 400 or len({episode.episode_id for episode in episodes}) != 400:
        raise SystemExit("expected 400 unique reused-holdout episodes")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "PREPARED_RETROSPECTIVE_REUSE",
        "evidence_class": "post_hoc_reuse_of_previously_opened_holdout",
        "independent_locked_test": False,
        "confirmatory": False,
        "eligible_to_replace_original_locked_result": False,
        "purpose": "diagnose whether frozen SHARE-V2 repairs the known old-holdout shift",
        "prohibitions": [
            "no fitting",
            "no recalibration",
            "no threshold selection",
            "no checkpoint replacement",
            "no further tuning based on these outcomes",
        ],
        "dataset": {
            "path": str(DATASET.relative_to(ROOT)),
            "sha256": DATASET_SHA256,
            "episodes": 400,
            "synthetic": True,
            "previously_opened": True,
        },
        "episode_order_sha256": sha256_json([episode.episode_id for episode in episodes]),
        "human_budget": 0.20,
        "hard_human_episode_cap": 80,
        "seeds": list(SEEDS),
        "training_reward": asdict(RewardConfig(rollback_penalty=0.60)),
        "common_evaluation_reward": asdict(RewardConfig()),
        "encoder": {"path": str(ENCODER.relative_to(ROOT)), "sha256": sha256_file(ENCODER)},
        "checkpoints": {
            str(seed): {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for seed, path in CHECKPOINTS.items()
        },
        "evaluator": {
            "script": str(Path(__file__).relative_to(ROOT)),
            "script_sha256": sha256_file(Path(__file__)),
            "hard_cap_source_sha256": sha256_file(ROOT / "scripts" / "run_share_online_hard_cap.py"),
            "evaluation_source_sha256": sha256_file(ROOT / "share" / "evaluation.py"),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "output": str(out.resolve().relative_to(ROOT)),
    }
    payload["protocol_sha256"] = sha256_json(payload)
    return payload


def prepare(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "pre_evaluation_manifest.json"
    payload = protocol(out)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SystemExit("existing retrospective manifest differs")
    else:
        atomic_json(manifest_path, payload)
    (out / "PRE_EVALUATION_SHA256").write_text(
        f"{sha256_file(manifest_path)}  pre_evaluation_manifest.json\n",
        encoding="utf-8",
    )
    return payload


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"method": "SHARE-V2", "seeds": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        output[f"{metric}_mean"] = mean(values)
        output[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output[f"{metric}_min"] = min(values)
        output[f"{metric}_max"] = max(values)
    output["all_runs_respected_cap"] = all(bool(row["cap_respected"]) for row in rows)
    return output


def execute(out: Path) -> None:
    manifest = prepare(out)
    if (out / "FINAL_RESULTS.json").exists():
        raise SystemExit("retrospective evaluation already executed; refusing to rerun")

    episodes = load_episodes(DATASET)
    encoder = FeatureEncoder.load(ENCODER)
    training_reward = RewardConfig(rollback_penalty=0.60)
    evaluation_reward = RewardConfig()
    cap = 80
    rows: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []

    for seed in SEEDS:
        policy = DQNPolicy.load(
            str(CHECKPOINTS[seed]),
            encoder,
            training_reward,
            require_reward_config=True,
        )
        policy.name = "share_v2"
        wrapper = OnlineHardHumanCapPolicy(policy, cap, binary_fallback=False)
        result = evaluate_policy(wrapper, episodes, evaluation_reward, 0.20, None)
        human_episodes = sum(int(record["human_used"]) for record in result["records"])
        human_actions = sum(
            int(action == int(AuthorityAction.HUMAN))
            for record in result["records"]
            for action in record["actions"]
        )
        if human_episodes != wrapper.human_interventions or human_episodes > cap:
            raise RuntimeError(f"seed {seed}: hard-cap audit failed")
        row = {key: value for key, value in result.items() if key != "records"}
        row.update(
            {
                "seed": seed,
                "hard_human_cap": cap,
                "human_interventions": human_episodes,
                "human_actions": human_actions,
                "cap_respected": human_episodes <= cap,
                "attempted_human_after_exhaustion": wrapper.attempted_human_after_exhaustion,
                "value_policy_replans": wrapper.value_policy_replans,
            }
        )
        rows.append(row)
        raw_results.append({"summary": row, "records": result["records"], "cap_events": wrapper.events})

    aggregate_row = aggregate(rows)
    result_payload = {
        "evidence_class": manifest["evidence_class"],
        "independent_locked_test": False,
        "confirmatory": False,
        "eligible_to_replace_original_locked_result": False,
        "protocol_sha256": manifest["protocol_sha256"],
        "per_seed": rows,
        "aggregate": aggregate_row,
        "interpretation_boundary": "Retrospective reused-holdout diagnostic only; a new untouched holdout is required for confirmatory claims.",
    }
    atomic_json(out / "FINAL_RESULTS.json", result_payload)
    atomic_json(out / "raw_records.json", raw_results)

    fields = list(rows[0].keys())
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_row.keys()))
        writer.writeheader()
        writer.writerow(aggregate_row)

    report = f"""# SHARE-V2 Retrospective Reused-Holdout Diagnostic

**Not an independent locked test. Not confirmatory.**

The frozen post-locked SHARE-V2 checkpoints were evaluated once on the previously opened 400-episode replayable synthetic holdout under the original 20% online hard Human cap. No fitting or recalibration occurred.

| Repair.SR | HIR | UAR | Stop | Return | Regret |
|---:|---:|---:|---:|---:|---:|
| {aggregate_row['repair_sr_mean']:.3f} | {aggregate_row['hir_mean']:.3f} | {aggregate_row['uar_mean']:.3f} | {aggregate_row['stopped_rate_mean']:.3f} | {aggregate_row['normalized_return_mean']:.3f} | {aggregate_row['collaboration_regret_mean']:.3f} |

All five runs respected the hard cap: `{aggregate_row['all_runs_respected_cap']}`.

Because V2 was designed after the old holdout's aggregate failure pattern was known, these numbers are retrospective diagnostics and cannot replace the original locked result as independent evidence. A new untouched workflow-group-disjoint holdout remains necessary.
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    checksum_paths = [
        out / "pre_evaluation_manifest.json",
        out / "FINAL_RESULTS.json",
        out / "raw_records.json",
        out / "per_seed.csv",
        out / "aggregate.csv",
        out / "REPORT.md",
    ]
    (out / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare_only == args.execute:
        raise SystemExit("choose exactly one of --prepare-only or --execute")
    if args.prepare_only:
        payload = prepare(args.out)
        print(json.dumps({"prepared": str(args.out), "protocol_sha256": payload["protocol_sha256"]}, indent=2))
    else:
        execute(args.out)
        print(json.dumps({"completed": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
