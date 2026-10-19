from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "random_budget": "Random",
    "confidence_threshold": "Confidence",
    "risk_rule": "Risk rule",
    "static_benefit_classifier": "Static classifier",
    "contextual_bandit": "Bandit",
    "vanilla_dqn": "DQN",
    "unconstrained_double_dqn": "Double DQN",
    "share_controller": "SHARE",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the synthetic SHARE budget frontier.")
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.aggregate.read_text(encoding="utf-8"))
    if not rows or not all(bool(row.get("synthetic_evidence", False)) for row in rows):
        raise SystemExit("this exporter accepts only explicitly labelled synthetic aggregate rows")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "m3_aggregate.csv", rows)
    primary = [row for row in rows if abs(float(row["target_budget"]) - 0.2) < 1e-12]
    write_csv(args.out / "m3_primary_budget.csv", primary)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for policy, label in LABELS.items():
        items = sorted((row for row in rows if row["policy"] == policy), key=lambda row: float(row["hir_mean"]))
        if not items:
            continue
        xs = [float(row["hir_mean"]) for row in items]
        ys = [float(row["repair_sr_mean"]) for row in items]
        ax.plot(xs, ys, marker="o", linewidth=1.3, markersize=4, label=label)
    ax.set_xlabel("Human intervention rate (mean over five seeds)")
    ax.set_ylabel("Repair success rate")
    ax.set_title("Synthetic development budget frontier")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "m3_budget_frontier.pdf")
    fig.savefig(args.out / "m3_budget_frontier.png", dpi=300)
    plt.close(fig)
    print(json.dumps({"out": str(args.out), "rows": len(rows), "primary_rows": len(primary)}, indent=2))


if __name__ == "__main__":
    main()
