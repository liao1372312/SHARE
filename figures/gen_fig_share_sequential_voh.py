from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "share_static": "#E69F00",
    "share_bandit": "#56B4E9",
    "share_ddqn": "#009E73",
    "share_h": "#D55E00",
}
LABELS = {
    "share_static": "SHARE-Static",
    "share_bandit": "SHARE-Bandit",
    "share_ddqn": "SHARE-DDQN",
    "share_h": "SHARE",
}
MARKERS = {"share_static": "o", "share_bandit": "s", "share_ddqn": "^", "share_h": "D"}
METRICS = (
    ("repair_sr", "Repair\nsuccess", (0.65, 1.01)),
    ("uar", "Unsafe autonomy\nrate", (0.0, 0.55)),
    ("stopped_rate", "Stop rate", (0.0, 0.18)),
    ("collaboration_regret", "Collaboration\nregret", (0.0, 1.85)),
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mechanism_panels(rows: list[dict[str, str]], stratum: str, order: list[str], labels: list[str], out: Path, stem: str) -> None:
    indexed = {(row[stratum], row["policy"]): row for row in rows}
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.15), constrained_layout=True)
    x = np.arange(len(order))
    for ax, (metric, title, ylim) in zip(axes, METRICS):
        for policy in COLORS:
            means = [float(indexed[(level, policy)][f"{metric}_mean"]) for level in order]
            stds = [float(indexed[(level, policy)][f"{metric}_std"]) for level in order]
            ax.errorbar(
                x, means, yerr=stds, color=COLORS[policy], marker=MARKERS[policy],
                markersize=3.5, linewidth=1.15, capsize=2, label=LABELS[policy],
            )
        ax.set_title(title)
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def voh_plot(per_seed: list[dict[str, str]], out: Path) -> None:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in per_seed:
        grouped[int(row["voh_bin"])].append(row)
    x = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(3.55, 2.55), constrained_layout=True)
    for seed in sorted({int(row["seed"]) for row in per_seed}):
        values = [
            float(next(row for row in grouped[b] if int(row["seed"]) == seed)["realized_downstream_reward_advantage"])
            for b in x
        ]
        ax.plot(x, values, color="#999999", alpha=0.42, linewidth=0.8)
    means = np.array([
        np.mean([float(row["realized_downstream_reward_advantage"]) for row in grouped[b]]) for b in x
    ])
    stds = np.array([
        np.std([float(row["realized_downstream_reward_advantage"]) for row in grouped[b]], ddof=1) for b in x
    ])
    ax.errorbar(x, means, yerr=stds, color="#D55E00", marker="o", linewidth=1.6,
                markersize=4.5, capsize=3, label="Mean ± SD across seeds")
    ax.axhline(0, color="#333333", linewidth=0.7, linestyle="--")
    ax.set_xticks(x, ["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"])
    ax.set_xlabel(r"Predicted $VoH=Q(H)-Q(AI)$ rank quintile")
    ax.set_ylabel(r"Realized downstream advantage $G_H-G_{AI}$")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(out / "fig_share_voh_validity.pdf")
    fig.savefig(out / "fig_share_voh_validity.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/share_sequential_voh_mechanism_20260826"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mechanism_panels(
        read_csv(args.results / "complexity_aggregate.csv"), "complexity",
        ["C_low", "C_mid", "C_high"], ["Low\n(n=55)", "Mid\n(n=49)", "High\n(n=16)"],
        args.out, "fig_share_complexity_mechanism",
    )
    mechanism_panels(
        read_csv(args.results / "horizon_aggregate.csv"), "horizon",
        ["H2", "H3"], ["2 steps\n(n=103)", "3 steps\n(n=16)"],
        args.out, "fig_share_horizon_mechanism",
    )
    voh_plot(read_csv(args.results / "voh_bins_per_seed.csv"), args.out)
    print(f"Wrote mechanism figures to {args.out}")


if __name__ == "__main__":
    main()
