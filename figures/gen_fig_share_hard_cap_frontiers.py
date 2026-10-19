from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("share_conf", "share_static", "share_bandit", "share_ddqn", "share_h")
LABELS = {
    "share_conf": "SHARE-Conf",
    "share_static": "SHARE-Static",
    "share_bandit": "SHARE-Bandit",
    "share_ddqn": "SHARE-DDQN",
    "share_h": "SHARE-H",
}
COLORS = {
    "share_conf": "#E69F00",
    "share_static": "#56B4E9",
    "share_bandit": "#009E73",
    "share_ddqn": "#0072B2",
    "share_h": "#D55E00",
}
MARKERS = {"share_conf": "o", "share_static": "s", "share_bandit": "^", "share_ddqn": "D", "share_h": "*"}
LINESTYLES = {"share_conf": ":", "share_static": "--", "share_bandit": "-.", "share_ddqn": "--", "share_h": "-"}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "lines.linewidth": 1.8,
    }
)


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: (METHODS.index(row["policy"]), float(row["nominal_budget"])))


def two_dimensional_front(rows, y: str, maximize_y: bool):
    points = []
    for row in rows:
        x = float(row["hir_mean"])
        value = float(row[f"{y}_mean"])
        dominated = False
        for other in rows:
            ox = float(other["hir_mean"])
            oy = float(other[f"{y}_mean"])
            weak = ox <= x + 1e-12 and (oy >= value - 1e-12 if maximize_y else oy <= value + 1e-12)
            strict = ox < x - 1e-12 or (oy > value + 1e-12 if maximize_y else oy < value - 1e-12)
            if weak and strict:
                dominated = True
                break
        if not dominated:
            points.append((x, value))
    return sorted(set(points))


def plot(rows, metric: str, ylabel: str, filename: str, maximize: bool, out: Path):
    fig, ax = plt.subplots(figsize=(5.5, 3.45))
    for method in METHODS:
        group = [row for row in rows if row["policy"] == method]
        x = np.asarray([float(row["hir_mean"]) for row in group])
        y = np.asarray([float(row[f"{metric}_mean"]) for row in group])
        yerr = np.asarray([float(row[f"{metric}_std"]) for row in group])
        order = np.argsort(x)
        ax.errorbar(
            x[order], y[order], yerr=yerr[order], label=LABELS[method],
            color=COLORS[method], marker=MARKERS[method], linestyle=LINESTYLES[method],
            markersize=7 if method == "share_h" else 4.5, capsize=2, alpha=1.0 if method == "share_h" else 0.88,
            zorder=4 if method == "share_h" else 2,
        )
    front = two_dimensional_front(rows, metric, maximize)
    if len(front) > 1:
        ax.plot([point[0] for point in front], [point[1] for point in front], color="#2E3440", linewidth=1.0, linestyle=(0, (2, 2)), label="2D frontier", zorder=1)
    ax.set_xlabel("Realized human intervention rate (HIR)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=-0.01)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.985), columnspacing=1.2, handlelength=2.4)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.20, top=0.72)
    fig.savefig(out / f"{filename}.pdf")
    fig.savefig(out / f"{filename}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load(args.aggregate)
    plot(rows, "repair_sr", "Repair success rate", "fig_share_repair_hir_frontier", True, args.out)
    plot(rows, "uar", "Unsafe autonomy rate", "fig_share_uar_hir_frontier", False, args.out)
    plot(rows, "collaboration_regret", "Collaboration regret", "fig_share_regret_hir_frontier", False, args.out)
    print(f"wrote 6 figure files to {args.out}")


if __name__ == "__main__":
    main()
