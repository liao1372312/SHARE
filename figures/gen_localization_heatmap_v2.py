#!/usr/bin/env python3
"""Render the publication heatmap from the frozen localization CSV.

This script changes presentation only. It never recomputes or edits the source
measurements in localization_heatmap_data.csv.  The historical display label
``FLARE`` is mapped to the unified manuscript name ``SHARE``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "figures" / "localization_heatmap_data.csv"
OUTPUT_STEM = ROOT / "TOSEM" / "TOSEM_FLARE" / "figures" / "localization_heatmap_v2"

METHOD_ORDER = [
    "Last-Agent",
    "Last-Failed-Step",
    "Critical-Path",
    "Backward-Slice-Last",
    "Slice-Centrality",
    "Failure-Signature Rule",
    "EPF Attribution",
    "FLARE",
]
FAILURE_ORDER = [
    "Planning",
    "Service Selection",
    "Parameter Binding",
    "Tool Execution",
    "Supervisor",
    "Answer Grounding",
]
METHOD_LABELS = {
    "Last-Agent": "Last-Agent",
    "Last-Failed-Step": "Last-Failed-Step",
    "Critical-Path": "Critical-Path",
    "Backward-Slice-Last": "Backward-Slice-Last",
    "Slice-Centrality": "Slice-Centrality",
    "Failure-Signature Rule": "Failure-Signature",
    "EPF Attribution": "EPF Attribution",
    "FLARE": "SHARE",
}
FAILURE_LABELS = {
    "Planning": "Planning",
    "Service Selection": "Service\nselection",
    "Parameter Binding": "Parameter\nbinding",
    "Tool Execution": "Tool\nexecution",
    "Supervisor": "Supervisor",
    "Answer Grounding": "Answer\ngrounding",
}


def load_frozen_values() -> tuple[np.ndarray, list[int]]:
    values: dict[tuple[str, str], float] = {}
    counts: dict[str, int] = {}
    with SOURCE_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], row["failure_type"])
            values[key] = float(row["faulty_decision_acc"]) * 100.0
            count = int(row["n"])
            previous = counts.setdefault(row["failure_type"], count)
            if previous != count:
                raise ValueError(f"Inconsistent n for {row['failure_type']}")

    expected = {(m, f) for m in METHOD_ORDER for f in FAILURE_ORDER}
    missing = expected - set(values)
    if missing:
        raise ValueError(f"Missing frozen cells: {sorted(missing)}")

    matrix = np.array(
        [[values[(method, failure)] for failure in FAILURE_ORDER] for method in METHOD_ORDER],
        dtype=float,
    )
    return matrix, [counts[failure] for failure in FAILURE_ORDER]


def render(matrix: np.ndarray, counts: list[int]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9.2,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        }
    )

    # A restrained blue sequential scale remains legible in grayscale.  The
    # 35--95 range covers every frozen cell, unlike the clipped legacy figure.
    cmap = LinearSegmentedColormap.from_list(
        "share_blue",
        ["#F7FBFF", "#DCEAF4", "#9ECAE1", "#4292C6", "#1764A0", "#08306B"],
        N=256,
    )
    fig, ax = plt.subplots(figsize=(7.25, 3.55), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, vmin=35.0, vmax=95.0, aspect="auto")

    ax.set_xticks(np.arange(len(FAILURE_ORDER)))
    ax.set_xticklabels(
        [f"{FAILURE_LABELS[f]}\n$n={n}$" for f, n in zip(FAILURE_ORDER, counts)],
        linespacing=1.05,
    )
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.tick_params(axis="both", which="major", length=0, pad=7)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if value >= 69.0 else "#17212B"
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.0,
                fontweight="bold" if row == matrix.shape[0] - 1 else "normal",
            )

    # Fine white cell boundaries and a colorblind-safe outline distinguish the
    # proposed method without distorting the quantitative color encoding.
    ax.set_xticks(np.arange(-0.5, len(FAILURE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.15)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    share_row = matrix.shape[0] - 1
    ax.add_patch(
        Rectangle(
            (-0.49, share_row - 0.49),
            len(FAILURE_ORDER) - 0.02,
            0.98,
            fill=False,
            edgecolor="#D55E00",
            linewidth=2.0,
            clip_on=False,
            zorder=5,
        )
    )
    share_tick = ax.get_yticklabels()[-1]
    share_tick.set_color("#B54F00")
    share_tick.set_fontweight("bold")

    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.025, ticks=np.arange(40, 100, 10))
    colorbar.set_label("Faulty-decision accuracy (%)", rotation=270, labelpad=15)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=2.5, width=0.7, labelsize=8)

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    matrix, counts = load_frozen_values()
    render(matrix, counts)
    print(f"Source unchanged: {SOURCE_CSV}")
    print(f"Saved: {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Saved: {OUTPUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
