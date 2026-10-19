#!/usr/bin/env python3
"""Generate the SHARE architecture overview used by the TOSEM manuscript."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Preserve selectable/searchable text and avoid Type 3 fonts in the ACM PDF.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


COLORS = {
    "ink": "#233044",
    "muted": "#58677C",
    "flare": "#E8F1FB",
    "state": "#EAF7F0",
    "policy": "#F3ECFC",
    "ai": "#E6F4FF",
    "human": "#FFF1DC",
    "stop": "#FDE9E7",
    "outcome": "#EDF3F8",
    "evidence": "#FFF8D8",
}


def box(ax, xy, width, height, text, color, *, fontsize=10, weight="semibold"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.09",
        linewidth=1.5,
        edgecolor=COLORS["ink"],
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.25,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, *, label=None, color=None, bend=0.0, style="-|>"):
    color = color or COLORS["ink"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=13,
        linewidth=1.55,
        color=color,
        connectionstyle=f"arc3,rad={bend}",
        zorder=4,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + (0.18 if bend >= 0 else -0.18)
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=COLORS["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
            zorder=5,
        )


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.6, 7.0))
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    # Inherited repair backbone.
    group = FancyBboxPatch(
        (0.25, 3.75),
        6.25,
        2.75,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.4,
        linestyle="--",
        edgecolor="#3977B7",
        facecolor="#F8FBFE",
        zorder=0,
    )
    ax.add_patch(group)
    ax.text(0.48, 6.22, "Inherited FLARE repair backbone", fontsize=11, fontweight="bold", color="#285D90")
    box(ax, (0.55, 4.55), 1.55, 1.15, "Failed agentic\nworkflow trace", COLORS["flare"])
    box(ax, (2.55, 4.55), 1.55, 1.15, "Outcome-aware\nexecution graph", COLORS["flare"])
    box(ax, (4.55, 4.35), 1.65, 1.55, "Root cause\nCandidate repair\nAffected slice\nRuntime evidence", COLORS["flare"], fontsize=9.3)
    arrow(ax, (2.10, 5.12), (2.55, 5.12))
    arrow(ax, (4.10, 5.12), (4.55, 5.12))

    # State and controller.
    box(
        ax,
        (6.95, 4.35),
        2.05,
        1.55,
        "Mixed-autonomy state $s_t$\n\nrepair evidence · history\nrisk · remaining budget",
        COLORS["state"],
        fontsize=9.2,
    )
    box(
        ax,
        (9.55, 4.25),
        2.25,
        1.75,
        "Constrained Double DQN\n\n$Q(s_t,a_{AI})$   $Q(s_t,a_H)$\n$Q(s_t,a_S)$\n\n$VoH=Q_H-Q_{AI}$",
        COLORS["policy"],
        fontsize=9.1,
    )
    arrow(ax, (6.20, 5.12), (6.95, 5.12), label="features")
    arrow(ax, (9.00, 5.12), (9.55, 5.12))

    # Authority actions are aligned in one row so that their consequences remain
    # visually distinct and the feedback paths do not cross other action boxes.
    ax.text(11.75, 3.82, "Authority action $a_t$", ha="center", fontsize=11, fontweight="bold", color=COLORS["ink"])
    box(ax, (8.35, 2.65), 2.05, 0.82, "Autonomous repair\n$a_{AI}$", COLORS["ai"], fontsize=9.2)
    box(ax, (10.75, 2.65), 2.05, 0.82, "Human escalation\n$a_H$", COLORS["human"], fontsize=9.2)
    box(ax, (13.15, 2.65), 2.05, 0.82, "Stop / rollback\n$a_S$", COLORS["stop"], fontsize=9.2)
    arrow(ax, (10.25, 4.25), (9.38, 3.47), bend=0.08)
    arrow(ax, (10.68, 4.25), (11.78, 3.47))
    arrow(ax, (11.15, 4.25), (14.18, 3.47), bend=-0.10)

    # Outcomes and feedback loop.
    box(ax, (8.35, 1.05), 2.05, 0.92, "Execute and evaluate\nrepair outcome", COLORS["outcome"])
    box(ax, (10.75, 1.05), 2.05, 0.92, "Human-validated\nevidence store", COLORS["evidence"])
    ax.text(14.18, 1.51, "Terminal safe exit", ha="center", va="center", fontsize=9.2, color=COLORS["muted"])
    arrow(ax, (9.38, 2.65), (9.38, 1.97), label="apply")
    arrow(ax, (11.78, 2.65), (11.78, 1.97), label="correction")
    arrow(ax, (10.75, 1.51), (10.40, 1.51), label="validated repair")
    arrow(ax, (14.18, 2.65), (14.18, 1.76))
    arrow(ax, (8.35, 1.48), (7.75, 4.35), label="next state $s_{t+1}$", bend=-0.30)
    ax.text(
        6.20,
        0.38,
        "Safety constraint: unsafe autonomous repair ≤ $\\epsilon_U$",
        ha="center",
        fontsize=9.2,
        color=COLORS["muted"],
    )
    ax.text(
        11.35,
        0.38,
        "Attention constraint: expected human use ≤ $B_H$",
        ha="center",
        fontsize=9.2,
        color=COLORS["muted"],
    )

    fig.tight_layout(pad=0.35)
    pdf_metadata = {
        "Creator": "scripts/generate_share_overview.py",
        "CreationDate": datetime(2026, 8, 21, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 8, 21, tzinfo=timezone.utc),
    }
    fig.savefig(
        output_dir / "share_overview.pdf",
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
        metadata=pdf_metadata,
    )
    fig.savefig(
        output_dir / "share_overview.png",
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "scripts/generate_share_overview.py"},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("TOSEM/TOSEM_FLARE/figures"),
    )
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
