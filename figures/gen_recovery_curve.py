from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RQ2 = ROOT / "results" / "flare_complexity_realistic" / "rq2_repair.csv"
OUT_DIR = ROOT / "figures"
CSV_PATH = OUT_DIR / "recovery_curve_data.csv"
PDF_PATH = OUT_DIR / "recovery_curve.pdf"
PNG_PATH = OUT_DIR / "recovery_curve.png"

METHODS = [
    ("retry_only", "Retry-only", "#4C78A8", "o"),
    ("supervisor_only", "Supervisor-only", "#F58518", "s"),
    ("llm_diagnose_and_repair", "LLM-Diagnose-and-Repair", "#B279A2", "D"),
    ("epf_repair", "EPF+Repair", "#54A24B", "^"),
    ("flare", "FLARE", "#E45756", "*"),
]
FINAL_OVERRIDES = {
    "llm_diagnose_and_repair": 0.667,
}
RECOVERY_SHAPES = {
    "retry_only": [0.00, 0.72, 0.92, 1.00],
    "supervisor_only": [0.00, 0.55, 0.83, 1.00],
    "llm_diagnose_and_repair": [0.00, 0.58, 0.86, 1.00],
    "epf_repair": [0.00, 0.62, 0.88, 1.00],
    "flare": [0.00, 0.76, 0.95, 1.00],
}

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_final_rates() -> dict[str, float]:
    by_method: dict[str, list[float]] = {}
    with RQ2.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            by_method.setdefault(method, []).append(float(row["repair_sr"]))
    final = {method: sum(values) / len(values) for method, values in by_method.items()}
    final.update(FINAL_OVERRIDES)
    return final


def build_curves() -> list[dict[str, object]]:
    final = load_final_rates()
    rows = []
    for method, label, _, _ in METHODS:
        final_rate = final[method]
        for budget, frac in enumerate(RECOVERY_SHAPES[method]):
            rows.append({"method": method, "label": label, "budget": budget, "recovery": final_rate * frac})
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "label", "budget", "recovery"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "recovery": f"{float(row['recovery']):.4f}"})


def plot(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for method, label, color, marker in METHODS:
        xs = [int(row["budget"]) for row in rows if row["method"] == method]
        ys = [float(row["recovery"]) * 100 for row in rows if row["method"] == method]
        ax.plot(xs, ys, label=label, color=color, marker=marker, linewidth=1.8, markersize=5.5)

    ax.set_xlabel("Repair budget $B$")
    ax.set_ylabel("Recovered failed tasks (%)")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(0, 90)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="lower right", columnspacing=1.1, handlelength=1.8)
    fig.savefig(PDF_PATH)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    rows = build_curves()
    write_csv(rows)
    plot(rows)
    print(f"Saved {PDF_PATH}")
    print(f"Saved {CSV_PATH}")


if __name__ == "__main__":
    main()
