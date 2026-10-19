from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "results" / "flare_complexity_realistic" / "predictions.jsonl"
DATA_DIR = ROOT / "datasets" / "flare_complexity"
OUT_DIR = ROOT / "figures"
CSV_PATH = OUT_DIR / "localization_heatmap_data.csv"
PDF_PATH = OUT_DIR / "localization_heatmap.pdf"
PNG_PATH = OUT_DIR / "localization_heatmap.png"

METHODS = [
    ("last_agent", "Last-Agent"),
    ("last_failed_step", "Last-Failed-Step"),
    ("critical_path", "Critical-Path"),
    ("backward_slice_last", "Backward-Slice-Last"),
    ("slice_centrality", "Slice-Centrality"),
    ("failure_signature", "Failure-Signature Rule"),
    ("epf_lite", "EPF Attribution"),
    ("flare", "FLARE"),
]
FAILURES = [
    ("planning_failure", "Planning"),
    ("service_selection_failure", "Service\nSelection"),
    ("parameter_binding_failure", "Parameter\nBinding"),
    ("tool_execution_failure", "Tool\nExecution"),
    ("supervisor_failure", "Supervisor"),
    ("answer_grounding_failure", "Answer\nGrounding"),
]

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_labels() -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                label = row.get("repair_label", {})
                if label.get("failed"):
                    labels[row["id"]] = {
                        "failure_type": label["failure_type"],
                        "faulty_decision_node": label["faulty_decision_node"],
                    }
    return labels


def compute_matrix() -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    labels = load_labels()
    totals = {(method, failure): 0 for method, _ in METHODS for failure, _ in FAILURES}
    correct = {(method, failure): 0 for method, _ in METHODS for failure, _ in FAILURES}
    with PRED_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            pred = json.loads(line)
            gold = labels.get(pred["sample_id"])
            if gold is None:
                continue
            method = pred["method"]
            failure = gold["failure_type"]
            if (method, failure) not in totals:
                continue
            totals[(method, failure)] += 1
            correct[(method, failure)] += int(pred.get("pred_decision") == gold["faulty_decision_node"])

    matrix = np.full((len(METHODS), len(FAILURES)), np.nan, dtype=float)
    for i, (method, _) in enumerate(METHODS):
        for j, (failure, _) in enumerate(FAILURES):
            total = totals[(method, failure)]
            matrix[i, j] = correct[(method, failure)] / total if total else np.nan
    return matrix, totals


def write_csv(matrix: np.ndarray, totals: dict[tuple[str, str], int]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "failure_type", "faulty_decision_acc", "n"])
        for i, (method, method_label) in enumerate(METHODS):
            for j, (failure, failure_label) in enumerate(FAILURES):
                writer.writerow([method_label, failure_label.replace("\n", " "), f"{matrix[i, j]:.4f}", totals[(method, failure)]])


def plot(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    cmap = plt.get_cmap("YlGnBu")
    image = ax.imshow(matrix * 100.0, cmap=cmap, vmin=35, vmax=75, aspect="auto")

    ax.set_xticks(np.arange(len(FAILURES)))
    ax.set_xticklabels([label for _, label in FAILURES])
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels([label for _, label in METHODS])
    ax.tick_params(axis="both", length=0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j] * 100.0
            color = "white" if value >= 62 else "#1a1a1a"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=7.5)

    ax.set_xticks(np.arange(-0.5, len(FAILURES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Faulty decision accuracy (%)", rotation=270, labelpad=12)
    cbar.outline.set_visible(False)

    fig.savefig(PDF_PATH)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    matrix, totals = compute_matrix()
    write_csv(matrix, totals)
    plot(matrix)
    print(f"Saved {PDF_PATH}")
    print(f"Saved {CSV_PATH}")


if __name__ == "__main__":
    main()
