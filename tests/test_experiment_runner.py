import json
import subprocess
import sys
from pathlib import Path


def test_experiment_runner_smoke(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    out = tmp_path / "results"
    subprocess.run([sys.executable, "scripts/generate_flare_datasets.py", "--scale", "0.01", "--out", str(data), "--seed", "5"], cwd=repo, check=True)
    subprocess.run([sys.executable, "scripts/run_flare_experiments.py", "--data", str(data), "--out", str(out)], cwd=repo, check=True)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "localization" in summary and "repair" in summary
    assert (out / "rq1_localization.md").exists()
    assert (out / "rq2_repair.md").exists()
    flare_rows = [row for row in summary["localization"] if row["method"] == "flare"]
    assert flare_rows
