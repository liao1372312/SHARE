import json
import subprocess
import sys
from pathlib import Path


def test_dataset_generator_smoke(tmp_path):
    out = tmp_path / "flare_synth"
    subprocess.run(
        [sys.executable, "scripts/generate_flare_datasets.py", "--scale", "0.01", "--out", str(out), "--seed", "3"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    subprocess.run(
        [sys.executable, "scripts/validate_flare_datasets.py", str(out)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert {item["dataset"] for item in summary} == {"public_api", "enterprise_service", "mixed_stress"}
    public_rows = (out / "public_api.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(public_rows[0])
    assert first["synthetic"] is True
    assert {"events", "repair_label", "candidate_api_ids"}.issubset(first)


def test_share_augmentation_binds_source_variants_to_one_group(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    flare = tmp_path / "flare"
    episodes = tmp_path / "share" / "episodes.jsonl"
    subprocess.run(
        [sys.executable, "scripts/generate_flare_datasets.py", "--scale", "0.02", "--out", str(flare), "--seed", "13"],
        check=True,
        cwd=repository,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/augment_share_synthetic.py",
            "--data", str(flare),
            "--out", str(episodes),
            "--seed", "31",
            "--failure-types", "planning_failure,parameter_binding_failure,answer_grounding_failure",
            "--limit-source-rows", "3",
        ],
        check=True,
        cwd=repository,
    )
    rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()]
    summary = json.loads((episodes.parent / "summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 9
    assert summary["synthetic"] is True
    assert summary["workflow_groups"] == 3
    grouped = {}
    for row in rows:
        source = row["episode_id"].split("__share_aug__", 1)[0]
        grouped.setdefault(source, set()).add((row["group_id"], row["split"]))
    assert all(len(values) == 1 for values in grouped.values())
    assert "not real paired authority outcomes" in (episodes.parent / "DATASET_CARD.md").read_text(encoding="utf-8")
