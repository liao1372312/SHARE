from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.prepare_share_swebench import prepare_rows
from scripts.collect_share_real_llm_executions import load_real_workflows
from scripts.create_share_swebench_pilot import select_explicit_instances, select_pilot
from scripts.archive_share_swebench_stop_result import validate_binding, validate_result
from scripts.generate_share_swebench_ai_patch import extract_json_object, read_instance
from scripts.pull_share_swebench_image import pull_image
from scripts.summarize_share_swebench_cross_backbones import sha256_file, verify_backbone_archive
from scripts.summarize_share_swebench_r002 import exact_mcnemar_pvalue, wilson_interval


def swebench_row(instance_id: str = "owner__repo-123") -> dict:
    return {
        "instance_id": instance_id,
        "repo": "owner/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix the failing behavior.",
        "patch": "diff --git a/code.py b/code.py\n+fixed = True\n",
        "test_patch": "diff --git a/test_code.py b/test_code.py\n+def test_fixed(): ...\n",
        "FAIL_TO_PASS": '["test_code.py::test_fixed"]',
        "PASS_TO_PASS": ["test_code.py::test_existing"],
        "version": "1.0",
        "environment_setup_commit": "def456",
    }


def test_prepare_rows_withholds_gold_patch_from_model_workflow() -> None:
    workflows, references = prepare_rows(
        [swebench_row()],
        dataset_name="princeton-nlp/SWE-bench_Verified",
        dataset_split="test",
        limit=None,
    )
    assert len(workflows) == len(references) == 1
    workflow = workflows[0]
    reference = references[0]
    assert workflow["workflow_id"] == reference["workflow_id"]
    assert workflow["metadata"]["gold_patch_withheld"] is True
    assert "patch" not in workflow["context"]
    assert reference["patch"] == swebench_row()["patch"]
    assert reference["execution_status"] == "not_executed"
    assert reference["human_minutes"] is None


def test_prepare_rows_rejects_duplicate_and_invalid_test_ids() -> None:
    with pytest.raises(ValueError, match="duplicate instance_id"):
        prepare_rows([swebench_row(), swebench_row()], dataset_name="test", dataset_split="test", limit=None)
    invalid = swebench_row()
    invalid["FAIL_TO_PASS"] = "not-json"
    with pytest.raises(ValueError, match="non-JSON FAIL_TO_PASS"):
        prepare_rows([invalid], dataset_name="test", dataset_split="test", limit=None)


def test_explicit_replacement_selection_is_bound_and_eligible() -> None:
    rows = [
        {"instance_id": "owner__one-1", "repo": "owner/one", "difficulty": "<15 min fix"},
        {"instance_id": "owner__two-2", "repo": "owner/two", "difficulty": "1-4 hour fix"},
    ]
    assert select_explicit_instances(rows, ["owner__one-1"])[0]["repo"] == "owner/one"
    with pytest.raises(ValueError, match="not an eligible"):
        select_explicit_instances(rows, ["owner__two-2"])
    with pytest.raises(ValueError, match="absent"):
        select_explicit_instances(rows, ["missing__repo-3"])


def test_ai_patch_input_requires_explicit_gold_withholding(tmp_path: Path) -> None:
    path = tmp_path / "workflows.jsonl"
    record = {
        "workflow_id": "owner__repo-123",
        "source": "swe-bench",
        "synthetic": False,
        "task": "Fix it",
        "context": {"repository": "owner/repo", "base_commit": "abc123"},
        "metadata": {"gold_patch_withheld": True},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert read_instance(path, record["workflow_id"])["task"] == "Fix it"
    record["metadata"]["gold_patch_withheld"] = False
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gold patch is withheld"):
        read_instance(path, record["workflow_id"])


def test_ai_patch_agent_accepts_plain_and_fenced_json_only() -> None:
    expected = {"action": "shell", "command": "sed -n '1,80p' code.py"}
    assert extract_json_object(json.dumps(expected)) == expected
    assert extract_json_object("```json\n" + json.dumps(expected) + "\n```") == expected
    with pytest.raises(ValueError, match="one JSON object"):
        extract_json_object("I would inspect code.py")


def test_ai_patch_hash_is_stable_for_empty_baseline() -> None:
    from scripts.generate_share_swebench_ai_patch import sha256_bytes

    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_cross_backbone_archive_verifies_identity_image_and_raw_hashes(tmp_path: Path) -> None:
    instance = "owner__repo-123"
    profile = "backbone_2"
    root = tmp_path / "observed_ai_backbones" / instance / profile
    raw = root / "raw" / "report.json"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"resolved": true}\n', encoding="utf-8")
    transcript = root / "raw" / "generation_transcript.jsonl"
    transcript.write_text('{"response":{"action":"replace"}}\n', encoding="utf-8")
    artifact = {
        "schema_version": 1,
        "kind": "share_swebench_ai_execution_artifact",
        "instance_id": instance,
        "branch": "ai",
        "profile": profile,
        "model": "model-b",
        "run_id": "run-r1",
        "resolved": True,
        "patch_successfully_applied": True,
        "evaluated_patch_exactly_matches_model_patch": True,
        "clean_tracked_generation_baseline_verified": True,
        "gold_patch_withheld_during_generation": True,
        "test_patch_withheld_during_generation": True,
        "image": {"id": "sha256:image"},
        "tests": {
            "fail_to_pass_success": 1,
            "fail_to_pass_failure": 0,
            "pass_to_pass_success": 2,
            "pass_to_pass_failure": 0,
        },
        "raw_files": {
            "report": {"file": "raw/report.json", "sha256": sha256_file(raw), "bytes": raw.stat().st_size},
            "generation_transcript.jsonl": {
                "file": "raw/generation_transcript.jsonl",
                "sha256": sha256_file(transcript),
                "bytes": transcript.stat().st_size,
            },
        },
    }
    (root / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    row = verify_backbone_archive(tmp_path, instance, profile, "sha256:image")
    assert row["resolved"] is True
    assert row["model"] == "model-b"
    assert row["transcript_action_counts"] == {"replace": 1}

    raw.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_backbone_archive(tmp_path, instance, profile, "sha256:image")


def test_image_pull_retries_same_source_and_reuses_docker_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    return_codes = iter((1, 1, 0))
    calls = []
    delays = []

    def fake_run(command):
        calls.append(command)
        return type("Result", (), {"returncode": next(return_codes)})()

    monkeypatch.setattr("scripts.pull_share_swebench_image.subprocess.run", fake_run)
    monkeypatch.setattr("scripts.pull_share_swebench_image.time.sleep", delays.append)
    assert pull_image("registry.example/image:latest", 3) is True
    assert calls == [["docker", "pull", "registry.example/image:latest"]] * 3
    assert delays == [3, 6]


def test_r002_paired_statistics_are_exact_and_bounded() -> None:
    assert exact_mcnemar_pvalue(3, 0) == pytest.approx(0.25)
    assert exact_mcnemar_pvalue(0, 5) == pytest.approx(0.0625)
    assert exact_mcnemar_pvalue(0, 0) == 1.0
    low, high = wilson_interval(5, 11)
    assert low == pytest.approx(0.21271, abs=1e-5)
    assert high == pytest.approx(0.71991, abs=1e-5)


def test_stop_validation_requires_expected_unresolved_test_pattern() -> None:
    instance = "owner__repo-123"
    binding = {
        "schema_version": 1,
        "kind": "share_swebench_stop_run_binding",
        "instance_id": instance,
        "run_id": "stop-r1",
        "semantics": "base_commit_tests_without_solution_patch",
        "solution_patch_applied": False,
        "harness_skip_patch": True,
        "resolved": False,
    }
    validate_binding(binding, instance, "stop-r1")
    summary = {
        "schema_version": 2,
        "submitted_ids": [instance],
        "completed_ids": [instance],
        "resolved_ids": [],
        "unresolved_ids": [instance],
        "error_ids": [],
        "infra_failure_ids": [],
        "ambiguous_failure_ids": [],
    }
    detail = {
        instance: {
            "resolved": False,
            "infra_failure": False,
            "tests_status": {
                "FAIL_TO_PASS": {"success": [], "failure": ["test_target"]},
                "PASS_TO_PASS": {"success": ["test_existing"], "failure": []},
            },
        }
    }
    assert validate_result(summary, detail, instance)["resolved"] is False
    detail[instance]["tests_status"]["PASS_TO_PASS"]["failure"] = ["test_existing"]
    with pytest.raises(ValueError, match="regressed"):
        validate_result(summary, detail, instance)


def test_prepare_cli_writes_bound_staging_files_and_refuses_overwrite(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "swebench.jsonl"
    source.write_text(json.dumps(swebench_row()) + "\n", encoding="utf-8")
    output = tmp_path / "staging"
    command = [
        sys.executable,
        "scripts/prepare_share_swebench.py",
        "--input",
        str(source),
        "--out",
        str(output),
    ]
    subprocess.run(command, cwd=repository, check=True)
    report = json.loads((output / "preparation_report.json").read_text(encoding="utf-8"))
    workflow = json.loads((output / "workflows.jsonl").read_text(encoding="utf-8"))
    assert report["selected_tasks"] == 1
    assert report["gold_patch_leakage_check"] is True
    assert report["evidence_status"] == "staging_only_not_paired_outcomes"
    assert report["source_fingerprint"]
    assert workflow["task"] == "Fix the failing behavior."
    loaded = load_real_workflows(output / "workflows.jsonl")
    assert loaded[0]["workflow_id"] == workflow["workflow_id"]
    repeated = subprocess.run(command, cwd=repository, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr


def test_prepare_cli_rejects_insecure_hub_endpoint(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_share_swebench.py",
            "--hf-endpoint",
            "http://untrusted.example",
            "--out",
            str(tmp_path / "staging"),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "must use HTTPS" in result.stderr


def test_prepare_cli_accepts_local_parquet(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "swebench.parquet"
    parquet.write_table(pa.Table.from_pylist([swebench_row()]), source)
    output = tmp_path / "parquet-staging"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_share_swebench.py",
            "--input",
            str(source),
            "--out",
            str(output),
        ],
        cwd=repository,
        check=True,
    )
    report = json.loads((output / "preparation_report.json").read_text(encoding="utf-8"))
    assert report["selected_tasks"] == 1
    assert report["source_kind"] == "local_export"
    assert report["hub_endpoint"] is None


def test_prepare_local_input_records_declared_source_provenance(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "swebench.jsonl"
    source.write_text(json.dumps(swebench_row()) + "\n", encoding="utf-8")
    output = tmp_path / "provenance-staging"
    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_share_swebench.py",
            "--input",
            str(source),
            "--source-dataset",
            "SWE-bench/SWE-bench_Verified",
            "--source-revision",
            "revision-123",
            "--out",
            str(output),
        ],
        cwd=repository,
        check=True,
    )
    report = json.loads((output / "preparation_report.json").read_text(encoding="utf-8"))
    workflow = json.loads((output / "workflows.jsonl").read_text(encoding="utf-8"))
    assert report["dataset"] == "SWE-bench/SWE-bench_Verified"
    assert report["source_revision"] == "revision-123"
    assert workflow["metadata"]["source_binding"]["dataset_revision"] == "revision-123"


def test_pilot_selection_is_easy_cross_repository_and_deterministic() -> None:
    rows = [
        {"instance_id": "repo-b-2", "repo": "owner/repo-b", "difficulty": "<15 min fix"},
        {"instance_id": "repo-a-2", "repo": "owner/repo-a", "difficulty": "<15 min fix"},
        {"instance_id": "repo-a-1", "repo": "owner/repo-a", "difficulty": "<15 min fix"},
        {"instance_id": "repo-c-1", "repo": "owner/repo-c", "difficulty": "1-4 hours"},
    ]
    selected = select_pilot(rows, 2)
    assert [row["instance_id"] for row in selected] == ["repo-a-1", "repo-b-2"]
    with pytest.raises(ValueError, match="only 2 repositories"):
        select_pilot(rows, 3)
