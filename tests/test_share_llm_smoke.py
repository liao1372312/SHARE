from __future__ import annotations

import importlib.util
import json
import socket
import ssl
from pathlib import Path

from share.llm_profiles import LLMProfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("share_llm_smoke", ROOT / "scripts" / "smoke_share_llm_profiles.py")
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_smoke_payload_carries_model_and_reasoning_effort():
    profile = LLMProfile("backbone_2", "https://example.test/v1", "test-key", "gpt-5.6-luna", "low")

    payload = smoke.request_payload(profile, "probe", 128)

    assert payload["model"] == "gpt-5.6-luna"
    assert payload["reasoning_effort"] == "low"
    assert payload["max_tokens"] == 128


def test_smoke_profile_selection_limits_calls_to_named_profile():
    profiles = (
        LLMProfile("primary", "https://example.test/v1", "key-1", "primary-model"),
        LLMProfile("backbone_2", "https://example.test/v1", "key-2", "second-model"),
        LLMProfile("backbone_3", "https://example.test/v1", "key-3", "third-model"),
    )

    selected = smoke.select_profiles(profiles, ["backbone_3"])

    assert [profile.name for profile in selected] == ["backbone_3"]


def test_smoke_profile_skips_second_probe_after_failure(monkeypatch):
    profile = LLMProfile("primary", "https://example.test/v1", "test-key", "model")
    calls = []

    def failed_probe(*_args, **_kwargs):
        calls.append(1)
        return {"probe": "connectivity", "status": "error", "error_kind": "http_error", "http_status": 401, "latency_ms": 1}

    monkeypatch.setattr(smoke, "run_probe", failed_probe)

    result = smoke.profile_result(profile, requests_per_profile=2, max_tokens=128, timeout_seconds=1)

    assert len(calls) == 1
    assert result["successful_probes"] == 0
    assert result["probes"][1] == {"probe": "structured_response", "status": "skipped", "reason": "previous_probe_failed"}


def test_smoke_json_response_requires_valid_json():
    assert smoke.json_response('{"status": "ok"}')
    assert not smoke.json_response("not json")
    assert not smoke.json_response(None)


def test_smoke_classifies_transport_errors_without_recording_endpoint_details():
    assert smoke.transport_error_kind(ssl.SSLCertVerificationError()) == "tls_certificate_error"
    assert smoke.transport_error_kind(socket.gaierror()) == "dns_error"
    assert smoke.transport_error_kind(ConnectionRefusedError()) == "connection_refused"
    assert smoke.transport_error_kind(TimeoutError()) == "timeout"


def test_smoke_report_does_not_store_response_content(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "LLM_BASE_URL=https://example.test/v1",
                "LLM_API_KEY=primary-key",
                "LLM_MODEL=primary-model",
                "LLM_BACKBONE_2_BASE_URL=https://example.test/v1",
                "LLM_BACKBONE_2_API_KEY=second-key",
                "LLM_BACKBONE_2_MODEL=second-model",
                "LLM_BACKBONE_3_BASE_URL=https://example.test/v1",
                "LLM_BACKBONE_3_API_KEY=third-key",
                "LLM_BACKBONE_3_MODEL=third-model",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    monkeypatch.setattr(smoke, "parse_args", lambda: type("Args", (), {"env": env, "out": out, "requests_per_profile": 1, "max_tokens": 128, "timeout_seconds": 1, "overwrite": False, "profile": None})())
    monkeypatch.setattr(
        smoke,
        "run_probe",
        lambda *_args, **_kwargs: {
            "probe": "connectivity",
            "status": "success",
            "error_kind": None,
            "latency_ms": 1,
            "response_id_configured": True,
            "content_is_json": True,
            "content_sha256": "a" * 64,
            "usage": {},
        },
    )

    smoke.main()

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    encoded = json.dumps(report)
    assert report["successful_probes"] == 3
    assert "primary-key" not in encoded
    assert "second-key" not in encoded
    assert "third-key" not in encoded
    assert "example.test" not in encoded
