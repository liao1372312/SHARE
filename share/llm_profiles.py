from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


VALID_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})


def is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class LLMProfile:
    name: str
    base_url: str
    api_key: str = field(repr=False)
    model: str
    reasoning_effort: str | None = None

    def public_summary(self) -> dict[str, object]:
        parsed = urlparse(self.base_url)
        return {
            "name": self.name,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "base_url_scheme": parsed.scheme,
            "base_url_configured": bool(self.base_url),
            "api_key_configured": bool(self.api_key),
        }


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_llm_profiles(path: Path, *, allow_insecure_http: bool = False) -> tuple[LLMProfile, ...]:
    values = read_dotenv(path)
    layouts = (
        ("primary", "LLM"),
        ("backbone_2", "LLM_BACKBONE_2"),
        ("backbone_3", "LLM_BACKBONE_3"),
    )
    return tuple(_load_profile(values, name=name, prefix=prefix, allow_insecure_http=allow_insecure_http) for name, prefix in layouts)


def _load_profile(
    values: Mapping[str, str],
    *,
    name: str,
    prefix: str,
    allow_insecure_http: bool = False,
) -> LLMProfile:
    base_url = values.get(f"{prefix}_BASE_URL", "")
    api_key = values.get(f"{prefix}_API_KEY", "")
    model = values.get(f"{prefix}_MODEL", "")
    effort = values.get(f"{prefix}_REASONING_EFFORT", "").lower() or None
    missing = [field for field, value in (("BASE_URL", base_url), ("API_KEY", api_key), ("MODEL", model)) if not value]
    if missing:
        raise ValueError(f"LLM profile {name} is missing {', '.join(missing)}")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"LLM profile {name} has an invalid BASE_URL")
    if parsed.scheme == "http" and not allow_insecure_http:
        raise ValueError(f"LLM profile {name} requires HTTPS; use an explicit insecure-HTTP acknowledgement only for a trusted loopback proxy")
    if parsed.scheme == "http" and not is_loopback_host(parsed.hostname):
        raise ValueError(f"LLM profile {name} only permits insecure HTTP for a loopback proxy")
    if parsed.path.rstrip("/").endswith(("/chat/completions", "/responses")):
        raise ValueError(f"LLM profile {name} BASE_URL must be an API root, not a terminal endpoint")
    if effort is not None and effort not in VALID_REASONING_EFFORTS:
        raise ValueError(f"LLM profile {name} has unsupported REASONING_EFFORT: {effort}")
    return LLMProfile(name, base_url, api_key, model, effort)


def load_annotation_profile(path: Path) -> LLMProfile:
    """Load the independent LLM used only to prefill expert-review defaults.

    This profile deliberately has a separate namespace from the three evaluated
    proposal generators, so a study can disclose and audit the assistance path.
    """
    return _load_profile(read_dotenv(path), name="annotation_judge", prefix="LLM_ANNOTATION")


def public_profile_report(path: Path, *, allow_insecure_http: bool = False) -> Mapping[str, object]:
    profiles = load_llm_profiles(path, allow_insecure_http=allow_insecure_http)
    return {
        "schema_version": 1,
        "kind": "share_llm_profile_preflight",
        "profiles": [profile.public_summary() for profile in profiles],
        "credentials_configured": True,
        "insecure_http_acknowledged": allow_insecure_http,
        "warning": "This local preflight does not connect to an LLM endpoint or verify credentials. HTTPS is required unless insecure HTTP was explicitly acknowledged for a trusted local proxy.",
    }
