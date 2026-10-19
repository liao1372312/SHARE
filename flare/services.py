from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass
class ParameterSpec:
    name: str
    required: bool = False
    type: str = "str"
    default: Any = None
    description: str = ""


@dataclass
class ServiceSpec:
    record_id: str
    api_name: str
    description: str = ""
    domain: str = ""
    category: str = ""
    endpoint: str = ""
    method: str = "POST"
    parameters: List[ParameterSpec] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    executable: bool = True

    @property
    def required_parameters(self) -> List[str]:
        return [parameter.name for parameter in self.parameters if parameter.required]


class ServiceRepository:
    def __init__(self, services: Iterable[ServiceSpec] = ()): 
        self.services: Dict[str, ServiceSpec] = {service.record_id: service for service in services}

    def add(self, service: ServiceSpec) -> None:
        self.services[service.record_id] = service

    def get(self, record_id: str) -> ServiceSpec:
        return self.services[record_id]

    def select(self, subtask: str, domain: str = "", exclude: Sequence[str] = ()) -> Optional[ServiceSpec]:
        excluded = set(exclude)
        candidates = [service for service in self.services.values() if service.record_id not in excluded]
        if not candidates:
            return None
        query_terms = _terms(f"{subtask} {domain}")
        ranked = sorted(candidates, key=lambda service: (_score_service(service, query_terms, domain), service.record_id), reverse=True)
        return ranked[0] if _score_service(ranked[0], query_terms, domain) > 0 else ranked[0]

    @classmethod
    def from_jsonl(cls, path: str | Path, limit: int | None = None) -> "ServiceRepository":
        services: List[ServiceSpec] = []
        with Path(path).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if limit is not None and index >= limit:
                    break
                if line.strip():
                    services.append(service_from_mapping(json.loads(line)))
        return cls(services)

    @classmethod
    def builtin(cls) -> "ServiceRepository":
        return cls(
            [
                ServiceSpec(
                    record_id="weather.current",
                    api_name="Current Weather API",
                    description="Get current weather for a city or location",
                    domain="weather",
                    category="Weather",
                    endpoint="/weather/current",
                    method="GET",
                    parameters=[ParameterSpec("city", required=True), ParameterSpec("unit", required=False, default="metric")],
                    outputs=["city", "condition", "temperature"],
                ),
                ServiceSpec(
                    record_id="weather.forecast",
                    api_name="Weather Forecast API",
                    description="Get multi-day weather forecast for a city",
                    domain="weather",
                    category="Weather",
                    endpoint="/weather/forecast",
                    method="GET",
                    parameters=[ParameterSpec("city", required=True), ParameterSpec("days", required=False, default=3)],
                    outputs=["city", "forecast"],
                ),
                ServiceSpec(
                    record_id="finance.quote",
                    api_name="Stock Quote API",
                    description="Get latest quote for a stock symbol",
                    domain="finance",
                    category="Finance",
                    endpoint="/finance/quote",
                    method="GET",
                    parameters=[ParameterSpec("symbol", required=True)],
                    outputs=["symbol", "price", "currency"],
                ),
                ServiceSpec(
                    record_id="translation.translate",
                    api_name="Translation API",
                    description="Translate text into a target language",
                    domain="translation",
                    category="Translation",
                    endpoint="/translate",
                    method="POST",
                    parameters=[ParameterSpec("text", required=True), ParameterSpec("target_language", required=True)],
                    outputs=["translated_text"],
                ),
            ]
        )


def service_from_mapping(item: Mapping[str, Any]) -> ServiceSpec:
    raw_parameters = item.get("parameters") or item.get("required_parameters") or []
    required_names = set(item.get("required_parameters") or [])
    parameters: List[ParameterSpec] = []
    for parameter in raw_parameters:
        if isinstance(parameter, str):
            parameters.append(ParameterSpec(parameter, required=parameter in required_names))
        elif isinstance(parameter, Mapping):
            name = str(parameter.get("name") or parameter.get("key") or "")
            if name:
                parameters.append(
                    ParameterSpec(
                        name=name,
                        required=bool(parameter.get("required", name in required_names)),
                        type=str(parameter.get("type") or "str"),
                        default=parameter.get("default"),
                        description=str(parameter.get("description") or ""),
                    )
                )
    outputs = []
    for output in item.get("outputs") or []:
        outputs.append(str(output.get("name") if isinstance(output, Mapping) else output))
    return ServiceSpec(
        record_id=str(item.get("record_id") or item.get("id") or item.get("name") or item.get("api_name")),
        api_name=str(item.get("api_name") or item.get("name") or item.get("record_id")),
        description=str(item.get("description") or item.get("desc") or ""),
        domain=str(item.get("domain") or ""),
        category=str(item.get("category") or ""),
        endpoint=str(item.get("endpoint") or item.get("path") or ""),
        method=str(item.get("method") or "POST"),
        parameters=parameters,
        outputs=outputs,
        executable=bool(item.get("executable", True)),
    )


def extract_parameters(text: str) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    city = _find_after(text, ["city", "城市", "location", "地点"])
    if city:
        values["city"] = city
    symbol = _find_after(text, ["symbol", "ticker", "股票"])
    if symbol:
        values["symbol"] = symbol.upper()
    target = _find_after(text, ["target language", "目标语言", "translate to", "翻译成"])
    if target:
        values["target_language"] = target
    query = _find_after(text, ["query", "查询", "检索"])
    if query:
        values["query"] = query
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
    if quoted:
        values.setdefault("text", quoted[0])
    if "Beijing" in text or "北京" in text:
        values.setdefault("city", "Beijing")
    if "Shanghai" in text or "上海" in text:
        values.setdefault("city", "Shanghai")
    if "AAPL" in text.upper():
        values.setdefault("symbol", "AAPL")
    return values


def bind_parameters(service: ServiceSpec, task_text: str, prior_outputs: Sequence[Mapping[str, Any]] = ()) -> tuple[Dict[str, Any], List[str]]:
    values = extract_parameters(task_text)
    for output in prior_outputs:
        for key, value in output.items():
            values.setdefault(key, value)
    bound: Dict[str, Any] = {}
    missing: List[str] = []
    for parameter in service.parameters:
        if parameter.name in values and values[parameter.name] not in (None, ""):
            bound[parameter.name] = values[parameter.name]
        elif parameter.default is not None:
            bound[parameter.name] = parameter.default
        elif parameter.required:
            missing.append(parameter.name)
    return bound, missing


def invoke_service(service: ServiceSpec, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [name for name in service.required_parameters if not parameters.get(name)]
    if missing:
        return {"ok": False, "status_code": 400, "error": f"missing required parameters: {', '.join(missing)}"}
    if not service.executable:
        return {"ok": False, "status_code": 503, "error": "service unavailable"}
    if service.record_id.startswith("weather.current"):
        city = parameters.get("city", "Unknown")
        return {"ok": True, "status_code": 200, "city": city, "condition": "sunny", "temperature": 22, "unit": parameters.get("unit", "metric")}
    if service.record_id.startswith("weather.forecast"):
        city = parameters.get("city", "Unknown")
        return {"ok": True, "status_code": 200, "city": city, "forecast": ["sunny", "cloudy", "rain"][: int(parameters.get("days", 3))]}
    if service.record_id.startswith("finance.quote"):
        symbol = parameters.get("symbol", "UNKNOWN")
        return {"ok": True, "status_code": 200, "symbol": symbol, "price": 123.45, "currency": "USD"}
    if service.record_id.startswith("translation.translate"):
        return {"ok": True, "status_code": 200, "translated_text": f"translated({parameters.get('text')})", "target_language": parameters.get("target_language")}
    payload = {name: parameters.get(name, f"{name}_value") for name in service.outputs}
    payload.update({"ok": True, "status_code": 200, "api": service.api_name})
    return payload


def _score_service(service: ServiceSpec, query_terms: set[str], domain: str) -> float:
    haystack = _terms(f"{service.api_name} {service.description} {service.domain} {service.category}")
    score = len(query_terms & haystack)
    if domain and domain.lower() in f"{service.domain} {service.category}".lower():
        score += 3
    return float(score)


def _terms(text: str) -> set[str]:
    return {term.lower() for term in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text)}


def _find_after(text: str, keys: Sequence[str]) -> str | None:
    for key in keys:
        pattern = re.compile(rf"{re.escape(key)}\s*[:=为是]?\s*([A-Za-z0-9_\-\u4e00-\u9fff]+)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if value.lower() not in {key.lower(), "is", "为", "是"}:
                return value
    return None
