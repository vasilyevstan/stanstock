from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stanstock.data.management.config_loader import (
    config_hash,
    default_sec_cik_mapping_path,
    default_sec_fundamentals_config_path,
    load_yaml_mapping,
)
from stanstock.data.providers.sec import format_cik


@dataclass(frozen=True, slots=True)
class SecConceptRule:
    canonical_concept: str
    period_type: str
    units: tuple[str, ...]
    source_concepts: tuple[str, ...]
    explanation_only: bool


@dataclass(frozen=True, slots=True)
class SecFundamentalsConfig:
    config_version: str
    mapping_endpoint: str
    submissions_reconciliation_days: int
    companyfacts_lag_retry_days: int
    requests_per_second: int
    allowed_taxonomies: tuple[str, ...]
    allowed_forms: tuple[str, ...]
    concept_rules: tuple[SecConceptRule, ...]
    raw: dict[str, Any]
    config_hash: str

    @property
    def source_concept_rules(self) -> dict[tuple[str, str], SecConceptRule]:
        return {
            (taxonomy, source_concept): rule
            for taxonomy in self.allowed_taxonomies
            for rule in self.concept_rules
            for source_concept in rule.source_concepts
        }


@dataclass(frozen=True, slots=True)
class SecCikMapping:
    symbol: str
    cik: str
    official_ticker: str
    exchange: str
    company_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class SecCikConfig:
    config_version: str
    universe_config_version: str
    source_sha256: str
    mappings: dict[str, SecCikMapping]
    excluded: dict[str, str]
    raw: dict[str, Any]
    config_hash: str


def load_sec_fundamentals_config(
    path: Path | None = None,
) -> SecFundamentalsConfig:
    raw = load_yaml_mapping(path or default_sec_fundamentals_config_path())
    if _required_int(raw, "schema_version") != 1:
        raise ValueError("SEC fundamentals config schema_version must be 1")
    if _required_text(raw, "provider") != "sec":
        raise ValueError("SEC fundamentals config provider must be 'sec'")
    taxonomies = _required_text_list(raw, "allowed_taxonomies")
    forms = _required_text_list(raw, "allowed_forms")
    rules_raw = raw.get("concepts")
    if not isinstance(rules_raw, dict) or not rules_raw:
        raise ValueError("SEC fundamentals config requires a concepts mapping")
    rules: list[SecConceptRule] = []
    seen_sources: set[str] = set()
    for canonical, payload in rules_raw.items():
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError("SEC canonical concept names must be non-empty strings")
        if not isinstance(payload, dict):
            raise ValueError(f"SEC concept {canonical!r} must be a mapping")
        period_type = _required_text(payload, "period_type")
        if period_type not in {"instant", "duration", "unclassified"}:
            raise ValueError(f"SEC concept {canonical!r} has invalid period_type")
        units = _required_text_list(payload, "units")
        source_concepts = _required_text_list(payload, "source_concepts")
        for source_concept in source_concepts:
            if source_concept in seen_sources:
                raise ValueError(f"SEC source concept {source_concept!r} is mapped more than once")
            seen_sources.add(source_concept)
        explanation_only = payload.get("explanation_only", False)
        if not isinstance(explanation_only, bool):
            raise ValueError(f"SEC concept {canonical!r} explanation_only must be boolean")
        rules.append(
            SecConceptRule(
                canonical_concept=canonical.strip(),
                period_type=period_type,
                units=units,
                source_concepts=source_concepts,
                explanation_only=explanation_only,
            )
        )
    reconciliation_days = _required_int(raw, "submissions_reconciliation_days")
    lag_retry_days = _required_int(raw, "companyfacts_lag_retry_days")
    requests_per_second = _required_int(raw, "requests_per_second")
    if not 1 <= reconciliation_days <= 365:
        raise ValueError("submissions_reconciliation_days must be between 1 and 365")
    if not 1 <= lag_retry_days <= 30:
        raise ValueError("companyfacts_lag_retry_days must be between 1 and 30")
    if not 1 <= requests_per_second <= 10:
        raise ValueError("requests_per_second must be between 1 and 10")
    return SecFundamentalsConfig(
        config_version=_required_text(raw, "config_version"),
        mapping_endpoint=_required_text(raw, "mapping_endpoint"),
        submissions_reconciliation_days=reconciliation_days,
        companyfacts_lag_retry_days=lag_retry_days,
        requests_per_second=requests_per_second,
        allowed_taxonomies=taxonomies,
        allowed_forms=forms,
        concept_rules=tuple(rules),
        raw=raw,
        config_hash=config_hash(raw),
    )


def load_sec_cik_config(path: Path | None = None) -> SecCikConfig:
    raw = load_yaml_mapping(path or default_sec_cik_mapping_path())
    if _required_int(raw, "schema_version") != 1:
        raise ValueError("SEC CIK config schema_version must be 1")
    mappings_raw = raw.get("mappings")
    if not isinstance(mappings_raw, dict) or not mappings_raw:
        raise ValueError("SEC CIK config requires a mappings mapping")
    mappings: dict[str, SecCikMapping] = {}
    seen_ciks: set[str] = set()
    for raw_symbol, payload in mappings_raw.items():
        symbol = _normalize_symbol(raw_symbol)
        if not isinstance(payload, dict):
            raise ValueError(f"SEC CIK mapping for {symbol} must be a mapping")
        cik = format_cik(_required_text(payload, "cik"))
        if cik in seen_ciks:
            raise ValueError(f"SEC CIK config maps more than one symbol to CIK {cik}")
        seen_ciks.add(cik)
        mappings[symbol] = SecCikMapping(
            symbol=symbol,
            cik=cik,
            official_ticker=_normalize_symbol(payload.get("official_ticker", symbol)),
            exchange=_required_text(payload, "exchange"),
            company_name=_required_text(payload, "company_name"),
            reason=str(payload.get("reason") or "").strip(),
        )
    excluded_raw = raw.get("excluded", {})
    if not isinstance(excluded_raw, dict):
        raise ValueError("SEC CIK config excluded must be a mapping")
    excluded = {
        _normalize_symbol(symbol): _required_text({"reason": reason}, "reason")
        for symbol, reason in excluded_raw.items()
    }
    overlap = set(mappings) & set(excluded)
    if overlap:
        raise ValueError(f"SEC CIK symbols cannot be both mapped and excluded: {sorted(overlap)}")
    source_sha256 = _required_text(raw, "source_sha256")
    if len(source_sha256) != 64:
        raise ValueError("SEC CIK source_sha256 must be a 64-character SHA-256")
    return SecCikConfig(
        config_version=_required_text(raw, "config_version"),
        universe_config_version=_required_text(raw, "universe_config_version"),
        source_sha256=source_sha256,
        mappings=mappings,
        excluded=excluded,
        raw=raw,
        config_hash=config_hash(raw),
    )


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_text_list(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key} entries must be non-empty strings")
        normalized.append(item.strip())
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{key} contains duplicate values")
    return tuple(normalized)


def _normalize_symbol(value: object) -> str:
    symbol = str(value).strip().upper()
    if not symbol:
        raise ValueError("SEC mapping symbol must be non-empty")
    return symbol
