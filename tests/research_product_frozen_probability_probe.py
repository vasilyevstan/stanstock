"""Emit one complete `us-price-fhs-v1` frozen-surface payload for one source tree.

`tests/test_research_product_frozen_probability_contract.py` executes this
module twice -- once against the exact frozen base revision exported from Git,
once against the current working tree -- in two separate subprocesses, then
compares the two emitted payloads.  Two processes are required because the two
revisions define the *same* dotted module names; one interpreter cannot hold
both.

Determinism
-----------
A byte-for-byte comparison of complete payloads is only meaningful when every
identity, clock, and revision is controlled *identically* in both processes,
so nothing has to be masked afterwards:

* ``uuid.uuid4`` is replaced with a deterministic sequence **before**
  ``django.setup()``, so every ``default=uuid.uuid4`` primary key and every
  explicit ``uuid4()`` issuance identity is reproducible.
* ``JobRun`` and ``DataAsset`` primary keys are instead bound to their own
  natural keys, and their field defaults are neutralized so they consume
  nothing from that sequence.  The candidate revision registers one additional
  frequency ``DataAsset`` and runs one additional ``JobRun`` child; a plain
  counter would shift every later identity and turn additive work into a fake
  frozen-contract break.
* ``django.utils.timezone.now`` is frozen, and the execution revision is a
  declared synthetic constant supplied through ``STANSTOCK_CODE_REVISION`` and
  a matching ``clean_git_revision`` binding, so no Git state leaks in.

Nothing else is normalized.  The frozen writer, reader, math, RNG, seeds, and
manifest bytes all execute for real.

Boundaries
----------
The probe never contacts a provider or the network: outbound sockets are
blocked, provider credential resolution and every fetch entry point raise, and
the price history is generated locally by the repository's own synthetic
`twelve_data` parsers.  All state lives in a caller-supplied disposable
directory.

Usage::

    python tests/research_product_frozen_probability_probe.py \
        --mode daily --workdir <dir> --output <file.json>

``--mode scheduled_recovery`` is different in kind: it constructs nothing and
instead reuses a working directory that an *earlier* probe process left
behind, so one revision's canonical replay and skip-recovery can be executed
against another revision's genuinely retained database, assets, jobs, and
immutable rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

#: Fixed namespace so every controlled identity is identical in both
#: processes and on every machine.
FIXTURE_NAMESPACE = uuid.UUID("7c0f5e4a-0000-4000-8000-5354414e5354")

#: The frozen product fixture's clocks and target, matching the repository's
#: existing `us-price-fhs-v1` product fixtures.
NOW = datetime(2026, 9, 12, 1, tzinfo=UTC)
TARGET = date(2026, 9, 11)

#: Declared synthetic execution revision. Both processes use it, so neither
#: the exported base tree nor the dirty working tree consults Git.
SYNTHETIC_REVISION = "a" * 40

OWNER_USERNAME = "frozen-probability-probe-owner"

#: A qualified saved listing whose history is valid, complete, and strictly
#: positive but perfectly flat, so the frozen filter reaches
#: `filter_variance_degenerate` and the frozen forecast is genuinely withheld.
WITHHELD_SYMBOL = "FLAT"
#: A qualified saved listing that produces the successful frozen payload.
SUCCESSFUL_SYMBOL = "CHEAP"

MODES = ("daily", "daily_derived", "scheduled", "scheduled_recovery")

#: ``full`` qualifies both the successful and the genuinely withheld listing.
#: ``successful`` omits the withheld listing so a surface that a revision
#: cannot currently execute for withheld evidence can still be compared on the
#: evidence it does support, without weakening the ``full`` comparison.
COHORTS = ("full", "successful")


class ProbeError(RuntimeError):
    """The probe could not produce a complete frozen-surface payload."""


# --------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------


def _install_boundary_guards() -> None:
    """Refuse network access and provider credentials inside the probe."""

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise ProbeError("The frozen-contract probe must never open a network connection")

    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    for name in list(os.environ):
        upper = name.upper()
        if any(marker in upper for marker in ("API_KEY", "TOKEN", "CREDENTIAL", "PASSWORD")):
            del os.environ[name]


# --------------------------------------------------------------------------
# Controlled identities
# --------------------------------------------------------------------------


class _DeterministicUuid:
    """A reproducible stand-in for ``uuid.uuid4`` within one probe process."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._index = 0

    def __call__(self) -> uuid.UUID:
        self._index += 1
        return uuid.uuid5(FIXTURE_NAMESPACE, f"{self._label}:{self._index}")


def _natural_identity(label: str, *parts: object) -> uuid.UUID:
    return uuid.uuid5(FIXTURE_NAMESPACE, "|".join([label, *(str(part) for part in parts)]))


_UNASSIGNED = uuid.uuid5(FIXTURE_NAMESPACE, "natural-key-placeholder")


def _neutralize_pk_default(model: Any) -> None:
    """Stop a natural-key model from consuming the shared uuid4 sequence."""

    field = model._meta.pk
    field.default = lambda: _UNASSIGNED
    field.__dict__.pop("_get_default", None)


def _bind_natural_key_identities() -> None:
    from django.db.models.signals import pre_save

    from stanstock.core.models import JobRun
    from stanstock.data.models import DataAsset

    occurrences: dict[tuple[object, ...], int] = {}

    def _job_identity(sender: Any, instance: Any, **_kwargs: object) -> None:
        if not instance._state.adding:
            return
        instance.pk = _natural_identity(
            "JobRun",
            instance.job_name,
            instance.region,
            instance.target_date,
            instance.attempt,
        )

    def _asset_identity(sender: Any, instance: Any, **_kwargs: object) -> None:
        if not instance._state.adding:
            return
        key = (
            instance.provider,
            instance.kind,
            instance.subject,
            instance.sha256,
            instance.relative_path,
        )
        index = occurrences.get(key, 0)
        occurrences[key] = index + 1
        instance.pk = _natural_identity("DataAsset", *key, index)

    _neutralize_pk_default(JobRun)
    _neutralize_pk_default(DataAsset)
    pre_save.connect(_job_identity, sender=JobRun, weak=False)
    pre_save.connect(_asset_identity, sender=DataAsset, weak=False)


def _freeze_clock() -> None:
    from django.utils import timezone

    timezone.now = lambda: NOW  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Payload serialization
# --------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return f"Decimal:{value}"
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes | bytearray | memoryview):
        return f"sha256:{hashlib.sha256(bytes(value)).hexdigest()}"
    if isinstance(value, float):
        return f"float:{value!r}"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _row(instance: Any) -> dict[str, Any]:
    """Serialize every concrete field of one row -- nothing is dropped."""

    return {
        field.attname: _jsonable(getattr(instance, field.attname))
        for field in instance._meta.concrete_fields
    }


# --------------------------------------------------------------------------
# Synthetic source construction
# --------------------------------------------------------------------------


def _synthetic_series(symbol: str, *, closes: int = 757) -> Any:
    """Build one deterministic synthetic provider payload for `symbol`.

    `FLAT` is a valid, complete, strictly positive history with zero
    variance, which is what drives the frozen filter into a genuine
    withholding reason rather than mocking the writer or the math.
    """
    from exchange_calendars import get_calendar

    from stanstock.data.providers import twelve_data

    calendar = get_calendar("XNYS")
    sessions = calendar.sessions_window(calendar.date_to_session(TARGET), -closes)
    base = 5.0 if symbol == SUCCESSFUL_SYMBOL else 80.0 if symbol == "SPY" else 20.0
    values = []
    for index, session in enumerate(sessions):
        if symbol == WITHHELD_SYMBOL:
            value = 25.0
        else:
            value = base * math.exp(0.0002 * index + 0.007 * math.sin(index / 11))
        values.append(
            {
                "datetime": session.date().isoformat(),
                "open": str(value),
                "high": str(value),
                "low": str(value),
                "close": str(value),
                "volume": "1000000",
            }
        )
    payload = {
        "status": "ok",
        "meta": {
            "symbol": symbol,
            "interval": "1day",
            "currency": "USD",
            "exchange": "NYSE ARCA" if symbol == "SPY" else "NASDAQ",
            "mic_code": "ARCX" if symbol == "SPY" else "XNAS",
            "type": "ETF" if symbol == "SPY" else "Common Stock",
        },
        "values": values,
    }
    return twelve_data.parse_daily_price_series(
        json.dumps(payload).encode(),
        symbol=symbol,
        retrieved_at=NOW,
        source_url="https://api.twelvedata.com/time_series",
        end_date=TARGET,
    )


def _build_environment(store: Any, *, cohort: str) -> tuple[Any, Any]:
    """Create the identical synthetic owner, catalog, and price evidence."""
    from django.contrib.auth import get_user_model

    from stanstock.data.live_us import (
        _persist_catalog,
        _persist_price_series,
        load_us_universe_config,
    )
    from stanstock.data.management.config_loader import default_us_universe_config_path
    from stanstock.data.models import ProviderRecord
    from stanstock.data.provider_policy import PRIVATE_USAGE_SCOPE
    from stanstock.data.providers import twelve_data
    from stanstock.data.providers.contracts import StockCatalog
    from stanstock.portfolio.models import TrackedSymbol

    owner = get_user_model().objects.create_user(username=OWNER_USERNAME, date_joined=NOW)
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        terms_url="https://twelvedata.com/terms",
        usage_scope=PRIVATE_USAGE_SCOPE,
        metadata={
            "internal_display_rights_confirmed": True,
            "plan": "grow",
            "daily_credit_limit": 800,
            "credits_per_minute": 8,
        },
    )
    config_path = default_us_universe_config_path()
    config = load_us_universe_config(config_path)
    saved_symbols = (
        (SUCCESSFUL_SYMBOL, WITHHELD_SYMBOL) if cohort == "full" else (SUCCESSFUL_SYMBOL,)
    )
    rows = [
        {
            "symbol": symbol,
            "name": f"Synthetic {symbol}",
            "currency": "USD",
            "exchange": "NASDAQ",
            "mic_code": "XNAS",
            "country": "United States",
            "type": ("Common Stock" if symbol in {"AAPL", "MSFT", *saved_symbols} else "ETF"),
        }
        for symbol in (*config.symbols, *saved_symbols)
    ]
    nyse_rows = [
        {
            "symbol": "VENUE",
            "name": "Synthetic venue",
            "currency": "USD",
            "exchange": "NYSE",
            "mic_code": "XNYS",
            "country": "United States",
            "type": "Common Stock",
        }
    ]
    for exchange, catalog_rows in (("NASDAQ", rows), ("NYSE", nyse_rows)):
        raw_catalog = json.dumps(
            {"status": "ok", "count": len(catalog_rows), "data": catalog_rows}
        ).encode()
        references, count = twelve_data.parse_stock_catalog_references(
            raw_catalog, exchange=exchange, require_complete=True
        )
        _persist_catalog(
            store,
            StockCatalog(
                provider="twelve_data",
                exchange=exchange,
                references=references,
                count=count,
                retrieved_at=NOW,
                source_url=f"https://api.twelvedata.com/stocks?exchange={exchange}",
                raw_bytes=raw_catalog,
            ),
        )
    for symbol in ("AAPL", "MSFT", "SPY", *saved_symbols):
        _persist_price_series(store=store, series=_synthetic_series(symbol), listing=None)
    for symbol in saved_symbols:
        TrackedSymbol.objects.create(owner=owner, symbol=symbol)

    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise ProbeError("The frozen-contract probe must never reach a provider")

    twelve_data.resolve_api_key = _refuse  # type: ignore[assignment]
    twelve_data.fetch_daily_price_series = _refuse  # type: ignore[assignment]
    twelve_data.fetch_stock_catalog = _refuse  # type: ignore[assignment]
    return owner, config_path


# --------------------------------------------------------------------------
# Frozen-surface capture
# --------------------------------------------------------------------------


def _config_payload() -> dict[str, Any]:
    from dataclasses import asdict

    from stanstock.research import price_product_config as module

    path = module.default_price_product_config_path()
    raw = path.read_bytes()
    config = module.load_price_product_config(path)
    return {
        "config_file_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "config_file_text": raw.decode("utf-8"),
        "declared_config_file_sha256": module.PRODUCT_CONFIG_FILE_SHA256,
        "declared_effective_config_hash": module.PRODUCT_EFFECTIVE_CONFIG_HASH,
        "recomputed_effective_config_hash": module.price_product_config_hash(config),
        "product_version": module.PRODUCT_VERSION,
        "fhs_method_version": module.FHS_METHOD_VERSION,
        "momentum_method_version": module.MOMENTUM_METHOD_VERSION,
        "payload_schema": module.PRODUCT_PAYLOAD_SCHEMA,
        "typed_config": _jsonable(asdict(config)),
    }


def _asset_key(asset: Any) -> str:
    return f"{asset.kind}|{asset.subject}|{asset.sha256}"


def _document_assets(store: Any) -> dict[str, Any]:
    """Serialize every registered asset row and every JSON document body."""
    from stanstock.data.models import DataAsset

    documents: dict[str, Any] = {}
    for asset in DataAsset.objects.order_by("kind", "subject", "sha256"):
        payload: dict[str, Any] = {"row": _row(asset)}
        raw = store.read_bytes(asset.relative_path)
        payload["stored_bytes_sha256"] = hashlib.sha256(raw).hexdigest()
        payload["stored_byte_count"] = len(raw)
        if asset.relative_path.endswith(".json"):
            payload["document"] = _jsonable(json.loads(raw.decode("utf-8")))
        documents[_asset_key(asset)] = payload
    return documents


def _prediction_key(prediction: Any) -> str:
    return (
        f"{prediction.listing.provider_symbol}|"
        f"{prediction.evidence_role}|{prediction.horizon}|{prediction.method_version}"
    )


def _frozen_output(*, run: Any, store: Any) -> dict[str, Any]:
    from stanstock.data.research_product import product_membership_payload
    from stanstock.research.models import Prediction, StockAnalysis
    from stanstock.research.price_product import deterministic_seed
    from stanstock.research.price_product_config import (
        FHS_METHOD_VERSION,
        PRODUCT_EFFECTIVE_CONFIG_HASH,
    )
    from stanstock.research.product_pipeline import verify_price_product_output

    analyses = list(StockAnalysis.objects.filter(run=run).select_related("listing"))
    predictions = list(
        Prediction.objects.filter(analysis__run=run).select_related("listing", "analysis")
    )
    verify_price_product_output(run=run, store=store, replay=True)
    return {
        "analysis_run": _row(run),
        "universe_snapshot": _row(run.universe_snapshot),
        "stock_analyses": {
            analysis.listing.provider_symbol: _row(analysis) for analysis in analyses
        },
        "predictions": {
            _prediction_key(prediction): _row(prediction) for prediction in predictions
        },
        "prediction_count": len(predictions),
        "membership_payload": _jsonable(
            product_membership_payload(run.universe_snapshot, store=store)
        ),
        "deterministic_seeds": {
            analysis.listing.provider_symbol: deterministic_seed(
                method_version=FHS_METHOD_VERSION,
                effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
                listing_id=analysis.listing_id,
                target_date=run.target_date,
            )
            for analysis in analyses
        },
        "output_verification": "verified",
    }


def _require_expected_cohort(output: dict[str, Any], *, cohort: str) -> None:
    """Fail loudly unless the fixture really produced both frozen states."""
    symbols = set(output["stock_analyses"])
    expected = {SUCCESSFUL_SYMBOL, WITHHELD_SYMBOL} if cohort == "full" else {SUCCESSFUL_SYMBOL}
    if not expected <= symbols or (cohort != "full" and WITHHELD_SYMBOL in symbols):
        raise ProbeError(
            f"The frozen {cohort} fixture did not qualify exactly the intended saved "
            f"listings; qualified symbols were {sorted(symbols)}"
        )
    if output["prediction_count"] != len(symbols) * 5:
        raise ProbeError("The frozen fixture did not write exactly five rows per listing")
    if cohort == "full":
        withheld = [
            row
            for key, row in output["predictions"].items()
            if key.startswith(f"{WITHHELD_SYMBOL}|") and row["method_version"] == "us-price-fhs-v1"
        ]
        if len(withheld) != 4 or any(not row["insufficiency_reason"] for row in withheld):
            raise ProbeError(
                "The withheld fixture did not produce four genuinely withheld FHS rows "
                "with stored reasons"
            )
    successful = [
        row
        for key, row in output["predictions"].items()
        if key.startswith(f"{SUCCESSFUL_SYMBOL}|") and row["method_version"] == "us-price-fhs-v1"
    ]
    if len(successful) != 4 or any(row["insufficiency_reason"] for row in successful):
        raise ProbeError("The successful fixture did not produce four available FHS rows")


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


def _run_daily(*, store: Any, derive: bool, cohort: str) -> Any:
    """Execute the frozen daily writer, excluding additive derived work."""
    import inspect

    from stanstock.data.research_product_jobs import execute_daily_research_job
    from stanstock.research.models import AnalysisRun

    owner, config_path = _build_environment(store, cohort=cohort)
    kwargs: dict[str, Any] = {}
    supports_flag = "derive_frequencies" in inspect.signature(execute_daily_research_job).parameters
    if supports_flag and not derive:
        kwargs["derive_frequencies"] = False
    if derive and not supports_flag:
        raise ProbeError("This revision has no additive frequency derivation to exercise")
    job = execute_daily_research_job(
        target_date=TARGET,
        owner=owner,
        store=store,
        core_config_path=config_path,
        enforce_rate_limit=False,
        **kwargs,
    )
    if job.status != "success":
        raise ProbeError(f"The frozen daily writer did not succeed: {job.status}")
    return AnalysisRun.objects.get(id=job.details["analysis_run_id"])


def _bind_synthetic_revision() -> None:
    from stanstock.core import research_product_refresh
    from stanstock.research import product_pipeline

    research_product_refresh.clean_git_revision = lambda _root: SYNTHETIC_REVISION
    product_pipeline.clean_git_revision = lambda _root: SYNTHETIC_REVISION


def _run_scheduled(*, store: Any, cohort: str) -> dict[str, Any]:
    """Execute and canonically replay the frozen scheduled parent."""
    from django.conf import settings

    from stanstock.core import research_product_refresh
    from stanstock.core.models import JobRun
    from stanstock.research.models import AnalysisRun

    owner, config_path = _build_environment(store, cohort=cohort)
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.OWNER_USERNAME = owner.username
    settings.DATA_DIR = store.root
    _bind_synthetic_revision()

    execution = research_product_refresh.execute_scheduled_research_refresh(
        core_config_path=config_path,
        store=store,
        enforce_rate_limit=False,
    )
    parent = JobRun.objects.get(pk=execution.parent.pk)
    if parent.status != JobRun.Status.SUCCESS:
        raise ProbeError(f"The scheduled parent did not succeed: {parent.status}")
    identity = research_product_refresh.scheduled_identity(owner)
    recomputed = research_product_refresh.verify_scheduled_research_refresh(
        target_date=TARGET,
        owner=owner,
        code_revision=parent.details["code_revision"],
        stages=parent.details["stages"],
        store=store,
        core_config_path=config_path,
    )
    replayed = research_product_refresh.replay_recorded_research_product_refresh(
        parent, store=store
    )
    run = AnalysisRun.objects.get(pk=recomputed["analysis_run_id"])
    return {
        "scheduled_identity": _jsonable(identity),
        "parent_job_run": _row(parent),
        "job_runs": {
            f"{job.job_name}|{job.region}|{job.target_date}|{job.attempt}": _row(job)
            for job in JobRun.objects.order_by("job_name", "attempt")
        },
        "recorded_verification": _jsonable(parent.details["verification"]),
        "recomputed_verification": _jsonable(recomputed),
        "replayed_verification": _jsonable(replayed.verification),
        "replayed_parent_id": str(replayed.parent.pk),
        "replayed_snapshot_id": str(replayed.snapshot.pk),
        "replayed_analysis_run_id": str(replayed.analysis_run.pk),
        "replayed_catalog_asset_ids": sorted(str(asset.id) for asset in replayed.catalog_assets),
        "analysis_count": execution.analysis_count,
        "prediction_count": execution.prediction_count,
        "_run": run,
    }


def _recover_scheduled(*, store: Any) -> dict[str, Any]:
    """Recover and canonically replay a *retained* old-source parent.

    Nothing is constructed here.  The caller supplies a working directory
    whose SQLite database and asset store were written by an earlier
    scheduled run of a *different* source revision, so the parent, its
    children, its immutable rows, and its registered assets are genuinely
    that revision's own output rather than a hand-built stand-in.

    The probe refuses outright if the retained parent already carries the
    additive frequency bindings: a recovery proof is only meaningful against
    a parent that actually predates them.
    """
    from django.conf import settings
    from django.contrib.auth import get_user_model

    from stanstock.core import research_product_refresh
    from stanstock.core.models import JobRun
    from stanstock.data.management.config_loader import default_us_universe_config_path
    from stanstock.data.research_product_jobs import SCHEDULED_RESEARCH_JOB, product_job_name
    from stanstock.research.models import AnalysisRun

    owner = get_user_model().objects.get(username=OWNER_USERNAME)
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.OWNER_USERNAME = owner.username
    settings.DATA_DIR = store.root
    _bind_synthetic_revision()
    config_path = default_us_universe_config_path()

    identity = research_product_refresh.scheduled_identity(owner)
    retained = JobRun.objects.get(
        job_name=product_job_name(SCHEDULED_RESEARCH_JOB, identity),
        region="us",
        target_date=TARGET,
        status=JobRun.Status.SUCCESS,
    )
    retained_details_before = _jsonable(dict(retained.details))
    forbidden = {"frequencies", "frequency_verification"} & set(retained.details)
    if forbidden:
        raise ProbeError(
            "The retained parent already carries the additive bindings "
            f"{sorted(forbidden)}, so it is not a frozen old-source parent"
        )
    if JobRun.objects.filter(job_name="research_product_frequency_v1").exists():
        raise ProbeError("The retained state already contains an additive frequency child")

    execution = research_product_refresh.execute_scheduled_research_refresh(
        core_config_path=config_path,
        store=store,
        enforce_rate_limit=False,
    )
    if execution.parent.status != JobRun.Status.SKIPPED:
        raise ProbeError(
            "Recovering an already-successful retained target must skip, not reissue; "
            f"got {execution.parent.status}"
        )
    parent = JobRun.objects.get(pk=retained.pk)
    recomputed = research_product_refresh.verify_scheduled_research_refresh(
        target_date=TARGET,
        owner=owner,
        code_revision=parent.details["code_revision"],
        stages=parent.details["stages"],
        store=store,
        core_config_path=config_path,
    )
    replayed = research_product_refresh.replay_recorded_research_product_refresh(
        parent, store=store
    )
    run = AnalysisRun.objects.get(pk=recomputed["analysis_run_id"])
    return {
        "scheduled_identity": _jsonable(identity),
        "retained_parent_details_before_recovery": retained_details_before,
        "parent_job_run": _row(parent),
        "recovery_job_run": _row(JobRun.objects.get(pk=execution.parent.pk)),
        "job_runs": {
            f"{job.job_name}|{job.region}|{job.target_date}|{job.attempt}": _row(job)
            for job in JobRun.objects.order_by("job_name", "attempt")
        },
        "recorded_verification": _jsonable(parent.details["verification"]),
        "recomputed_verification": _jsonable(recomputed),
        "replayed_verification": _jsonable(replayed.verification),
        "replayed_parent_id": str(replayed.parent.pk),
        "replayed_snapshot_id": str(replayed.snapshot.pk),
        "replayed_analysis_run_id": str(replayed.analysis_run.pk),
        "replayed_catalog_asset_ids": sorted(str(asset.id) for asset in replayed.catalog_assets),
        "analysis_count": execution.analysis_count,
        "prediction_count": execution.prediction_count,
        "_run": run,
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _configure_django(workdir: Path) -> None:
    import django
    from django.conf import settings

    settings.DATABASES["default"]["NAME"] = str(workdir / "frozen-probe.sqlite3")
    settings.DATA_DIR = workdir / "data"
    django.setup()


def run(*, mode: str, cohort: str, workdir: Path) -> dict[str, Any]:
    _configure_django(workdir)

    from django.core.management import call_command

    call_command("migrate", verbosity=0)
    _bind_natural_key_identities()
    _freeze_clock()

    from stanstock.data.assets import AssetStore

    store = AssetStore(workdir / "assets")
    # The emitted payload deliberately carries no probe-only metadata beyond
    # the cohort, so the two revisions' payloads are compared whole, with no
    # field excluded from the comparison.
    payload: dict[str, Any] = {
        "cohort": cohort,
        "target_date": TARGET.isoformat(),
        "generated_at": NOW.isoformat(),
        "config": _config_payload(),
    }
    if mode in ("scheduled", "scheduled_recovery"):
        scheduled = (
            _run_scheduled(store=store, cohort=cohort)
            if mode == "scheduled"
            else _recover_scheduled(store=store)
        )
        run_row = scheduled.pop("_run")
        payload["scheduled"] = scheduled
        payload["output"] = _frozen_output(run=run_row, store=store)
    else:
        run_row = _run_daily(store=store, derive=mode == "daily_derived", cohort=cohort)
        payload["output"] = _frozen_output(run=run_row, store=store)
    _require_expected_cohort(payload["output"], cohort=cohort)
    payload["assets"] = _document_assets(store)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--cohort", choices=COHORTS, default="full")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    _install_boundary_guards()
    os.environ["STANSTOCK_CODE_REVISION"] = SYNTHETIC_REVISION
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stanstock.settings.test")
    uuid.uuid4 = _DeterministicUuid("uuid4")  # type: ignore[assignment]

    args.workdir.mkdir(parents=True, exist_ok=True)
    payload = run(mode=args.mode, cohort=args.cohort, workdir=args.workdir)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
