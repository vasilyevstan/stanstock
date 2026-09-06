"""Safely probe configured representative provider endpoints.

Calls one small, representative request per configured provider (Twelve
Data and Stooq daily prices, SEC submissions, filings.xbrl.org filings
index, ECB EXR CSV),
classifies each outcome, updates `ProviderRecord` status (never `enabled` -
the provider stop/go gate is a manual decision, see `LEARNINGS.md`), and
writes a private JSON report under `STANSTOCK_DATA_DIR`. It never retries
past a challenge, never scrapes HTML, and never stores secrets or full
response bodies in the report.

Usage::

    python manage.py source_spike
    python manage.py source_spike --skip sec,ecb

This command reports two distinct things per provider, and they must not be
confused:

1. A **runtime classification** of *this run's* probe (see below) - whether
   the request could be completed right now, from this environment.
2. A static **capability verdict** (``PROVIDER_CAPABILITY_VERDICTS``) - the
   already-completed research conclusion about whether StanStock is willing
   to rely on that provider at all, independent of any single run. This
   verdict only changes when someone updates the constant below after new
   research, alongside `docs/source-spike.md`; it is never inferred from a
   probe result.

Runtime classification buckets, applied per provider based on what a
failure at that provider actually indicates (see `docs/source-spike.md` for
the evidence behind each provider's mapping):

- ``ok``: the representative request succeeded and returned usable data.
- ``configuration_missing``: required local configuration (for example
  ``SEC_USER_AGENT``) is absent; this blocks the probe but says nothing
  about the provider itself.
- ``environment_blocked``: this execution environment could not complete
  the request as designed (network egress failure, or a provider that
  returned an access-denial response StanStock's own spike attributes to
  network/egress policy rather than the provider's own automation stance).
- ``provider_incompatible``: the provider itself returned something
  StanStock cannot use for its documented, non-bypassing access path (an
  HTML/JavaScript verification challenge, a malformed payload, or a clear
  rate/subscription message).
- ``quota_exhausted``: an approved provider reported that the configured
  request or daily credit allowance is exhausted.
- ``unexpected_error``: any other exception; always investigated before the
  report is trusted.

Completed provider research (updated 2026-09-06, see
`docs/source-spike.md`) fixes
each provider's capability verdict as:

- ``twelve_data`` -> ``CONDITIONAL_GO``: its official API permits the
  reduced private US-only scope, subject to the configured plan's market,
  quota, storage, and non-redistribution limits.
- ``stooq`` -> ``NO_GO``: its public CSV download is automation-blocked
  (StanStock will never attempt to bypass the JS verification gate) and its
  automation/private-retention terms could not be independently verified.
  This is fixed regardless of whether a given probe run happens to succeed.
- ``sec`` -> ``GO``: SEC submissions/companyfacts are an approved source in
  general, even though this execution environment's IP observed HTTP 403.
- ``filings_xbrl_org`` -> ``CONDITIONAL_GO``: usable, but with explicit,
  documented gaps (Germany and Ireland are missing from the index) and
  repository ingestion lag relative to the filing authority.
- ``ecb`` -> ``GO``: usable via the SDMX CSV/XML API (which is what
  StanStock's client uses); the legacy bulk history CSV showed anomalous
  rows in research and is deliberately not used.

The overall CONDITIONAL_GO/NO_GO decision is driven only by whether a
price-capability provider both (a) has a capability verdict other than
``NO_GO`` and (b) probed ``ok`` in this run. Twelve Data can satisfy that
gate for the reduced US-only scope when a non-demo personal API key is
configured. Stooq remains fixed ``NO_GO`` regardless of a one-off technical
success. Fundamentals/FX classifications are reported separately.
"""

from __future__ import annotations

import json
import stat
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_timezone

from stanstock.data.live_us import ProviderCreditBudget
from stanstock.data.models import ProviderRecord
from stanstock.data.providers import ecb, filings_xbrl, sec, stooq, twelve_data
from stanstock.data.providers.exceptions import (
    ProviderBlockedError,
    ProviderConfigurationError,
    ProviderError,
    ProviderNetworkError,
    ProviderQuotaError,
    ProviderResponseError,
)

#: A well-known, long-public CIK (Apple Inc.) used only to exercise the SEC
#: endpoint shape; StanStock does not claim any relationship with the filer.
PROBE_SEC_CIK = "0000320193"
PROBE_STOOQ_SYMBOL = "aapl.us"
PROBE_TWELVE_DATA_SYMBOL = "AAPL"
PROBE_ECB_QUOTE_CURRENCY = "USD"
TWELVE_DATA_USAGE_SCOPE = "personal_internal_display_requires_entitlement"

CLASSIFICATION_OK = "ok"
CLASSIFICATION_CONFIG_MISSING = "configuration_missing"
CLASSIFICATION_ENV_BLOCKED = "environment_blocked"
CLASSIFICATION_PROVIDER_INCOMPATIBLE = "provider_incompatible"
CLASSIFICATION_QUOTA_EXHAUSTED = "quota_exhausted"
CLASSIFICATION_UNEXPECTED = "unexpected_error"

DECISION_CONDITIONAL_GO = "CONDITIONAL_GO"
DECISION_NO_GO = "NO_GO"

#: Static, manually-researched capability verdicts (see `docs/source-spike.md`
#: for the evidence behind each). These are deliberately NOT derived from a
#: probe's runtime classification: a provider that happens to respond ``ok``
#: on a given run is not thereby "approved" if its terms/automation rights
#: were never verified (Stooq), and a provider that returns an
#: `environment_blocked` result on this run is not thereby "rejected" if its
#: own policy generally permits compliant automated access (SEC). Update
#: these constants only after new research, together with
#: `docs/source-spike.md`.
CAPABILITY_VERDICT_GO = "GO"
CAPABILITY_VERDICT_CONDITIONAL_GO = "CONDITIONAL_GO"
CAPABILITY_VERDICT_NO_GO = "NO_GO"

PROVIDER_CAPABILITY_VERDICTS: dict[str, str] = {
    "twelve_data": CAPABILITY_VERDICT_CONDITIONAL_GO,
    "stooq": CAPABILITY_VERDICT_NO_GO,
    "sec": CAPABILITY_VERDICT_GO,
    "filings_xbrl_org": CAPABILITY_VERDICT_CONDITIONAL_GO,
    "ecb": CAPABILITY_VERDICT_GO,
}


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    provider: str
    endpoint: str
    classification: str
    detail: str
    elapsed_ms: int
    is_price_capability: bool = False
    capability_verdict: str = CAPABILITY_VERDICT_NO_GO


def _probe_twelve_data() -> ProbeOutcome:
    started = dj_timezone.now()
    try:
        api_key = twelve_data.resolve_api_key()
        ProviderRecord.objects.get_or_create(provider=twelve_data.PROVIDER)
        budget = ProviderCreditBudget(require_enabled=False)
        budget.preflight(1)
        budget.consume()
        series = twelve_data.fetch_daily_price_series(
            PROBE_TWELVE_DATA_SYMBOL,
            outputsize=2,
            api_key=api_key,
        )
    except ProviderConfigurationError as exc:
        return _outcome(
            "twelve_data",
            twelve_data.TIME_SERIES_URL,
            CLASSIFICATION_CONFIG_MISSING,
            exc,
            started,
            price=True,
        )
    except ProviderQuotaError as exc:
        return _outcome(
            "twelve_data",
            twelve_data.TIME_SERIES_URL,
            CLASSIFICATION_QUOTA_EXHAUSTED,
            exc,
            started,
            price=True,
        )
    except ProviderBlockedError as exc:
        return _outcome(
            "twelve_data",
            twelve_data.TIME_SERIES_URL,
            CLASSIFICATION_PROVIDER_INCOMPATIBLE,
            exc,
            started,
            price=True,
        )
    except ProviderNetworkError as exc:
        return _outcome(
            "twelve_data",
            twelve_data.TIME_SERIES_URL,
            CLASSIFICATION_ENV_BLOCKED,
            exc,
            started,
            price=True,
        )
    except ProviderResponseError as exc:
        return _outcome(
            "twelve_data",
            twelve_data.TIME_SERIES_URL,
            CLASSIFICATION_PROVIDER_INCOMPATIBLE,
            exc,
            started,
            price=True,
        )
    except ProviderError as exc:  # pragma: no cover - defensive catch-all
        return _outcome(
            "twelve_data",
            twelve_data.TIME_SERIES_URL,
            CLASSIFICATION_UNEXPECTED,
            exc,
            started,
            price=True,
        )
    return _outcome(
        "twelve_data",
        twelve_data.TIME_SERIES_URL,
        CLASSIFICATION_OK,
        f"Received {len(series.bars)} daily bars for {PROBE_TWELVE_DATA_SYMBOL!r}",
        started,
        price=True,
    )


def _probe_stooq() -> ProbeOutcome:
    started = dj_timezone.now()
    try:
        series = stooq.fetch_daily_price_series(PROBE_STOOQ_SYMBOL)
    except ProviderBlockedError as exc:
        return _outcome(
            "stooq", stooq.BASE_URL, CLASSIFICATION_PROVIDER_INCOMPATIBLE, exc, started, price=True
        )
    except ProviderConfigurationError as exc:
        return _outcome(
            "stooq", stooq.BASE_URL, CLASSIFICATION_CONFIG_MISSING, exc, started, price=True
        )
    except ProviderNetworkError as exc:
        return _outcome(
            "stooq", stooq.BASE_URL, CLASSIFICATION_ENV_BLOCKED, exc, started, price=True
        )
    except ProviderResponseError as exc:
        return _outcome(
            "stooq", stooq.BASE_URL, CLASSIFICATION_PROVIDER_INCOMPATIBLE, exc, started, price=True
        )
    except ProviderError as exc:  # pragma: no cover - defensive catch-all
        return _outcome(
            "stooq", stooq.BASE_URL, CLASSIFICATION_UNEXPECTED, exc, started, price=True
        )
    return _outcome(
        "stooq",
        stooq.BASE_URL,
        CLASSIFICATION_OK,
        f"Received {len(series.bars)} daily bars for {PROBE_STOOQ_SYMBOL!r}",
        started,
        price=True,
    )


def _probe_sec() -> ProbeOutcome:
    started = dj_timezone.now()
    try:
        payload = sec.fetch_submissions(PROBE_SEC_CIK)
    except ProviderConfigurationError as exc:
        return _outcome("sec", sec.SUBMISSIONS_URL, CLASSIFICATION_CONFIG_MISSING, exc, started)
    except ProviderNetworkError as exc:
        return _outcome("sec", sec.SUBMISSIONS_URL, CLASSIFICATION_ENV_BLOCKED, exc, started)
    except ProviderBlockedError as exc:
        # StanStock's own spike reproduced HTTP 403 here with a compliant
        # SEC_USER_AGENT; that pattern is attributed to environment/network
        # egress blocking rather than SEC rejecting a well-formed request.
        # See docs/source-spike.md for the evidence behind this mapping.
        return _outcome("sec", sec.SUBMISSIONS_URL, CLASSIFICATION_ENV_BLOCKED, exc, started)
    except ProviderResponseError as exc:
        return _outcome(
            "sec", sec.SUBMISSIONS_URL, CLASSIFICATION_PROVIDER_INCOMPATIBLE, exc, started
        )
    except ProviderError as exc:  # pragma: no cover - defensive catch-all
        return _outcome("sec", sec.SUBMISSIONS_URL, CLASSIFICATION_UNEXPECTED, exc, started)
    return _outcome(
        "sec",
        sec.SUBMISSIONS_URL,
        CLASSIFICATION_OK,
        f"Received submissions payload ({len(payload.content)} bytes)",
        started,
    )


def _probe_filings_xbrl() -> ProbeOutcome:
    started = dj_timezone.now()
    try:
        page = filings_xbrl.fetch_filings_page(page_size=1)
    except ProviderNetworkError as exc:
        return _outcome(
            "filings_xbrl_org",
            filings_xbrl.FILINGS_API_URL,
            CLASSIFICATION_ENV_BLOCKED,
            exc,
            started,
        )
    except ProviderResponseError as exc:
        return _outcome(
            "filings_xbrl_org",
            filings_xbrl.FILINGS_API_URL,
            CLASSIFICATION_PROVIDER_INCOMPATIBLE,
            exc,
            started,
        )
    except ProviderError as exc:  # pragma: no cover - defensive catch-all
        return _outcome(
            "filings_xbrl_org",
            filings_xbrl.FILINGS_API_URL,
            CLASSIFICATION_UNEXPECTED,
            exc,
            started,
        )
    return _outcome(
        "filings_xbrl_org",
        filings_xbrl.FILINGS_API_URL,
        CLASSIFICATION_OK,
        f"Received {len(page.records)} filing index entries",
        started,
    )


def _probe_ecb() -> ProbeOutcome:
    started = dj_timezone.now()
    try:
        result = ecb.fetch_exr_csv(PROBE_ECB_QUOTE_CURRENCY)
    except ProviderNetworkError as exc:
        return _outcome("ecb", ecb.BASE_URL, CLASSIFICATION_ENV_BLOCKED, exc, started)
    except ProviderResponseError as exc:
        return _outcome("ecb", ecb.BASE_URL, CLASSIFICATION_PROVIDER_INCOMPATIBLE, exc, started)
    except ProviderError as exc:  # pragma: no cover - defensive catch-all
        return _outcome("ecb", ecb.BASE_URL, CLASSIFICATION_UNEXPECTED, exc, started)
    return _outcome(
        "ecb",
        ecb.BASE_URL,
        CLASSIFICATION_OK,
        f"Received {len(result.observations)} EXR observations",
        started,
    )


def _outcome(
    provider: str,
    endpoint: str,
    classification: str,
    detail: Exception | str,
    started: datetime,
    *,
    price: bool = False,
) -> ProbeOutcome:
    elapsed_ms = int((dj_timezone.now() - started).total_seconds() * 1000)
    message = str(detail)
    return ProbeOutcome(
        provider=provider,
        endpoint=endpoint,
        classification=classification,
        detail=message[:300],
        elapsed_ms=elapsed_ms,
        is_price_capability=price,
        capability_verdict=PROVIDER_CAPABILITY_VERDICTS.get(provider, CAPABILITY_VERDICT_NO_GO),
    )


PROBES = (
    _probe_twelve_data,
    _probe_stooq,
    _probe_sec,
    _probe_filings_xbrl,
    _probe_ecb,
)


class Command(BaseCommand):
    help = "Probe representative provider endpoints and record a source-capability report."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--skip",
            default="",
            help=(
                "Comma-separated provider names to skip "
                "(twelve_data,stooq,sec,filings_xbrl_org,ecb)"
            ),
        )

    def handle(self, *args: object, **options: object) -> None:
        skip = {name.strip() for name in str(options.get("skip") or "").split(",") if name.strip()}
        outcomes: list[ProbeOutcome] = []
        for probe in PROBES:
            provider_guess = probe.__name__.removeprefix("_probe_")
            if provider_guess in skip or provider_guess.replace("_", "") in skip:
                continue
            outcomes.append(probe())

        price_outcomes = [outcome for outcome in outcomes if outcome.is_price_capability]
        # An "approved" price capability requires BOTH a successful probe on
        # this run AND a capability verdict other than NO_GO. A technical
        # success alone (classification == ok) never grants CONDITIONAL_GO
        # by itself: Stooq's verdict remains fixed NO_GO, while Twelve Data
        # must both have a reviewed CONDITIONAL_GO verdict and succeed with
        # the configured non-demo account.
        price_capability = any(
            outcome.classification == CLASSIFICATION_OK
            and outcome.capability_verdict != CAPABILITY_VERDICT_NO_GO
            for outcome in price_outcomes
        )
        decision = DECISION_CONDITIONAL_GO if price_capability else DECISION_NO_GO

        run_at = dj_timezone.now()
        report = {
            "run_at": run_at.isoformat(),
            "decision": decision,
            "price_capability": price_capability,
            "probes": [asdict(outcome) for outcome in outcomes],
        }
        report_path = self._write_report(run_at, report)
        self._update_provider_records(outcomes, run_at)

        for outcome in outcomes:
            summary = (
                f"{outcome.provider}: {outcome.classification} "
                f"verdict={outcome.capability_verdict} "
                f"({outcome.elapsed_ms}ms) - {outcome.detail}"
            )
            self.stdout.write(summary)
        style = self.style.SUCCESS if decision == DECISION_CONDITIONAL_GO else self.style.WARNING
        self.stdout.write(style(f"Decision: {decision} (price_capability={price_capability})"))
        self.stdout.write(f"Report written to {report_path}")

    def _write_report(self, run_at: datetime, report: dict[str, object]) -> Path:
        reports_dir = settings.DATA_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"source_spike_{run_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return report_path

    def _update_provider_records(self, outcomes: list[ProbeOutcome], run_at: datetime) -> None:
        for outcome in outcomes:
            record, _created = ProviderRecord.objects.get_or_create(provider=outcome.provider)
            metadata = {
                **record.metadata,
                "last_probe_endpoint": outcome.endpoint,
                "last_probe_elapsed_ms": outcome.elapsed_ms,
                "last_probe_detail": outcome.detail,
                "last_probe_at": run_at.isoformat(),
                "capability_verdict": outcome.capability_verdict,
            }
            last_error = (
                ""
                if outcome.classification == CLASSIFICATION_OK
                else f"{outcome.classification}: {outcome.detail}"
            )
            record.status = outcome.classification
            record.metadata = metadata
            record.last_error = last_error
            if outcome.provider == twelve_data.PROVIDER:
                record.terms_url = twelve_data.TERMS_URL
                if not record.usage_scope:
                    record.usage_scope = TWELVE_DATA_USAGE_SCOPE
            if outcome.classification == CLASSIFICATION_OK:
                record.last_success_at = run_at
            record.save(
                update_fields=[
                    "status",
                    "metadata",
                    "last_error",
                    "last_success_at",
                    "terms_url",
                    "usage_scope",
                ]
            )
