"""Read-only offline audit of SEC evidence selection for long forecasts.

Usage::

    python manage.py audit_long_evidence \\
        --listing-ids 0e2f0f4e-0000-4000-8000-000000000001 \\
        --target-date 2026-02-27 \\
        --available-through 2026-03-01T12:00:00+00:00 \\
        --decision-time 2026-03-01T12:00:00+00:00 \\
        --json

The command reads only already-persisted immutable rows through `AsOfData`.
It makes no HTTP or provider request, consumes no credits, and writes no
database row, asset, or file. It exists so an operator can see, before any
`us-sec-long-v3` replay, which source alias anchors each canonical TTM
concept, which real filed fact controls that choice, whether that alias
supplies a homogeneous four-quarter tail, which aliases collide or are
stale-but-complete alternatives, and whether a compatible beginning/end
invested-capital pair exists.

All three forecast boundaries are separate, required, and never guessed:
``--target-date`` bounds the reporting periods, ``--available-through`` is
the historical data cutoff for fact availability, and ``--decision-time`` is
the as-of visibility boundary. A naive timestamp is rejected rather than
silently assigned a timezone.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from datetime import date, datetime
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from stanstock.research.evidence_audit import (
    AmbiguousListingSymbolError,
    audit_long_evidence,
)
from stanstock.research.long_forecast_config import (
    LongForecastConfigParseError,
    load_long_forecast_config,
    long_forecast_config_path,
)

DEFAULT_LONG_CONFIG_VERSION = "us-sec-long-v3"


class Command(BaseCommand):
    help = (
        "Audit persisted SEC alias selection and invested-capital pair "
        "compatibility for the requested listings. Read-only and offline."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--listing-ids",
            default="",
            help=(
                "Comma-separated immutable listing UUIDs. This is the canonical "
                "input: a ticker is not an identity."
            ),
        )
        parser.add_argument(
            "--symbols",
            default="",
            help=(
                "Comma-separated tickers, for operator convenience only. A "
                "ticker that matches more than one listing is rejected; audit "
                "it by listing ID instead."
            ),
        )
        parser.add_argument(
            "--target-date",
            required=True,
            help=(
                "Forecast target date (YYYY-MM-DD). Reporting periods ending "
                "after this date are excluded, exactly as in a forecast run."
            ),
        )
        parser.add_argument(
            "--available-through",
            required=True,
            help=(
                "Timezone-aware ISO-8601 historical data cutoff applied to SEC "
                "fact availability, e.g. 2026-03-01T12:00:00+00:00."
            ),
        )
        parser.add_argument(
            "--decision-time",
            required=True,
            help=(
                "Timezone-aware ISO-8601 as-of decision time bounding evidence "
                "and source-asset visibility, e.g. 2026-03-01T12:00:00+00:00."
            ),
        )
        parser.add_argument(
            "--long-config",
            default=None,
            help=(
                "Path to the long forecast configuration to audit against "
                f"(default: config/forecasts/{DEFAULT_LONG_CONFIG_VERSION}.yml)."
            ),
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit the complete report as JSON for aggregate replay.",
        )

    def handle(self, *args: object, **options: object) -> None:
        listing_ids = _csv(str(options.get("listing_ids") or ""))
        symbols = tuple(symbol.upper() for symbol in _csv(str(options.get("symbols") or "")))
        if not listing_ids and not symbols:
            raise CommandError("Provide at least one of --listing-ids or --symbols")
        target_date = _target_date(str(options["target_date"]))
        available_through = _aware_datetime(
            str(options["available_through"]),
            flag="--available-through",
        )
        decision_time = _aware_datetime(str(options["decision_time"]), flag="--decision-time")
        raw_config_path = options.get("long_config")
        config_path = (
            Path(str(raw_config_path))
            if raw_config_path
            else long_forecast_config_path(DEFAULT_LONG_CONFIG_VERSION)
        )
        try:
            config = load_long_forecast_config(config_path)
        except LongForecastConfigParseError:
            # Fail closed without quoting the file: a YAML parser message
            # includes the offending source line, which may hold
            # credential-shaped content when the wrong file is passed.
            raise CommandError(
                f"Could not parse {config_path}: the file is not valid YAML. "
                "The parser message is withheld because it would echo the "
                "offending line from the file."
            ) from None
        except (OSError, ValueError) as exc:
            raise CommandError(f"Could not load {config_path}: {exc}") from exc
        try:
            report = audit_long_evidence(
                listing_ids=listing_ids,
                symbols=symbols,
                target_date=target_date,
                available_through=available_through,
                decision_time=decision_time,
                config=config,
            )
        except AmbiguousListingSymbolError as exc:
            raise CommandError(str(exc)) from exc
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        if options.get("json"):
            self.stdout.write(json.dumps(report, sort_keys=True, indent=2))
            return
        for line in _summary_lines(report):
            self.stdout.write(line)


def _csv(raw: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


def _target_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise CommandError(
            f"--target-date must be an ISO-8601 date (YYYY-MM-DD); got {raw!r}"
        ) from exc


def _aware_datetime(raw: str, *, flag: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise CommandError(
            f"{flag} must be an ISO-8601 datetime with a timezone offset; got {raw!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise CommandError(
            f"{flag} must include an explicit timezone offset so the point-in-time "
            "boundary is unambiguous (e.g. 2026-03-01T12:00:00+00:00)"
        )
    return parsed


def _summary_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        f"target_date={report['target_date']} "
        f"available_through={report['available_through']} "
        f"decision_time={report['decision_time']} "
        f"long_config={report['long_forecast_config_version']} "
        f"fundamentals={report['fundamentals_config_version']} "
        f"ttm_policy={report['ttm_selection_policy']}"
    ]
    for entry in report["listings"]:
        label = entry["listing_id"] or entry["requested_listing_id"] or entry["requested_symbol"]
        lines.append(f"{label} ({entry['symbol']}): {entry['status']}")
        if entry["status"] != "audited":
            lines.append(f"  reason: {entry['reason']}")
            continue
        for concept_entry in entry["ttm_alias_selection"]:
            selection = concept_entry.get("selection") or {}
            controlling = selection.get("controlling_source_fact") or {}
            lines.append(
                f"  {concept_entry['concept']}: "
                f"ttm={'yes' if concept_entry['ttm_available'] else 'no'} "
                f"alias={selection.get('selected_source_concept')} "
                f"newest_quarter={selection.get('newest_quarter_end')} "
                f"controlling_fact={controlling.get('fact_id')} "
                f"homogeneous_tail={concept_entry.get('homogeneous_four_quarter_tail')} "
                f"alias_collision={concept_entry.get('alias_collision')} "
                f"stale_complete={concept_entry.get('stale_complete_alternatives')}"
            )
            reason = selection.get("reason")
            if reason:
                lines.append(f"    withheld: {reason}")
        capital = entry["invested_capital"]
        if capital["status"] != "audited":
            lines.append(f"  invested_capital: {capital['status']} ({capital['reason']})")
            continue
        lines.append(
            "  invested_capital: "
            f"compatible_pair={capital['compatible_pair_available']} "
            f"assessment={capital['assessment_status']} "
            f"independent_nearest_compatible={capital['independent_nearest_pair_compatible']} "
            f"joint_recovers_missed_pair={capital['joint_selection_recovers_missed_pair']}"
        )
        if capital["reason"]:
            lines.append(f"    withheld: {capital['reason']}")
    return lines
