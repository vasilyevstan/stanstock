"""SYNTHETIC-ONLY bounded vertical-flow demo command.

Runs the "seed -> analyze" flow against nothing but deterministic synthetic
demo data: (1) unconditionally calls the idempotent ``seed_demo`` command
(cheap: it is itself a no-op for anything already seeded), (2) resolves the
one research-grade synthetic universe snapshot it creates
(``UniverseSnapshot.Grade.RESEARCH`` for the demo universe config),
(3) resolves and validates the logical ``target_date`` against that
snapshot and the synthetic benchmark's own observed price history (see
below), and (4) runs `analyze_snapshot` for that validated target against
the snapshot with ``provider="synthetic_demo"`` and the demo benchmark
subject (``ZZBENCH01``, from
``config/benchmarks/demo_synthetic_balanced_v1.yaml``).

This command NEVER calls, claims to call, or approximates any live provider
(Stooq/SEC/filings.xbrl.org/ECB/etc.) -- see ``docs/source-spike.md`` for why
unattended live price ingestion is NO_GO in this environment. It exists
purely to exercise the demo/data + research vertical slice end-to-end with
synthetic data, for local development, demos, and tests.

Target-date semantics (this is the part that must not fabricate data):
``seed_demo``'s synthetic history is generated only up to a fixed historical
date (the snapshot's ``as_of_date``); it is not regenerated relative to
"today" each time this command happens to run. So:

- ``--target-date`` defaults to ``snapshot.as_of_date`` -- not calendar
  "today" -- since "today" may be long after the synthetic history ends (a
  weekend, or simply a later date than any synthetic data exists for).
- An explicit ``--target-date`` after ``snapshot.as_of_date`` is rejected:
  there is no synthetic history past that date, so honoring it would mean
  analyzing a "future" logical target using only prior-close data.
- Any ``--target-date`` (explicit or defaulted) that is not one of the
  synthetic benchmark's own *observed* session dates (its `DataAsset`
  price frame's ``date`` column) is rejected -- e.g. a weekend/holiday that
  was never a generated trading session. This command never fabricates a
  session that `seed_demo` did not actually generate.

Only the analysis step (`analyze_snapshot`) -- run against the validated
target -- is wrapped in `execute_target_job` (job_name="refresh_demo",
region="all"), so a target_date that has already succeeded is idempotently
skipped rather than re-running `analyze_snapshot` (and therefore never
creates additional `AnalysisRun`/`StockAnalysis`/`Prediction` rows for a
target that already succeeded) -- mirroring how a real scheduled job would
behave, without ever touching a live provider. `seed_demo` and target-date
resolution/validation always run first, outside that guard, since they must
happen before a target can even be validated or looked up.

The "as-of" decision moment used for analysis generation (`decision_time`,
i.e. what `AsOfData` gates history visibility against) is always the actual
current time (`timezone.now()`) when this command runs; only the *logical*
``target_date`` recorded on the run/predictions is the validated value
above.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.data.asof import AsOfData
from stanstock.data.management.config_loader import (
    default_benchmark_config_path,
    default_universe_config_path,
    load_yaml_mapping,
)
from stanstock.data.models import UniverseSnapshot
from stanstock.research.service import analyze_snapshot

JOB_NAME = "refresh_demo"
JOB_REGION = "all"
PROVIDER = "synthetic_demo"


class Command(BaseCommand):
    help = (
        "SYNTHETIC-ONLY demo vertical flow: seed_demo -> resolve the "
        "research-grade synthetic universe snapshot -> validate target_date "
        "against its observed synthetic history -> analyze_snapshot "
        "(provider=synthetic_demo, benchmark=ZZBENCH01). Never calls any "
        "live provider."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--target-date",
            help=(
                "Logical market date to persist on the run/predictions and "
                "to key the target-job idempotency check on (YYYY-MM-DD); "
                "defaults to the synthetic snapshot's as_of_date. Must be an "
                "observed session in the synthetic benchmark's price history "
                "and no later than the snapshot's as_of_date."
            ),
        )

    def handle(self, *args: object, **options: object) -> None:
        call_command("seed_demo")
        benchmark_config = load_yaml_mapping(default_benchmark_config_path())
        benchmark_subject = str(benchmark_config["benchmark_subject"])
        snapshot = _resolve_synthetic_snapshot()
        target_date = _resolve_and_validate_target_date(
            raw=options.get("target_date"),
            snapshot=snapshot,
            benchmark_subject=benchmark_subject,
        )

        def _task(run: JobRun) -> JobExecutionResult:
            results = analyze_snapshot(
                universe_snapshot=snapshot,
                decision_time=timezone.now(),
                target_date=target_date,
                provider=PROVIDER,
                benchmark_subject=benchmark_subject,
            )
            analyses = len(results)
            predictions = sum(len(result.predictions) for result in results)
            return JobExecutionResult(
                details={
                    "snapshot_id": str(snapshot.pk),
                    "snapshot_grade": snapshot.grade,
                    "provider": PROVIDER,
                    "benchmark_subject": benchmark_subject,
                    "analyses": analyses,
                    "predictions": predictions,
                }
            )

        job_run = execute_target_job(
            job_name=JOB_NAME,
            region=JOB_REGION,
            target_date=target_date,
            task=_task,
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"refresh_demo job_run={job_run.pk} status={job_run.status} "
                f"target_date={target_date.isoformat()} details={job_run.details!r}"
            )
        )


def _parse_target_date(raw: object) -> date | None:
    if raw is None:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError as exc:
        raise CommandError("--target-date must use YYYY-MM-DD") from exc


def _resolve_synthetic_snapshot() -> UniverseSnapshot:
    universe_config = load_yaml_mapping(default_universe_config_path())
    slug = str(universe_config["slug"])
    try:
        return UniverseSnapshot.objects.get(
            universe__slug=slug, grade=UniverseSnapshot.Grade.RESEARCH
        )
    except UniverseSnapshot.DoesNotExist as exc:
        raise CommandError(
            f"No research-grade UniverseSnapshot found for universe {slug!r} "
            "after seed_demo; this should not happen."
        ) from exc
    except UniverseSnapshot.MultipleObjectsReturned as exc:
        raise CommandError(
            f"Multiple research-grade UniverseSnapshots found for universe "
            f"{slug!r}; expected exactly one from seed_demo."
        ) from exc


def _resolve_and_validate_target_date(
    *,
    raw: object,
    snapshot: UniverseSnapshot,
    benchmark_subject: str,
) -> date:
    explicit = _parse_target_date(raw)
    target_date = explicit if explicit is not None else snapshot.as_of_date

    if target_date > snapshot.as_of_date:
        raise CommandError(
            f"--target-date {target_date.isoformat()} is after the synthetic "
            f"snapshot's as_of_date ({snapshot.as_of_date.isoformat()}); this "
            "demo flow only has synthetic history up to that date and refuses "
            "to analyze a future logical target with only prior-close data."
        )

    observed_dates = _observed_benchmark_dates(benchmark_subject)
    if target_date not in observed_dates:
        raise CommandError(
            f"--target-date {target_date.isoformat()} is not an observed "
            f"session in the synthetic benchmark {benchmark_subject!r}'s "
            "price history (e.g. a weekend/holiday); refusing to fabricate a "
            "trading session that seed_demo never generated."
        )
    return target_date


def _observed_benchmark_dates(benchmark_subject: str) -> set[date]:
    frame = AsOfData(timezone.now()).price_frame(provider=PROVIDER, subject=benchmark_subject)
    return set(frame["date"].to_list())
