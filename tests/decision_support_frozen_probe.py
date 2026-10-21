"""Only the frozen surfaces missing from the unchanged original probe.

Retained recovery copies an entire f800-produced database and asset store:
no parent, child, row, binding, or asset is synthesized or removed. The other
modes execute the legacy declaration/default CLI/rejection and demo producers
with the original probe's deterministic identities, clocks and source fixtures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from io import StringIO
from pathlib import Path
from subprocess import CompletedProcess

import research_product_frozen_probability_probe as frozen


def _prepare(workdir: Path):
    frozen._configure_django(workdir)

    from django.core.management import call_command

    call_command("migrate", verbosity=0)
    frozen._bind_natural_key_identities()
    frozen._freeze_clock()
    from stanstock.data.assets import AssetStore

    return AssetStore(workdir / "assets")


def retained_recovery(workdir: Path) -> dict[str, object]:
    store = _prepare(workdir)

    from django.conf import settings
    from django.contrib.auth import get_user_model

    from stanstock.core import research_product_refresh
    from stanstock.core.models import JobRun
    from stanstock.data.management.config_loader import default_us_universe_config_path
    from stanstock.data.models import ProviderRecord
    from stanstock.data.providers import twelve_data
    from stanstock.data.research_product_jobs import SCHEDULED_RESEARCH_JOB, product_job_name
    from stanstock.research.models import AnalysisRun

    owner = get_user_model().objects.get(username=frozen.OWNER_USERNAME)
    settings.DATA_DIR = store.root
    settings.OWNER_USERNAME = owner.username
    settings.RESEARCH_PRODUCT_ENABLED = False
    settings.RESEARCH_PRODUCT_VERSION = "research-product-v1"
    frozen._bind_synthetic_revision()
    calls = {"credentials": 0, "fetches": 0}

    def credentials(*_args, **_kwargs):
        calls["credentials"] += 1
        raise frozen.ProbeError("Retained recovery must not resolve credentials")

    def fetch(*_args, **_kwargs):
        calls["fetches"] += 1
        raise frozen.ProbeError("Retained recovery must not fetch provider data")

    twelve_data.resolve_api_key = credentials
    twelve_data.fetch_daily_price_series = fetch
    twelve_data.fetch_stock_catalog = fetch
    twelve_data.fetch = fetch
    record = ProviderRecord.objects.get(provider="twelve_data")
    record.enabled = False
    record.save(update_fields=["enabled"])
    identity = research_product_refresh.scheduled_identity(owner)
    retained = JobRun.objects.get(
        job_name=product_job_name(SCHEDULED_RESEARCH_JOB, identity),
        region="us",
        target_date=frozen.TARGET,
        status=JobRun.Status.SUCCESS,
    )
    before = frozen._jsonable(retained.details)
    if not {"frequencies", "frequency_verification"} <= retained.details.keys():
        raise frozen.ProbeError("This supplement requires the genuine retained frequency bindings")
    if not JobRun.objects.filter(
        job_name__startswith="research_product_frequency_v1", status=JobRun.Status.SUCCESS
    ).exists():
        raise frozen.ProbeError("The old producer did not retain its real frequency child")
    revision_calls = []

    def dirty_checkout(_root):
        revision_calls.append(True)
        raise frozen.ProbeError("Complete retained recovery must use its recorded revision")

    research_product_refresh.clean_git_revision = dirty_checkout
    execution = research_product_refresh.execute_scheduled_research_refresh(
        core_config_path=default_us_universe_config_path(),
        store=store,
        enforce_rate_limit=False,
    )
    if execution.parent.status != JobRun.Status.SKIPPED or revision_calls or any(calls.values()):
        raise frozen.ProbeError("Complete frequency-bound recovery was not a local no-op")
    retained.refresh_from_db()
    if frozen._jsonable(retained.details) != before:
        raise frozen.ProbeError("Recovery mutated the retained parent's original proof")
    settings.RESEARCH_PRODUCT_VERSION = "research-product-v2"
    replayed = research_product_refresh.replay_recorded_research_product_refresh(
        retained, store=store
    )
    if replayed.verification != retained.details["verification"]:
        raise frozen.ProbeError("Captured-v1 replay was reinterpreted by the current selector")
    settings.RESEARCH_PRODUCT_VERSION = "research-product-v1"
    run = AnalysisRun.objects.get(pk=replayed.analysis_run.pk)
    output = frozen._frozen_output(run=run, store=store)
    frozen._require_expected_cohort(output, cohort="full")
    return {
        "cohort": "full",
        "target_date": frozen.TARGET.isoformat(),
        "generated_at": frozen.NOW.isoformat(),
        "config": frozen._config_payload(),
        "scheduled": {
            "scheduled_identity": frozen._jsonable(identity),
            "retained_parent_details_before_recovery": before,
            "parent_job_run": frozen._row(retained),
            "recovery_job_run": frozen._row(execution.parent),
            "job_runs": {
                f"{job.job_name}|{job.region}|{job.target_date}|{job.attempt}": frozen._row(job)
                for job in JobRun.objects.order_by("job_name", "attempt")
            },
            "replayed_verification": frozen._jsonable(replayed.verification),
            "replayed_parent_id": str(replayed.parent.pk),
            "replayed_snapshot_id": str(replayed.snapshot.pk),
            "replayed_analysis_run_id": str(replayed.analysis_run.pk),
            "replayed_catalog_asset_ids": sorted(
                str(asset.pk) for asset in replayed.catalog_assets
            ),
            "analysis_count": execution.analysis_count,
            "prediction_count": execution.prediction_count,
            "credential_and_fetch_calls": calls,
            "dirty_checkout_reads": len(revision_calls),
        },
        "output": output,
        "assets": frozen._document_assets(store),
    }


def legacy_declaration(workdir: Path) -> dict[str, object]:
    store = _prepare(workdir)

    from django.conf import settings
    from django.contrib.auth import get_user_model
    from django.core.management import call_command
    from django.core.management.base import CommandError

    from stanstock.core.models import JobRun
    from stanstock.data.models import DataAsset
    from stanstock.portfolio.models import Portfolio, PortfolioHolding
    from stanstock.portfolio.planner import _qualified_satellite
    from stanstock.portfolio.service import PortfolioValuationError, build_sample_portfolio
    from stanstock.research import forward_scorecard
    from stanstock.research.models import Prediction
    from stanstock.simulation.builders import SimulationWorkflowError, build_signals_for_backtest

    settings.DATA_DIR = store.root
    settings.RESEARCH_PRODUCT_ENABLED = True
    if (
        getattr(settings, "RESEARCH_PRODUCT_VERSION", "research-product-v1")
        != "research-product-v1"
    ):
        raise frozen.ProbeError("Legacy default selection was changed")
    settings.DEMO_MODE = False
    frozen._bind_synthetic_revision()
    run = frozen._run_daily(store=store, derive=True, cohort="full")
    owner = get_user_model().objects.get(username=frozen.OWNER_USERNAME)

    def synthetic_git(_root, *args):
        if args != ("rev-parse", "HEAD"):
            raise frozen.ProbeError("Only deterministic revision substitution is allowed")
        return CompletedProcess(args, 0, stdout=frozen.SYNTHETIC_REVISION + "\n", stderr="")

    forward_scorecard._git = synthetic_git
    outputs = []
    for _ in range(2):
        output = StringIO()
        call_command("forecast_scorecard", owner=owner.username, declare=True, stdout=output)
        outputs.append(output.getvalue())
    declaration = forward_scorecard.read_forward_declaration(user=owner, store=store)
    declared = DataAsset.objects.filter(kind=forward_scorecard.FORWARD_PROTOCOL_KIND)
    if declared.count() != 1 or outputs[0] != outputs[1]:
        raise frozen.ProbeError("Default legacy CLI declaration was not immutable/idempotent")
    report = forward_scorecard.read_forward_scorecard(user=owner, store=store)
    counts = (
        Portfolio.objects.count(),
        PortfolioHolding.objects.count(),
        Prediction.objects.count(),
    )
    rejections = {}
    for name, attempt, expected in (
        (
            "sample_portfolio",
            lambda: build_sample_portfolio(owner=owner, source_run=run),
            PortfolioValuationError,
        ),
        (
            "numeric_simulation",
            lambda: build_signals_for_backtest(
                snapshot=run.universe_snapshot,
                start_date=run.target_date,
                end_date=run.target_date,
            ),
            SimulationWorkflowError,
        ),
        (
            "predict",
            lambda: call_command("predict", analysis=str(run.stocks.first().pk), stdout=StringIO()),
            CommandError,
        ),
    ):
        try:
            attempt()
        except expected as error:
            chain = []
            while error is not None:
                chain.append(
                    {
                        "type": f"{type(error).__module__}.{type(error).__qualname__}",
                        "text": str(error),
                    }
                )
                error = error.__cause__
            rejections[name] = chain
        else:
            raise frozen.ProbeError(f"Legacy scoreless {name} was not rejected")
    satellite, reason = _qualified_satellite(as_of_date=run.target_date, today=run.target_date)
    if satellite is not None or counts != (
        Portfolio.objects.count(),
        PortfolioHolding.objects.count(),
        Prediction.objects.count(),
    ):
        raise frozen.ProbeError("Legacy rejection allocated holdings or appended a signal")
    return {
        "config": frozen._config_payload(),
        "declaration": {
            "row": frozen._row(declaration.asset),
            "declared_at": declaration.declared_at.isoformat(),
            "document": declaration.document,
            "scorecard": report,
            "default_cli_outputs": outputs,
        },
        "legacy_rejections": rejections,
        "satellite": {"listing": None, "reason": reason},
        "job_runs": {
            f"{job.job_name}|{job.region}|{job.target_date}|{job.attempt}": frozen._row(job)
            for job in JobRun.objects.order_by("job_name", "attempt")
        },
        "output": frozen._frozen_output(run=run, store=store),
        "assets": frozen._document_assets(store),
    }


def legacy_demo(workdir: Path) -> dict[str, object]:
    store = _prepare(workdir)

    from django.conf import settings
    from django.core.management import call_command

    from stanstock.core.models import JobRun
    from stanstock.research.models import AnalysisRun

    settings.DATA_DIR = store.root
    settings.RESEARCH_PRODUCT_ENABLED = True
    if (
        getattr(settings, "RESEARCH_PRODUCT_VERSION", "research-product-v1")
        != "research-product-v1"
    ):
        raise frozen.ProbeError("Legacy default selection was changed")
    settings.DEMO_MODE = True
    frozen._bind_synthetic_revision()
    outputs = []
    for _ in range(2):
        output = StringIO()
        call_command("refresh_demo", stdout=output)
        outputs.append(output.getvalue())
    run = AnalysisRun.objects.get(config_version="research-product-v1")
    return {
        "config": frozen._config_payload(),
        "default_cli_outputs": outputs,
        "job_runs": {
            f"{job.job_name}|{job.region}|{job.target_date}|{job.attempt}": frozen._row(job)
            for job in JobRun.objects.order_by("job_name", "attempt")
        },
        "output": frozen._frozen_output(run=run, store=store),
        "assets": frozen._document_assets(store),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("retained-frequency", "declaration", "demo"),
        default="retained-frequency",
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen._install_boundary_guards()
    os.environ["STANSTOCK_CODE_REVISION"] = frozen.SYNTHETIC_REVISION
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stanstock.settings.test")
    uuid.uuid4 = frozen._DeterministicUuid("uuid4")
    if args.mode == "retained-frequency":
        payload = retained_recovery(args.workdir)
    elif args.mode == "declaration":
        payload = legacy_declaration(args.workdir)
    else:
        payload = legacy_demo(args.workdir)
    args.output.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
