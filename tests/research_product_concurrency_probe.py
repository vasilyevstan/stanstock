"""Commit threaded synthetic issuances without weakening immutable-row guards.

A disposable SQLite process avoids TransactionTestCase's DELETE-based flush,
which correctly cannot delete the immutable predictions being exercised.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import django
import pytest
from django.conf import settings


def main() -> None:
    root = Path(sys.argv[1])
    settings.DATABASES["default"]["NAME"] = str(root / "concurrency.sqlite3")
    django.setup()

    from django.contrib.auth import get_user_model
    from django.core.management import call_command
    from django.db import OperationalError, connections

    from stanstock.data import research_product_jobs
    from stanstock.data.models import DataAsset, Listing
    from stanstock.research.models import AnalysisRun, Prediction
    from stanstock.research.product_frequency_evidence import register_product_frequencies
    from test_research_product_jobs import _run, _series, make_product_environment

    call_command("migrate", verbosity=0)
    with pytest.MonkeyPatch.context() as monkeypatch:
        environment = make_product_environment(root, monkeypatch, get_user_model())
        _owner, _store, _path, resolve, fetch = environment
        resolve.side_effect = None
        resolve.return_value = "synthetic-test-token"
        fetch.side_effect = lambda symbol, **_kwargs: _series(symbol)
        original_frequency_writer = research_product_jobs.register_product_frequencies
        monkeypatch.setattr(
            research_product_jobs,
            "register_product_frequencies",
            lambda **_kw: None,
        )

        def execute(key: str) -> str:
            try:
                return _run(environment, issuance_key=key).status
            finally:
                connections["default"].close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            statuses = list(executor.map(execute, ("first", "second")))

        assert statuses == ["success", "success"]
        monkeypatch.setattr(
            research_product_jobs,
            "register_product_frequencies",
            original_frequency_writer,
        )
        source_run = AnalysisRun.objects.order_by("generated_at").first()
        assert source_run is not None

        def derive_frequency() -> str:
            try:
                return str(register_product_frequencies(run=source_run, store=_store).id)
            except OperationalError:
                # SQLite may reject the loser while its winner has not yet
                # committed. A bounded explicit retry must recover it.
                time.sleep(0.1)
                return str(register_product_frequencies(run=source_run, store=_store).id)
            finally:
                connections["default"].close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            frequency_ids = list(executor.map(lambda _value: derive_frequency(), (1, 2)))
        frequency_assets = DataAsset.objects.filter(
            kind="research_product_frequency_evidence",
            subject=f"research-product-v1:frequencies:{source_run.id}",
        )
        assert frequency_assets.count() == 1
        assert len(set(frequency_ids)) == 1
        assert DataAsset.objects.filter(kind="price_history", subject="CHEAP").count() == 1
        resolve.assert_called_once()
        fetch.assert_called_once()
        print(
            json.dumps(
                {
                    "analyses": AnalysisRun.objects.count(),
                    "predictions": Prediction.objects.count(),
                    "listings": Listing.objects.filter(
                        provider_symbol__in=["AAPL", "MSFT", "CHEAP"]
                    ).count(),
                    "history_requests": fetch.call_count,
                    "frequency_assets": frequency_assets.count(),
                }
            )
        )
    connections["default"].close()


if __name__ == "__main__":
    main()
