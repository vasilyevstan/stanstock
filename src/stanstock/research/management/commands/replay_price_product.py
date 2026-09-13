from __future__ import annotations

import os
from argparse import ArgumentParser
from pathlib import Path
from uuid import UUID

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.data.assets import open_asset_store
from stanstock.research.models import AnalysisRun
from stanstock.research.price_product_study import (
    render_price_product_study,
    study_price_product_run,
)


class Command(BaseCommand):
    help = (
        "Read-only retrospective replay of research-product-v1 against the exact immutable "
        "selected source run. This command never contacts a provider, resolves credentials, "
        "creates observed output, or writes database/asset evidence."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--run", required=True, help="Immutable AnalysisRun UUID to replay")
        parser.add_argument(
            "--all-selected",
            action="store_true",
            help="Replay every listing selected in the source run",
        )
        parser.add_argument(
            "--listing-ids",
            default="",
            help=(
                "Comma-separated immutable Listing UUIDs to replay. "
                "Provide exactly one of --all-selected or --listing-ids."
            ),
        )
        parser.add_argument(
            "--format",
            required=True,
            choices=("json", "text"),
            help="Explicit output format",
        )
        parser.add_argument(
            "--output-file",
            type=Path,
            default=None,
            help="New private report file outside DATA_DIR; existing files are never overwritten",
        )

    def handle(self, *args: object, **options: object) -> None:
        try:
            run = AnalysisRun.objects.select_related("universe_snapshot").get(
                pk=UUID(str(options["run"]))
            )
            listing_ids = _parse_listing_ids(str(options.get("listing_ids") or ""))
            store = open_asset_store()
            output_file = options.get("output_file")
            if output_file is not None:
                if not isinstance(output_file, Path):
                    raise CommandError("--output-file must be a filesystem path")
                if output_file.exists() or output_file.is_symlink():
                    raise CommandError("Report output must be a new file")
                if output_file.resolve().is_relative_to(store.root):
                    raise CommandError("Report output must be outside the immutable asset store")
            report = study_price_product_run(
                run=run,
                store=store,
                all_selected=bool(options["all_selected"]),
                listing_ids=listing_ids,
                report_generated_at=timezone.now(),
            )
            rendered = render_price_product_study(
                report,
                output_format=str(options["format"]),
                include_generated_at=True,
            )
            if output_file is None:
                self.stdout.write(rendered)
                return
            _write_private_report(output_file, rendered)
        except AnalysisRun.DoesNotExist as exc:
            raise CommandError("Unknown AnalysisRun UUID") from exc
        except OSError as exc:
            raise CommandError("Could not write the requested report output file") from exc
        except (TypeError, ValueError) as exc:
            raise CommandError(str(exc)) from exc


def _write_private_report(path: Path, rendered: str) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(rendered)
    except OSError:
        path.unlink(missing_ok=True)
        raise


def _parse_listing_ids(raw: str) -> tuple[UUID, ...]:
    if not raw:
        return ()
    parsed: list[UUID] = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        try:
            parsed.append(UUID(item))
        except ValueError as exc:
            raise CommandError("--listing-ids must be comma-separated Listing UUIDs") from exc
    if len(set(parsed)) != len(parsed):
        raise CommandError("--listing-ids must not contain duplicates")
    return tuple(parsed)
