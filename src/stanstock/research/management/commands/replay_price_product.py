from __future__ import annotations

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
            help="Optional explicit output file; stdout is used when omitted",
        )

    def handle(self, *args: object, **options: object) -> None:
        try:
            run = AnalysisRun.objects.select_related("universe_snapshot").get(
                pk=UUID(str(options["run"]))
            )
            listing_ids = _parse_listing_ids(str(options.get("listing_ids") or ""))
            store = open_asset_store()
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
            output_file = options.get("output_file")
            if output_file is None:
                self.stdout.write(rendered)
                return
            if not isinstance(output_file, Path):
                raise CommandError("--output-file must be a filesystem path")
            output_file.write_text(rendered, encoding="utf-8")
        except AnalysisRun.DoesNotExist as exc:
            raise CommandError("Unknown AnalysisRun UUID") from exc
        except OSError as exc:
            raise CommandError("Could not write the requested report output file") from exc
        except (TypeError, ValueError) as exc:
            raise CommandError(str(exc)) from exc


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
