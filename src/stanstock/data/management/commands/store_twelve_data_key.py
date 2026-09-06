from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand, CommandError

from stanstock.data.provider_credentials import (
    delete_twelve_data_api_key,
    read_twelve_data_api_key,
    store_twelve_data_api_key,
)
from stanstock.data.providers.exceptions import ProviderConfigurationError


class Command(BaseCommand):
    help = (
        "Store, inspect, or delete the Twelve Data API key in the current "
        "macOS user's login keychain without writing it to Git, dotenv files, "
        "the database, or command-line arguments."
    )

    def add_arguments(self, parser: Any) -> None:
        action = parser.add_mutually_exclusive_group()
        action.add_argument(
            "--status",
            action="store_true",
            help="Report whether a key is stored without printing it.",
        )
        action.add_argument(
            "--delete",
            action="store_true",
            help="Delete the stored keychain item.",
        )

    def handle(self, *args: object, **options: object) -> None:
        try:
            if options["status"]:
                stored = read_twelve_data_api_key() is not None
                self.stdout.write(
                    "Twelve Data key is stored in macOS Keychain."
                    if stored
                    else "No Twelve Data key is stored in macOS Keychain."
                )
                return
            if options["delete"]:
                deleted = delete_twelve_data_api_key()
                self.stdout.write(
                    self.style.SUCCESS("Twelve Data key deleted from macOS Keychain.")
                    if deleted
                    else "No Twelve Data key was stored in macOS Keychain."
                )
                return
            store_twelve_data_api_key()
        except ProviderConfigurationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Twelve Data key stored in macOS Keychain. The provider remains "
                "disabled until its display-rights activation gate is satisfied."
            )
        )
