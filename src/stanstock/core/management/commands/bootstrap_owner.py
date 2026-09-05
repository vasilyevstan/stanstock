from __future__ import annotations

import os

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction


class Command(BaseCommand):
    help = "Create or update the single StanStock owner from environment variables."

    def handle(self, *args: object, **options: object) -> None:
        username = os.getenv("STANSTOCK_OWNER_USERNAME", "admin").strip()
        password = os.getenv("STANSTOCK_OWNER_PASSWORD", "")
        email = os.getenv("STANSTOCK_OWNER_EMAIL", "").strip()

        if not password:
            raise CommandError("STANSTOCK_OWNER_PASSWORD is required")

        user_model = get_user_model()
        candidate = user_model.objects.filter(username=username).first() or user_model(
            username=username,
            email=email,
        )
        try:
            validate_password(password, user=candidate)
        except ValidationError as exc:
            raise CommandError("; ".join(exc.messages)) from exc

        with transaction.atomic():
            user, created = user_model.objects.get_or_create(
                username=username,
                defaults={"email": email, "is_staff": True, "is_superuser": True},
            )
            user.email = email
            user.is_staff = True
            user.is_superuser = True
            user.set_password(password)
            user.save(update_fields=["email", "is_staff", "is_superuser", "password"])

        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} StanStock owner: {username}"))
