from __future__ import annotations

from django.contrib import admin
from django.db import models
from django.http import HttpRequest


class ReadOnlyModelAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    def has_add_permission(self, request: HttpRequest) -> bool:
        return False

    def has_change_permission(
        self,
        request: HttpRequest,
        obj: models.Model | None = None,
    ) -> bool:
        return False

    def has_delete_permission(
        self,
        request: HttpRequest,
        obj: models.Model | None = None,
    ) -> bool:
        return False

    def has_view_permission(
        self,
        request: HttpRequest,
        obj: models.Model | None = None,
    ) -> bool:
        return super().has_view_permission(request, obj)
