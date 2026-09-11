from typing import Any

from django.contrib import admin
from django.db import transaction
from django.db.models import QuerySet
from django.http import HttpRequest

from stanstock.core.admin_mixins import ReadOnlyModelAdmin
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioDeposit,
    PortfolioHolding,
    PortfolioPerformanceBaseline,
    PortfolioPlanExecution,
    PortfolioPurchase,
    PortfolioSnapshot,
    PortfolioSnapshotHolding,
    TrackedSymbol,
)


@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    _manual_update_fields = frozenset(
        {
            "name",
            "description",
            "monthly_contribution",
            "allow_fractional_shares",
            "archived_at",
        }
    )

    def get_readonly_fields(
        self,
        request: HttpRequest,
        obj: Portfolio | None = None,
    ) -> tuple[str, ...]:
        if obj is not None and obj.is_model_portfolio:
            return tuple(
                field.name for field in obj._meta.fields if field.name not in {"archived_at"}
            )
        if obj is not None:
            return (
                "owner",
                "base_currency",
                "cash_balance",
                "source_analysis_run",
                "construction_policy",
                "construction_metadata",
                "starting_capital",
                "created_at",
                "updated_at",
            )
        return ()

    def has_add_permission(self, request: HttpRequest) -> bool:
        return False

    def save_model(
        self,
        request: HttpRequest,
        obj: Portfolio,
        form: Any,
        change: bool,
    ) -> None:
        if not change:
            super().save_model(request, obj, form, change)
            return
        with transaction.atomic():
            locked = Portfolio.objects.select_for_update().get(pk=obj.pk)
            allowed_fields = (
                {"archived_at"} if locked.is_model_portfolio else self._manual_update_fields
            )
            update_fields = sorted(set(form.changed_data) & allowed_fields)
            for field_name in update_fields:
                setattr(locked, field_name, getattr(obj, field_name))
            if update_fields:
                locked.save(update_fields=[*update_fields, "updated_at"])
            obj.refresh_from_db()

    def has_delete_permission(
        self,
        request: HttpRequest,
        obj: Portfolio | None = None,
    ) -> bool:
        if obj is not None and obj.is_model_portfolio:
            return False
        return super().has_delete_permission(request, obj)


@admin.register(PortfolioHolding)
class PortfolioHoldingAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    def get_readonly_fields(
        self,
        request: HttpRequest,
        obj: PortfolioHolding | None = None,
    ) -> tuple[str, ...]:
        if obj is not None and obj.portfolio.is_model_portfolio:
            return tuple(field.name for field in obj._meta.fields)
        if obj is not None:
            return tuple(field.name for field in obj._meta.fields if field.name != "notes")
        return ()

    def has_add_permission(self, request: HttpRequest) -> bool:
        return False

    def has_delete_permission(
        self,
        request: HttpRequest,
        obj: PortfolioHolding | None = None,
    ) -> bool:
        return False

    def save_model(
        self,
        request: HttpRequest,
        obj: PortfolioHolding,
        form: Any,
        change: bool,
    ) -> None:
        if not change:
            super().save_model(request, obj, form, change)
            return
        with transaction.atomic():
            portfolio = Portfolio.objects.select_for_update().get(pk=obj.portfolio_id)
            locked = (
                PortfolioHolding.objects.select_for_update(of=("self",))
                .order_by("pk")
                .get(pk=obj.pk)
            )
            if portfolio.is_model_portfolio:
                obj.refresh_from_db()
                return
            if "notes" in form.changed_data:
                locked.notes = obj.notes.strip()
                locked.save(update_fields=["notes", "updated_at"])
            obj.refresh_from_db()


@admin.register(TrackedSymbol)
class TrackedSymbolAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    list_display = ("symbol", "owner", "created_at")
    search_fields = ("symbol", "owner__username")
    readonly_fields = ("owner", "symbol", "created_at")

    def get_queryset(self, request: HttpRequest) -> QuerySet[TrackedSymbol]:
        queryset = super().get_queryset(request)
        if request.user.is_superuser:
            return queryset
        return queryset.filter(owner=request.user)

    def has_add_permission(self, request: HttpRequest) -> bool:
        return False

    def has_change_permission(
        self,
        request: HttpRequest,
        obj: TrackedSymbol | None = None,
    ) -> bool:
        return False

    def has_delete_permission(
        self,
        request: HttpRequest,
        obj: TrackedSymbol | None = None,
    ) -> bool:
        return obj is None or request.user.is_superuser or obj.owner_id == request.user.pk


admin.site.register(PortfolioSnapshot, ReadOnlyModelAdmin)
admin.site.register(PortfolioSnapshotHolding, ReadOnlyModelAdmin)
admin.site.register(PortfolioDeposit, ReadOnlyModelAdmin)
admin.site.register(PortfolioPlanExecution, ReadOnlyModelAdmin)
admin.site.register(PortfolioPurchase, ReadOnlyModelAdmin)
admin.site.register(PortfolioPerformanceBaseline, ReadOnlyModelAdmin)
