from django.contrib import admin
from django.http import HttpRequest

from stanstock.core.admin_mixins import ReadOnlyModelAdmin
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioHolding,
    PortfolioSnapshot,
    PortfolioSnapshotHolding,
)


@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    def get_readonly_fields(
        self,
        request: HttpRequest,
        obj: Portfolio | None = None,
    ) -> tuple[str, ...]:
        if obj is not None and obj.is_model_portfolio:
            return tuple(
                field.name for field in obj._meta.fields if field.name not in {"archived_at"}
            )
        return ()

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
        return ()

    def has_delete_permission(
        self,
        request: HttpRequest,
        obj: PortfolioHolding | None = None,
    ) -> bool:
        if obj is not None and obj.portfolio.is_model_portfolio:
            return False
        return super().has_delete_permission(request, obj)


admin.site.register(PortfolioSnapshot, ReadOnlyModelAdmin)
admin.site.register(PortfolioSnapshotHolding, ReadOnlyModelAdmin)
