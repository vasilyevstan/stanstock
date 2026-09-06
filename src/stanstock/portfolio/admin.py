from django.contrib import admin

from stanstock.core.admin_mixins import ReadOnlyModelAdmin
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioHolding,
    PortfolioSnapshot,
    PortfolioSnapshotHolding,
)

admin.site.register(Portfolio)
admin.site.register(PortfolioHolding)
admin.site.register(PortfolioSnapshot, ReadOnlyModelAdmin)
admin.site.register(PortfolioSnapshotHolding, ReadOnlyModelAdmin)
