from django.contrib import admin

from stanstock.core.admin_mixins import ReadOnlyModelAdmin
from stanstock.data.models import (
    Company,
    DataAsset,
    FundamentalFact,
    FxRate,
    LatestMarketData,
    Listing,
    ProviderRecord,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)

for editable_model in (
    Company,
    Security,
    Listing,
    Universe,
    UniverseSnapshot,
    UniverseMembership,
    ProviderRecord,
):
    admin.site.register(editable_model)

for readonly_model in (
    DataAsset,
    FundamentalFact,
    FxRate,
    LatestMarketData,
):
    admin.site.register(readonly_model, ReadOnlyModelAdmin)
