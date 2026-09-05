from django.contrib import admin

from stanstock.core.admin_mixins import ReadOnlyModelAdmin
from stanstock.simulation.models import (
    SimulationDefinition,
    SimulationHolding,
    SimulationRun,
    SimulationTrade,
)

admin.site.register(SimulationDefinition)
admin.site.register(SimulationRun, ReadOnlyModelAdmin)
admin.site.register(SimulationHolding, ReadOnlyModelAdmin)
admin.site.register(SimulationTrade, ReadOnlyModelAdmin)
