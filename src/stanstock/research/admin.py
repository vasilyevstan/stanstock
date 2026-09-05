from django.contrib import admin

from stanstock.core.admin_mixins import ReadOnlyModelAdmin
from stanstock.research.models import AnalysisRun, Prediction, PredictionOutcome, StockAnalysis

for model in (AnalysisRun, StockAnalysis, Prediction, PredictionOutcome):
    admin.site.register(model, ReadOnlyModelAdmin)
