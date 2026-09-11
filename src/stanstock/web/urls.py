from __future__ import annotations

from django.urls import path

from stanstock.web import views

urlpatterns = [
    path("", views.index, name="index"),
    path("healthz", views.health, name="health"),
    path("status", views.status_page, name="status"),
    path("opportunities", views.opportunities_page, name="opportunities"),
    path("stocks/<uuid:listing_id>", views.stock_detail_page, name="stock-detail"),
    path("etfs/<uuid:listing_id>", views.etf_detail_page, name="etf-detail"),
    path("market", views.market_overview_page, name="market"),
    path("predictions", views.prediction_history_page, name="predictions"),
    path("performance", views.performance_page, name="performance"),
    path("my-list", views.my_list_page, name="my-list"),
    path(
        "my-list/<uuid:tracked_symbol_id>/delete",
        views.tracked_symbol_delete,
        name="tracked-symbol-delete",
    ),
    path("portfolios", views.portfolios_page, name="portfolios"),
    path(
        "portfolios/<uuid:portfolio_id>",
        views.portfolio_detail_page,
        name="portfolio-detail",
    ),
    path(
        "portfolios/<uuid:portfolio_id>/holdings/<int:holding_id>/delete",
        views.portfolio_holding_delete,
        name="portfolio-holding-delete",
    ),
    path("simulations", views.simulations_page, name="simulations"),
    path(
        "simulations/<uuid:run_id>",
        views.simulation_detail_page,
        name="simulation-detail",
    ),
    path("methodology", views.methodology_page, name="methodology"),
]
