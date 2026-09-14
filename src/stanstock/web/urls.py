from __future__ import annotations

from django.urls import path

from stanstock.web import product_views, views

urlpatterns = [
    path("", views.index, name="index"),
    path("healthz", views.health, name="health"),
    path("status", product_views.status_page, name="status"),
    path("opportunities", product_views.opportunities_page, name="opportunities"),
    path("stocks/<uuid:listing_id>", product_views.stock_detail_page, name="stock-detail"),
    path("etfs/<uuid:listing_id>", views.etf_detail_page, name="etf-detail"),
    path("market", views.market_overview_page, name="market"),
    path("predictions", product_views.prediction_history_page, name="predictions"),
    path("performance", product_views.performance_page, name="performance"),
    path("my-list", product_views.my_list_page, name="my-list"),
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
    path(
        "archive/opportunities",
        product_views.archive_opportunities_page,
        name="archive-opportunities",
    ),
    path(
        "archive/stocks/<uuid:listing_id>",
        product_views.archive_stock_detail_page,
        name="archive-stock-detail",
    ),
    path(
        "archive/predictions",
        product_views.archive_prediction_history_page,
        name="archive-predictions",
    ),
    path(
        "archive/performance",
        product_views.archive_performance_page,
        name="archive-performance",
    ),
]
