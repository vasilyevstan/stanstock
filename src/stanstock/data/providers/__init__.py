"""Raw provider clients for StanStock.

Modules under this package talk to external data sources over HTTP and return
small, explicit dataclasses (see ``contracts``) or raw bytes plus metadata.
They never normalize into research/simulation domain models directly, never
solve or bypass bot/anti-automation challenges, and never scrape HTML in
place of a documented machine-readable endpoint.

``research`` and ``simulation`` must not import from this package; an
architecture test enforces that boundary. Callers in ``stanstock.data``
normalize provider output into ``DataAsset``/``FundamentalFact``/``FxRate``
rows via ``stanstock.data.assets`` and ``stanstock.data.asof``.
"""

from __future__ import annotations
