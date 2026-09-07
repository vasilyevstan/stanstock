from __future__ import annotations

from stanstock.data.models import Listing, Security

STOCK_RESEARCH_SECURITY_TYPES = frozenset(
    {
        Security.SecurityType.COMMON_STOCK,
        Security.SecurityType.ADR,
    }
)


def require_stock_research_listing(
    listing: Listing,
    *,
    operation: str,
) -> None:
    security_type = listing.security.security_type
    if security_type in STOCK_RESEARCH_SECURITY_TYPES:
        return
    security_label = listing.security.get_security_type_display().lower()
    raise ValueError(
        f"{operation} supports common stocks and depositary receipts; "
        f"{listing.ticker} is an {security_label}."
    )
