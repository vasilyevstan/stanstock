from __future__ import annotations

from urllib.parse import urlencode

from django.http import HttpRequest, HttpResponse, HttpResponseForbidden
from django.template import loader
from django.urls import reverse
from django.utils.cache import add_never_cache_headers
from django.utils.http import url_has_allowed_host_and_scheme


def csrf_failure(request: HttpRequest, reason: str = "") -> HttpResponse:
    """Render a public, non-replaying recovery page for rejected CSRF requests."""

    del reason
    login_url = reverse("login")
    is_login_recovery = request.path == login_url
    recovery_url = login_url if is_login_recovery else reverse("index")

    if is_login_recovery:
        next_url = request.GET.get("next")
        if next_url and url_has_allowed_host_and_scheme(
            next_url,
            allowed_hosts={request.get_host()},
            require_https=request.is_secure(),
        ):
            recovery_url = f"{login_url}?{urlencode({'next': next_url})}"

    content = loader.get_template("403_csrf.html").render(
        {
            "recovery_url": recovery_url,
            "is_login_recovery": is_login_recovery,
        }
    )
    response = HttpResponseForbidden(content)
    add_never_cache_headers(response)
    return response
