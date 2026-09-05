from __future__ import annotations

import hashlib
from collections.abc import Callable

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse


class SecurityHeadersMiddleware:
    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "base-uri 'self'; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'; "
            "img-src 'self' data:; "
            "object-src 'none'; "
            "script-src 'self'; "
            "style-src 'self'"
        )
        response.headers["Permissions-Policy"] = (
            "camera=(), geolocation=(), microphone=(), payment=(), usb=()"
        )
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
        user = getattr(request, "user", None)
        if (user is not None and user.is_authenticated) or request.path_info in {
            "/healthz",
            "/accounts/login/",
            "/accounts/logout/",
        }:
            response.headers["Cache-Control"] = "no-store"
        return response


class LoginRateLimitMiddleware:
    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        if request.method != "POST" or request.path_info not in settings.LOGIN_RATE_LIMIT_PATHS:
            return self.get_response(request)

        cache_key = self._cache_key(request)
        attempts = int(cache.get(cache_key, 0))
        if attempts >= settings.LOGIN_RATE_LIMIT_ATTEMPTS:
            response = HttpResponse(
                "Too many sign-in attempts. Try again later.",
                status=429,
                content_type="text/plain",
            )
            response.headers["Retry-After"] = str(settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS)
            return response

        response = self.get_response(request)
        if request.user.is_authenticated:
            cache.delete(cache_key)
        elif response.status_code == 200:
            if not cache.add(cache_key, 1, settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS):
                cache.incr(cache_key)
        return response

    @staticmethod
    def _cache_key(request: HttpRequest) -> str:
        remote_address = LoginRateLimitMiddleware._client_address(request)
        digest = hashlib.sha256(
            f"{request.path_info}:{remote_address}".encode(),
        ).hexdigest()
        return f"stanstock:login-attempts:{digest}"

    @staticmethod
    def _client_address(request: HttpRequest) -> str:
        remote_address = str(request.META.get("REMOTE_ADDR", "unknown"))
        if remote_address not in settings.LOGIN_RATE_LIMIT_TRUSTED_PROXY_IPS:
            return remote_address
        forwarded_for = str(request.META.get("HTTP_X_FORWARDED_FOR", ""))
        client_address = forwarded_for.split(",", maxsplit=1)[0].strip()
        return client_address or remote_address
