from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parents[3]


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


DEBUG = env_bool("DJANGO_DEBUG", False)

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = "stanstock-development-only-secret-key"
    else:
        raise ImproperlyConfigured("DJANGO_SECRET_KEY is required when DJANGO_DEBUG is false")

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1]")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "stanstock.core",
    "stanstock.data",
    "stanstock.research",
    "stanstock.simulation",
    "stanstock.portfolio",
    "stanstock.web",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "stanstock.web.middleware.SecurityHeadersMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "stanstock.web.middleware.PersonalProviderAccessMiddleware",
    "stanstock.web.middleware.LoginRateLimitMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "stanstock.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "stanstock.web.context_processors.stanstock_runtime",
            ],
        },
    },
]

WSGI_APPLICATION = "stanstock.wsgi.application"
ASGI_APPLICATION = "stanstock.asgi.application"


_SQLITE_OPTIONS: dict[str, object] = {
    "timeout": 20,
    "transaction_mode": "IMMEDIATE",
    "init_command": ("PRAGMA journal_mode=WAL;PRAGMA synchronous=NORMAL;PRAGMA busy_timeout=20000"),
}


def _resolve_sqlite_path(raw_path: str) -> Path:
    expanded = Path(raw_path).expanduser()
    if not expanded.is_absolute():
        raise ImproperlyConfigured(
            "STANSTOCK_SQLITE_PATH must be an absolute path (after expanding '~')"
        )
    return expanded.resolve()


def database_config() -> dict[str, object]:
    database_url = os.getenv("DATABASE_URL", "").strip()
    sqlite_path_raw = os.getenv("STANSTOCK_SQLITE_PATH", "").strip()
    if database_url and sqlite_path_raw:
        raise ImproperlyConfigured(
            "DATABASE_URL and STANSTOCK_SQLITE_PATH cannot both be set; "
            "choose exactly one database backend."
        )
    if not database_url:
        sqlite_name = (
            _resolve_sqlite_path(sqlite_path_raw)
            if sqlite_path_raw
            else BASE_DIR / "stanstock.sqlite3"
        )
        return {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": sqlite_name,
            "OPTIONS": dict(_SQLITE_OPTIONS),
        }

    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ImproperlyConfigured("DATABASE_URL must use postgres:// or postgresql://")

    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": parsed.path.lstrip("/"),
        "USER": unquote(parsed.username or ""),
        "PASSWORD": unquote(parsed.password or ""),
        "HOST": parsed.hostname or "",
        "PORT": parsed.port or 5432,
        "CONN_MAX_AGE": int(os.getenv("DATABASE_CONN_MAX_AGE", "60")),
        "OPTIONS": {"connect_timeout": 5},
    }


DATABASES = {"default": database_config()}

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "stanstock",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "var" / "static"
STATICFILES_DIRS = [BASE_DIR / "static"]
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "status"
LOGOUT_REDIRECT_URL = "login"

DATA_DIR = Path(os.getenv("STANSTOCK_DATA_DIR", BASE_DIR / "var" / "data")).resolve()
BACKUP_DIR = Path(os.getenv("STANSTOCK_BACKUP_DIR", BASE_DIR / "var" / "backups")).resolve()
DEMO_MODE = env_bool("STANSTOCK_DEMO_MODE", DEBUG)
OWNER_USERNAME = os.getenv("STANSTOCK_OWNER_USERNAME", "admin")
LOGIN_RATE_LIMIT_ATTEMPTS = int(os.getenv("STANSTOCK_LOGIN_RATE_LIMIT_ATTEMPTS", "5"))
LOGIN_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("STANSTOCK_LOGIN_RATE_LIMIT_WINDOW_SECONDS", "900"))
LOGIN_RATE_LIMIT_PATHS = {"/accounts/login/", "/admin/login/"}
LOGIN_RATE_LIMIT_TRUSTED_PROXY_IPS = {
    value.strip()
    for value in os.getenv("STANSTOCK_LOGIN_TRUSTED_PROXY_IPS", "").split(",")
    if value.strip()
}

SECURE_SSL_REDIRECT = env_bool("DJANGO_SECURE_SSL_REDIRECT", False)
SECURE_REDIRECT_EXEMPT = [r"^healthz$"]
SESSION_COOKIE_SECURE = env_bool("DJANGO_SESSION_COOKIE_SECURE", not DEBUG)
CSRF_COOKIE_SECURE = env_bool("DJANGO_CSRF_COOKIE_SECURE", not DEBUG)
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_AGE = int(os.getenv("DJANGO_SESSION_COOKIE_AGE", "43200"))
SESSION_EXPIRE_AT_BROWSER_CLOSE = env_bool("DJANGO_SESSION_EXPIRE_AT_BROWSER_CLOSE", True)
CSRF_COOKIE_SAMESITE = "Lax"
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
SECURE_REFERRER_POLICY = "same-origin"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structured": {
            "()": "stanstock.core.logging.JsonFormatter",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "structured",
        }
    },
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
}
