import os

os.environ.setdefault("DJANGO_DEBUG", "true")

from .base import *  # noqa: F403

SESSION_COOKIE_SECURE = env_bool("DJANGO_SESSION_COOKIE_SECURE", False)  # noqa: F405
CSRF_COOKIE_SECURE = env_bool("DJANGO_CSRF_COOKIE_SECURE", False)  # noqa: F405
