from __future__ import annotations

import os
import re
import shlex
import stat
from pathlib import Path

_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def validate_private_environment_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError("The scheduled environment file is missing")
    metadata = path.stat()
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid != os.getuid():
        raise ValueError("The scheduled environment file must be owned by the current user")
    if mode & 0o077:
        raise ValueError(
            "The scheduled environment file must not be readable or writable "
            "by group or other users"
        )


def load_private_environment_file(path: Path) -> None:
    validate_private_environment_file(path)
    assignments: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise ValueError(
                f"The scheduled environment file has an invalid assignment on line {line_number}"
            )
        assignments[name] = _parse_environment_value(raw_value, line_number=line_number)

    os.environ.update(assignments)


def _parse_environment_value(raw_value: str, *, line_number: int) -> str:
    lexer = shlex.shlex(raw_value, posix=True)
    lexer.commenters = "#"
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise ValueError(
            f"The scheduled environment file has invalid quoting on line {line_number}"
        ) from exc
    if len(tokens) > 1:
        raise ValueError(
            f"The scheduled environment file has an unquoted value on line {line_number}"
        )
    return tokens[0] if tokens else ""
