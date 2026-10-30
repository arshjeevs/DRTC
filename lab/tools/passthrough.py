"""Pass-through of upstream CLI flags (lerobot-record, lerobot-train) with override-by-key."""

from __future__ import annotations

import sys


def parse_passthrough(tokens: list[str]) -> dict[str, str]:
    """['--a.b=1', '--c', '2', '--flag'] -> {'--a.b': '1', '--c': '2', '--flag': 'true'}."""
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            sys.exit(f"Unexpected argument {token!r}: upstream flags look like --key=value")
        if "=" in token:
            key, value = token.split("=", 1)
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            key, value = token, tokens[i + 1]
            i += 1
        else:
            key, value = token, "true"
        flags[key] = value
        i += 1
    return flags
