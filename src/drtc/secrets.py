"""Tokens for the machine lab runs on: the host's `secrets` file in lab/hosts.yaml (KEY=VALUE lines, not in git).

Loaded by every `lab` command at start, so HF_TOKEN and WANDB_API_KEY reach training on each server without being
sourced by hand. A key already in the environment wins; values are never printed.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, sep, value = line.partition("=")
        if sep and key.strip().isidentifier():
            out[key.strip()] = value.strip().strip("'\"")
    return out


def load(reg=None) -> list[str]:
    """Set the current host's secrets that are not set yet; -> their names."""
    try:
        from drtc.registry import Registry

        host = (reg or Registry()).current_host()
    except Exception:  # noqa: BLE001 - no registry or unknown host: nothing to load
        return []
    if host is None or not host.secrets:
        return []
    path = Path(os.path.expanduser(host.secrets))
    if not path.is_file():
        return []
    added = []
    for k, v in parse(path.read_text()).items():
        if k not in os.environ:
            os.environ[k] = v
            added.append(k)
    return added
