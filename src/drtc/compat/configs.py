"""Old checkpoint config.json -> a config the new policy class accepts.

Old configs (LeRobot 0.4.3 fork) can carry fields the 0.7 class no longer has, and GR00T N1.5's type `groot` now
belongs to N1.7 upstream. Fields are only dropped, never invented; every drop is logged, and the parity check
(tests/parity) shows whether a dropped field mattered.
"""

from __future__ import annotations

import dataclasses
import logging

TYPE_MAP = {"groot": "groot_n15"}


def upgrade_config(raw: dict, config_class) -> dict:
    out = dict(raw)
    out["type"] = TYPE_MAP.get(out.get("type"), out.get("type"))
    known = {f.name for f in dataclasses.fields(config_class)} | {"type"}
    dropped = sorted(set(out) - known)
    for k in dropped:
        out.pop(k)
    if dropped:
        print(f"config: dropped fields unknown to {config_class.__name__}: {', '.join(dropped)}", flush=True)
    return out
