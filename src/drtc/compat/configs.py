"""Old checkpoint config.json -> a config the new policy class accepts.

Old configs (LeRobot 0.4.3 fork) can carry fields the 0.7 class no longer has, and GR00T N1.5's type `groot` now
belongs to N1.7 upstream. Fields are only dropped, never invented; every drop is logged, and the parity check
(tests/parity) shows whether a dropped field mattered.
"""

from __future__ import annotations

import dataclasses
import logging

TYPE_MAP = {"groot": "groot_n15"}
# Type `groot` is ambiguous: our old GR00T N1.5 checkpoints (the 0.4.3 fork) and GR00T N1.7 trained with 0.7 (upstream's
# `groot`) both say it. Upstream N1.7 configs always carry fields the fork never had; any of them means N1.7, which
# loads as it is (2026-11-25: N1.7 was served as N1.5 and failed, `no attribute 'backbone_cfg'`).
N17_FIELDS = {"use_relative_actions", "num_inference_timesteps", "action_decode_transform", "rtc_ramp_rate",
              "model_params_fp32", "tune_vlln", "tune_top_llm_layers"}


def is_old_type(raw: dict) -> bool:
    """An old policy type that 0.7 reads differently (TYPE_MAP), and not a 0.7 config of that same type."""
    if raw.get("type") not in TYPE_MAP:
        return False
    return not (raw.get("type") == "groot" and N17_FIELDS & set(raw))


def upgrade_config(raw: dict, config_class) -> dict:
    out = dict(raw)
    if is_old_type(out):
        out["type"] = TYPE_MAP[out["type"]]
    known = {f.name for f in dataclasses.fields(config_class)} | {"type"}
    dropped = sorted(set(out) - known)
    for k in dropped:
        out.pop(k)
    if dropped:
        print(f"config: dropped fields unknown to {config_class.__name__}: {', '.join(dropped)}", flush=True)
    return out
