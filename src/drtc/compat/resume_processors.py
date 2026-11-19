"""`lerobot-train --resume` works for policies whose processor pipeline has no (un)normalizer step.

When lerobot-train loads a saved processor (on resume), it always passes overrides for the steps
`normalizer_processor` and `unnormalizer_processor`, and PolicyProcessorPipeline.from_pretrained refuses
overrides that match no step. GR00T N1.5 (our plugin) normalizes inside its own steps (groot_n15_pack_inputs_v3,
groot_n15_action_unpack_unnormalize_v1), so its resume failed at startup (round 1, 2026-11-19). On resume those
two overrides only carry features and norm_map (no stats: the checkpoint's saved stats are kept), so dropping them
when the pipeline has no such step changes nothing. Any other unmatched override still raises. Upstream code is
not edited.
"""

from __future__ import annotations

import logging

DROPPABLE = frozenset({"normalizer_processor", "unnormalizer_processor"})


def install() -> None:
    from lerobot.processor.pipeline import DataProcessorPipeline

    original = DataProcessorPipeline.__dict__["_validate_overrides_used"]
    if getattr(original.__func__, "_drtc", False):
        return
    validate = original.__func__

    def _validate_overrides_used(cls, remaining_override_keys, loaded_config):
        dropped = set(remaining_override_keys) & DROPPABLE
        if dropped:
            logging.info(f"processor: no step for overrides {sorted(dropped)} (normalization is inside the "
                         f"policy's own steps); ignored")
        return validate(cls, set(remaining_override_keys) - DROPPABLE, loaded_config)

    _validate_overrides_used._drtc = True
    DataProcessorPipeline._validate_overrides_used = classmethod(_validate_overrides_used)
