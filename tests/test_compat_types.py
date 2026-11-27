"""Type `groot`: our old GR00T N1.5 checkpoints need the compat copy; GR00T N1.7 trained with 0.7 loads as it is."""

from __future__ import annotations

import pytest

from drtc.compat.configs import is_old_type, upgrade_config


def test_old_groot_config_is_old():
    assert is_old_type({"type": "groot", "chunk_size": 16, "tune_llm": False})


def test_n17_groot_config_is_not_old():
    assert not is_old_type({"type": "groot", "chunk_size": 16, "use_relative_actions": True,
                            "num_inference_timesteps": 4})


@pytest.mark.parametrize("t", ["groot_n15", "smolvla", "pi05", "molmoact2", None])
def test_other_types_are_not_old(t):
    assert not is_old_type({"type": t})


def test_upgrade_keeps_n17_type():
    from lerobot.policies.groot.configuration_groot import GrootConfig

    out = upgrade_config({"type": "groot", "use_relative_actions": True}, GrootConfig)
    assert out["type"] == "groot"


@pytest.mark.parametrize("hub,old", [("orangemoon599/grootn17_sort_one_v1_full", False),
                                     ("orangemoon599/grootn17_cubebox_v1_full", False),
                                     ("orangemoon599/groot_cubebox_v1_lora_raw", True),
                                     ("orangemoon599/groot_sort_one_v1_full", False)])
def test_hub_checkpoints(hub, old):
    pytest.importorskip("huggingface_hub")
    from drtc.compat.serve import needs_compat

    assert needs_compat(hub) is old
