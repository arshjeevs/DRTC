"""Diffusion Policy chunk serving on LeRobot 0.7 (drtc.control.chunk_adapters), on a tiny CPU model."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from drtc.control import chunk_adapters


def tiny(n_obs_steps=1):
    cfg = DiffusionConfig(
        device="cpu", n_obs_steps=n_obs_steps, horizon=16, n_action_steps=8, drop_n_last_frames=8 - n_obs_steps,
        down_dims=(32, 64), crop_shape=None, pretrained_backbone_weights=None, use_group_norm=True,
        num_inference_steps=2, noise_scheduler_type="DDIM",
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (6,)),
                        "observation.images.scene": PolicyFeature(FeatureType.VISUAL, (3, 64, 64))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
    )
    return DiffusionPolicy(cfg).eval()


def server_batch():
    # What the policy server hands predict_action_chunk: one observation, no time axis.
    return {"observation.state": torch.zeros(1, 6), "observation.images.scene": torch.rand(1, 3, 64, 64)}


def test_spec_and_chunk_shape():
    chunk_adapters.install()
    policy = tiny()
    spec = policy.chunk_inference_spec()
    assert (spec.prediction_steps, spec.execution_steps) == (8, 8)
    with torch.no_grad():
        actions = policy.predict_action_chunk(server_batch())
    assert actions.shape == (1, 8, 6)


def test_observation_history_is_refused():
    chunk_adapters.install()
    with pytest.raises(ValueError, match="n_obs_steps=1"):
        tiny(n_obs_steps=2).chunk_inference_spec()
