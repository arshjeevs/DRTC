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


# --- π0.5 / MolmoAct 2 feature admission (the checkpoint keeps its base model's nominal input_features) ----------

from types import SimpleNamespace

from lerobot.utils.feature_spec import FeatureSpec


def feat(name, shape, kind):
    return FeatureSpec(name=name, shape=tuple(shape), dtype="uint8" if kind == "rgb" else "float32", kind=kind,
                       semantics="test-v1")


PI05_BASE = {  # lerobot/pi05_base input_features
    "observation.images.base_0_rgb": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
    "observation.images.left_wrist_0_rgb": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
    "observation.images.right_wrist_0_rgb": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
    "observation.state": PolicyFeature(FeatureType.STATE, (32,)),
}
OURS_PI05 = (feat("observation.state", (6,), "tensor"), feat("observation.images.right_wrist_0_rgb", (600, 800, 3), "rgb"),
             feat("observation.images.base_0_rgb", (600, 800, 3), "rgb"))


def test_pi05_admits_our_rig():
    fake = SimpleNamespace(config=SimpleNamespace(input_features=PI05_BASE))
    chunk_adapters.validate_pi05_features(fake, OURS_PI05)  # 2 of 3 slots, 800x600, 6-d state


def test_pi05_refusals():
    fake = SimpleNamespace(config=SimpleNamespace(input_features=PI05_BASE))
    with pytest.raises(ValueError, match="unknown"):
        chunk_adapters.validate_pi05_features(fake, OURS_PI05 + (feat("observation.images.cam9", (600, 800, 3), "rgb"),))
    with pytest.raises(ValueError, match="does not fit"):
        chunk_adapters.validate_pi05_features(fake, (feat("observation.state", (40,), "tensor"),) + OURS_PI05[1:])
    with pytest.raises(ValueError, match="Missing non-camera"):
        chunk_adapters.validate_pi05_features(fake, OURS_PI05[1:])


def test_molmoact2_admits_any_resolution_but_exact_names():
    base = {"observation.images.cam0": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.images.cam1": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.state": PolicyFeature(FeatureType.STATE, (6,))}
    fake = SimpleNamespace(config=SimpleNamespace(input_features=base))
    ours = (feat("observation.state", (6,), "tensor"), feat("observation.images.cam1", (600, 800, 3), "rgb"),
            feat("observation.images.cam0", (600, 800, 3), "rgb"))
    chunk_adapters.validate_molmoact2_features(fake, ours)
    with pytest.raises(ValueError, match="exactly match"):
        chunk_adapters.validate_molmoact2_features(fake, ours[:2])


def test_installed_on_the_policies():
    chunk_adapters.install()
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    assert PI05Policy.validate_chunk_input_features is chunk_adapters.validate_pi05_features
    assert MolmoAct2Policy.validate_chunk_input_features is chunk_adapters.validate_molmoact2_features
    from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import ActionExpert

    assert ActionExpert.get_or_prepare_modulation_cache.__wrapped__ is not None


def test_molmoact2_modulation_cache_survives_chunk_then_rtc():
    # The H200 serve check: chunk warmup under inference_mode, then guided RTC differentiates through the expert.
    from lerobot.policies.molmoact2.molmoact2_hf_model.configuration_molmoact2 import MolmoAct2ActionExpertConfig
    from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import ActionExpert

    chunk_adapters.install()
    cfg = MolmoAct2ActionExpertConfig(max_action_horizon=4, max_action_dim=6, hidden_size=32, num_layers=1,
                                      num_heads=2, ffn_multiple_of=8, timestep_embed_dim=16)
    expert = ActionExpert(cfg, llm_dim=16, llm_kv_dim=16, llm_num_layers=1).eval()
    steps = [torch.full((1,), i / 2) for i in range(2)]
    key = (2, 1, torch.device("cpu"), torch.float32)

    def modulated(x):
        mod = expert.get_or_prepare_modulation_cache(steps, cache_key=key)[0]
        return (x * mod.conditioning).sum()  # saves the cached tensor for backward, as the expert's gates do

    with torch.inference_mode():
        modulated(torch.ones(1, 32))  # chunk mode: the cache is created as inference tensors
    with torch.no_grad(), torch.enable_grad():  # guided RTC's correction
        x = torch.ones(1, 32, requires_grad=True)
        (grad,) = torch.autograd.grad(modulated(x), x)
    assert torch.isfinite(grad).all()
    first = expert._modulation_cache_value
    with torch.inference_mode():
        modulated(torch.ones(1, 32))
    assert expert._modulation_cache_value is first  # rebuilt once, then reused in both modes
