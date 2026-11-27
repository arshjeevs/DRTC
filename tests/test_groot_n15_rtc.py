"""GR00T N1.5 plugin: guided RTC on LeRobot 0.7 (supports_rtc, RTCProcessor wiring, padded action dims).

Runs without the 3B weights: the policy is built without __init__ and the action head's Euler loop is replayed
with a toy velocity field, exactly as `FlowmatchingActionHead.get_action` calls the guidance.
"""

import pytest
import torch

pytest.importorskip("lerobot_policy_groot_n15", reason="the GPU extra (uv sync --extra gpu) installs the plugin")

from lerobot.inference.rtc import supports_rtc_inference  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot_policy_groot_n15.configuration_groot_n15 import GrootN15Config  # noqa: E402
from lerobot_policy_groot_n15.modeling_groot_n15 import GrootN15Policy  # noqa: E402

H, WIDTH, PADDED, STEPS = 16, 6, 32, 4


def shell(rtc_config):
    policy = GrootN15Policy.__new__(GrootN15Policy)
    torch.nn.Module.__init__(policy)
    policy.config = GrootN15Config(chunk_size=H, n_action_steps=H, rtc_config=rtc_config)
    policy.init_rtc_processor()
    return policy


def integrate(guidance, noise, target):
    """GR00T's loop: t 0 (noise) -> 1 (clean), x += dt * v; the toy field flows straight to `target`."""

    def velocity(x):
        return target - noise + 0.0 * x.sum()  # depends on x so autograd has a graph, like the real head

    actions, dt = noise.clone(), 1.0 / STEPS
    for step in range(STEPS):
        t = step / STEPS
        v = velocity(actions) if guidance is None else guidance(actions, t, STEPS, velocity)
        actions = actions + dt * v
    return actions


def test_declares_rtc_and_accepts_the_server_call():
    policy = shell(RTCConfig(execution_horizon=8))
    assert policy.supports_rtc()
    assert supports_rtc_inference(policy)
    assert shell(None).rtc_processor is None


def test_no_prefix_or_no_config_means_no_guidance():
    assert shell(RTCConfig(execution_horizon=8))._velocity_guidance({"prev_chunk_left_over": None}) is None
    assert shell(None)._velocity_guidance({"prev_chunk_left_over": torch.zeros(1, H, WIDTH)}) is None


def test_guidance_pulls_the_frozen_prefix_and_leaves_padding_alone():
    torch.manual_seed(0)
    noise = torch.randn(1, H, PADDED)
    target = torch.randn(1, H, PADDED)
    prefix = torch.randn(1, 10, WIDTH)  # the previous chunk's unexecuted actions, robot width
    delay = 4

    plain = integrate(None, noise, target)
    policy = shell(RTCConfig(execution_horizon=8, max_guidance_weight=10.0))
    guided = integrate(
        policy._velocity_guidance({"prev_chunk_left_over": prefix, "inference_delay": delay}), noise, target
    )

    # The first `delay` steps (executed while this chunk was computed) move towards the previous chunk.
    before = (plain[0, :delay, :WIDTH] - prefix[0, :delay]).abs().mean()
    after = (guided[0, :delay, :WIDTH] - prefix[0, :delay]).abs().mean()
    assert after < 0.5 * before
    # Past the execution horizon the prefix weight is 0, and the padded dims are never guided.
    torch.testing.assert_close(guided[0, 8:, :WIDTH], plain[0, 8:, :WIDTH])
    torch.testing.assert_close(guided[..., WIDTH:], plain[..., WIDTH:])


def test_config_round_trip_keeps_rtc(tmp_path):
    cfg = GrootN15Config(chunk_size=H, n_action_steps=H, rtc_config=RTCConfig(execution_horizon=8))
    cfg._save_pretrained(tmp_path)
    loaded = GrootN15Config.from_pretrained(tmp_path)
    assert loaded.rtc_config is not None and loaded.rtc_config.execution_horizon == 8
    plain = GrootN15Config(chunk_size=H, n_action_steps=H)
    (tmp_path / "plain").mkdir()
    plain._save_pretrained(tmp_path / "plain")
    assert GrootN15Config.from_pretrained(tmp_path / "plain").rtc_config is None


def test_postprocessor_keeps_the_whole_chunk():
    """LeRobot 0.7's chunk server postprocesses (1, H, D_model) and requires (1, H, robot width) back."""
    from lerobot.processor import TransitionKey
    from lerobot_policy_groot_n15.processor_groot_n15 import GrootActionUnpackUnnormalizeStep

    stats = {"action": {"min": torch.full((WIDTH,), -10.0), "max": torch.full((WIDTH,), 30.0)}}
    step = GrootActionUnpackUnnormalizeStep(env_action_dim=WIDTH, stats=stats)
    chunk = torch.linspace(-1, 1, H).view(1, H, 1).expand(1, H, PADDED).clone()
    out = step({TransitionKey.ACTION: chunk})[TransitionKey.ACTION]
    assert out.shape == (1, H, WIDTH)
    torch.testing.assert_close(out[0, 0], torch.full((WIDTH,), -10.0))  # y=-1 -> min
    torch.testing.assert_close(out[0, -1], torch.full((WIDTH,), 30.0))  # y=+1 -> max
    single = step({TransitionKey.ACTION: chunk[:, 0]})[TransitionKey.ACTION]  # one action: unchanged behaviour
    assert single.shape == (1, WIDTH)
