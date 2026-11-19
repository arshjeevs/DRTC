"""drtc.compat.wandb_mirror: Trackio's calls are mirrored to wandb (a stand-in module, nothing goes online)."""

import sys
import types

import pytest


@pytest.fixture
def fake_wandb(monkeypatch):
    calls = []

    class Run:
        url = "https://wandb.example/run"

        def log(self, data, step=None):
            calls.append(("log", data, step))

        def finish(self):
            calls.append(("finish",))

    mod = types.SimpleNamespace(api=types.SimpleNamespace(api_key="x"),
                                init=lambda **kw: calls.append(("init", kw)) or Run())
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return calls


def test_trackio_calls_reach_wandb(fake_wandb, monkeypatch):
    from lerobot.common import trackio_utils

    from drtc.compat import wandb_mirror

    class Logger:  # stands in for TrackioLogger: same methods, no Trackio
        def __init__(self, cfg):
            self.cfg = types.SimpleNamespace(run_name="act_cubebox_v0_full-1a2b3c4d")
            self._group = "act"

        def log_dict(self, d, step=None, mode="train", custom_step_key=None):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(trackio_utils, "TrackioLogger", Logger)
    wandb_mirror.install("drtc-lab")
    cfg = types.SimpleNamespace(to_dict=lambda: {"steps": 10})
    lg = trackio_utils.TrackioLogger(cfg)
    lg.log_dict({"loss": 0.5, "note": "text"}, step=200)
    lg.log_dict({"eval_loss": 0.2}, step=1000, mode="eval")
    lg.finish()
    init = fake_wandb[0][1]
    assert init["project"] == "drtc-lab" and init["name"] == "act_cubebox_v0_full-1a2b3c4d" and init["resume"] == "allow"
    assert ("log", {"train/loss": 0.5}, 200) in fake_wandb
    assert ("log", {"eval/eval_loss": 0.2}, 1000) in fake_wandb
    assert fake_wandb[-1] == ("finish",)
