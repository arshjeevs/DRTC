"""Recipe train.stop_at: the run stops cleanly at that step while the schedule keeps the recipe's `steps`."""

from __future__ import annotations

import types

import pytest

from drtc import train
from drtc.registry import Registry


def fake_lerobot_train():
    return types.SimpleNamespace(should_save_checkpoint=lambda step, freq, total: step % freq == 0 or step == total,
                                 update_last_checkpoint=lambda d: None)


def test_hook_requests_a_save_at_stop_at(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "STOPS", tmp_path / ".stop")
    lt = fake_lerobot_train()
    state = train.install_clean_stop(lt, "m", stop_at=1500)
    assert not lt.should_save_checkpoint(1499, 1000, 2000)
    assert lt.should_save_checkpoint(1500, 1000, 2000)
    assert state["requested"]
    ckpt = tmp_path / "checkpoints" / "001500"
    ckpt.mkdir(parents=True)
    with pytest.raises(train.StopTraining):
        lt.update_last_checkpoint(ckpt)


def test_no_stop_at_trains_to_the_end(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "STOPS", tmp_path / ".stop")
    lt = fake_lerobot_train()
    state = train.install_clean_stop(lt, "m")
    assert not lt.should_save_checkpoint(1500, 1000, 2000)
    assert not state["requested"]


@pytest.mark.parametrize("model,steps,stop_at", [("pi05_sort_one_v1_full", 20000, 15000),
                                                  ("pi05_cubebox_v1_full", 20000, 15000),
                                                  ("molmoact2_cubebox_v1_lora", 50000, 30000),
                                                  ("smolvla_sort_one_v1_full", 20000, None)])
def test_recipes_keep_steps_and_cut(model, steps, stop_at):
    flags, info = train.build(Registry(), model, download=False)
    assert flags["steps"] == steps and info["stop_at"] == stop_at
    assert "stop_at" not in flags  # not a lerobot-train flag


def test_short_run_ignores_stop_at():
    flags, info = train.build(Registry(), "pi05_sort_one_v1_full", steps=200, download=False)
    assert flags["steps"] == 200 and info["stop_at"] is None
