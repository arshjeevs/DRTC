"""A base checkpoint written for an older policy config is passed to lerobot-train as an upgraded local copy."""

import json
from pathlib import Path

import drtc.compat.serve
from drtc.train import upgraded_base

FIXTURE = Path(__file__).parent / "fixtures" / "molmoact2_so100_101_config.json"  # lerobot/MolmoAct2-SO100_101-LeRobot


def test_molmoact2_base_is_upgraded(tmp_path, monkeypatch):
    monkeypatch.setattr(drtc.compat.serve, "CACHE", tmp_path / "cache" / "compat_checkpoints")
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text(FIXTURE.read_text())
    (snap / "model.safetensors").write_bytes(b"weights")
    out = upgraded_base(snap, "molmo@rev")
    assert out != snap
    cfg = json.loads((out / "config.json").read_text())
    for gone in ("enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only", "model_dtype"):
        assert gone not in cfg
    assert cfg["type"] == "molmoact2" and cfg["chunk_size"] == 30
    assert (out / "model.safetensors").is_symlink() and (out / "model.safetensors").read_bytes() == b"weights"
    assert json.loads((snap / "config.json").read_text()) == json.loads(FIXTURE.read_text())  # snapshot untouched
    from lerobot.configs import PreTrainedConfig

    assert PreTrainedConfig.from_pretrained(out).type == "molmoact2"  # 0.7 parses it


def test_current_config_is_passed_through(tmp_path):
    snap = tmp_path / "snapshot"
    snap.mkdir()
    raw = json.loads(FIXTURE.read_text())
    for k in ("enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only", "model_dtype"):
        raw.pop(k)
    (snap / "config.json").write_text(json.dumps(raw))
    assert upgraded_base(snap, "molmo@rev") == snap
