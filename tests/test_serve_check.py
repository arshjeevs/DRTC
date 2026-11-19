"""tests/smoke/serve_check.py: config overrides for timing inference settings without retraining."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("serve_check", Path(__file__).parent / "smoke" / "serve_check.py")
serve_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_check)


def test_override_copies_checkpoint_without_writing_it(tmp_path):
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    original = {"type": "molmoact2", "num_inference_steps": None, "compile_model": True}
    (ckpt / "config.json").write_text(json.dumps(original))
    (ckpt / "model.safetensors").write_bytes(b"weights")
    sets = serve_check.parse_sets(["num_inference_steps=5", "compile_model=false"])
    assert sets == {"num_inference_steps": 5, "compile_model": False}
    (tmp_path / "copy").mkdir()
    out = serve_check.with_config(ckpt, sets, tmp_path / "copy")
    assert json.loads((out / "config.json").read_text())["num_inference_steps"] == 5
    assert (out / "model.safetensors").is_symlink()
    assert json.loads((ckpt / "config.json").read_text()) == original


def test_override_rejects_unknown_keys(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(SystemExit):
        serve_check.with_config(tmp_path, {"num_inference_step": 5}, tmp_path)
    with pytest.raises(SystemExit):
        serve_check.parse_sets(["no_equals"])
