"""Frame cache (drtc.data.prepare FrameCache, drtc.compat.frame_cache) and the Phase 3 comparison runner."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from drtc.compat import frame_cache
from drtc.data import prepare
from drtc.registry import Registry, RegistryError

DIFF = {"policy.resize_shape": [240, 320]}


def test_frames_need_matching_resize_and_policy():
    assert prepare.needs(DIFF, "diffusion", {"frames": [240, 320]}) == [prepare.FrameCache(240, 320)]
    assert prepare.needs(DIFF, "diffusion") == []
    with pytest.raises(RegistryError):
        prepare.needs(DIFF, "diffusion", {"frames": [224, 224]})  # the model's resize would change the frame
    with pytest.raises(RegistryError):
        prepare.needs(DIFF, "pi05", {"frames": [240, 320]})  # other policies resize differently
    with pytest.raises(RegistryError):
        prepare.needs({**DIFF, "dataset.image_transforms.enable": True}, "diffusion", {"frames": [240, 320]})


def test_groot_n15_frames_are_the_eagle_tile():
    need = prepare.needs({}, "groot_n15", {"frames": [224, 224]})
    assert need == [prepare.FrameCache(224, 224, "eagle")] and need[0].tag == "frames224x224_eagle"
    with pytest.raises(RegistryError):
        prepare.needs({}, "groot_n15", {"frames": [256, 256]})
    with pytest.raises(RegistryError):
        prepare.needs({}, "groot", {"frames": [224, 224]})  # N1.7's processor is different


def test_recipes_with_frames_are_valid():
    reg = Registry()
    with_frames = {rid for rid, r in reg.recipes.items() if r.data.get("frames")}
    assert {"diffusion_original", "groot_n15_nvidia"} <= with_frames
    for rid in with_frames:
        r = reg.recipes[rid]
        assert any(isinstance(n, prepare.FrameCache) for n in prepare.needs(r.flags, r.policy, r.data))


def test_variant_names_combine_and_keep_the_relative_stats_name():
    rel = prepare.RelativeStats(50, ("gripper",))
    fc = prepare.FrameCache(240, 320)
    single = prepare.variant_root("u/ds", "3ef1f445d952ae6b", rel)
    assert single == prepare.variant_root("u/ds", "3ef1f445d952ae6b", [rel])
    assert single.name == "3ef1f445d9_rel_c50_xgripper"  # variants made before the frame cache stay valid
    assert prepare.variant_root("u/ds", "3ef1f445d952ae6b", [rel, fc]).name.endswith("rel_c50_xgripper_frames240x320")
    old_marker = {"repo": "u/ds", "revision": "r", "relative_action": True, "chunk_size": 50, "exclude_joints": ["gripper"]}
    assert prepare._want("u/ds", "r", [rel]) == old_marker
    extra = prepare._want("u/ds", "r", [fc])["extra"]
    assert extra == [{"type": "FrameCache", "height": 240, "width": 320, "decoder": prepare.video_backend()}]


def test_cached_frames_are_read_by_episode_and_timestamp(tmp_path):
    frames = np.arange(10 * 3 * 2 * 2, dtype=np.uint8).reshape(10, 3, 2, 2)
    (tmp_path / "drtc_frames").mkdir()
    np.save(tmp_path / "drtc_frames" / "cam.npy", frames)
    (tmp_path / "drtc_frames" / "meta.json").write_text('{"keys": ["cam"], "fps": 30.0}')
    cache = frame_cache.cache_for(tmp_path)
    one = frame_cache.read_frames(cache, {"cam": [2 / 30]}, from_index=4, tolerance_s=1e-4)
    assert one["cam"].shape == (3, 2, 2) and one["cam"].dtype == torch.uint8
    assert torch.equal(one["cam"], torch.from_numpy(frames[6]))
    two = frame_cache.read_frames(cache, {"cam": [0.0, 1 / 30]}, from_index=4, tolerance_s=1e-4, return_uint8=False)
    assert two["cam"].shape == (2, 3, 2, 2) and torch.allclose(two["cam"] * 255, torch.from_numpy(frames[4:6]).float())
    with pytest.raises(ValueError):
        frame_cache.read_frames(cache, {"cam": [0.5 / 30]}, from_index=4, tolerance_s=1e-4)  # between frames
    assert frame_cache.cache_for(tmp_path / "elsewhere") is None  # other roots decode as before


def test_comparison_variants_resolve():
    spec = importlib.util.spec_from_file_location("round_", Path(__file__).parent / "bench" / "round.py")
    rnd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rnd)
    variants = rnd.load()
    models = set(Registry().models)
    for name, v in variants.items():
        for model, env in rnd.jobs(name, v):
            assert model in models, (name, model)
            assert env["LABEL"] == (f"{name}/{model}" if v.get("parallel") else name)
    assert dict(rnd.jobs("compile", variants["compile"]))["pi05_cubebox_v1_full"]["EXTRA"] == "--policy.compile_model=true"


def test_cached_reader_survives_spawned_workers(tmp_path):
    """lerobot-train's workers are spawned: the reader's class, not a patch, must carry the cache across a pickle."""
    import pickle

    from drtc.compat.frame_cache import CachedDatasetReader

    r = CachedDatasetReader.__new__(CachedDatasetReader)
    r.root = tmp_path
    clone = pickle.loads(pickle.dumps(r))
    assert type(clone) is CachedDatasetReader
