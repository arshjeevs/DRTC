"""Dataset preparation (drtc.data.prepare), checkpoint pruning and flag values (drtc.train)."""

import pytest

from drtc.data import prepare
from drtc.registry import Registry, RegistryError
from drtc.train import build, flag_value, prune_training_state


def test_relative_actions_need_stats_per_chunk_and_exclusion():
    assert prepare.needs({}) == []
    assert prepare.needs({"policy.use_relative_actions": False}) == []
    need = prepare.needs({"policy.use_relative_actions": True, "policy.chunk_size": 16,
                          "policy.relative_exclude_joints": ["gripper"]})
    assert need == [prepare.RelativeStats(16, ("gripper",))]
    assert need[0].tag == "rel_c16_xgripper"


@pytest.mark.parametrize("missing", ["policy.chunk_size", "policy.relative_exclude_joints"])
def test_relative_actions_must_be_explicit(missing):
    flags = {"policy.use_relative_actions": True, "policy.chunk_size": 50, "policy.relative_exclude_joints": ["gripper"]}
    del flags[missing]
    with pytest.raises(RegistryError):
        prepare.needs(flags)


def test_variants_are_keyed_by_revision_and_need():
    a = prepare.variant_root("u/ds", "3ef1f445d952ae6b", prepare.RelativeStats(50, ("gripper",)))
    b = prepare.variant_root("u/ds", "3ef1f445d952ae6b", prepare.RelativeStats(16, ("gripper",)))
    c = prepare.variant_root("u/ds", "408fa7e3939b35176", prepare.RelativeStats(50, ("gripper",)))
    assert len({a, b, c}) == 3
    assert a.parent.parent.name == ".drtc_variants" or ".drtc_variants" in a.parts


def test_pi05_and_groot_n17_train_on_their_variant():
    reg = Registry()
    for model, tag in (("pi05_cubebox_v1_full", "rel_c50_xgripper"), ("grootn17_cubebox_v1_full", "rel_c16_xgripper")):
        flags, info = build(reg, model, download=False)
        assert info["variants"] == [tag]
        assert flags["dataset.root"].endswith(tag)
        assert flags["dataset.repo_id"] == info["resolved"].dataset.repo  # same dataset, only the stats differ
    flags, info = build(reg, "act_cubebox_v1_full", download=False)
    assert "dataset.root" not in flags and info["variants"] == []


def test_every_recipe_in_use_is_preparable():
    reg = Registry()
    for rid in prepare.recipes_in_use(reg):
        recipe = reg.recipes[rid]
        prepare.needs(recipe.flags, recipe.policy)  # raises if a relative recipe is incomplete


def test_prune_keeps_weights_everywhere_and_state_only_in_the_latest(tmp_path):
    ckpts = tmp_path / "checkpoints"
    for step in ("001000", "002000", "003000"):
        (ckpts / step / "pretrained_model").mkdir(parents=True)
        (ckpts / step / "training_state").mkdir()
    (ckpts / "last").symlink_to(ckpts / "003000")
    removed = prune_training_state(ckpts / "003000")
    assert sorted(p.parent.name for p in removed) == ["001000", "002000"]
    for step in ("001000", "002000", "003000"):
        assert (ckpts / step / "pretrained_model").is_dir()
    assert (ckpts / "003000" / "training_state").is_dir()


def test_none_flags_reach_draccus_as_none():
    assert flag_value(None) == "null"
    assert flag_value([216, 288]) == "[216, 288]"
    assert flag_value(True) == "true"
