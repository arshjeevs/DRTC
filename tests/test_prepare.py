"""Dataset preparation (drtc.data.prepare), checkpoint pruning and flag values (drtc.train)."""

import pytest

from drtc.data import prepare
from drtc.registry import Registry, RegistryError
from drtc.train import EvalLosses, best_checkpoint, build, flag_value, prune_checkpoints


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


def test_prune_keeps_the_last_two_and_the_best(tmp_path):
    ckpts = tmp_path / "checkpoints"
    steps = ("001000", "002000", "003000", "004000", "005000")
    for step in steps:
        (ckpts / step / "pretrained_model").mkdir(parents=True)
        (ckpts / step / "training_state").mkdir()
    (ckpts / "last").symlink_to(ckpts / "005000")
    losses = {1000: 0.9, 2000: 0.3, 3000: 0.5, 4000: 0.6, 5000: 0.7}
    prune_checkpoints(ckpts / "005000", losses)
    assert sorted(c.name for c in ckpts.iterdir() if not c.is_symlink()) == ["002000", "004000", "005000"]
    assert best_checkpoint(ckpts, losses).name == "002000"
    assert (ckpts / "005000" / "training_state").is_dir()  # the latest resumes
    assert not (ckpts / "002000" / "training_state").exists() and not (ckpts / "004000" / "training_state").exists()
    assert (ckpts / "last").resolve() == (ckpts / "005000").resolve()


def test_prune_without_eval_keeps_the_last_two(tmp_path):
    ckpts = tmp_path / "checkpoints"
    for step in ("001000", "002000", "003000"):
        (ckpts / step / "pretrained_model").mkdir(parents=True)
    prune_checkpoints(ckpts / "003000", {})
    assert sorted(c.name for c in ckpts.iterdir()) == ["002000", "003000"]


def test_eval_losses_are_read_from_lerobot_train_logs():
    import logging

    h = EvalLosses()
    log = logging.getLogger("test_eval_losses")
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.info("step 1000: eval_loss=0.2190")
    log.info("step:1K smpl:64K loss:0.5")
    assert h.losses == {1000: 0.219}


def test_none_flags_reach_draccus_as_none():
    assert flag_value(None) == "null"
    assert flag_value([216, 288]) == "[216, 288]"
    assert flag_value(True) == "true"


def test_molmoact2_lora_from_full_weights_loads_before_wrapping():
    # Stand-in with MolmoAct2Policy's order: __init__ wraps for LoRA, then from_pretrained loads strictly.
    from types import SimpleNamespace

    from drtc.compat.molmoact2 import load_then_lora

    events = []

    class Base:
        @classmethod
        def from_pretrained(cls, path, *, config=None, **kwargs):
            policy = cls(config)
            if policy.wrapped:
                raise RuntimeError("Error(s) in loading state_dict")  # plain keys into a wrapped model
            events.append("load")
            return policy

    class Policy(Base):
        def __init__(self, config):
            self.config = config
            self.wrapped = config.train_mode_vlm == "lora"
            events.append(f"init {config.train_mode_vlm} compile={config.compile_model}")

        def _apply_lora_adapters(self):
            self.wrapped = True
            events.append("lora")

        def _apply_compile(self):
            events.append(f"compile={self.config.compile_model}")

    cfg = lambda: SimpleNamespace(train_mode_vlm="lora", compile_model=True, lora_rank=64)
    with pytest.raises(RuntimeError):
        Policy.from_pretrained("full", config=cfg())  # upstream order fails
    events.clear()
    Policy.from_pretrained = load_then_lora(Policy, Base.from_pretrained.__func__, has_lora=lambda p, r: p == "lora")
    p = Policy.from_pretrained("full", config=cfg())
    assert events == ["init fft compile=False", "load", "lora", "compile=True"]
    assert p.wrapped and p.config.train_mode_vlm == "lora" and p.config.compile_model
    events.clear()
    with pytest.raises(RuntimeError):  # a LoRA checkpoint takes the upstream path unchanged
        Policy.from_pretrained("lora", config=cfg())
    assert events == ["init lora compile=True"]
