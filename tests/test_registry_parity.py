"""The models carried over from the minor must behave exactly as in the old setup (tag minor-final, frozen in
tests/fixtures/legacy), apart from the names changed on 2026-11-18 (lab/renames.yaml):

- each kept model's entry equals the old models.yaml entry, with the renamed Hub repos substituted;
- every experiment composes the same robot-run config as the old lab/tools/config.py did.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from drtc.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "tests" / "fixtures" / "legacy"
RENAMES = yaml.safe_load((ROOT / "lab" / "renames.yaml").read_text())
U = "orangemoon599"
KEPT = {"groot_cubebox_v1_lora": "groot_cubebox_v1_lora_raw",
        "smolvla_train_new_1_lora": "smolvla_cubebox_v0_lora_raw",
        "jack_smolvla_cube": "jack_smolvla_cube"}


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry(ROOT / "lab")


def old_models() -> dict:
    raw = yaml.safe_load((LEGACY / "models.yaml").read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def renamed(entry: dict) -> dict:
    """An old entry with the 2026-11-18 Hub renames applied."""
    e = {**entry}
    hub = e["pretrained_name_or_path"].removeprefix(f"{U}/")
    e["pretrained_name_or_path"] = f"{U}/{RENAMES['models'].get(hub, hub)}" if "/" not in hub else e["pretrained_name_or_path"]
    if "train" in e:
        ds = e["train"]["dataset"].removeprefix(f"{U}/")
        e["train"] = {**e["train"], "dataset": f"{U}/{RENAMES['datasets'].get(ds, ds)}"}
    return e


def test_registry_is_consistent(reg):
    assert reg.check() == []


def test_renames_table_covers_the_kept_models():
    for old, new in KEPT.items():
        assert RENAMES["models"].get(old, old) == new


@pytest.mark.parametrize("old,new", sorted(KEPT.items()))
def test_kept_model_identical(reg, old, new):
    assert reg.resolve(new).legacy_entry() == renamed(old_models()[old])


def test_old_name_says_where_it_went(reg):
    with pytest.raises(Exception, match="renamed to 'groot_cubebox_v1_lora_raw'"):
        reg.resolve("groot_cubebox_v1_lora")


def _old_config(rig_file: str, experiments: Path):
    """The old lab/tools/config.py from minor-final, pointed at the frozen files."""
    src = ROOT / "tests" / "fixtures" / "legacy_config.py"
    spec = importlib.util.spec_from_file_location(f"legacy_config_{rig_file}", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.RIG_PATH = LEGACY / rig_file
    mod.MODELS_PATH = LEGACY / "models.yaml"
    mod.EXPERIMENTS_DIR = experiments
    mod.DEFAULTS_PATH = experiments / "_defaults.yaml"
    return mod


@pytest.fixture(scope="module")
def old_experiments(tmp_path_factory) -> Path:
    """lab/experiments with the model names mapped back to the old ones."""
    back = {new: old for old, new in KEPT.items()}
    out = tmp_path_factory.mktemp("experiments")
    shutil.copytree(ROOT / "lab" / "experiments", out, dirs_exist_ok=True)
    for p in out.rglob("*.yaml"):
        data = yaml.safe_load(p.read_text()) or {}
        if data.get("model") in back:
            p.write_text(p.read_text().replace(f"model: {data['model']}", f"model: {back[data['model']]}"))
    return out


# The groups the minor had (its composer is the reference); groups added later (bench/, 2026-11-20) use models and keys
# the minor never knew, so there is nothing to compare them with.
MINOR_GROUPS = {"replication", "smoothing", "smoke"}
EXPERIMENTS = sorted(
    str(p.relative_to(ROOT / "lab" / "experiments").with_suffix(""))
    for p in (ROOT / "lab" / "experiments").rglob("*.yaml")
    if not p.name.startswith("_") and p.relative_to(ROOT / "lab" / "experiments").parts[0] in MINOR_GROUPS
)


@pytest.mark.parametrize("rig_file,host", [("rig.yaml", "pi"), ("rig.windows.yaml", "windows-laptop")])
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_compose_identical(reg, monkeypatch, old_experiments, rig_file, host, experiment):
    old = _old_config(rig_file, old_experiments)
    monkeypatch.setenv("LAB_HOST", host)
    expected = renamed(old.compose(old_experiments / f"{experiment}.yaml"))
    expected["model"] = KEPT[expected["model"]]
    assert reg.compose(experiment) == expected


def test_compose_without_a_known_host(reg, monkeypatch):
    """On a machine that is not in hosts.yaml the client defaults to the network's client host."""
    monkeypatch.delenv("LAB_HOST", raising=False)
    monkeypatch.setattr(type(reg), "current_host", lambda self: None)
    merged = reg.compose("smoothing/baseline")
    assert merged["client_host"] == reg.hosts["pi"].label
