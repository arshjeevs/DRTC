"""Serve the models carried over from the minor (old checkpoint format) with LeRobot 0.7's policy server.

Their checkpoints have an old config (e.g. type `groot`, now N1.7 upstream; fields 0.7 dropped), old processor step
names (GR00T N1.5) and our own LoRA adapter format. prepare_checkpoint() makes a local copy that 0.7 reads: the
config upgraded (drtc.compat.configs), the processors with mapped step names (drtc.compat.processors); the adapter
files unchanged. install_loader() makes the server build the policy with our old-format LoRA loader
(drtc.compat.lora, the one the parity check proved). Everything else is upstream's server.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .configs import upgrade_config
from .lora import is_adapter
from .models import policy_class_for
from .processors import STEP_MAP

CACHE = Path(os.environ.get("DRTC_CACHE", Path.home() / ".cache" / "drtc")) / "compat_checkpoints"


def needs_compat(hub: str, revision: str | None = None) -> bool:
    """Old-format checkpoint: our LoRA adapter (adapter_config with lerobot_trainable) or an old policy type."""
    from huggingface_hub import hf_hub_download

    from .configs import TYPE_MAP

    cfg = json.loads(Path(hf_hub_download(hub, "config.json", revision=revision)).read_text())
    if cfg.get("type") in TYPE_MAP:
        return True
    if is_adapter(hub, revision):
        from huggingface_hub import hf_hub_download as dl

        adapter = json.loads(Path(dl(hub, "adapter_config.json", revision=revision)).read_text())
        return "lerobot_trainable" in adapter
    return False


def prepare_checkpoint(hub: str, revision: str | None = None) -> Path:
    """A local copy of an old-format checkpoint that LeRobot 0.7's server can read."""
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(hub, revision=revision))
    dst = CACHE / hub.replace("/", "--") / src.name  # src.name: the snapshot's commit
    if (dst / ".complete").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name == "config.json":
            raw = json.loads(f.read_text())
            cls = policy_class_for(raw["type"])
            (dst / f.name).write_text(json.dumps(upgrade_config(raw, cls.config_class), indent=2))
        elif f.suffix == ".json" and f.name.startswith("policy_") and f.name.endswith("processor.json"):
            data = json.loads(f.read_text())
            for step in data.get("steps", []):
                if step.get("registry_name") in STEP_MAP:
                    step["registry_name"] = STEP_MAP[step["registry_name"]]
            (dst / f.name).write_text(json.dumps(data, indent=2))
        elif f.is_file():
            shutil.copyfile(f, dst / f.name)
    (dst / ".complete").write_text(hub + "\n")
    return dst


def install_loader() -> None:
    """Make the server build old-format checkpoints with our LoRA loader (others load as usual)."""
    import lerobot.remote_inference.deployment as deployment

    from .lora import is_adapter as _is_adapter
    from .lora import load_lora_policy

    original = deployment.get_policy_class

    def get_policy_class(name):
        cls = original(name)

        class _OldFormat(cls):  # only from_pretrained differs; the instance is the real policy class's
            @classmethod
            def from_pretrained(klass, path, config=None, **kwargs):
                if _is_adapter(str(path)):
                    return load_lora_policy(cls, str(path), config=config, device=getattr(config, "device", None))
                return cls.from_pretrained(path, config=config, **kwargs)

        _OldFormat.__name__ = cls.__name__
        return _OldFormat

    deployment.get_policy_class = get_policy_class
    # Our adapter format is not upstream's PEFT format: never take upstream's PEFT branch for it.
    deployment.has_peft_adapter_config = lambda path: False
