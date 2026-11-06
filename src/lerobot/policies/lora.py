"""LoRA fine-tuning for LeRobot policies: train adapters, save only them, load them onto the base.

A LoRA checkpoint is a normal policy folder (config.json, processors) where model.safetensors is
replaced by:
    adapter_config.json        peft LoraConfig + base_model_name_or_path + the fully-trained modules
    adapter_model.safetensors  the trained tensors only (LoRA A/B matrices + fully-trained modules)

The base weights come from `base_model_name_or_path` at load time, so the folder stays small (~21 MB
for SmolVLA at r=32, against ~900 MB for a full checkpoint).
`load_policy(..., finetune="lora")` rebuilds the policy with the adapters live, as in training. They are
not merged: SmolVLA's weights are bf16, and folding a small LoRA update into them rounds it away.
"""

from __future__ import annotations

import json
import logging
import os
import re
import types
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file, save_file

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
FINETUNE_METHODS = ("full", "lora")

# Where adapters go, per policy type. `targets` are named presets of LoRA module regexes (matched
# against full module names with re.fullmatch); `trainable` are small modules trained in full.
_SMOLVLA_LAYERS = r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
PRESETS = {
    "smolvla": {
        "targets": {
            # The action expert: the part SmolVLA trains by default (train_expert_only).
            "expert": rf".*\.lm_expert\.layers\.\d+\.{_SMOLVLA_LAYERS}",
            # Also adapt the VLM's language layers (more memory: backprop runs through the VLM).
            "vlm+expert": rf".*\.(lm_expert|text_model)\.layers\.\d+\.{_SMOLVLA_LAYERS}",
        },
        "trainable": r"model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)\..*",
    },
}
DEFAULTS = {"r": 32, "alpha": 32, "dropout": 0.05, "targets": "expert"}


def resolve_spec(policy_type: str, spec: dict | None = None) -> dict:
    """Fill defaults and expand a target preset into the regexes stored with the adapter."""
    spec = {**DEFAULTS, **(spec or {})}
    preset = PRESETS.get(policy_type, {})
    targets = preset.get("targets", {}).get(spec["targets"], spec["targets"])
    if targets == spec["targets"] and not any(ch in targets for ch in r".*\()"):
        known = ", ".join(preset.get("targets", {})) or "none (pass a module-name regex)"
        raise ValueError(f"Unknown LoRA targets {spec['targets']!r} for {policy_type}. Presets: {known}")
    trainable = spec.get("trainable", preset.get("trainable"))
    return {"r": int(spec["r"]), "alpha": int(spec["alpha"]), "dropout": float(spec["dropout"]),
            "target_modules": targets, "trainable": trainable}


def _lora_config(spec: dict, base: str | None):
    from peft import LoraConfig

    return LoraConfig(r=spec["r"], lora_alpha=spec["alpha"], lora_dropout=spec["dropout"],
                      target_modules=spec["target_modules"], base_model_name_or_path=base)


def _inject(policy: torch.nn.Module, spec: dict, base: str | None) -> None:
    from peft import inject_adapter_in_model

    inject_adapter_in_model(_lora_config(spec, base), policy)
    if not any("lora_A" in name for name, _ in policy.named_parameters()):
        raise ValueError(f"LoRA target_modules {spec['target_modules']!r} matched no layer")


def apply_lora(policy: torch.nn.Module, spec: dict, base: str | None) -> torch.nn.Module:
    """Freeze the policy, add trainable adapters, and make save_pretrained write adapters only."""
    for param in policy.parameters():
        param.requires_grad = False
    _inject(policy, spec, base)  # new LoRA parameters are created trainable
    if spec.get("trainable"):
        for name, param in policy.named_parameters():
            if re.fullmatch(spec["trainable"], name):
                param.requires_grad = True
    policy._lora = {"spec": spec, "base": base}
    policy._save_pretrained = types.MethodType(_save_adapter, policy)
    trained = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    logging.info(f"LoRA r={spec['r']} alpha={spec['alpha']} on {spec['target_modules']}: "
                 f"training {trained / 1e6:.2f}M of {total / 1e6:.1f}M parameters")
    return policy


def _save_adapter(self, save_directory: Path) -> None:
    """Replaces PreTrainedPolicy._save_pretrained for LoRA-wrapped policies."""
    save_directory = Path(save_directory)
    self.config._save_pretrained(save_directory)
    spec, base = self._lora["spec"], self._lora["base"]
    config = _lora_config(spec, base).to_dict()
    config["lerobot_trainable"] = spec.get("trainable")
    config["target_modules"] = spec["target_modules"]
    (save_directory / ADAPTER_CONFIG).write_text(json.dumps(config, indent=2, default=str))
    tensors = {name: p.detach().cpu().contiguous() for name, p in self.named_parameters() if p.requires_grad}
    save_file(tensors, str(save_directory / ADAPTER_WEIGHTS))


def _fetch(path: str, filename: str) -> str | None:
    if os.path.isdir(path):
        file = Path(path) / filename
        return str(file) if file.is_file() else None
    try:
        return hf_hub_download(repo_id=path, filename=filename)
    except EntryNotFoundError:
        return None


def is_adapter(path: str | Path) -> bool:
    return _fetch(str(path), ADAPTER_CONFIG) is not None


def read_adapter(path: str | Path) -> tuple[dict, str, str]:
    """-> (spec, base_model_name_or_path, local adapter weights file)."""
    config_file = _fetch(str(path), ADAPTER_CONFIG)
    if config_file is None:
        raise FileNotFoundError(f"{path} has no {ADAPTER_CONFIG}: it is not a LoRA checkpoint (use finetune: full)")
    config = json.loads(Path(config_file).read_text())
    spec = {"r": config["r"], "alpha": config["lora_alpha"], "dropout": config["lora_dropout"],
            "target_modules": config["target_modules"], "trainable": config.get("lerobot_trainable")}
    return spec, config["base_model_name_or_path"], _fetch(str(path), ADAPTER_WEIGHTS)


def load_adapter_weights(policy: torch.nn.Module, weights_file: str) -> None:
    tensors = load_file(weights_file)
    result = policy.load_state_dict(tensors, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Adapter does not fit this policy; unknown tensors: {result.unexpected_keys[:5]}")


def load_policy(policy_class, path: str, finetune: str = "full", **kwargs):
    """Load a policy for inference, from a full checkpoint or a LoRA adapter on its base model."""
    if finetune not in FINETUNE_METHODS:
        raise ValueError(f"finetune must be one of {FINETUNE_METHODS}, got {finetune!r}")
    if finetune == "full":
        try:
            return policy_class.from_pretrained(path, **kwargs)
        except FileNotFoundError as e:
            if is_adapter(path):
                raise FileNotFoundError(f"{path} is a LoRA adapter, not a full model: set finetune: lora") from e
            raise

    from lerobot.configs.policies import PreTrainedConfig

    spec, base, weights = read_adapter(path)
    # The fine-tuned config (this dataset's features), with the base model's weights underneath.
    config = PreTrainedConfig.from_pretrained(path)
    policy = policy_class.from_pretrained(base, config=config, **kwargs)
    _inject(policy, spec, base)
    load_adapter_weights(policy, weights)
    policy.to(config.device).eval()
    logging.info(f"Loaded LoRA adapter {path} (r={spec['r']}) on base {base}")
    return policy
