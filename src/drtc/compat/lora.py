"""Load LoRA checkpoints saved by the old setup (tag minor-final, lerobot/policies/lora.py there).

Format: a policy folder whose model.safetensors is replaced by
    adapter_config.json        PEFT LoraConfig + base_model_name_or_path + `lerobot_trainable`
    adapter_model.safetensors  LoRA A/B tensors + the modules trained in full (+ buffers when there is no base)

The base weights come from base_model_name_or_path (None: the policy builds itself from its config, e.g. GR00T,
which loads its own base). Adapters are injected with PEFT exactly as in training and kept unmerged (merging into
bf16 weights rounds small updates away). New training uses LeRobot's own PEFT support instead; this is only for
loading the checkpoints carried over from the minor.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"


def _fetch(path: str, filename: str, revision: str | None = None) -> str | None:
    if os.path.isdir(path):
        f = Path(path) / filename
        return str(f) if f.is_file() else None
    try:
        return hf_hub_download(repo_id=path, filename=filename, revision=revision)
    except EntryNotFoundError:
        return None


def is_adapter(path: str, revision: str | None = None) -> bool:
    return _fetch(path, ADAPTER_CONFIG, revision) is not None


def read_adapter(path: str, revision: str | None = None) -> tuple[dict, str | None, str]:
    """-> (LoRA settings, base_model_name_or_path, local weights file)."""
    cfg_file = _fetch(path, ADAPTER_CONFIG, revision)
    if cfg_file is None:
        raise FileNotFoundError(f"{path} has no {ADAPTER_CONFIG}: not a LoRA checkpoint")
    cfg = json.loads(Path(cfg_file).read_text())
    spec = {"r": cfg["r"], "alpha": cfg["lora_alpha"], "dropout": cfg["lora_dropout"],
            "target_modules": cfg["target_modules"], "trainable": cfg.get("lerobot_trainable")}
    return spec, cfg.get("base_model_name_or_path"), _fetch(path, ADAPTER_WEIGHTS, revision)


def inject(policy: torch.nn.Module, spec: dict, base: str | None) -> None:
    from peft import LoraConfig, inject_adapter_in_model

    config = LoraConfig(r=spec["r"], lora_alpha=spec["alpha"], lora_dropout=spec["dropout"],
                        target_modules=spec["target_modules"], base_model_name_or_path=base)
    inject_adapter_in_model(config, policy)
    if not any("lora_A" in name for name, _ in policy.named_parameters()):
        raise ValueError(f"LoRA target_modules {spec['target_modules']!r} matched no layer")


def load_adapter_weights(policy: torch.nn.Module, weights_file: str) -> None:
    tensors = load_file(weights_file)
    result = policy.load_state_dict(tensors, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"adapter does not fit this policy; unknown tensors: {result.unexpected_keys[:5]}")
    lora_missing = [k for k in result.missing_keys if "lora_" in k]
    if lora_missing:
        raise ValueError(f"adapter is missing LoRA tensors: {lora_missing[:5]}")


def load_lora_policy(policy_class, path: str, config=None, revision: str | None = None, device: str | None = None):
    """Base policy + old-format adapter, ready for inference.

    `config`: the fine-tuned policy config (this dataset's features); read from `path` when not given.
    """
    spec, base, weights = read_adapter(path, revision)
    if config is None:
        from lerobot.configs import PreTrainedConfig

        config = PreTrainedConfig.from_pretrained(path, revision=revision)
    if device:
        config.device = device
    policy = policy_class.from_pretrained(base, config=config) if base else policy_class(config)
    inject(policy, spec, base)
    load_adapter_weights(policy, weights)
    policy.to(config.device).eval()
    logging.info("loaded old-format LoRA adapter %s (r=%s) on base %s", path, spec["r"], base)
    return policy
