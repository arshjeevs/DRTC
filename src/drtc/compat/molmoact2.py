"""MolmoAct 2 LoRA fine-tuning from a full-weight LeRobot checkpoint (training only).

LeRobot 0.7's MolmoAct2Policy applies its LoRA adapters in `__init__` (`train_mode_vlm: lora`), and
`from_pretrained` then loads the checkpoint's weights strictly. A checkpoint without LoRA weights (e.g.
`lerobot/MolmoAct2-SO100_101-LeRobot`, the base our recipe starts from) has the plain key names, so the load into the
adapter-wrapped model fails ("Error(s) in loading state_dict for MolmoAct2Policy"). Full fine-tuning never hit this.

`install()` makes such a load happen in the order the official recipe implies: build and load as `fft` (without
compiling), then apply the LoRA adapters exactly as `__init__` would (`_apply_lora_adapters`: VLM frozen, adapters
initialised to an identity update, action expert trainable), then compile. A checkpoint that already holds LoRA
weights (a LoRA-trained model of ours) loads unchanged through the upstream path. Upstream code is not edited.
"""

from __future__ import annotations

from pathlib import Path


def checkpoint_has_lora(pretrained_name_or_path, revision: str | None = None) -> bool:
    from safetensors import safe_open

    path = Path(str(pretrained_name_or_path))
    if path.is_dir():
        model_file = path / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        model_file = Path(hf_hub_download(str(pretrained_name_or_path), "model.safetensors", revision=revision))
    with safe_open(str(model_file), framework="pt") as f:
        return any("lora_" in k for k in f.keys())


def load_then_lora(policy_cls, original, has_lora=checkpoint_has_lora):
    """A `from_pretrained` for `policy_cls` that loads full weights before applying LoRA (see the module doc)."""

    def from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
        if config is None or getattr(config, "train_mode_vlm", None) != "lora" or \
                has_lora(pretrained_name_or_path, kwargs.get("revision")):
            return original(cls, pretrained_name_or_path, config=config, **kwargs)
        compile_model = config.compile_model
        config.train_mode_vlm, config.compile_model = "fft", False
        try:
            policy = original(cls, pretrained_name_or_path, config=config, **kwargs)
        finally:
            config.train_mode_vlm, config.compile_model = "lora", compile_model
        policy.config.train_mode_vlm, policy.config.compile_model = "lora", compile_model
        print(f"MolmoAct 2: loaded full weights from {pretrained_name_or_path}, then applied LoRA "
              f"(rank {config.lora_rank}) to the VLM", flush=True)
        policy._apply_lora_adapters()
        policy._apply_compile()
        return policy

    return classmethod(from_pretrained)


def install() -> None:
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    if getattr(MolmoAct2Policy, "_drtc_load_then_lora", False):
        return
    MolmoAct2Policy.from_pretrained = load_then_lora(MolmoAct2Policy, MolmoAct2Policy.from_pretrained.__func__)
    MolmoAct2Policy._drtc_load_then_lora = True
