"""Load the models carried over from the minor (lab/models/kept.yaml) in the new setup.

They were trained with the old setup (LeRobot 0.4.3 fork, our LoRA format). Each needs:
- the policy class for its type: LeRobot 0.7's own, or our plugin where 0.7 dropped it (GR00T N1.5);
- its config.json read by that class (old configs can carry fields 0.7 no longer has);
- the old-format LoRA adapter (drtc.compat.lora).
"""

from __future__ import annotations

from drtc.registry import Registry

from .lora import is_adapter, load_lora_policy

# Old policy type in the checkpoint -> policy type that loads it now.
TYPE_MAP = {"groot": "groot_n15"}


def policy_class_for(policy_type: str):
    t = TYPE_MAP.get(policy_type, policy_type)
    if t == "groot_n15":
        from lerobot_policy_groot_n15 import GrootN15Policy

        return GrootN15Policy
    from lerobot.policies.factory import get_policy_class

    return get_policy_class(t)


def load_config(policy_class, hub: str, revision: str | None = None):
    """The checkpoint's config.json as the (new) policy class's config."""
    import json

    from huggingface_hub import hf_hub_download

    cfg_cls = policy_class.config_class
    raw = json.loads(open(hf_hub_download(hub, "config.json", revision=revision)).read())
    from .configs import upgrade_config

    raw = upgrade_config(raw, cfg_cls)
    import draccus

    from lerobot.configs import PreTrainedConfig

    with draccus.config_type("json"):
        return draccus.decode(PreTrainedConfig, raw)


def load_kept_model(model_id: str, device: str = "cuda", registry: Registry | None = None):
    """-> (policy ready for inference, Hub repo id)."""
    reg = registry or Registry()
    r = reg.resolve(model_id)
    cls = policy_class_for(r.policy)
    config = load_config(cls, r.hub)
    config.device = device
    if (r.finetune == "lora" and r.lora is not None) or is_adapter(r.hub):
        policy = load_lora_policy(cls, r.hub, config=config, device=device)
    else:
        policy = cls.from_pretrained(r.hub, config=config)
        policy.to(device).eval()
    return policy, r.hub
