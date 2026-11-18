"""Reference outputs of a kept model in the OLD setup (tag minor-final), for the old-vs-new parity check.

Runs with the old venvs (~/Projects/drtc/.venv for SmolVLA, .venv-groot for GR00T N1.5), loading the model exactly as
the old policy server did (lerobot.policies.lora.load_policy + make_pre_post_processors). For a few dataset frames
it saves:
  - the exact input (images as uint8, state, task),
  - every random draw the model makes (torch.randn / torch.normal), so the new setup can replay the same noise,
  - the preprocessed batch's state and the predicted action chunk before and after the postprocessor.

    ~/Projects/drtc/.venv-groot/bin/python tests/parity/make_reference.py groot_cubebox_v1_lora_raw
    ~/Projects/drtc/.venv/bin/python tests/parity/make_reference.py smolvla_cubebox_v0_lora_raw

Output: ~/Projects/drtc-snapshots/parity/<model>.pt
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

import torch

U = "orangemoon599"
# Where the datasets are (LeRobot's HF_LEROBOT_HOME) and where results go (DRTC_PARITY_DIR); laptop defaults.
LEROBOT = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot")) / U
OUT = Path(os.environ.get("DRTC_PARITY_DIR", Path.home() / "Projects/drtc-snapshots/parity"))
# model -> (old policy type, finetune, dataset folder (renamed 2026-11-18), frames to use)
MODELS = {
    "groot_cubebox_v1_lora_raw": ("groot", "lora", "so101_cubebox_v1_raw", [0, 7000, 14762, 22000, 29524]),
    "smolvla_cubebox_v0_lora_raw": ("smolvla", "lora", "so101_cubebox_v0_raw", [0, 3000, 7487, 11000, 14974]),
}


@contextlib.contextmanager
def record_random(draws: list):
    """Record every torch.randn / torch.normal result made inside the block."""
    real_randn, real_normal = torch.randn, torch.normal

    def randn(*args, **kwargs):
        out = real_randn(*args, **kwargs)
        draws.append(("randn", tuple(out.shape), out.detach().cpu().clone()))
        return out

    def normal(*args, **kwargs):
        out = real_normal(*args, **kwargs)
        draws.append(("normal", tuple(out.shape), out.detach().cpu().clone()))
        return out

    torch.randn, torch.normal = randn, normal
    try:
        yield
    finally:
        torch.randn, torch.normal = real_randn, real_normal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    policy_type, finetune, ds_dir, frames = MODELS[a.model]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.lora import load_policy

    hub = f"{U}/{a.model}"
    policy = load_policy(get_policy_class(policy_type), hub, finetune)
    policy.config.device = a.device
    policy.to(a.device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=hub,
                                         preprocessor_overrides={"device_processor": {"device": a.device}})
    ds = LeRobotDataset(f"{U}/{ds_dir}", root=LEROBOT / ds_dir, video_backend="pyav")
    torch.manual_seed(0)

    samples = []
    for idx in frames:
        item = ds[idx]
        raw = {}
        for k, v in item.items():
            if k.startswith("observation.images."):
                u8 = (v * 255).round().to(torch.uint8)
                assert torch.equal(u8.float() / 255, v), f"{k}: frame is not exactly uint8/255"
                raw[k] = u8
            elif k == "observation.state":
                raw[k] = v.clone()
        raw["task"] = item["task"]
        batch = {k: (v.float() / 255 if v.dtype == torch.uint8 else v).unsqueeze(0) for k, v in raw.items()
                 if k != "task"}
        batch["task"] = [raw["task"]]
        draws: list = []
        with torch.no_grad():
            proc = pre(batch)
            policy.reset()
            with record_random(draws):
                chunk = policy.predict_action_chunk(proc)
            actions = post(chunk)
        proc_saved = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in proc.items()
                      if torch.is_tensor(v) or isinstance(v, (str, list, int, float))}
        samples.append({"index": idx, "episode": int(item["episode_index"]), "input": raw, "draws": draws,
                        "proc": proc_saved,
                        "chunk": chunk.detach().float().cpu(), "actions": actions.detach().float().cpu()})
        print(f"frame {idx}: {len(draws)} random draws {[d[1] for d in draws]}, chunk {tuple(chunk.shape)}, "
              f"actions {tuple(actions.shape)} first {[round(x, 3) for x in actions.reshape(-1)[:6].tolist()]}")

    import lerobot, transformers  # noqa: E401
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.model}.{a.device}.pt"
    torch.save({"model": a.model, "hub": hub, "policy_type": policy_type, "device": a.device,
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                             "lerobot": getattr(lerobot, "__version__", "?")},
                "samples": samples}, out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
