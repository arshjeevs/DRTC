"""Old-vs-new parity of a kept model: the new setup must reproduce the old setup's actions.

Replays the inputs and random draws saved by make_reference.py (old setup) through the model loaded the new way
(LeRobot 0.7 + our plugins and compat loader), and compares the action chunk and the final actions.

    .venv/bin/python tests/parity/check.py smolvla_cubebox_v0_lora_raw [--device cuda]
Exit code 0 when every frame is within tolerance.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

import torch

REF = Path(os.environ.get("DRTC_PARITY_DIR", Path.home() / "Projects/drtc-snapshots/parity"))
# Tolerances: how much the OLD setup differs from itself between two GPUs (laptop RTX 4060 vs H200) on the same
# inputs and noise, measured with compare_refs.py on 2026-11-18 and rounded up. Old vs new must stay within it:
# anything beyond is a change in what the model computes, not GPU numerics. Units: normalized (chunk) and robot
# units, degrees / gripper % (actions). SmolVLA's 50-step chunks amplify rounding most at the rest pose (frame 0).
NOISE_FLOOR = {  # measured: chunk, actions
    "groot_cubebox_v1_lora_raw": {"chunk": 7.8e-3, "actions": 0.31},
    "smolvla_cubebox_v0_lora_raw": {"chunk": 1.64e-1, "actions": 2.84},
}
DEFAULT_TOL = {"chunk": 2e-2, "actions": 0.5}


def tolerance(model: str) -> dict:
    floor = NOISE_FLOOR.get(model)
    if floor is None:
        return DEFAULT_TOL
    return {k: max(DEFAULT_TOL[k], round(v * 1.1, 3)) for k, v in floor.items()}


@contextlib.contextmanager
def replay_random(draws: list, device_hint: torch.device):
    """Return the recorded torch.randn / torch.normal results, in order, instead of drawing new ones."""
    real_randn, real_normal = torch.randn, torch.normal
    queue = list(draws)

    def take(kind, kwargs):
        if not queue:
            raise AssertionError(f"new code draws more random tensors than the old ({kind})")
        rkind, shape, value = queue.pop(0)
        if rkind != kind:
            raise AssertionError(f"draw order differs: old {rkind}{shape}, new {kind}")
        dev = kwargs.get("device") or device_hint
        dtype = kwargs.get("dtype") or value.dtype
        return value.to(device=dev, dtype=dtype)

    def randn(*args, **kwargs):
        out = take("randn", kwargs)
        size = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)) else args
        if tuple(size) and tuple(size) != tuple(out.shape):
            raise AssertionError(f"randn shape differs: old {tuple(out.shape)}, new {tuple(size)}")
        return out

    def normal(*args, **kwargs):
        out = take("normal", kwargs)
        size = kwargs.get("size")
        if size is not None and tuple(size) != tuple(out.shape):
            raise AssertionError(f"normal shape differs: old {tuple(out.shape)}, new {tuple(size)}")
        return out

    torch.randn, torch.normal = randn, normal
    try:
        yield queue
    finally:
        torch.randn, torch.normal = real_randn, real_normal


def load_new(model: str, device: str):
    """The kept model, loaded the new way. -> (policy, preprocessor, postprocessor)."""
    from drtc.compat.models import load_kept_model
    from drtc.compat.processors import load_processors

    policy, hub = load_kept_model(model, device=device)
    pre, post = load_processors(policy.config, hub, device)
    return policy, pre, post


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ref-device", help="compare against the reference made on this device (default: --device)")
    a = ap.parse_args()
    ref = torch.load(REF / f"{a.model}.{a.ref_device or a.device}.pt", weights_only=False)
    print(f"reference: {ref['versions']} on {ref['device']}")
    policy, pre, post = load_new(a.model, a.device)

    worst = {"chunk": 0.0, "actions": 0.0}
    for s in ref["samples"]:
        raw = s["input"]
        batch = {k: (v.float() / 255 if v.dtype == torch.uint8 else v).unsqueeze(0) for k, v in raw.items() if k != "task"}
        batch["task"] = [raw["task"]]
        with torch.no_grad():
            proc = pre(batch)
            policy.reset()
            with replay_random(s["draws"], torch.device(a.device)) as left:
                chunk = policy.predict_action_chunk(proc)
            if left:
                raise AssertionError(f"new code draws fewer random tensors than the old ({len(left)} unused)")
            actions = post(chunk)
        for k, ref_v in (s.get("proc") or {}).items():
            new_v = proc.get(k)
            if torch.is_tensor(ref_v) and torch.is_tensor(new_v):
                if tuple(ref_v.shape) != tuple(new_v.shape):
                    print(f"    proc {k}: SHAPE old {tuple(ref_v.shape)} new {tuple(new_v.shape)}")
                elif ref_v.dtype.is_floating_point:
                    print(f"    proc {k}: max diff {(new_v.float().cpu() - ref_v.float()).abs().max().item():.2e}")
                else:
                    print(f"    proc {k}: {'equal' if torch.equal(new_v.cpu(), ref_v) else 'DIFFERENT'}")
            elif torch.is_tensor(ref_v):
                print(f"    proc {k}: missing in new")
        d_chunk = (chunk.float().cpu() - s["chunk"]).abs().max().item()
        d_act = (actions.float().cpu().reshape(s["actions"].shape) - s["actions"]).abs().max().item()
        worst = {"chunk": max(worst["chunk"], d_chunk), "actions": max(worst["actions"], d_act)}
        print(f"frame {s['index']:>6}: max |chunk diff| {d_chunk:.2e}  max |action diff| {d_act:.3f}")
    tol = tolerance(a.model)
    ok = worst["chunk"] <= tol["chunk"] and worst["actions"] <= tol["actions"]
    print(f"{'PASS' if ok else 'FAIL'}: worst chunk {worst['chunk']:.2e} (tol {tol['chunk']}), "
          f"worst action {worst['actions']:.3f} (tol {tol['actions']}; old setup's own GPU-to-GPU noise)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
