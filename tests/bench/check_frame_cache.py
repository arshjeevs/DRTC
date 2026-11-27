"""Check a frame cache against the normal path, and time both DataLoaders.

    .venv/bin/python tests/bench/check_frame_cache.py <model> [--samples 200] [--batches 20] [--workers 8]

1. Equivalence, for random samples, of what the model gets on the normal path (decoded full-size frames) and the
   cached path:
   - Diffusion (`resize`): the image after its own torchvision Resize; expected difference ≤ 0.5/255 (rounding).
   - GR00T N1.5 (`eagle`): the plugin's steps from the float batch on (uint8 conversion, Eagle encode and collate),
     compared as eagle_pixel_values; expected identical.
2. Speed: DataLoader samples/s at the recipe's batch size, both paths; for `eagle` also the Eagle preprocessing time
   per batch (it runs in the training process).
"""

from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    import numpy as np
    import torch
    import torchvision
    from lerobot.datasets import LeRobotDataset

    from drtc.compat import frame_cache
    from drtc.data import prepare
    from drtc.registry import Registry
    from drtc.train import build

    reg = Registry()
    flags, info = build(reg, a.model, download=True)
    r = info["resolved"]
    need = [n for n in prepare.needs(flags, r.policy, r.recipe.data) if isinstance(n, prepare.FrameCache)]
    if not need:
        print(f"{a.model}: its recipe has no data.frames", file=sys.stderr)
        return 1
    fc = need[0]
    root = flags["dataset.root"]
    frame_cache.install()
    # Speed first: DataLoader workers must not inherit video decoders opened in this process (LeRobot's warning),
    # so nothing is decoded here before they start.
    delta = None
    if r.policy == "diffusion":
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.policies.factory import make_policy_config

        cfg = make_policy_config(r.policy, **{k.removeprefix("policy."): v for k, v in flags.items()
                                              if k.startswith("policy.") and k.count(".") == 1
                                              and k not in ("policy.repo_id", "policy.push_to_hub", "policy.device",
                                                            "policy.type", "policy.path")})
        delta = resolve_delta_timestamps(cfg, LeRobotDataset(r.dataset.repo, revision=r.dataset.revision).meta)
    keys = LeRobotDataset(r.dataset.repo, revision=r.dataset.revision).meta.video_keys
    batch = int(flags["batch_size"])
    for name, rt in (("decode", None), ("cache", root)):
        ds = LeRobotDataset(r.dataset.repo, root=rt, revision=r.dataset.revision, return_uint8=True,
                            delta_timestamps=delta)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=a.workers,
                                         pin_memory=torch.cuda.is_available(), prefetch_factor=4,
                                         persistent_workers=True)
        it = iter(dl)
        for _ in range(3):
            next(it)
        t = time.perf_counter()
        for _ in range(a.batches):
            b = next(it)
        dt = (time.perf_counter() - t) / a.batches
        del it, dl
        print(f"{name}: {dt * 1000:.0f} ms per batch of {batch} = {batch / dt:.0f} samples/s "
              f"(images {tuple(b[keys[0]].shape)})", flush=True)
    plain = LeRobotDataset(r.dataset.repo, revision=r.dataset.revision, return_uint8=True)
    cached = LeRobotDataset(r.dataset.repo, root=root, revision=r.dataset.revision, return_uint8=True)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(plain), size=min(a.samples, len(plain)), replace=False).tolist()
    if fc.method == "resize":
        resize = torchvision.transforms.Resize([fc.height, fc.width])
        worst, total, count = 0.0, 0.0, 0
        for i in picks:
            p, c = plain[i], cached[i]
            for k in keys:
                want = resize(p[k].to(torch.float32) / 255.0)
                got = resize(c[k].to(torch.float32) / 255.0)
                d = (want - got).abs()
                worst, total, count = max(worst, d.max().item()), total + d.sum().item(), count + d.numel()
        print(f"equivalence over {len(picks)} samples x {len(keys)} cameras: max {worst * 255:.3f}/255, "
              f"mean {total / count * 255:.4f}/255 ({'ok' if worst <= 0.5 / 255 + 1e-6 else 'LARGER THAN ROUNDING'})")
        return 0
    eagle = eagle_steps()
    same = 0
    for s0 in range(0, len(picks), 32):
        part = picks[s0:s0 + 32]
        want = eagle([plain[i] for i in part], keys)
        got = eagle([cached[i] for i in part], keys)
        same += int(torch.equal(want, got))
        if not torch.equal(want, got):
            print(f"  batch {s0 // 32}: max difference {(want - got).abs().max().item():.6f}")
    batches = (len(picks) + 31) // 32
    print(f"equivalence over {len(picks)} samples x {len(keys)} cameras: eagle_pixel_values identical in "
          f"{same}/{batches} batches ({'ok' if same == batches else 'DIFFERENT'})")
    for name, ds in (("decode", plain), ("cache", cached)):
        items = [ds[i] for i in picks[:32]]
        eagle(items, keys)
        t = time.perf_counter()
        for _ in range(3):
            eagle(items, keys)
        print(f"Eagle preprocessing, batch {len(items)}, {name} frames: {(time.perf_counter() - t) / 3 * 1000:.0f} ms")
    return 0


def eagle_steps():
    """GR00T N1.5's image steps from the float batch on, as its preprocessor runs them; returns pixel values."""
    import numpy as np
    import torch
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from lerobot_policy_groot_n15.groot_n1 import DEFAULT_VENDOR_EAGLE_PATH
    from lerobot_policy_groot_n15.processor_groot_n15 import (
        DEFAULT_TOKENIZER_ASSETS_REPO, GrootEagleEncodeStep, TransitionKey, _build_eagle_processor,
        _to_uint8_np_bhwc, collate)
    from lerobot_policy_groot_n15.utils import ensure_eagle_cache_ready

    ensure_eagle_cache_ready(DEFAULT_VENDOR_EAGLE_PATH, HF_LEROBOT_HOME / "groot_n15_plugin" / DEFAULT_TOKENIZER_ASSETS_REPO,
                             DEFAULT_TOKENIZER_ASSETS_REPO)
    proc = _build_eagle_processor()
    step = GrootEagleEncodeStep()
    step._proc = proc

    def run(items, keys):
        # Training: uint8 frames -> float / 255 (lerobot-train) -> the plugin's pack step back to uint8 numpy.
        cams = [_to_uint8_np_bhwc(torch.stack([it[k] for it in items]).to(torch.float32) / 255.0) for k in keys]
        video = np.transpose(np.expand_dims(np.stack(cams, axis=1), axis=1), (0, 1, 2, 5, 3, 4))
        tr = {TransitionKey.OBSERVATION: {"video": video}, TransitionKey.COMPLEMENTARY_DATA: {"language": "task"}}
        out = step(tr)
        feats = [{"eagle_content": c} for c in out[TransitionKey.COMPLEMENTARY_DATA]["eagle_content"]]
        return collate(feats, proc)["eagle_pixel_values"]

    return run


if __name__ == "__main__":
    sys.exit(main())
