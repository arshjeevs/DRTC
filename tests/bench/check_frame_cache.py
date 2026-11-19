"""Check a frame cache against the normal path, and time both DataLoaders.

    .venv/bin/python tests/bench/check_frame_cache.py <model> [--samples 200] [--batches 20] [--workers 8]

1. Equivalence: for random samples, the image Diffusion's encoder gets on the normal path (decoded frame / 255,
   then its own torchvision Resize) against the cached path (cached frame / 255; its Resize leaves it unchanged).
   Prints the largest difference per pixel (expected ≤ 0.5/255 from the uint8 rounding) and the mean.
2. Speed: DataLoader samples/s at the recipe's batch size with the recipe's delta timestamps, both paths.
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
    resize = torchvision.transforms.Resize([fc.height, fc.width])
    rng = np.random.default_rng(0)
    worst, total, count = 0.0, 0.0, 0
    for i in rng.choice(len(plain), size=min(a.samples, len(plain)), replace=False).tolist():
        p, c = plain[i], cached[i]
        for k in keys:
            want = resize(p[k].to(torch.float32) / 255.0)
            got = resize(c[k].to(torch.float32) / 255.0)
            d = (want - got).abs()
            worst, total, count = max(worst, d.max().item()), total + d.sum().item(), count + d.numel()
    print(f"equivalence over {min(a.samples, len(plain))} samples x {len(keys)} cameras: max {worst * 255:.3f}/255, "
          f"mean {total / count * 255:.4f}/255 ({'ok' if worst <= 0.5 / 255 + 1e-6 else 'LARGER THAN ROUNDING'})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
