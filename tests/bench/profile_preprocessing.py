"""Where a model's preprocessing time goes: its real preprocessor, fed real training batches, timed step by step.

    .venv/bin/python tests/bench/profile_preprocessing.py <model> [--batches 5] [--workers 4] [--device cuda]

Builds the policy config from the model's recipe flags (as lab train does) and its preprocessor from scratch, loads
the dataset the way training does (the recipe's prepared variant, frame cache included), converts each batch like
lerobot-train (uint8 frames / 255) and runs the preprocessor with a timer around every step. Prints the median time
per step and the functions that take the most time (cProfile, cumulative). No model weights are needed.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import sys
import time


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args(argv)

    import torch
    from lerobot.configs.types import FeatureType
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.factory import make_policy_config, make_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.utils.import_utils import register_third_party_plugins

    from drtc.compat import frame_cache
    from drtc.registry import Registry
    from drtc.train import build

    register_third_party_plugins()
    frame_cache.install()
    flags, info = build(Registry(), a.model, download=True)
    r = info["resolved"]
    skip = {"policy.repo_id", "policy.push_to_hub", "policy.device", "policy.type", "policy.path"}
    kwargs = {k.removeprefix("policy."): v for k, v in flags.items()
              if k.startswith("policy.") and k.count(".") == 1 and k not in skip}
    cfg = make_policy_config(r.policy, device=a.device if torch.cuda.is_available() else "cpu", **kwargs)
    meta_ds = LeRobotDataset(r.dataset.repo, root=flags.get("dataset.root"), revision=r.dataset.revision)
    feats = dataset_to_policy_features(meta_ds.meta.features)
    cfg.output_features = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    cfg.input_features = {k: f for k, f in feats.items() if k not in cfg.output_features}
    pre, _post = make_pre_post_processors(cfg, dataset_stats=meta_ds.meta.stats)
    ds = LeRobotDataset(r.dataset.repo, root=flags.get("dataset.root"), revision=r.dataset.revision,
                        delta_timestamps=resolve_delta_timestamps(cfg, meta_ds.meta), return_uint8=True)
    keys = ds.meta.camera_keys
    dl = torch.utils.data.DataLoader(ds, batch_size=int(flags["batch_size"]), shuffle=True, num_workers=a.workers)

    names = [type(s).__name__ for s in pre.steps]
    times: dict[int, list[float]] = {i: [] for i in range(len(names))}
    start: dict[int, float] = {}
    pre.before_step_hooks.append(lambda i, _t: start.__setitem__(i, time.perf_counter()))
    pre.after_step_hooks.append(lambda i, _t: times[i].append(time.perf_counter() - start[i]))

    def convert(batch):
        for k in keys:
            if batch[k].dtype == torch.uint8:
                batch[k] = batch[k].to(torch.float32) / 255.0
        return batch

    it = iter(dl)
    pre(convert(next(it)))  # warm up (processor caches, first CUDA use)
    for t in times.values():
        t.clear()
    prof = cProfile.Profile()
    totals = []
    for _ in range(a.batches):
        batch = convert(next(it))
        t0 = time.perf_counter()
        prof.enable()
        pre(batch)
        prof.disable()
        totals.append(time.perf_counter() - t0)
    print(f"{a.model}: preprocessing {statistics.median(totals) * 1000:.0f} ms per batch of {flags['batch_size']} "
          f"(median of {a.batches}; torch threads {torch.get_num_threads()})")
    for i, n in enumerate(names):
        if times[i]:
            print(f"  {statistics.median(times[i]) * 1000:8.1f} ms  {i:2d} {n}")
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(a.top)
    print("\n".join(line for line in s.getvalue().splitlines() if line.strip())[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
