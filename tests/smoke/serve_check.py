"""Serve check for one trained checkpoint: does LeRobot 0.7's policy server admit it, with RTC where the policy has
it, and how long does a chunk take compared with DRTC's margin?

    .venv/bin/python tests/smoke/serve_check.py <model> [<checkpoint folder>] [--calls 20] [--device cuda]

It builds the same deployment `lab serve <model> --path <folder>` would, adds the `rtc_guided` mode for RTC
policies (lab registry RTC_POLICIES), and loads it with upstream's `load_deployment`. That runs the server's own
admission checks (chunk spec: one observation frame, no temporal ensembling; RTC call interface; RTC horizon within
the prediction horizon) and its warmup, whose second RTC call passes the first chunk back as the prefix. No Zenoh
session is opened. Then it times `runner.predict` (preprocess + policy + postprocess, the server's work per request)
on random camera images, in each mode, and compares the median with the DRTC margin H - s_min (in actions at the
dataset's fps): the next chunk must arrive before the queue runs out.

`--set key=value` (repeatable) serves a copy of the checkpoint with those keys of its config.json changed, to time
inference settings without retraining, e.g. `--set num_inference_steps=5` for MolmoAct 2's flow steps. The copy is
a temporary folder of symlinks plus the edited config.json; the checkpoint is not written. `--tag` names the output
(serve_<model>_<tag>.json) so several settings can be compared.

Prints one summary line and writes logs/smoke/serve_<model>[_<tag>].json. Exit code 1 if the deployment is refused.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import yaml


def parse_sets(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set wants KEY=VALUE, got {item!r}")
        out[key] = yaml.safe_load(value)
    return out


def with_config(checkpoint: Path, overrides: dict, dst: Path) -> Path:
    """`dst`: symlinks to every file of `checkpoint` and its config.json with `overrides` applied."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise SystemExit(f"not in {checkpoint / 'config.json'}: {', '.join(unknown)}")
    cfg.update(overrides)
    for f in checkpoint.iterdir():
        if f.name != "config.json":
            (dst / f.name).symlink_to(f.resolve())
    (dst / "config.json").write_text(json.dumps(cfg, indent=4))
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("checkpoint", type=Path, nargs="?",
                    help="local checkpoint folder; omitted: the model's Hub repo, old-format (minor) checkpoints "
                         "converted as `lab serve` does")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--s-min", type=int, help="DRTC s_min (default: the recipe's serve.s_min, else 20)")
    ap.add_argument("--rtc-horizon", type=int, default=10, help="execution.rtc.execution_horizon for rtc_guided")
    ap.add_argument("--out", type=Path, default=Path("logs/smoke"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="change a key of the checkpoint's config.json for this check (value parsed as YAML)")
    ap.add_argument("--tag", help="suffix for the output file name")
    a = ap.parse_args(argv)

    import draccus
    import torch
    from lerobot.inference.contracts import ObservationSnapshot
    from lerobot.policies.chunk import ExecutionMode
    from lerobot.remote_inference.configs import ServerConfig
    from lerobot.remote_inference.deployment import load_deployment
    from lerobot.utils.import_utils import register_third_party_plugins

    from drtc.control.serve import server_config
    from drtc.registry import Registry
    from drtc.registry.registry import RTC_POLICIES

    register_third_party_plugins()
    from drtc.control.chunk_adapters import install as install_chunk_adapters

    install_chunk_adapters()  # as `lab serve` does
    reg = Registry()
    r = reg.resolve(a.model)
    overrides = parse_sets(a.set)
    if a.checkpoint is None:
        from drtc.compat.serve import install_loader, needs_compat, prepare_checkpoint

        checkpoint = r.hub
        if needs_compat(r.hub):
            checkpoint = prepare_checkpoint(r.hub)
            install_loader()
            print(f"old-format checkpoint {r.hub}: checking the converted copy {checkpoint}", flush=True)
        if overrides:
            raise SystemExit("--set needs a local checkpoint folder")
    else:
        checkpoint = a.checkpoint.resolve()
        if overrides:
            checkpoint = with_config(checkpoint, overrides, Path(tempfile.mkdtemp(prefix="serve_check_")))
    rtc = r.policy in RTC_POLICIES
    modes = ["chunk", "rtc_guided"] if rtc else ["chunk"]
    cfg = server_config(reg, a.model, str(checkpoint), a.device, modes=modes)
    if rtc:
        cfg["execution"]["rtc"] = {"enabled": True, "mode": "guided", "execution_horizon": a.rtc_horizon}
    s_min = a.s_min or int(r.serve.get("s_min") or 20)
    fps = cfg["execution"]["action_fps"]
    result = {"model": a.model, "policy": r.policy, "checkpoint": str(a.checkpoint or r.hub), "modes": modes, "s_min": s_min,
              "fps": fps, "config_overrides": overrides, "admitted": False}
    out_name = f"serve_{a.model}{'_' + a.tag if a.tag else ''}.json"
    a.out.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    try:
        server_cfg = draccus.parse(ServerConfig, config_path=f.name, args=[])
        t0 = time.perf_counter()
        runner, _identity = load_deployment(server_cfg)  # admission + warmup in every mode
        result["load_and_warmup_s"] = round(time.perf_counter() - t0, 1)
    except Exception as e:  # noqa: BLE001 - the refusal is the result
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        print(f"{a.model}{' ' + str(overrides) if overrides else ''}: REFUSED  {result['error']}", flush=True)
        (a.out / out_name).write_text(json.dumps(result, indent=1))
        return 1

    caps = runner.capabilities
    H = caps.execution_steps
    result.update(admitted=True, prediction_steps=caps.prediction_steps, execution_steps=H,
                  rtc_horizon=caps.rtc_horizon, peak_gpu_gb=None)
    rng = np.random.default_rng(0)
    obs = {}
    for feat in server_cfg.features:
        if feat.dtype == "uint8":
            obs[feat.name] = rng.integers(0, 256, size=feat.shape, dtype=np.uint8)
        else:
            obs[feat.name] = np.zeros(feat.shape, dtype=feat.dtype)
    task = r.dataset.task if r.dataset else ""

    for mode in caps.modes:
        totals, policy_ms, previous = [], [], None
        for i in range(a.calls + 1):
            snap = ObservationSnapshot(obs, time.time(), task, observation_id=f"bench-{mode.value}-{i}")
            kwargs = {}
            if mode is not ExecutionMode.CHUNK and previous is not None:
                kwargs = {"inference_delay": min(3, caps.rtc_horizon), "model_continuation": previous.model_actions,
                          "canonical_continuation": previous.canonical_actions}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            chunk = runner.predict(snap, mode=mode, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i:  # the first call is extra warmup
                totals.append((time.perf_counter() - t0) * 1000)
                policy_ms.append(chunk.server_durations["policy"] * 1000)
            previous = chunk
        med = statistics.median(totals)
        p90 = sorted(totals)[max(0, math.ceil(0.9 * len(totals)) - 1)]
        steps = math.ceil(med / 1000 * fps)
        result[mode.value] = {"median_ms": round(med, 1), "p90_ms": round(p90, 1),
                              "policy_median_ms": round(statistics.median(policy_ms), 1),
                              "latency_steps": steps, "margin_steps": H - s_min, "fits": H - s_min > steps}
    if torch.cuda.is_available():
        result["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    label = a.model + (f" [{', '.join(f'{k}={v}' for k, v in overrides.items())}]" if overrides else "")
    parts = [f"{label}: admitted {'+'.join(m.value for m in caps.modes)}  H={H} (predicts {caps.prediction_steps})"]
    for mode in caps.modes:
        m = result[mode.value]
        parts.append(f"{mode.value} {m['median_ms']} ms (p90 {m['p90_ms']}) = {m['latency_steps']} steps vs "
                     f"H-s_min {m['margin_steps']}: {'fits' if m['fits'] else 'DOES NOT FIT'}")
    parts.append(f"peak {result['peak_gpu_gb']} GB")
    print("  |  ".join(parts), flush=True)
    (a.out / out_name).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
