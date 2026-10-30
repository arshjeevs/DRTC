"""Train a model from lab/models.yaml on the laptop GPU and push it to the Hugging Face Hub.

The entry's `train:` block says what to train; its `pretrained_name_or_path` is where the result is
pushed. The policy server loads models from that same field, so once training finishes
`lab/bin/run <experiment> --model <name>` runs the new model.

Checkpoints go to outputs/train/<name>/. Any other lerobot-train flag is passed through and
overrides the generated value (`lerobot-train --help`).

Usage:
    python -m lab.tools.train <model> [--dry-run] [--no-push] [--resume] [--<lerobot-train flag>=value ...]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

from lab.tools.config import REPO_ROOT, load_models
from lab.tools.passthrough import parse_passthrough

TRAIN_DIR = REPO_ROOT / "outputs/train"
HUB_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def check_local_dataset(repo_id: str) -> None:
    """A local copy shadows the Hub; an empty one (left by a failed record) would train on nothing."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    info = HF_LEROBOT_HOME / repo_id / "meta/info.json"
    if info.is_file() and json.loads(info.read_text())["total_episodes"] == 0:
        sys.exit(f"Local copy {info.parent.parent} has 0 episodes and would be used instead of the Hub "
                 f"dataset.\nMove it aside, or pass --dataset.root=<empty dir> to download a fresh copy.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a registry model (lerobot-train)", allow_abbrev=False)
    parser.add_argument("model", help="Name in lab/models.yaml with a train: block")
    parser.add_argument("--dry-run", action="store_true", help="Print the lerobot-train command and exit")
    parser.add_argument("--no-push", action="store_true", help="Keep the model local (no Hub upload)")
    parser.add_argument("--resume", action="store_true", help="Continue from outputs/train/<model>/checkpoints/last")
    args, passthrough = parser.parse_known_args()

    models = load_models()
    if args.model not in models:
        sys.exit(f"Unknown model {args.model!r}. Registered: {', '.join(models)}")
    entry = models[args.model]
    train = entry.get("train")
    if not train:
        sys.exit(f"{args.model!r} has no train: block in lab/models.yaml")

    out = TRAIN_DIR / args.model
    target = entry["pretrained_name_or_path"]
    push = not args.no_push and bool(HUB_ID.match(target)) and not target.startswith("outputs/")

    if args.resume:
        config = out / "checkpoints/last/pretrained_model/train_config.json"
        if not config.is_file():
            sys.exit(f"Nothing to resume: {config} not found")
        flags = {"--config_path": str(config), "--resume": "true"}
    else:
        if out.exists():
            sys.exit(f"{out} exists. Continue it with --resume, or delete it to start over.")
        base = {"--policy.path": train["from"]} if "from" in train else {"--policy.type": entry["policy_type"]}
        flags = {
            "--dataset.repo_id": train["dataset"],
            **base,
            "--policy.device": "cuda",
            "--policy.push_to_hub": str(push).lower(),
            "--output_dir": str(out),
            "--job_name": args.model,
            "--batch_size": train.get("batch_size", 8),
            "--steps": train.get("steps", 20_000),
            "--save_freq": train.get("save_freq", 5_000),
            "--wandb.enable": "false",
            # LeRobot picks torchcodec whenever it is installed, but it cannot load with FFmpeg 8.
            "--dataset.video_backend": "pyav",
        }
        if push:
            flags["--policy.repo_id"] = target
    overrides = parse_passthrough(passthrough)
    flags.update(overrides)
    if not args.resume and "--dataset.root" not in flags:
        check_local_dataset(flags["--dataset.repo_id"])

    cmd = ["lerobot-train", *(f"{key}={value}" for key, value in flags.items())]
    print(" \\\n    ".join(shlex.quote(part) for part in cmd), flush=True)
    if overrides:
        print(f"overrides: {', '.join(overrides)}", flush=True)
    print(f"push to Hub: {target}" if push else "push to Hub: no (local only)", flush=True)
    if args.dry_run:
        return
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
