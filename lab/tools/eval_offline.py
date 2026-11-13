"""Offline evaluation: score a trained registry model on held-out episodes, without the robot.

Uses the same metrics as validation during training (lerobot_train.validate): the training loss and
the error of predicted action chunks (normalized units) on frames from the held-out episodes. Useful
for models trained before validation existed, after training, or on other episodes.

Episodes: the entry's held-out split (train.split) by default, or --episodes. The dataset and policy
settings come from the checkpoint's train_config.json. Results go to results/offline/<model>.json.

Usage:
    python -m lab.tools.eval_offline <model> [--episodes "45-49"] [--batches 20] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import torch

from lab.tools.config import RESULTS_DIR, load_models
from lab.tools.train import parse_episodes, select_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a registry model on held-out episodes")
    parser.add_argument("model")
    parser.add_argument("--episodes", help='episodes to score, e.g. "45-49" (default: the held-out split)')
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    models = load_models()
    if args.model not in models:
        sys.exit(f"Unknown model {args.model!r}. Registered: {', '.join(models)}")
    entry = models[args.model]
    if args.episodes:
        episodes = parse_episodes(args.episodes)
    else:
        held_out = select_episodes(entry.get("train") or {})[1]
        if not held_out:
            sys.exit(f"{args.model!r} has no held-out split (train.split); pass --episodes")
        episodes = held_out

    from accelerate import Accelerator

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.lora import load_policy
    from lerobot.scripts.lerobot_train import make_val_loader, validate

    path, finetune = entry["pretrained_name_or_path"], entry.get("finetune", "full")
    cfg = TrainPipelineConfig.from_pretrained(path)  # dataset + policy settings used in training
    cfg.val_episodes, cfg.val_batches = episodes, args.batches
    cfg.batch_size, cfg.num_workers = args.batch_size, args.num_workers
    cfg.dataset.root, cfg.dataset.video_backend = None, "pyav"
    trained_on = set(cfg.dataset.episodes or [])
    if trained_on & set(episodes):
        print(f"WARNING: episodes {sorted(trained_on & set(episodes))} were used in training; "
              "the score is not a held-out score.", flush=True)

    policy = load_policy(get_policy_class(entry["policy_type"]), path, finetune)
    policy.config.device = args.device
    policy.to(args.device)
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=path,
                                      preprocessor_overrides={"device_processor": {"device": args.device}})
    accelerator = Accelerator(cpu=args.device == "cpu")
    loader = make_val_loader(cfg, accelerator.device)
    metrics = validate(policy, loader, pre, accelerator, actions=True)

    record = {"model": args.model, "dataset": cfg.dataset.repo_id, "episodes": episodes,
              "frames": len(loader.dataset), "date": datetime.now().isoformat(timespec="seconds"), **metrics}
    out = RESULTS_DIR / "offline" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"{args.model}: val_loss {metrics['val_loss']:.4f}, val_action_mse {metrics['val_action_mse']:.4f} "
          f"on {len(loader.dataset)} frames of episodes {episodes} -> {out}")


if __name__ == "__main__":
    main()
