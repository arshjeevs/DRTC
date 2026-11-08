"""Train a model from lab/models.yaml on the laptop GPU and push it to the Hugging Face Hub.

The entry's `train:` block says what to train; its `pretrained_name_or_path` is where the result is
pushed. The policy server loads models from that same field, so once training finishes
`lab/bin/run <experiment> --model <name>` runs the new model.

Fine-tuning method: the entry's `finetune:` (default full), the same field the client sends to the
policy server so it knows how to load the model.
    full  update the weights SmolVLA normally trains (action expert + projections; VLM frozen) and
          save the whole model (~900 MB)
    lora  freeze the policy, train low-rank adapters (src/lerobot/policies/lora.py) and save only
          those (~21 MB at r=32); the server loads the base model named in the adapter, then the adapter
LoRA settings come from `train.lora:` (r, alpha, dropout, targets) and `--lora.<key>=value` flags.

Checkpoints go to outputs/train/<name>/. Any other lerobot-train flag is passed through and
overrides the generated value (`lerobot-train --help`).

A run stopped early pushes nothing; `--push [checkpoint]` uploads a saved checkpoint instead
(default: the latest), e.g. after Ctrl+C at step 5000.

Usage:
    python -m lab.tools.train <model> [--dry-run] [--no-push] [--resume] [--<lerobot-train flag>=value ...]
    python -m lab.tools.train <model> --finetune lora --lora.r=16 --lora.targets=vlm+expert --no-push
    python -m lab.tools.train <model> --push [last | 005000]
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

from lab.tools.config import REPO_ROOT, load_models
from lab.tools.passthrough import parse_passthrough
from lerobot.policies import lora

TRAIN_DIR = REPO_ROOT / "outputs/train"
HUB_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
# Every run (new or resumed) logs to wandb; checkpoints stay local and go to the Hub, not to wandb.
WANDB = {"--wandb.enable": "true", "--wandb.project": "drtc-smolvla", "--wandb.disable_artifact": "true"}


def check_local_dataset(repo_id: str) -> None:
    """A local copy shadows the Hub; an empty one (left by a failed record) would train on nothing."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    info = HF_LEROBOT_HOME / repo_id / "meta/info.json"
    if info.is_file() and json.loads(info.read_text())["total_episodes"] == 0:
        sys.exit(f"Local copy {info.parent.parent} has 0 episodes and would be used instead of the Hub "
                 f"dataset.\nMove it aside, or pass --dataset.root=<empty dir> to download a fresh copy.")


def push_checkpoint(out: Path, name: str, target: str) -> None:
    """Upload checkpoints/<name>/pretrained_model (weights, config, processors) to the Hub."""
    from huggingface_hub import HfApi

    checkpoints = out / "checkpoints"
    folder = checkpoints / name / "pretrained_model"
    if not folder.is_dir() and name.isdigit():
        folder = checkpoints / f"{int(name):06d}" / "pretrained_model"
    if not folder.is_dir():
        saved = sorted(p.name for p in checkpoints.glob("*")) if checkpoints.is_dir() else []
        sys.exit(f"No checkpoint {name!r} in {checkpoints}. Saved: {', '.join(saved) or 'none'}")
    if not HUB_ID.match(target) or target.startswith("outputs/"):
        sys.exit(f"pretrained_name_or_path {target!r} is not a Hub repo id")
    step = folder.resolve().parent.name
    kind = "lora" if (folder / lora.ADAPTER_CONFIG).is_file() else "full"
    print(f"pushing {kind} checkpoint {folder} (step {step}) -> https://huggingface.co/{target}", flush=True)
    api = HfApi()
    api.create_repo(target, exist_ok=True)
    # Drop the other method's weights, so the repo never holds a stale model next to the new one.
    stale = ["model.safetensors"] if kind == "lora" else [lora.ADAPTER_CONFIG, lora.ADAPTER_WEIGHTS]
    api.upload_folder(repo_id=target, folder_path=folder, delete_patterns=stale,
                      commit_message=f"{out.name}: {kind} checkpoint {step}")
    print("done", flush=True)


def lora_make_policy(make_policy, spec: dict):
    """Wrap lerobot-train's make_policy: base policy + adapters (or, on resume, the saved adapters)."""
    def make(cfg, **kwargs):
        checkpoint = cfg.pretrained_path
        if checkpoint and lora.is_adapter(checkpoint):  # --resume: rebuild base, then load adapters
            saved, base, weights = lora.read_adapter(checkpoint)
            cfg.pretrained_path = base
            try:
                policy = make_policy(cfg, **kwargs)
            finally:
                cfg.pretrained_path = checkpoint
            lora.apply_lora(policy, saved, base)
            lora.load_adapter_weights(policy, weights)
            return policy
        return lora.apply_lora(make_policy(cfg, **kwargs), spec, checkpoint)
    return make


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a registry model (lerobot-train)", allow_abbrev=False)
    parser.add_argument("model", help="Name in lab/models.yaml with a train: block")
    parser.add_argument("--dry-run", action="store_true", help="Print the lerobot-train command and exit")
    parser.add_argument("--no-push", action="store_true", help="Keep the model local (no Hub upload)")
    parser.add_argument("--resume", action="store_true", help="Continue from outputs/train/<model>/checkpoints/last")
    parser.add_argument("--push", nargs="?", const="last", metavar="CHECKPOINT",
                        help="Don't train: upload a saved checkpoint (default: last) to the Hub")
    parser.add_argument("--finetune", choices=lora.FINETUNE_METHODS,
                        help="Override the entry's finetune: (only with --no-push; the server reads the entry)")
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
    if args.push:
        return push_checkpoint(out, args.push, target)
    push = not args.no_push and bool(HUB_ID.match(target)) and not target.startswith("outputs/")
    finetune = args.finetune or entry.get("finetune", "full")
    if finetune != entry.get("finetune", "full") and push:
        sys.exit(f"models.yaml says {args.model!r} is finetune: {entry.get('finetune', 'full')}, and the server "
                 f"loads it that way. Change the entry (or add a new one with its own Hub id), or add --no-push.")

    overrides = parse_passthrough(passthrough)
    lora_flags = {k.removeprefix("--lora."): overrides.pop(k) for k in list(overrides) if k.startswith("--lora.")}
    if lora_flags and finetune != "lora":
        sys.exit(f"--lora.* flags given, but {args.model!r} trains with finetune: {finetune}")

    if args.resume:
        last = out / "checkpoints/last/pretrained_model"
        config = last / "train_config.json"
        if not config.is_file():
            sys.exit(f"Nothing to resume: {config} not found")
        saved = "lora" if (last / lora.ADAPTER_CONFIG).is_file() else "full"
        if saved != finetune:
            sys.exit(f"{last} is a {saved} checkpoint, but {args.model!r} is finetune: {finetune}")
        # --resume runs up to the run's saved step target; a finished run needs a larger --steps.
        done = json.loads((last.parent / "training_state/training_step.json").read_text())["step"]
        goal = int(overrides.get("--steps", json.loads(config.read_text())["steps"]))
        if done >= goal:
            sys.exit(f"{args.model!r} already finished: checkpoint at step {done}, target {goal}.\n"
                     f"To train further, raise the target, e.g. lab/bin/train {args.model} --resume --steps={done + 10_000}")
        print(f"resuming at step {done}, training to {goal}", flush=True)
        flags = {"--config_path": str(config), "--resume": "true", **WANDB}
    else:
        if finetune == "lora" and "from" not in train:
            sys.exit(f"finetune: lora needs a base checkpoint: set train.from for {args.model!r}")
        if out.exists():
            sys.exit(f"{out} exists. Continue it with --resume, or delete it to start over.")
        base = {"--policy.path": train["from"]} if "from" in train else {"--policy.type": entry["policy_type"]}
        flags = {
            "--dataset.repo_id": train["dataset"],
            **base,
            "--policy.device": "cuda",
            # Pushed below from the last checkpoint, the same way for both methods.
            "--policy.push_to_hub": "false",
            "--output_dir": str(out),
            "--job_name": args.model,
            "--batch_size": train.get("batch_size", 8),
            "--steps": train.get("steps", 20_000),
            "--save_freq": train.get("save_freq", 5_000),
            **WANDB,
            # LeRobot picks torchcodec whenever it is installed, but it cannot load with FFmpeg 8.
            "--dataset.video_backend": "pyav",
        }
    flags.update(overrides)
    if not args.resume and "--dataset.root" not in flags:
        check_local_dataset(flags["--dataset.repo_id"])

    cmd = ["lerobot-train", *(f"{key}={value}" for key, value in flags.items())]
    print(" \\\n    ".join(shlex.quote(part) for part in cmd), flush=True)
    if overrides:
        print(f"overrides: {', '.join(overrides)}", flush=True)
    spec = None
    if finetune == "lora":
        spec = lora.resolve_spec(entry["policy_type"], {**train.get("lora", {}), **lora_flags})
        print(f"finetune: lora r={spec['r']} alpha={spec['alpha']} dropout={spec['dropout']}\n"
              f"    adapters on {spec['target_modules']}\n    trained in full: {spec['trainable']}", flush=True)
    else:
        print("finetune: full", flush=True)
    print(f"push to Hub: {target}" if push else "push to Hub: no (local only)", flush=True)
    if args.dry_run:
        return

    from lerobot.scripts import lerobot_train

    if spec:  # on --resume the saved adapter settings win (lora_make_policy)
        lerobot_train.make_policy = lora_make_policy(lerobot_train.make_policy, spec)
    sys.argv = cmd
    lerobot_train.main()
    if push:
        push_checkpoint(out, "last", target)


if __name__ == "__main__":
    main()
