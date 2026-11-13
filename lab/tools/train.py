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
import math
import random
import re
import shlex
import sys
from pathlib import Path

from lab.tools.config import REPO_ROOT, legacy_models, load_models, naming_problems
from lab.tools.passthrough import parse_passthrough
from lerobot.policies import lora

TRAIN_DIR = REPO_ROOT / "outputs/train"
HUB_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
# New runs log to wandb (resumed runs keep their own settings); checkpoints go to the Hub, not to wandb.
WANDB = {"--wandb.enable": "true", "--wandb.project": "drtc", "--wandb.disable_artifact": "true"}


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


def parse_episodes(spec) -> list[int]:
    """[0, 1, 5], "0-44" or "0-9,20-29" -> episode indices."""
    if isinstance(spec, list):
        return [int(i) for i in spec]
    episodes = []
    for part in str(spec).replace(" ", "").split(","):
        first, _, last = part.partition("-")
        episodes += range(int(first), int(last or first) + 1)
    return episodes


def split_episodes(split: dict, count: int | None) -> tuple[list[int], list[int]]:
    """Seeded random split -> (training episodes, held-out episodes).

    `count` takes the first `count` of one fixed shuffle of the training episodes, so subsets are
    nested (10 within 25 within 50) and every model sees the same held-out episodes.
    """
    order = random.Random(split["seed"]).sample(range(split["total"]), split["total"])
    held_out, pool = order[:split["held_out"]], order[split["held_out"]:]
    if count is not None and count > len(pool):
        sys.exit(f"episodes: {count} > the {len(pool)} training episodes of the split")
    return sorted(pool[:count] if count is not None else pool), sorted(held_out)


def select_episodes(train: dict) -> tuple[list[int] | None, list[int] | None]:
    """(training episodes, held-out episodes) from an entry's train: block; None = not set."""
    if "split" in train:
        count = train.get("episodes")
        if count is not None and not isinstance(count, int):
            sys.exit("with split:, episodes must be a number of episodes (a random subset of the training ones)")
        return split_episodes(train["split"], count)
    if "episodes" in train:
        return parse_episodes(train["episodes"]), None
    return None, None


def check_episode_count(repo_id: str, root: str | None, split: dict) -> None:
    """Refuse to train on a dataset that does not have every episode the split expects (yet)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    found = LeRobotDatasetMetadata(repo_id, root=root).total_episodes
    if found < split["total"]:
        sys.exit(f"{repo_id} has {found} episodes; the split expects {split['total']}. Recording not finished?")
    if found > split["total"]:
        print(f"note: {repo_id} has {found} episodes; episodes {split['total']}-{found - 1} are not used", flush=True)


def resolve_base(source: str, download: bool) -> tuple[str, str]:
    """`repo` or `repo@revision` -> (what --policy.path loads, the repo id recorded as the LoRA base).

    A pinned revision is downloaded and loaded from its local snapshot (lerobot-train has no revision
    flag). Pin pi0 / pi05 to the commit before the Hub added processor steps this fork does not have.
    """
    repo, _, revision = source.partition("@")
    if not revision:
        return source, source
    if not download:
        return f"<{repo} at {revision[:10]}, downloaded at start>", repo
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision), repo


def epochs_to_steps(repo_id: str, root: str | None, episodes: list[int] | None, epochs: float,
                    batch_size: int) -> tuple[int, int]:
    """Training steps for `epochs` passes over the training frames -> (steps, frames)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    lengths = dict(zip(meta.episodes["episode_index"], meta.episodes["length"]))
    frames = sum(lengths[e] for e in episodes) if episodes is not None else meta.total_frames
    return max(1, math.ceil(epochs * frames / batch_size)), frames


def flag_value(value):
    """YAML value -> lerobot-train flag text (true/false; dicts such as rename_map as JSON)."""
    if isinstance(value, bool):
        return str(value).lower()
    return json.dumps(value) if isinstance(value, dict) else value


def lora_make_policy(make_policy, spec: dict, base_id: str | None = None):
    """Wrap lerobot-train's make_policy: base policy + adapters (or, on resume, the saved adapters).

    `base_id` is recorded as the adapter's base when the base was loaded from a local snapshot.
    """
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
        return lora.apply_lora(make_policy(cfg, **kwargs), spec, base_id or checkpoint)
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
    if args.model not in legacy_models() and (problems := naming_problems(args.model, entry, models)):
        sys.exit(f"{args.model!r} breaks the naming convention (lab/models.yaml header):\n  - "
                 + "\n  - ".join(problems))

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
        # No wandb flags: the run continues with its saved wandb settings (project and run id), so a run
        # started in another project resumes there, and one trained without wandb stays without it.
        flags = {"--config_path": str(config), "--resume": "true"}
    else:
        if finetune == "lora" and "from" not in train:
            sys.exit(f"finetune: lora needs a base checkpoint: set train.from for {args.model!r}")
        if out.exists():
            sys.exit(f"{out} exists. Continue it with --resume, or delete it to start over.")
        if "epochs" in train and "steps" in train:
            sys.exit(f"{args.model!r}: set train.steps or train.epochs, not both")
        base_path, base_id = resolve_base(train["from"], download=not args.dry_run) if "from" in train else (None, None)
        base = {"--policy.path": base_path} if base_path else {"--policy.type": entry["policy_type"]}
        flags = {
            "--dataset.repo_id": train["dataset"],
            **base,
            "--policy.device": "cuda",
            # Pushed below from the last checkpoint, the same way for both methods.
            "--policy.push_to_hub": "false",
            "--output_dir": str(out),
            "--job_name": args.model,
            "--batch_size": train.get("batch_size", 8),
            "--steps": train.get("steps", 20_000),  # replaced below when train.epochs is set
            "--save_freq": train.get("save_freq", 5_000),
            **WANDB,
            # LeRobot picks torchcodec whenever it is installed, but it cannot load with FFmpeg 8.
            "--dataset.video_backend": "pyav",
            # Model-specific lerobot-train flags from the entry, e.g. {policy.dtype: bfloat16}
            **{f"--{key}": flag_value(value) for key, value in train.get("flags", {}).items()},
        }
        # Training episodes (held-out split, data-efficiency subsets) and the held-out ones for validation.
        episodes, held_out = select_episodes(train)
        if episodes is not None:
            flags["--dataset.episodes"] = json.dumps(episodes)
        if held_out:
            flags["--val_episodes"] = json.dumps(held_out)
    flags.update(overrides)
    if not args.resume and "--dataset.root" not in flags:
        check_local_dataset(flags["--dataset.repo_id"])
    if not args.resume and "split" in train:
        check_episode_count(flags["--dataset.repo_id"], flags.get("--dataset.root"), train["split"])
    if not args.resume and "epochs" in train and "--steps" not in overrides:
        episodes = json.loads(flags["--dataset.episodes"]) if "--dataset.episodes" in flags else None
        steps, frames = epochs_to_steps(flags["--dataset.repo_id"], flags.get("--dataset.root"), episodes,
                                        train["epochs"], int(flags["--batch_size"]))
        flags["--steps"] = steps
        print(f"epochs: {train['epochs']} x {frames} frames / batch {flags['--batch_size']} = {steps} steps", flush=True)
        if "--val_episodes" in flags and "--val_freq" not in overrides:
            flags["--val_freq"] = max(50, steps // 20)  # ~20 validation points however long the run is

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
    if not args.resume and "--val_episodes" in flags:
        print(f"episodes: {len(json.loads(flags['--dataset.episodes']))} for training, "
              f"held out for validation: {flags['--val_episodes']}", flush=True)
    if args.dry_run:
        return

    from lerobot.scripts import lerobot_train

    if spec:  # on --resume the saved adapter settings win (lora_make_policy)
        lerobot_train.make_policy = lora_make_policy(lerobot_train.make_policy, spec, None if args.resume else base_id)
    sys.argv = cmd
    lerobot_train.main()
    if push:
        push_checkpoint(out, "last", target)


if __name__ == "__main__":
    main()
