"""Train a registry model with LeRobot 0.7's lerobot-train: `lab train <model>`.

The registry (lab/) gives dataset (pinned revision), recipe and overrides; this builds the lerobot-train flags and
adds only what 0.7 lacks:
- our seeded held-out split (the same held-out episodes as in the minor), mapped onto 0.7's own eval split: the
  episode list is passed training-first, held-out-last, and eval_split selects exactly the held-out tail;
- epochs -> steps over the training frames;
- refusing a stale local dataset copy (a different version cached under the same name);
- checkpoints: the weights of every one are kept, the training state only for the latest (see
  prune_training_state);
- recipes that need more than the recorded dataset (relative-action stats, a frame cache) train on a prepared
  variant (drtc.data.prepare; drtc.compat.frame_cache reads the cached frames);
- a clean stop: Ctrl-C, SIGTERM or `lab stop <model>` saves a checkpoint at the next step, pushes it with its
  training state (resumable anywhere: `lab train <model> --resume`) and exits;
- logging to Trackio (lab/lab.yaml: project and the shared Hugging Face Space);
- pushing the final (or a chosen) checkpoint: model files at the repo root, the checkpoint for resuming under
  checkpoints/<step>/ (tagged).

Everything else (progress bar with time left, held-out eval loss, logging, mixed precision, multi-GPU) is 0.7's.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import shlex
import signal
import sys
from pathlib import Path
from typing import Any

from drtc.registry import Registry, RegistryError, Resolved

OUTPUTS = Path("outputs/train")
# Stop requests live outside the run folder: LeRobot 0.7 creates outputs/train/<model> only at the first checkpoint.
STOPS = OUTPUTS / ".stop"


def stop_file(model: str) -> Path:
    return STOPS / model


class StopTraining(Exception):
    pass


# --- episodes ------------------------------------------------------------------------------------------------------


def split_episodes(split: dict, count: int | None = None) -> tuple[list[int], list[int]]:
    """Seeded random split -> (training episodes, held-out episodes); identical to the old lab/tools/train.py.

    `count` takes the first `count` of the training episodes' fixed shuffle, so subsets are nested (10 within 25
    within 50) and every model sees the same held-out episodes.
    """
    total, held, seed = split["total"], split["held_out"], split.get("seed", 0)
    order = random.Random(seed).sample(range(total), total)
    held_out, pool = order[:held], order[held:]
    if count is not None:
        if count > len(pool):
            raise RegistryError(f"episodes: {count} > the {len(pool)} training episodes of the split")
        pool = pool[:count]
    return pool, held_out


def parse_episodes(spec) -> list[int]:
    if isinstance(spec, list):
        return [int(e) for e in spec]
    out: list[int] = []
    for part in str(spec).replace(" ", "").split(","):
        first, _, last = part.partition("-")
        out += range(int(first), int(last or first) + 1)
    return out


def select_episodes(train: dict) -> tuple[list[int] | None, list[int]]:
    if train.get("split"):
        count = train.get("episodes")
        if count is not None and not isinstance(count, int):
            raise RegistryError("with a split, train.episodes must be a number (a subset of the training episodes)")
        return split_episodes(train["split"], count)
    if train.get("episodes") is not None:
        return parse_episodes(train["episodes"]), []
    return None, []


# --- dataset pin ---------------------------------------------------------------------------------------------------


def check_local_dataset(r: Resolved) -> None:
    """A local copy under the dataset's name must be the registered version; LeRobot would use it as it is."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    info = HF_LEROBOT_HOME / r.dataset.repo / "meta" / "info.json"
    if not info.is_file():
        return
    meta = json.loads(info.read_text())
    got = (meta["total_episodes"], meta["total_frames"])
    want = (r.dataset.episodes, r.dataset.frames)
    if any(w is not None and w != g for w, g in zip(want, got)):
        raise RegistryError(
            f"local copy {info.parent.parent} has {got[0]} episodes / {got[1]} frames, but {r.dataset.id} is pinned "
            f"to {want[0]} / {want[1]} (revision {r.dataset.revision}). It is another version under the same name; "
            f"move it aside or set HF_LEROBOT_HOME to another folder.")


def training_frames(r: Resolved, episodes: list[int] | None) -> int:
    from lerobot.datasets import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(r.dataset.repo, revision=r.dataset.revision)
    lengths = dict(zip(meta.episodes["episode_index"], meta.episodes["length"]))
    return sum(lengths[e] for e in episodes) if episodes is not None else meta.total_frames


# --- flags ---------------------------------------------------------------------------------------------------------


def flag_value(v: Any) -> str:
    if v is None:
        return "null"  # draccus reads "null" as None; "None" would arrive as the string "None"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def resolve_base(base: str) -> str:
    """repo or repo@revision -> what --policy.path loads (a pinned revision is downloaded and passed as a folder).

    A base whose config.json has fields the LeRobot 0.7 policy class no longer has (e.g. MolmoAct 2's Hub
    conversion, written for an earlier MolmoAct2Config) is passed as a local copy: the files linked, the config
    upgraded by drtc.compat.configs (fields only dropped, each drop printed). The Hub snapshot is not modified.
    """
    repo, _, rev = base.partition("@")
    if not rev:
        return repo
    from huggingface_hub import snapshot_download

    return str(upgraded_base(Path(snapshot_download(repo, revision=rev)), f"{repo.replace('/', '--')}@{rev}"))


def upgraded_base(snapshot: Path, name: str) -> Path:
    from lerobot.configs import PreTrainedConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    from drtc.compat.configs import upgrade_config
    from drtc.compat.serve import CACHE

    raw = json.loads((snapshot / "config.json").read_text())
    register_third_party_plugins()
    if raw.get("type") not in PreTrainedConfig.get_known_choices():
        return snapshot
    upgraded = upgrade_config(raw, PreTrainedConfig.get_choice_class(raw["type"]))
    if upgraded == raw:
        return snapshot
    out = CACHE.parent / "compat_bases" / name
    out.mkdir(parents=True, exist_ok=True)
    for f in snapshot.rglob("*"):
        rel = f.relative_to(snapshot)
        dst = out / rel
        if f.is_dir() or rel == Path("config.json"):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(f.resolve())
    (out / "config.json").write_text(json.dumps(upgraded, indent=2))
    print(f"base {name}: config upgraded for LeRobot 0.7, loading the local copy {out}", flush=True)
    return out


def unknown_policy_flags(policy: str, flags: dict) -> list[str]:
    """policy.* flags the policy's config class does not have (would be rejected or, worse, mean nothing)."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    if policy not in PreTrainedConfig.get_known_choices():
        raise RegistryError(f"policy {policy!r} is not available in this setup (LeRobot 0.7 + our plugins); "
                            f"it needs a plugin port first")
    cls = PreTrainedConfig.get_choice_class(policy)
    known = {f.name for f in dataclasses.fields(cls)}
    special = {"path", "type"}  # draccus arguments (load from / choose the policy), not config fields
    return [k for k in flags if k.startswith("policy.") and k.split(".")[1] not in known | special]


def build(reg: Registry, model_id: str, steps: int | None = None, download: bool = True) -> tuple[dict, dict]:
    """-> (lerobot-train flags, info for the summary)."""
    r = reg.resolve(model_id)
    if not r.trainable:
        raise RegistryError(f"{model_id!r} has no recipe/dataset: it cannot be trained here")
    if r.status == "not_trainable":
        raise RegistryError(f"{model_id!r} is marked not_trainable ({reg.models[model_id].notes or 'see the registry'})")
    if r.finetune == "lora" and r.lora is not None:
        # Adapter-style LoRA (a `lora:` block, our old format). Policy-native LoRA (e.g. MolmoAct 2's
        # policy.train_mode_vlm: lora) is a recipe flag and trains as is.
        raise RegistryError(f"{model_id!r}: adapter LoRA recipes are not mapped to LeRobot 0.7's --peft.* yet")
    t = dict(r.train)
    if "epochs" in t and "steps" in t:
        raise RegistryError(f"{model_id!r}: set train.epochs or train.steps, not both")
    episodes, held_out = select_episodes(t)
    out = OUTPUTS / model_id
    flags: dict[str, Any] = {
        "dataset.repo_id": r.dataset.repo,
        "dataset.revision": r.dataset.revision,
        "policy.repo_id": r.hub,
        "policy.push_to_hub": False,  # pushed by us (final or chosen checkpoint), see push()
        "policy.device": "cuda",
        "output_dir": str(out),
        "job_name": model_id,
        "batch_size": t.get("batch_size", 8),
        "save_freq": t.get("save_freq", 5_000),
        "trackio.enable": True,
        "trackio.project": reg.settings.trackio_project,
    }
    if reg.settings.trackio_space:
        flags["trackio.space_id"] = reg.settings.trackio_space
        flags["trackio.private"] = reg.settings.trackio_private
    if r.base:
        flags["policy.path"] = resolve_base(r.base) if download else r.base
    else:
        flags["policy.type"] = r.policy
    if episodes is not None:
        # Training episodes first, held-out last: 0.7 holds out the last ceil(n * eval_split) episodes.
        flags["dataset.episodes"] = episodes + held_out
    if held_out:
        n = len(episodes) + len(held_out)
        flags["dataset.eval_split"] = (len(held_out) - 0.5) / n  # ceil(n * x) == len(held_out), robust to rounding
    flags.update({k: v for k, v in r.flags.items()})
    # What the recipe needs beyond the recorded dataset (relative-action stats, a frame cache): one prepared
    # variant folder.
    from drtc.data import prepare

    needs = prepare.needs(flags, r.policy, r.recipe.data if r.recipe else None)
    variants = [n.tag for n in needs]
    if needs:
        root = prepare.ensure(r.dataset.repo, r.dataset.revision, needs) if download else \
            prepare.variant_root(r.dataset.repo, r.dataset.revision, needs)
        flags["dataset.root"] = str(root)
    frames = None
    if steps is not None:
        flags["steps"] = steps
    elif "epochs" in t:
        frames = training_frames(r, episodes)
        flags["steps"] = max(1, math.ceil(t["epochs"] * frames / flags["batch_size"]))
    else:
        flags["steps"] = t.get("steps", 20_000)
    if held_out:
        flags["eval_steps"] = max(50, flags["steps"] // 20)  # ~20 held-out evaluations per run, as before
    info = {"model": model_id, "recipe": r.recipe.id, "dataset": f"{r.dataset.id}@{r.dataset.revision[:10]}",
            "episodes": len(episodes) if episodes is not None else "all", "held_out": held_out,
            "frames": frames, "epochs": t.get("epochs"), "unknown_flags": unknown_policy_flags(r.policy, flags),
            "variants": variants,
            "resolved": r}
    return flags, info


def command(flags: dict) -> list[str]:
    return ["lerobot-train", *(f"--{k}={flag_value(v)}" for k, v in flags.items())]


# --- clean stop and pushing ----------------------------------------------------------------------------------------


def push(checkpoint_dir: Path, repo_id: str) -> None:
    """Checkpoint (with training state, for resuming) under checkpoints/<step>/, model files at the repo root."""
    from huggingface_hub import HfApi
    from lerobot.common.train_utils import push_checkpoint_to_hub

    checkpoint_dir = checkpoint_dir.resolve()
    push_checkpoint_to_hub(checkpoint_dir, repo_id)
    HfApi().upload_folder(folder_path=str(checkpoint_dir / "pretrained_model"), repo_id=repo_id, repo_type="model",
                          commit_message=f"model from checkpoint {checkpoint_dir.name}")
    print(f"pushed checkpoint {checkpoint_dir.name} to https://huggingface.co/{repo_id}", flush=True)


def prune_training_state(latest: Path) -> list[Path]:
    """Keep every checkpoint's weights (to choose the best by held-out action error), but the training state
    (optimizer, scheduler, RNG: the bulk of a large model's checkpoint) only for the latest one, which resumes.
    LeRobot 0.7 keeps everything; for π0.5 that is tens of GB per checkpoint."""
    import shutil

    from lerobot.utils.constants import TRAINING_STATE_DIR

    removed = []
    latest = latest.resolve()
    for ckpt in latest.parent.iterdir():
        if ckpt.is_symlink() or not ckpt.is_dir() or ckpt.resolve() == latest:
            continue
        ts = ckpt / TRAINING_STATE_DIR
        if ts.is_dir():
            shutil.rmtree(ts)
            removed.append(ts)
    return removed


def install_clean_stop(lerobot_train, model: str) -> dict:
    """At the next step after a stop request: save a checkpoint, then stop (the caller pushes it)."""
    state = {"requested": False, "saved": None}
    stop = stop_file(model)
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.unlink(missing_ok=True)  # an old request must not stop this run

    def request(*_):
        if state["requested"]:
            print("\nsecond stop request: exiting without saving", flush=True)
            raise KeyboardInterrupt
        state["requested"] = True
        print("\nstop requested: saving a checkpoint at the next step (Ctrl-C again to quit without saving)",
              flush=True)

    signal.signal(signal.SIGINT, request)
    signal.signal(signal.SIGTERM, request)
    should_save = lerobot_train.should_save_checkpoint
    update_last = lerobot_train.update_last_checkpoint

    def should_save_checkpoint(step, save_freq, total_steps):
        if stop.exists() and not state["requested"]:
            stop.unlink(missing_ok=True)
            request()
        return state["requested"] or should_save(step, save_freq, total_steps)

    def update_last_checkpoint(checkpoint_dir):
        update_last(checkpoint_dir)
        prune_training_state(Path(checkpoint_dir))
        if state["requested"]:
            state["saved"] = Path(checkpoint_dir)
            raise StopTraining

    lerobot_train.should_save_checkpoint = should_save_checkpoint
    lerobot_train.update_last_checkpoint = update_last_checkpoint
    return state


# --- CLI -----------------------------------------------------------------------------------------------------------


def print_summary(flags: dict, info: dict) -> None:
    print(" \\\n    ".join(shlex.quote(p) for p in command(flags)), flush=True)
    r: Resolved = info["resolved"]
    print(f"\nmodel {info['model']}: recipe {info['recipe']}, dataset {info['dataset']}, policy {r.policy} "
          f"({r.finetune}), push to {r.hub}")
    if info["variants"]:
        print(f"dataset prepared for this recipe: {', '.join(info['variants'])} (--dataset.root)")
    if info["held_out"]:
        print(f"episodes: {info['episodes']} for training, held out: {info['held_out']}")
    if info["frames"]:
        print(f"epochs {info['epochs']} x {info['frames']} frames / batch {flags['batch_size']} = {flags['steps']} steps")
    if info["unknown_flags"]:
        print(f"WARNING: {r.policy} has no config field for: {', '.join(info['unknown_flags'])}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab train", description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--dry-run", action="store_true", help="print the lerobot-train command and exit")
    ap.add_argument("--steps", type=int, help="override the run length (e.g. a short test run)")
    ap.add_argument("--no-push", action="store_true", help="keep the model local")
    ap.add_argument("--resume", action="store_true", help="continue outputs/train/<model> (or the Hub checkpoint)")
    ap.add_argument("--push", nargs="?", const="last", metavar="CHECKPOINT", help="only push a saved checkpoint")
    args, extra = ap.parse_known_args(argv)
    reg = Registry()
    out = OUTPUTS / args.model
    try:
        r = reg.resolve(args.model)
    except RegistryError as e:
        print(f"lab train: {e}", file=sys.stderr)
        return 1

    if args.push:
        push(out / "checkpoints" / args.push, r.hub)
        return 0

    from lerobot.scripts import lerobot_train

    if args.resume:
        local = out / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        cfg_path = str(local) if local.is_file() else r.hub
        flags = {"config_path": cfg_path, "resume": True}
    else:
        try:
            check_local_dataset(r)
            flags, info = build(reg, args.model, steps=args.steps, download=not args.dry_run)
        except RegistryError as e:
            print(f"lab train: {e}", file=sys.stderr)
            return 1
        print_summary(flags, info)
        if info["unknown_flags"] and not args.dry_run:
            print("lab train: fix the recipe's flags first (or remove them)", file=sys.stderr)
            return 1
        if args.dry_run:
            return 0
        if out.exists():
            print(f"lab train: {out} exists; continue it with --resume", file=sys.stderr)
            return 1
    cmd = command(flags) + extra
    from drtc.compat import frame_cache, molmoact2

    molmoact2.install()  # LoRA fine-tuning from a full-weight MolmoAct 2 checkpoint
    frame_cache.install()  # cached frames for variants that have them (recipe data.frames)
    state = install_clean_stop(lerobot_train, args.model)
    sys.argv = cmd
    try:
        lerobot_train.main()
    except StopTraining:
        print(f"\nstopped at checkpoint {state['saved'].name}", flush=True)
    last = out / "checkpoints" / "last"
    if not args.no_push and last.exists():
        push(last.resolve(), r.hub)
    if state["requested"]:
        print(f"continue (here or on another machine): lab train {args.model} --resume", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
