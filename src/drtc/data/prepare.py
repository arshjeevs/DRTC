"""Dataset preparation that recipes need: `lab prepare <dataset>`, and automatically in `lab train`.

A recipe can need more than the recorded dataset. Today that is relative-action statistics: π0.5 and GR00T N1.7
train with `policy.use_relative_actions` (arm joints relative to the current state, the gripper absolute), and their
normalizers need action stats computed in that space, over chunks of the policy's `chunk_size`, with the same joints
excluded. LeRobot computes them with `lerobot-edit-dataset --operation.type recompute_stats --operation.relative_action
true`; here the same function (`compute_relative_action_stats`) runs on the pinned dataset, read-only.

Each need becomes a variant folder next to LeRobot's cache:
    $HF_LEROBOT_HOME/.drtc_variants/<repo>/<revision[:10]>_rel_c<chunk>_x<excluded>/
holding symlinks to every file of the pinned dataset and its own meta/stats.json (the dataset's stats with the action
entry replaced). The dataset itself is never written. `lab train` passes the variant as --dataset.root, so a new
dataset needs no manual step: the variant is made on first use on any host (or ahead of time with `lab prepare`,
which `lab record` runs after pushing).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from drtc.registry import Registry, RegistryError

MARKER = "drtc_variant.json"


@dataclass(frozen=True)
class RelativeStats:
    chunk_size: int
    exclude: tuple[str, ...]

    @property
    def tag(self) -> str:
        return f"rel_c{self.chunk_size}_x{'-'.join(self.exclude) or 'none'}"


def needs(flags: dict, policy: str | None = None) -> list[RelativeStats]:
    """What a recipe's (resolved) flags need from its dataset."""
    if not flags.get("policy.use_relative_actions"):
        return []
    chunk = flags.get("policy.chunk_size")
    if chunk is None:
        raise RegistryError(f"{policy or 'recipe'}: policy.use_relative_actions needs an explicit policy.chunk_size "
                            f"(the relative-action stats are computed over chunks of that length)")
    exclude = flags.get("policy.relative_exclude_joints")
    if exclude is None:
        raise RegistryError(f"{policy or 'recipe'}: policy.use_relative_actions needs an explicit "
                            f"policy.relative_exclude_joints (e.g. [gripper]); the policies' defaults differ")
    return [RelativeStats(int(chunk), tuple(exclude))]


def variant_root(repo: str, revision: str | None, need: RelativeStats) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / ".drtc_variants" / repo / f"{(revision or 'local')[:10]}_{need.tag}"


def ensure(repo: str, revision: str | None, need: RelativeStats) -> Path:
    """The variant folder, made if missing (downloads the pinned dataset like training would)."""
    root = variant_root(repo, revision, need)
    marker = root / MARKER
    want = {"repo": repo, "revision": revision, "relative_action": True, "chunk_size": need.chunk_size,
            "exclude_joints": list(need.exclude)}
    if marker.is_file() and json.loads(marker.read_text()) == want:
        return root
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.compute_stats import compute_relative_action_stats
    from lerobot.utils.constants import ACTION

    ds = LeRobotDataset(repo, revision=revision)
    src = Path(ds.root).resolve()
    print(f"prepare {repo}@{(revision or 'local')[:10]}: relative-action stats, chunk {need.chunk_size}, "
          f"absolute: {', '.join(need.exclude) or 'none'} -> {root}", flush=True)
    stats = compute_relative_action_stats(hf_dataset=ds.hf_dataset, features=ds.meta.features,
                                          chunk_size=need.chunk_size, exclude_joints=list(need.exclude),
                                          num_workers=4)
    merged = dict(ds.meta.stats)
    merged[ACTION] = stats
    root.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        if f.is_dir() or rel == Path("meta/stats.json") or rel.parts[0] == ".cache":
            continue
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(f.resolve())
    from lerobot.datasets.io_utils import write_stats

    stats_file = root / "meta" / "stats.json"
    if stats_file.is_symlink():
        stats_file.unlink()  # never write through a link into the dataset
    write_stats(merged, root)
    marker.write_text(json.dumps(want, indent=1))
    return root


def recipes_in_use(reg: Registry) -> list[str]:
    """Recipes of every model that is not retired: what a new dataset should be ready for."""
    return sorted({m.recipe for m in reg.models.values() if m.recipe and m.status not in ("retired", "external")})


def prepare(reg: Registry, dataset_id: str, recipe_ids: list[str], dry_run: bool = False) -> list[Path]:
    d = reg.datasets[dataset_id]
    done: dict[RelativeStats, Path] = {}
    for rid in recipe_ids:
        recipe = reg.recipes[rid]
        for need in needs(recipe.flags, recipe.policy):
            if need not in done:
                done[need] = variant_root(d.repo, d.revision, need) if dry_run else ensure(d.repo, d.revision, need)
            print(f"  {rid}: {need.tag} -> {done[need]}")
    if not done:
        print(f"{dataset_id}: nothing to prepare for {', '.join(recipe_ids)}")
    return list(done.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab prepare", description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", help="registry dataset id")
    ap.add_argument("--recipes", nargs="*", help="default: the recipes of every model that is not retired")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be prepared")
    a = ap.parse_args(argv)
    reg = Registry()
    try:
        if a.dataset not in reg.datasets:
            raise RegistryError(f"unknown dataset {a.dataset!r}")
        prepare(reg, a.dataset, a.recipes or recipes_in_use(reg), a.dry_run)
    except RegistryError as e:
        print(f"lab prepare: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
