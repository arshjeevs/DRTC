"""Dataset preparation that recipes need: `lab prepare <dataset>`, and automatically in `lab train`.

A recipe can need more than the recorded dataset:
- relative-action statistics: π0.5 and GR00T N1.7 train with `policy.use_relative_actions` (arm joints relative to
  the current state, the gripper absolute), and their normalizers need action stats computed in that space, over
  chunks of the policy's `chunk_size`, with the same joints excluded. LeRobot computes them with
  `lerobot-edit-dataset --operation.type recompute_stats --operation.relative_action true`; here the same function
  (`compute_relative_action_stats`) runs on the pinned dataset, read-only.
- a frame cache (recipe `data: {frames: [H, W]}`, a speed setting): every camera frame decoded once and resized to
  H×W with the policy's own first image step, stored as uint8 arrays. Training then reads frames instead of decoding
  AV1 video, moves H×W instead of full-size frames between processes, and the policy's own resize has nothing left
  to do (see drtc.compat.frame_cache). Per policy (FRAME_CACHE_METHODS):
  - Diffusion (`resize`): its first image step is `torchvision.transforms.Resize(resize_shape)` on float images,
    which leaves an image of that size unchanged; the model sees the same input up to the uint8 rounding of the
    resized frame (at most 0.5/255 per pixel).
  - GR00T N1.5 (`eagle`): its Eagle processor (one tile, no thumbnail) resizes the uint8 frame to 224×224 with
    torchvision's bicubic resize; the cache stores exactly that, and the processor leaves a 224×224 frame unchanged,
    so the model input is bit-identical (tests/bench/check_frame_cache.py checks it).
  Serving is unchanged: the checkpoint still declares the full-size cameras.

All needs of a recipe become one variant folder next to LeRobot's cache:
    $HF_LEROBOT_HOME/.drtc_variants/<repo>/<revision[:10]>_<tag>[_<tag>...]/
holding symlinks to every file of the pinned dataset, its own meta/stats.json if the stats change, and
drtc_frames/ if frames are cached. The dataset itself is never written. `lab train` passes the variant as
--dataset.root, so a new dataset needs no manual step: the variant is made on first use on any host (or ahead of
time with `lab prepare`, which `lab record` runs after pushing).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from drtc.registry import Registry, RegistryError

MARKER = "drtc_variant.json"
FRAMES_DIR = "drtc_frames"
# Policy -> how its first image step resizes (see the module doc); only these can use a frame cache.
FRAME_CACHE_METHODS = {"diffusion": "resize", "groot_n15": "eagle"}
EAGLE_TILE = (224, 224)  # the N1.5 plugin's Eagle processor: one 224x224 tile per camera


@dataclass(frozen=True)
class RelativeStats:
    chunk_size: int
    exclude: tuple[str, ...]

    @property
    def tag(self) -> str:
        return f"rel_c{self.chunk_size}_x{'-'.join(self.exclude) or 'none'}"


@dataclass(frozen=True)
class FrameCache:
    height: int
    width: int
    method: str = "resize"

    @property
    def tag(self) -> str:
        return f"frames{self.height}x{self.width}" + ("" if self.method == "resize" else f"_{self.method}")


def needs(flags: dict, policy: str | None = None, data: dict | None = None) -> list:
    """What a recipe's (resolved) flags and `data` settings need from its dataset, in a fixed order."""
    out: list = []
    if flags.get("policy.use_relative_actions"):
        chunk = flags.get("policy.chunk_size")
        if chunk is None:
            raise RegistryError(f"{policy or 'recipe'}: policy.use_relative_actions needs an explicit "
                                f"policy.chunk_size (the relative-action stats are computed over chunks of that length)")
        exclude = flags.get("policy.relative_exclude_joints")
        if exclude is None:
            raise RegistryError(f"{policy or 'recipe'}: policy.use_relative_actions needs an explicit "
                                f"policy.relative_exclude_joints (e.g. [gripper]); the policies' defaults differ")
        out.append(RelativeStats(int(chunk), tuple(exclude)))
    frames = (data or {}).get("frames")
    if frames is not None:
        h, w = (int(x) for x in frames)
        method = FRAME_CACHE_METHODS.get(policy or "")
        if method is None:
            raise RegistryError(f"{policy or 'recipe'}: data.frames is only valid for {sorted(FRAME_CACHE_METHODS)} "
                                f"(the cache must match the policy's own first image step)")
        if method == "resize" and [int(x) for x in flags.get("policy.resize_shape") or []] != [h, w]:
            raise RegistryError(f"{policy}: data.frames {[h, w]} must equal policy.resize_shape "
                                f"{flags.get('policy.resize_shape')} (the model's resize then leaves the frame unchanged)")
        if method == "eagle" and (h, w) != EAGLE_TILE:
            raise RegistryError(f"{policy}: data.frames must be {list(EAGLE_TILE)}, the Eagle processor's tile")
        if flags.get("dataset.image_transforms.enable"):
            raise RegistryError(f"{policy}: data.frames with dataset.image_transforms (they would run on the cached size)")
        out.append(FrameCache(h, w, method))
    return out


def variant_root(repo: str, revision: str | None, need) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    need = list(need) if isinstance(need, (list, tuple)) else [need]
    return HF_LEROBOT_HOME / ".drtc_variants" / repo / f"{(revision or 'local')[:10]}_{'_'.join(n.tag for n in need)}"


def _want(repo: str, revision: str | None, need: list) -> dict:
    want: dict = {"repo": repo, "revision": revision}
    for n in need:
        if isinstance(n, RelativeStats):  # the keys of the first version, so existing variants stay valid
            want.update({"relative_action": True, "chunk_size": n.chunk_size, "exclude_joints": list(n.exclude)})
        else:
            d = asdict(n)
            if d.get("method") == "resize":
                del d["method"]  # the default; caches made before methods existed stay valid
            if isinstance(n, FrameCache):
                d["decoder"] = video_backend()  # frames as this host's training decodes them
            want.setdefault("extra", []).append({"type": type(n).__name__, **d})
    return want


def video_backend() -> str:
    """The decoder lerobot-train uses on this host (dataset.video_backend's default): torchcodec, or pyav where
    torchcodec cannot load FFmpeg. torchcodec and pyav can differ slightly, so a cache is made with this one."""
    from lerobot.utils.import_utils import get_safe_default_video_backend

    return get_safe_default_video_backend()


def ensure(repo: str, revision: str | None, need) -> Path:
    """The variant folder for `need` (one need or a list), made if missing (downloads the pinned dataset like
    training would)."""
    need = list(need) if isinstance(need, (list, tuple)) else [need]
    root = variant_root(repo, revision, need)
    marker = root / MARKER
    want = _want(repo, revision, need)
    if marker.is_file() and json.loads(marker.read_text()) == want:
        return root
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset(repo, revision=revision)
    src = Path(ds.root).resolve()
    rel = next((n for n in need if isinstance(n, RelativeStats)), None)
    frames = next((n for n in need if isinstance(n, FrameCache)), None)
    print(f"prepare {repo}@{(revision or 'local')[:10]}: {', '.join(n.tag for n in need)} -> {root}", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    for f in src.rglob("*"):
        r = f.relative_to(src)
        if f.is_dir() or r.parts[0] in (".cache", FRAMES_DIR) or (rel and r == Path("meta/stats.json")):
            continue
        dst = root / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(f.resolve())
    if rel is not None:
        from lerobot.datasets.compute_stats import compute_relative_action_stats
        from lerobot.datasets.io_utils import write_stats
        from lerobot.utils.constants import ACTION

        stats = compute_relative_action_stats(hf_dataset=ds.hf_dataset, features=ds.meta.features,
                                              chunk_size=rel.chunk_size, exclude_joints=list(rel.exclude),
                                              num_workers=4)
        merged = dict(ds.meta.stats)
        merged[ACTION] = stats
        stats_file = root / "meta" / "stats.json"
        if stats_file.is_symlink():
            stats_file.unlink()  # never write through a link into the dataset
        write_stats(merged, root)
    if frames is not None:
        build_frames(ds, root / FRAMES_DIR, frames)
    marker.write_text(json.dumps(want, indent=1))
    return root


def build_frames(ds, out: Path, need: FrameCache, batch: int = 64) -> None:
    """drtc_frames/: one uint8 array (frames, 3, H, W) per camera, row = the dataset's frame `index`, decoded with
    LeRobot's own decoder and this host's backend (as training reads them), resized as the policy does."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    import torchvision
    from lerobot.datasets.utils import shift_timestamps
    from lerobot.datasets.video_utils import decode_video_frames

    meta = ds.meta
    if meta.depth_keys:
        raise RegistryError(f"{meta.repo_id}: frame cache for depth cameras is not supported")
    cols = ds.hf_dataset.select_columns(["index", "episode_index", "frame_index", "timestamp"]).with_format("numpy")[:]
    index, ep_col, fidx, ts_col = (np.asarray(cols[k]).reshape(-1)
                                   for k in ("index", "episode_index", "frame_index", "timestamp"))
    n = int(meta.total_frames)
    if sorted(index.tolist()) != list(range(n)):
        raise RegistryError(f"{meta.repo_id}: frame indices are not 0..{n - 1}; cannot cache frames by index")
    fps = float(meta.fps)
    if np.abs(ts_col * fps - fidx).max() > 1e-3:
        raise RegistryError(f"{meta.repo_id}: timestamps are not frame_index / fps; the cache lookup relies on it")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if need.method == "resize":  # Diffusion: torchvision Resize on float images in [0, 1], rounded back to uint8
        torch_resize = torchvision.transforms.Resize([need.height, need.width])

        def resize(raw):
            small = torch_resize(raw.to(torch.float32) / 255.0)
            return (small * 255.0).round().clamp(0, 255).to(torch.uint8)
    elif need.method == "eagle":  # GR00T N1.5: the Eagle fast processor's one-tile resize, uint8 in and out
        from torchvision.transforms.v2 import functional as tvf

        def resize(raw):
            return tvf.resize(raw, [need.height, need.width], interpolation=tvf.InterpolationMode.BICUBIC,
                              antialias=True)
    else:
        raise RegistryError(f"unknown frame cache method {need.method!r}")
    tolerance = float(getattr(ds, "tolerance_s", 1e-4))
    backend = video_backend()
    print(f"  frames: decoding with {backend} (as training on this host does)", flush=True)
    arrays = {k: np.lib.format.open_memmap(tmp / f"{k}.npy", mode="w+", dtype=np.uint8,
                                           shape=(n, 3, need.height, need.width)) for k in meta.video_keys}

    # Work per video file: LeRobot keeps one decoder per file (episodes share files), and a decoder must not be
    # used by two threads at once. Each thread decodes one file's episodes in order.
    episodes = sorted(set(ep_col.tolist()))
    per_file: dict[tuple[str, Path], list[int]] = {}
    for ep_idx in episodes:
        for key in meta.video_keys:
            per_file.setdefault((key, ds.root / meta.get_video_file_path(ep_idx, key)), []).append(ep_idx)

    def one_file(item: tuple[tuple[str, Path], list[int]]) -> int:
        (key, path), eps = item
        for ep_idx in eps:
            ep = meta.episodes[ep_idx]
            rows = np.nonzero(ep_col == ep_idx)[0]
            rows = rows[np.argsort(fidx[rows])]
            for s in range(0, len(rows), batch):
                part = rows[s:s + batch]
                ts = shift_timestamps(ts_col[part].tolist(), ep[f"videos/{key}/from_timestamp"])
                raw = decode_video_frames(path, ts, tolerance, backend, return_uint8=True)
                arrays[key][index[part]] = resize(raw).numpy()
        return len(eps)

    done = 0
    with ThreadPoolExecutor(max_workers=min(8, len(per_file))) as pool:
        for k in pool.map(one_file, per_file.items()):
            done += k
            print(f"  frames: {done}/{len(episodes) * len(meta.video_keys)} episode videos", flush=True)
    for a in arrays.values():
        a.flush()
    del arrays
    (tmp / "meta.json").write_text(json.dumps({"height": need.height, "width": need.width, "frames": n,
                                               "fps": fps, "keys": list(meta.video_keys),
                                               "method": need.method, "decoder": backend},
                                              indent=1))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    # Close the decoders opened here: DataLoader workers forked later (lab train builds the cache in its own process)
    # must not inherit them ("Could not push packet to decoder").
    from lerobot.datasets import video_utils

    video_utils._default_decoder_cache.clear()


def recipes_in_use(reg: Registry) -> list[str]:
    """Recipes of every model that is not retired: what a new dataset should be ready for."""
    return sorted({m.recipe for m in reg.models.values() if m.recipe and m.status not in ("retired", "external")})


def prepare(reg: Registry, dataset_id: str, recipe_ids: list[str], dry_run: bool = False) -> list[Path]:
    d = reg.datasets[dataset_id]
    done: dict[tuple, Path] = {}
    for rid in recipe_ids:
        recipe = reg.recipes[rid]
        need = tuple(needs(recipe.flags, recipe.policy, recipe.data))
        if not need:
            continue
        if need not in done:
            done[need] = variant_root(d.repo, d.revision, list(need)) if dry_run else \
                ensure(d.repo, d.revision, list(need))
        print(f"  {rid}: {', '.join(n.tag for n in need)} -> {done[need]}")
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
