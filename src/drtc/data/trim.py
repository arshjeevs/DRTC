"""Automatic trimming of the idle start and end of every episode (LeRobot v3.0 datasets).

    lab trim <raw dataset dir> <output dir> [--dry-run] [--move F] [--pad S] ...

Teleoperated episodes begin and end with the arm at rest. Those frames teach a policy that "do nothing" is a likely
action in the start pose. Every recording is trimmed before it is used: the untrimmed recording stays as
<name>_raw, the trimmed copy is <name> (lab/lab.yaml).

Detection: the algorithm of the minor's lab/tools/autotrim.py, unchanged, so it finds exactly the same crops.
Nothing in it is specific to one robot or dataset:
- joints are compared as a fraction of their own range (meta/stats.json), so units, joint count and gripper scale
  do not matter;
- the noise level that counts as "still" is measured from the dataset itself;
- padding and minimum lengths are in seconds, so any fps works;
- the START is found from `action` (the follower may still be catching up in the first frames), the END from
  `action` and `observation.state`, whichever settles later.

Applying the crops writes a NEW dataset folder; the raw one is never modified. In v3.0 an episode's frames are a
time window into a shared mp4, so trimming k frames off the front is `from_timestamp += k / fps` and k fewer data
rows: no re-encoding, frames stay pixel-identical, and the mp4 files are hardlinked from the raw dataset (no extra
disk). Numeric statistics are recomputed with LeRobot's own functions; image statistics are carried over per
episode (the frames did not change) and re-aggregated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Params:
    move: float = 0.03          # fraction of a joint's range that counts as "the arm has moved"
    noise_mult: float = 2.0     # step above (noise_mult x measured noise floor) = still moving
    pad: float = 0.1            # seconds of rest kept before the onset and after the offset
    min_keep_s: float = 1.0     # crops keeping less than this many seconds are not applied
    min_keep_frac: float = 0.5  # crops keeping less than this fraction are applied but listed for a look
    pause_s: float = 2.0        # still stretches inside the kept range longer than this are reported


@dataclass
class EpisodeResult:
    episode: int
    length: int
    start: int                  # first kept frame
    end: int                    # one past the last kept frame
    head_s: float
    tail_s: float
    status: str                 # trim | none | static | too_short
    longest_pause_s: float
    note: str = ""


# --- detection (unchanged from the minor's lab/tools/autotrim.py) -------------------------------------------------


def joint_ranges(stats: dict, key: str, sample: list[np.ndarray]) -> np.ndarray:
    """Per-joint scale: max - min from meta/stats.json (else from the frames), never below 1/4 of the median."""
    st = stats.get(key, {})
    if "min" in st and "max" in st:
        lo, hi = np.asarray(st["min"], float).ravel(), np.asarray(st["max"], float).ravel()
    else:
        allx = np.concatenate(sample)
        lo, hi = allx.min(axis=0), allx.max(axis=0)
    rng = hi - lo
    median = float(np.median(rng[rng > 0])) if np.any(rng > 0) else 1.0
    return np.maximum(rng, 0.25 * median)


def steps(x: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.abs(np.diff(x, axis=0)).max(axis=1)] if len(x) > 1 else np.zeros(len(x))


def measure_noise(series: list[np.ndarray], move: float) -> float:
    vals = []
    for x in series:
        s = steps(x)
        for seg, st in ((x, s[1:]), (x[::-1], s[1:][::-1])):
            near = np.abs(seg - seg[0]).max(axis=1) < move / 3
            k = int(np.argmin(near)) if not near.all() else len(near)
            vals.append(st[:max(k - 1, 0)])
    v = np.concatenate(vals) if vals else np.array([])
    v = v[v > 0]
    return float(np.percentile(v, 95)) if len(v) else 0.0


def onset(x: np.ndarray, move: float, floor: float) -> int | None:
    dev = np.abs(x - x[0]).max(axis=1)
    over = np.flatnonzero(dev > move)
    if not len(over):
        return None
    i = int(over[0])
    while i > 0 and dev[i] > floor:
        i -= 1
    return i


def offset(x: np.ndarray, move: float, floor: float) -> int | None:
    i = onset(x[::-1], move, floor)
    return None if i is None else len(x) - 1 - i


def longest_pause(x: np.ndarray, floor: float) -> int:
    still = steps(x) <= floor
    best = run = 0
    for v in still[1:]:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def trim_episode(ep: int, act: np.ndarray, state: np.ndarray | None, fps: int, p: Params, floor: float) -> EpisodeResult:
    n = len(act)
    pad = int(round(p.pad * fps))
    first = onset(act, p.move, floor)
    lasts = [offset(act, p.move, floor)] + ([offset(state, p.move, floor)] if state is not None else [])
    lasts = [v for v in lasts if v is not None]
    if first is None or not lasts:
        return EpisodeResult(ep, n, 0, n, 0.0, 0.0, "static", 0.0, "no joint leaves its start/end pose")
    start = max(0, first - pad)
    end = min(n, max(lasts) + 1 + pad)
    if end <= start:
        return EpisodeResult(ep, n, 0, n, 0.0, 0.0, "static", 0.0, "motion too short to locate")
    kept = end - start
    pause = longest_pause(act[start:end], floor) / fps
    res = EpisodeResult(ep, n, start, end, round(start / fps, 2), round((n - end) / fps, 2),
                        "trim" if (start > 0 or end < n) else "none", round(pause, 2))
    if kept / fps < p.min_keep_s:
        res.status, res.note = "too_short", f"would keep {kept} of {n} frames ({kept / fps:.1f} s): check it by hand"
    elif kept / n < p.min_keep_frac:
        res.note = f"removes {1 - kept / n:.0%} of the episode: long rest, worth a look"
    elif pause >= p.pause_s:
        res.note = f"still for {pause:.1f} s inside the kept range"
    return res


# --- reading a dataset ---------------------------------------------------------------------------------------------


class RawDataset:
    """The parts of a LeRobot v3.0 dataset folder that trimming reads."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError(f"{root}: codebase_version {self.info.get('codebase_version')}; only v3.0 is supported")
        self.fps = int(self.info["fps"])
        self.features = self.info["features"]
        self.stats = json.loads((self.root / "meta/stats.json").read_text())
        self.episodes = pd.concat([pd.read_parquet(f) for f in sorted((self.root / "meta/episodes").rglob("*.parquet"))],
                                  ignore_index=True).sort_values("episode_index").reset_index(drop=True)
        self.camera_keys = [k for k, f in self.features.items() if f["dtype"] == "video"]

    def data_file(self, row) -> Path:
        return self.root / self.info["data_path"].format(chunk_index=int(row["data/chunk_index"]),
                                                         file_index=int(row["data/file_index"]))

    def series(self) -> dict[int, dict[str, np.ndarray]]:
        """Per episode: action and observation.state, in frame order."""
        out: dict[int, dict[str, np.ndarray]] = {}
        cols = ["episode_index", "frame_index", "action"] + (["observation.state"] if "observation.state" in self.features else [])
        for path in sorted({self.data_file(r) for _, r in self.episodes.iterrows()}):
            df = pd.read_parquet(path, columns=cols)
            for ep, d in df.groupby("episode_index"):
                d = d.sort_values("frame_index")
                out[int(ep)] = {k: np.stack(d[k].to_numpy()).astype(float) for k in cols[2:]}
        return out


def detect(ds: RawDataset, p: Params) -> tuple[list[EpisodeResult], float, float]:
    """-> (per-episode results, measured noise, still threshold)."""
    if "action" not in ds.features:
        raise ValueError(f"{ds.root}: no 'action' feature")
    has_state = "observation.state" in ds.features and ds.features["observation.state"]["shape"] == ds.features["action"]["shape"]
    raw = ds.series()
    episodes = [int(e) for e in ds.episodes["episode_index"]]
    r_act = joint_ranges(ds.stats, "action", [raw[e]["action"] for e in episodes])
    r_state = joint_ranges(ds.stats, "observation.state", [raw[e]["observation.state"] for e in episodes]) if has_state else None
    act = {e: raw[e]["action"] / r_act for e in episodes}
    state = {e: raw[e]["observation.state"] / r_state for e in episodes} if has_state else {}
    noise = measure_noise(list(act.values()) + list(state.values()), p.move)
    floor = max(noise * p.noise_mult, 1e-6)
    return [trim_episode(e, act[e], state.get(e), ds.fps, p, floor) for e in episodes], noise, floor


# --- applying the crops into a new dataset folder --------------------------------------------------------------------


def _link_tree(src: Path, dst: Path) -> None:
    """Hardlink every file of src into dst (copy when hardlinks are impossible, e.g. across filesystems)."""
    for path in src.rglob("*"):
        target = dst / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def _num(v) -> np.ndarray:
    """A stats cell from the episodes parquet (possibly nested object arrays, e.g. (3, 1, 1) image stats) as floats."""
    if isinstance(v, np.ndarray) and v.dtype == object:
        return np.stack([_num(x) for x in v])
    return np.asarray(v, dtype=float)


def _to_list(v):
    return v.tolist() if isinstance(v, np.ndarray) else v


QUANTILE_CONVENTIONS = ("lerobot", "weighted_mean")


def _weighted_mean_quantiles(stats: dict, per_episode: list[dict]) -> dict:
    """LeRobot 0.4.3's dataset quantiles: the frame-count-weighted mean of the episodes' quantiles (the minor's
    datasets, including so101_cubebox_v1, use this). LeRobot 0.7 keeps an envelope instead (min of the lower, max of
    the upper quantiles)."""
    for feat in stats:
        eps = [e[feat] for e in per_episode if feat in e]
        if not eps:
            continue
        w = np.array([np.asarray(e["count"], float).reshape(-1)[0] for e in eps])
        for q in [k for k in stats[feat] if k.startswith("q") and k[1:].isdigit()]:
            vals = np.stack([np.asarray(e[q], float) for e in eps])
            stats[feat][q] = (vals * w.reshape((-1,) + (1,) * (vals.ndim - 1))).sum(0) / w.sum()
    return stats


def apply(ds: RawDataset, results: list[EpisodeResult], dst: Path, quantiles: str = "lerobot") -> dict:
    """Write the trimmed dataset to dst (must not exist). Episodes keep their indices; only crops are applied.

    quantiles: how dataset-level quantiles (q01..q99, used by QUANTILES normalization, e.g. pi0.5) are combined
    from the episodes': "lerobot" = the installed LeRobot's aggregate_stats, "weighted_mean" = LeRobot 0.4.3's.
    """
    if quantiles not in QUANTILE_CONVENTIONS:
        raise ValueError(f"quantiles must be one of {QUANTILE_CONVENTIONS}")
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    crops = {r.episode: (r.start, r.end) if r.status == "trim" else (0, r.length) for r in results}
    fps = ds.fps
    numeric = {k: f for k, f in ds.features.items() if f["dtype"] not in ("video", "image", "string")}
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        raise FileExistsError(f"{tmp} exists (an interrupted earlier run?)")
    tmp.mkdir(parents=True)

    # videos: unchanged, hardlinked
    if (ds.root / "videos").is_dir():
        _link_tree(ds.root / "videos", tmp / "videos")

    # new global row numbering
    cursor, new_rows = 0, {}
    for ep in ds.episodes["episode_index"]:
        s, e = crops[int(ep)]
        new_rows[int(ep)] = (cursor, cursor + (e - s))
        cursor += e - s

    # data files: crop rows, renumber frame_index / timestamp / index; per-episode numeric stats
    ep_stats: dict[int, dict] = {}
    for path in sorted({ds.data_file(r) for _, r in ds.episodes.iterrows()}):
        df = pd.read_parquet(path)
        pieces = []
        for ep, d in df.groupby("episode_index", sort=True):
            ep = int(ep)
            s, e = crops[ep]
            d = d.sort_values("frame_index").iloc[s:e].copy()
            n = len(d)
            d["frame_index"] = np.arange(n, dtype=d["frame_index"].dtype)
            d["timestamp"] = (np.arange(n) / fps).astype(d["timestamp"].dtype)
            lo, hi = new_rows[ep]
            d["index"] = np.arange(lo, hi, dtype=d["index"].dtype)
            pieces.append(d)
            data = {k: (np.stack(d[k].to_numpy()) if numeric[k]["shape"] not in ([1], []) else d[k].to_numpy())
                    for k in numeric if k in d.columns}
            ep_stats[ep] = compute_episode_stats(data, {k: numeric[k] for k in data})
        out = pd.concat(pieces, ignore_index=True)
        target = tmp / path.relative_to(ds.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)

    # episode metadata
    meta = ds.episodes.copy()
    for i, row in meta.iterrows():
        ep = int(row["episode_index"])
        s, e = crops[ep]
        meta.at[i, "length"] = e - s
        meta.at[i, "dataset_from_index"], meta.at[i, "dataset_to_index"] = new_rows[ep]
        for cam in ds.camera_keys:
            fc, tc = f"videos/{cam}/from_timestamp", f"videos/{cam}/to_timestamp"
            if fc in meta.columns:
                new_from = float(row[fc]) + s / fps
                meta.at[i, fc] = new_from
                meta.at[i, tc] = new_from + (e - s) / fps
        for feat, st in ep_stats[ep].items():
            for name, value in st.items():
                col = f"stats/{feat}/{name}"
                if col in meta.columns:
                    meta.at[i, col] = _to_list(np.asarray(value))
    for (chunk, file_index), group in meta.groupby(["meta/episodes/chunk_index", "meta/episodes/file_index"]):
        target = tmp / f"meta/episodes/chunk-{int(chunk):03d}/file-{int(file_index):03d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        group.reset_index(drop=True).to_parquet(target, index=False)

    # aggregate stats: numeric recomputed, images carried over per episode
    per_episode = []
    for _, row in meta.iterrows():
        entry = {k: {s: np.asarray(v) for s, v in st.items()} for k, st in ep_stats[int(row["episode_index"])].items()}
        for cam in ds.camera_keys:
            cols = [c for c in meta.columns if c.startswith(f"stats/{cam}/")]
            if cols:
                entry[cam] = {c.split("/", 2)[2]: _num(row[c]) for c in cols}
                entry[cam] = {k: (v.reshape(3, 1, 1) if v.size == 3 else v.reshape(-1)) for k, v in entry[cam].items()}
        per_episode.append(entry)
    stats = aggregate_stats(per_episode)
    if quantiles == "weighted_mean":
        stats = _weighted_mean_quantiles(stats, per_episode)
    (tmp / "meta").mkdir(parents=True, exist_ok=True)
    (tmp / "meta/stats.json").write_text(json.dumps({k: {s: _to_list(np.asarray(v)) for s, v in st.items()}
                                                     for k, st in stats.items()}, indent=4))
    shutil.copy2(ds.root / "meta/tasks.parquet", tmp / "meta/tasks.parquet")
    info = dict(ds.info)
    info["total_frames"] = cursor
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=4))
    tmp.rename(dst)
    return {"episodes": len(results), "frames_before": int(ds.episodes["length"].sum()), "frames_after": cursor,
            "quantiles": quantiles}


# --- CLI -------------------------------------------------------------------------------------------------------------


def report(ds: RawDataset, results: list[EpisodeResult], noise: float, floor: float, p: Params, show: int = 15) -> None:
    fps = ds.fps
    total = sum(r.length for r in results)
    cut = sum(r.length - (r.end - r.start) for r in results if r.status == "trim")
    by = {s: [r for r in results if r.status == s] for s in ("trim", "none", "static", "too_short")}
    print(f"{ds.root.name}: {len(results)} episodes, {total} frames at {fps} fps")
    print(f"still = step <= {floor:.4f} of range (measured noise {noise:.4f} x {p.noise_mult}); moved = > {p.move:.0%}"
          f" of range; pad {p.pad} s")
    heads = np.array([r.head_s for r in by["trim"]] or [0.0])
    tails = np.array([r.tail_s for r in by["trim"]] or [0.0])
    print(f"trim {len(by['trim'])}, unchanged {len(by['none'])}, static {len(by['static'])}, too short "
          f"{len(by['too_short'])}; removes {cut} frames ({cut / max(total, 1):.1%}, {cut / fps:.0f} s)")
    print(f"cut from start: median {np.median(heads):.2f} s, max {heads.max():.2f} s;  from end: median "
          f"{np.median(tails):.2f} s, max {tails.max():.2f} s")
    shown = sorted(results, key=lambda r: -(r.head_s + r.tail_s))[: show or None]
    flagged = [r for r in results if r.status in ("static", "too_short") or r.note]
    for r in shown + [r for r in flagged if r not in shown]:
        print(f"  {r.episode:>4} {r.length:>6} {r.start:>5}-{r.end:<5} {r.head_s:>6.2f} {r.tail_s:>6.2f}  {r.status:<10} {r.note}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab trim", description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", type=Path, help="raw (untrimmed) LeRobot v3.0 dataset folder; never modified")
    ap.add_argument("out", type=Path, nargs="?", help="trimmed dataset folder to create")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--move", type=float, default=Params.move)
    ap.add_argument("--noise-mult", type=float, default=Params.noise_mult)
    ap.add_argument("--pad", type=float, default=Params.pad)
    ap.add_argument("--min-keep", type=float, default=Params.min_keep_s)
    ap.add_argument("--min-keep-frac", type=float, default=Params.min_keep_frac)
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--quantiles", choices=QUANTILE_CONVENTIONS, default="lerobot",
                    help="how dataset quantiles combine the episodes' (default and the lab's convention from "
                         "2026-11-18: the installed LeRobot's)")
    a = ap.parse_args(argv)
    p = Params(a.move, a.noise_mult, a.pad, a.min_keep, a.min_keep_frac)
    ds = RawDataset(a.raw)
    results, noise, floor = detect(ds, p)
    report(ds, results, noise, floor, p, a.show)
    if a.dry_run or a.out is None:
        return 0
    res = apply(ds, results, a.out, quantiles=a.quantiles)
    rep = {"raw": str(ds.root), "params": asdict(p), "noise": noise, "floor": floor, "fps": ds.fps,
           "episodes": [asdict(r) for r in results], **res}
    (Path(a.out) / "meta" / "trim_report.json").write_text(json.dumps(rep, indent=2))
    print(f"wrote {a.out}: {res['frames_after']} of {res['frames_before']} frames (report: meta/trim_report.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
