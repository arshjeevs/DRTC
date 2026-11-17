"""Automatic trimming of the idle start and end of every episode in a LeRobot v3.0 dataset.

    python -m lab.tools.autotrim <dataset dir> [--apply] [--replace] [--pad S] [--move F] ...

Teleoperated episodes begin and end with the arm at rest (recording started before the operator moved, and
stopped a moment after the task was done). Those frames teach a policy that "do nothing" is a likely action
in the start pose, which hurts models such as pi0.5. This finds where motion starts and stops and crops each
episode to it, keeping `--pad` seconds of rest on either side.

Nothing in here is specific to one robot or dataset:
- Joints are compared as a fraction of their own range (min..max from meta/stats.json), so units (degrees,
  radians, metres), the number of joints and the gripper's scale do not matter.
- The noise level that counts as "still" is measured from the dataset itself (the jitter of frames that sit
  at the episode's first / last pose), not set by hand.
- Padding and minimum lengths are in seconds, so any fps works.
- The START is found from `action` (the commanded motion: in teleoperation the follower may still be catching
  up with the leader in the first frames, which is not task motion). The END uses `action` and
  `observation.state` (when present), whichever settles later, because the follower lags the command.

Detection, per episode and signal x (n frames x d joints, divided by the joint ranges):
  onset   = first frame whose distance from frame 0 exceeds --move (default 3% of the range in any joint),
            then walked back to the last frame still within the noise floor of frame 0 (where the motion began)
  offset  = the same from the last frame, backwards
Episodes where nothing exceeds --move are reported as static (candidates for deletion; deleted only with
--delete-static). Crops that would keep under --min-keep seconds are reported and not staged (more likely a
detection miss); crops that remove most of an episode are staged and listed (long rests do happen).

The crops are written as staged edits of github.com/acrosson/lerobot-dataset-editor (<dataset>/_editor/
session.json): review them in the editor (lab/bin/edit-dataset) and Save, or pass --apply to write them now
with the editor's own writer (backup in <dataset>/_backups/). A JSON report goes to <dataset>/_editor/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

EDITOR_DIR = Path(os.environ.get("LRDE_DIR", Path(__file__).resolve().parents[3] / "lerobot-dataset-editor"))


def _import_editor():
    if not (EDITOR_DIR / "editor" / "apply.py").is_file():
        sys.exit(f"lerobot-dataset-editor not found at {EDITOR_DIR} (run lab/bin/edit-dataset once, or set LRDE_DIR)")
    sys.path.insert(0, str(EDITOR_DIR))
    from editor import apply, dataset, session  # noqa: PLC0415

    return dataset, session, apply


@dataclass
class Params:
    move: float = 0.03          # fraction of a joint's range that counts as "the arm has moved"
    noise_mult: float = 2.0     # step above (noise_mult x measured noise floor) = still moving
    pad: float = 0.1            # seconds of rest kept before the onset and after the offset
    min_keep_s: float = 1.0     # crops keeping less than this many seconds are not staged
    min_keep_frac: float = 0.5  # crops keeping less than this fraction are staged but listed for a look
    pause_s: float = 2.0        # still stretches inside the kept range longer than this are reported


@dataclass
class EpisodeResult:
    episode: int
    length: int
    start: int                  # first kept frame
    end: int                    # one past the last kept frame
    head_s: float               # seconds cut from the start
    tail_s: float               # seconds cut from the end
    status: str                 # trim | none | static | too_short
    longest_pause_s: float      # longest still stretch inside the kept range
    note: str = ""


# ----------------------------------------------------------------------------- signals

def joint_ranges(ds, key: str, sample: list[np.ndarray]) -> np.ndarray:
    """Per-joint scale: the full range (max - min) from meta/stats.json, else from the frames themselves.

    Never below a quarter of the median joint's range: a joint the task barely uses (e.g. a wrist roll that
    stays within a few degrees) would otherwise turn its sensor noise into "motion".
    """
    stats_path = ds.path / "meta" / "stats.json"
    st = json.loads(stats_path.read_text()).get(key, {}) if stats_path.is_file() else {}
    if "min" in st and "max" in st:
        lo, hi = np.asarray(st["min"], float).ravel(), np.asarray(st["max"], float).ravel()
    else:
        allx = np.concatenate(sample)
        lo, hi = allx.min(axis=0), allx.max(axis=0)
    rng = hi - lo
    median = float(np.median(rng[rng > 0])) if np.any(rng > 0) else 1.0
    return np.maximum(rng, 0.25 * median)


def steps(x: np.ndarray) -> np.ndarray:
    """Frame-to-frame change, largest over joints; steps[t] is the change from t-1 to t (steps[0] = 0)."""
    return np.r_[0.0, np.abs(np.diff(x, axis=0)).max(axis=1)] if len(x) > 1 else np.zeros(len(x))


def measure_noise(series: list[np.ndarray], move: float) -> float:
    """The "still" jitter level: frame-to-frame steps in the rest at the very start and end of episodes.

    Only the unbroken run of frames from frame 0 (and back from the last frame) that stays within move/3 of
    that pose is used, so the measurement is of the arm at rest, not of slow motion that passes near the pose.
    """
    vals = []
    for x in series:
        s = steps(x)
        for seg, st in ((x, s[1:]), (x[::-1], s[1:][::-1])):
            near = np.abs(seg - seg[0]).max(axis=1) < move / 3
            k = int(np.argmin(near)) if not near.all() else len(near)  # length of the unbroken rest
            vals.append(st[:max(k - 1, 0)])
    v = np.concatenate(vals) if vals else np.array([])
    v = v[v > 0]  # exact repeats say nothing about the jitter's size
    return float(np.percentile(v, 95)) if len(v) else 0.0


def onset(x: np.ndarray, move: float, floor: float) -> int | None:
    """Last frame at the first pose before the motion that takes x away from it; None if it never leaves.

    The motion is found where x is more than `move` from frame 0; from there we walk back to the last frame
    still within `floor` (the noise level) of frame 0. Using the distance, not the speed, means slow starts
    are kept whole.
    """
    dev = np.abs(x - x[0]).max(axis=1)
    over = np.flatnonzero(dev > move)
    if not len(over):
        return None
    i = int(over[0])
    while i > 0 and dev[i] > floor:
        i -= 1
    return i


def offset(x: np.ndarray, move: float, floor: float) -> int | None:
    """Last frame of motion before x settles into its final pose (index into x); None if never away from it."""
    i = onset(x[::-1], move, floor)
    return None if i is None else len(x) - 1 - i


def longest_pause(x: np.ndarray, floor: float) -> int:
    """Longest run of frames whose step stays at or below the noise floor."""
    still = steps(x) <= floor
    best = run = 0
    for v in still[1:]:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


# ----------------------------------------------------------------------------- per episode

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
    if kept / fps < p.min_keep_s:  # almost nothing moved: more likely a detection miss than a real take
        res.status, res.note = "too_short", f"would keep {kept} of {n} frames ({kept / fps:.1f} s): check it by hand"
    elif kept / n < p.min_keep_frac:  # long rests do happen (measured: 14 s before a take); cropped, but listed
        res.note = f"removes {1 - kept / n:.0%} of the episode: long rest, worth a look"
    elif pause >= p.pause_s:
        res.note = f"still for {pause:.1f} s inside the kept range"
    return res


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="local LeRobot v3.0 dataset directory (edited in place: use a copy)")
    ap.add_argument("--apply", action="store_true", help="write the crops now (else: staged for review in the editor)")
    ap.add_argument("--dry-run", action="store_true", help="only print the report; write nothing (not even the report)")
    ap.add_argument("--replace", action="store_true", help="discard edits already staged in the editor session")
    ap.add_argument("--delete-static", action="store_true", help="also stage episodes with no motion for deletion")
    ap.add_argument("--move", type=float, default=Params.move, help="motion threshold, fraction of joint range (0.03)")
    ap.add_argument("--noise-mult", type=float, default=Params.noise_mult, help="x measured noise = still (2.0)")
    ap.add_argument("--pad", type=float, default=Params.pad, help="seconds of rest kept at each end (0.1)")
    ap.add_argument("--min-keep", type=float, default=Params.min_keep_s, help="don't crop to less than this, seconds (1.0)")
    ap.add_argument("--min-keep-frac", type=float, default=Params.min_keep_frac, help="list crops keeping less than this fraction (0.5)")
    ap.add_argument("--show", type=int, default=15, help="episodes listed in the table (largest cuts first; 0 = all)")
    args = ap.parse_args()
    p = Params(args.move, args.noise_mult, args.pad, args.min_keep, args.min_keep_frac)

    dataset_mod, session_mod, apply_mod = _import_editor()
    ds = dataset_mod.Dataset(args.dataset)
    if ds.info.get("codebase_version") != "v3.0":
        sys.exit(f"{ds.path}: codebase_version {ds.info.get('codebase_version')}; only v3.0 is supported")
    if "action" not in ds.features:
        sys.exit(f"{ds.path}: no 'action' feature")
    has_state = "observation.state" in ds.features and \
        ds.features["observation.state"].shape == ds.features["action"].shape
    fps = ds.fps

    session = session_mod.Session.load(ds.path)
    if session.dirty and not args.replace and not args.dry_run:
        sys.exit(f"{len(session.edits)} edits are already staged in {session.file}; pass --replace to discard them")

    # Read every episode once (two features per frame: small even for hundreds of episodes).
    episodes = [int(e) for e in ds.episodes["episode_index"]]
    raw = {e: ds.episode_series(e) for e in episodes}
    r_act = joint_ranges(ds, "action", [raw[e]["action"] for e in episodes])
    r_state = joint_ranges(ds, "observation.state", [raw[e]["observation.state"] for e in episodes]) if has_state else None
    act = {e: raw[e]["action"] / r_act for e in episodes}
    state = {e: raw[e]["observation.state"] / r_state for e in episodes} if has_state else {}

    noise = measure_noise(list(act.values()) + list(state.values()), p.move)
    floor = max(noise * p.noise_mult, 1e-6)
    results = [trim_episode(e, act[e], state.get(e), fps, p, floor) for e in episodes]

    # ---- report
    total = sum(r.length for r in results)
    cut = sum(r.length - (r.end - r.start) for r in results if r.status == "trim")
    by = {s: [r for r in results if r.status == s] for s in ("trim", "none", "static", "too_short")}
    print(f"{ds.path.name}: {len(results)} episodes, {total} frames at {fps} fps; joints: {ds.features['action'].shape}"
          f"{' (+ observation.state)' if has_state else ''}")
    print(f"still = step <= {floor:.4f} of range (measured noise {noise:.4f} x {p.noise_mult}); "
          f"moved = > {p.move:.0%} of range; pad {p.pad} s")
    heads = np.array([r.head_s for r in by["trim"]] or [0.0])
    tails = np.array([r.tail_s for r in by["trim"]] or [0.0])
    print(f"trim {len(by['trim'])}, unchanged {len(by['none'])}, static {len(by['static'])}, "
          f"too short {len(by['too_short'])}; removes {cut} frames ({cut / max(total, 1):.1%}, {cut / fps:.0f} s)")
    print(f"cut from start: median {np.median(heads):.2f} s, max {heads.max():.2f} s;  "
          f"from end: median {np.median(tails):.2f} s, max {tails.max():.2f} s")
    shown = sorted(results, key=lambda r: -(r.head_s + r.tail_s))
    shown = shown if args.show == 0 else shown[:args.show]
    print(f"\n  {'ep':>4} {'frames':>6} {'kept':>11} {'start -s':>8} {'end -s':>7}  status     note")
    for r in shown:
        print(f"  {r.episode:>4} {r.length:>6} {r.start:>5}-{r.end:<5} {r.head_s:>8.2f} {r.tail_s:>7.2f}  {r.status:<10} {r.note}")
    flagged = [r for r in results if r.status in ("static", "too_short") or r.note]
    for r in flagged:
        if r not in shown:
            print(f"  {r.episode:>4} {r.length:>6} {r.start:>5}-{r.end:<5} {r.head_s:>8.2f} {r.tail_s:>7.2f}  {r.status:<10} {r.note}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0
    report = {"dataset": str(ds.path), "params": asdict(p), "noise": noise, "floor": floor,
              "fps": fps, "episodes": [asdict(r) for r in results]}
    out = ds.path / session_mod.EDITOR_DIR / "autotrim.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    # ---- stage (and optionally apply)
    if args.replace:
        session.clear()
    for r in results:
        if r.status == "trim":
            session.set(r.episode, session_mod.EpisodeEdit(start=r.start, end=r.end), r.length)
        elif r.status == "static" and args.delete_static:
            session.set(r.episode, session_mod.EpisodeEdit(deleted=True), r.length)
    session.save()
    print(f"\nreport: {out}\nstaged {len(session.edits)} edits in {session.file}")
    if args.apply:
        result = apply_mod.apply_session(ds, session)
        print(f"applied: {result.get('episodes_after', '?')} episodes, {result.get('frames_after', '?')} frames"
              f" (backup: {result.get('backup', ds.path / '_backups')})")
    else:
        print("review them in the editor (lab/bin/edit-dataset) and press Save, or re-run with --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
