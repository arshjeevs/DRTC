"""Dataset operations: `lab data show|view|delete|trim|label|undo|push|remove <name>`.

One rule makes every operation simple: **the raw recording (<name>_raw) is the only copy that is edited**. The
trimmed copy (<name>, the one to train on) is derived from it and rebuilt after every change, never edited by hand:
autotrim (drtc.data.trim) plus the manual trims you set. What belongs to a dataset besides LeRobot's files is kept
inside the raw copy, in meta/drtc.json, so it travels with it to the Hub:

    {"layouts": {"<episode>": "<layout>"},      which layout an episode was recorded with (lab/layouts.yaml)
     "trim": {"<episode>": [start, end]}}         manual crop in frames, instead of autotrim's

Edits stay local until `lab data push` (it uploads both copies and updates lab/datasets/recorded.yaml; each GPU
server prepares the dataset for its recipes at its first `lab train`, or ahead of time with `lab prepare`). One undo level: before each edit the raw copy is snapshotted (hardlinks, so it costs almost
no disk) in $HF_LEROBOT_HOME/.drtc_edit/<repo>/undo; the snapshot is dropped at the next push or recording.
Deleting episodes renumbers the later ones (LeRobot keeps indices 0..N-1); layouts and manual trims follow.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from drtc.registry import Registry

SIDECAR = "meta/drtc.json"
LAYOUTS_FILE = "layouts.yaml"


# --- where a dataset lives ---------------------------------------------------------------------------------------


@dataclass
class Paths:
    name: str
    raw_repo: str
    trimmed_repo: str
    raw: Path
    trimmed: Path
    state: Path       # .drtc_edit/<raw repo>: undo snapshot and the list of unpushed changes

    @property
    def undo(self) -> Path:
        return self.state / "undo"

    @property
    def rebuild(self) -> Path:
        return self.trimmed.with_name(self.trimmed.name + ".rebuild")

    def variants(self) -> list[Path]:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        return [HF_LEROBOT_HOME / ".drtc_variants" / r for r in (self.raw_repo, self.trimmed_repo)]


def paths(reg: Registry, name: str) -> Paths:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    if name.endswith("_raw"):
        name = name[: -len("_raw")]  # either name means the dataset
    account = reg.settings.hub_account
    raw_repo, trimmed_repo = f"{account}/{name}_raw", f"{account}/{name}"
    return Paths(name, raw_repo, trimmed_repo, HF_LEROBOT_HOME / raw_repo, HF_LEROBOT_HOME / trimmed_repo,
                 HF_LEROBOT_HOME / ".drtc_edit" / raw_repo)


# --- meta/drtc.json, pending changes, undo -----------------------------------------------------------------------


def _write_atomic(path: Path, text: str) -> None:
    """A new file, never an in-place write: the undo snapshot hardlinks the old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_sidecar(raw: Path) -> dict:
    f = raw / SIDECAR
    d = json.loads(f.read_text()) if f.is_file() else {}
    return {"layouts": {int(k): v for k, v in d.get("layouts", {}).items()},
            "trim": {int(k): tuple(v) for k, v in d.get("trim", {}).items()}}


def write_sidecar(raw: Path, d: dict) -> None:
    out = {"layouts": {str(k): v for k, v in sorted(d["layouts"].items())},
           "trim": {str(k): list(v) for k, v in sorted(d["trim"].items())}}
    _write_atomic(raw / SIDECAR, json.dumps(out, indent=1) + "\n")


def pending(p: Paths) -> list[str]:
    f = p.state / "pending.json"
    return json.loads(f.read_text()) if f.is_file() else []


def add_pending(p: Paths, change: str) -> None:
    _write_atomic(p.state / "pending.json", json.dumps(pending(p) + [change], indent=1))


def clear_state(p: Paths) -> None:
    """After a push or a new recording session: nothing pending, no undo."""
    if p.state.exists():
        shutil.rmtree(p.state)


def snapshot(p: Paths, change: str) -> None:
    """Keep the raw copy as it is now, for `lab data undo` (one level)."""
    if p.undo.exists():
        shutil.rmtree(p.undo)
    p.state.mkdir(parents=True, exist_ok=True)
    shutil.copytree(p.raw, p.undo / "raw", copy_function=os.link)
    _write_atomic(p.undo / "change.txt", change)


# --- layouts (lab/layouts.yaml) -------------------------------------------------------------------------


def layouts_for(reg: Registry, name: str) -> list[str]:
    f = reg.lab / LAYOUTS_FILE
    d = yaml.safe_load(f.read_text()) if f.is_file() else {}
    return list((d or {}).get(name, []) or [])


def next_layout(rotation: list[str], labels: dict[int, str]) -> str | None:
    """The layout recorded least so far; on a tie, the next one in the rotation after the last episode's."""
    if not rotation:
        return None
    counts = {lay: 0 for lay in rotation}
    for lay in labels.values():
        if lay in counts:
            counts[lay] += 1
    low = min(counts.values())
    last = labels[max(labels)] if labels else None
    start = (rotation.index(last) + 1) % len(rotation) if last in rotation else 0
    order = rotation[start:] + rotation[:start]
    return next(lay for lay in order if counts[lay] == low)


def remap(d: dict, deleted: set[int], n: int) -> dict:
    """meta/drtc.json after deleting episodes: the kept ones are renumbered 0..; entries of deleted ones go."""
    new_index = {old: new for new, old in enumerate(i for i in range(n) if i not in deleted)}
    return {key: {new_index[i]: v for i, v in d[key].items() if i in new_index} for key in ("layouts", "trim")}


def parse_episodes(spec: list[str], n: int) -> list[int]:
    """'3', '5-9', '3 5-9 12' -> sorted indices, each checked against 0..n-1."""
    out: set[int] = set()
    for part in " ".join(spec).replace(",", " ").split():
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    bad = sorted(i for i in out if not 0 <= i < n)
    if bad:
        raise SystemExit(f"lab data: no episode {', '.join(map(str, bad))} (the dataset has 0-{n - 1})")
    return sorted(out)


# --- the trimmed copy: always rebuilt from the raw one -----------------------------------------------------------


def trim_results(raw: Path):
    """Autotrim's result per episode, with the manual trims of meta/drtc.json applied over it."""
    from drtc.data import trim

    ds = trim.RawDataset(raw)
    params = trim.Params()
    results, noise, floor = trim.detect(ds, params)
    manual = read_sidecar(raw)["trim"]
    fps = ds.fps
    for r in results:
        if r.episode in manual:
            r.start, r.end = manual[r.episode]
            r.head_s, r.tail_s = r.start / fps, (r.length - r.end) / fps
            r.status, r.note = "trim", "manual"
    return ds, results, noise, floor, params


def rebuild(p: Paths, quiet: bool = False) -> None:
    """(Re)make the trimmed copy from the raw one. The old copy is replaced only once the new one is complete."""
    from drtc.data import trim

    ds, results, noise, floor, params = trim_results(p.raw)
    if not quiet:
        trim.report(ds, results, noise, floor, params, show=5)
    for leftover in (p.rebuild, p.rebuild.with_name(p.rebuild.name + ".partial")):
        if leftover.exists():
            shutil.rmtree(leftover)  # an interrupted earlier rebuild: derived data, made again below
    trim.apply(ds, results, p.rebuild)
    if p.trimmed.exists():
        shutil.rmtree(p.trimmed)
    p.rebuild.rename(p.trimmed)
    print(f"trimmed copy rebuilt from all {len(results)} episodes: {p.trimmed}")


# --- the Hub -------------------------------------------------------------------------------------------------------


def push_one(repo: str, root: Path) -> None:
    """LeRobot's push_to_hub, after deleting files the Hub has but the local copy no longer has (e.g. a video file
    that a deletion removed), so the Hub copy is exactly the local one."""
    from huggingface_hub import HfApi
    from lerobot.datasets import LeRobotDataset

    api = HfApi()
    if api.repo_exists(repo, repo_type="dataset"):
        local = {str(f.relative_to(root)) for f in root.rglob("*") if f.is_file()}
        remote = [f for f in api.list_repo_files(repo, repo_type="dataset")
                  if f.split("/", 1)[0] in ("data", "meta", "videos")]
        stale = [f for f in remote if f not in local]
        if stale:
            api.delete_files(repo, delete_patterns=stale, repo_type="dataset",
                             commit_message=f"lab data push: remove {len(stale)} files the local copy no longer has")
    LeRobotDataset(repo, root=root).push_to_hub(allow_patterns=["data/*", "meta/*", "videos/*"])
    print(f"pushed https://huggingface.co/datasets/{repo}")


def task_of(raw: Path) -> str:
    import pandas as pd

    tasks = pd.read_parquet(raw / "meta/tasks.parquet")
    names = list(tasks.index) if "task" not in tasks.columns else list(tasks["task"])
    return names[0] if len(names) == 1 else " | ".join(map(str, names))


def publish(reg: Registry, p: Paths, held_out: int | None = None) -> None:
    """Push both copies and write their registry entries (pinned to the pushed revisions). Not prepared here: the
    prepared copies (frame caches, relative stats) only serve the machine that makes them, so each training host
    makes its own at its first `lab train` (or `lab prepare <name>` ahead of time)."""
    from drtc.data.record import REGISTRY_FILE, check_recording, registry_entries

    info = json.loads((p.raw / "meta/info.json").read_text())
    issues = check_recording(p.raw, int(info["fps"]))
    if issues:
        for i in issues:
            print(f"  CHECK {i}", file=sys.stderr)
        raise SystemExit("lab data: the raw copy has problems; not pushing. Look at them first.")
    if not p.trimmed.exists():
        rebuild(p)
    for repo, root in ((p.raw_repo, p.raw), (p.trimmed_repo, p.trimmed)):
        push_one(repo, root)
    path = reg.lab / "datasets" / REGISTRY_FILE
    existing = (yaml.safe_load(path.read_text()) if path.is_file() else {}) or {}
    if held_out is None:
        held_out = (existing.get(p.name, {}).get("split") or {}).get("held_out") or max(1, round(info["total_episodes"] / 10))
    entries = registry_entries(reg, p.name, p.raw_repo, p.trimmed_repo, task_of(p.raw), int(info["fps"]), held_out)
    existing.update(entries)
    path.write_text("# Written by lab record / lab data; review and commit.\n\n" + yaml.safe_dump(existing, sort_keys=False))
    print(f"registry: {', '.join(entries)} in {path} (review and commit)")
    clear_state(p)
    print(f"ready to train: commit lab/datasets/{REGISTRY_FILE}, git pull on the GPU server; its first lab train "
          f"prepares the dataset there (or ahead of time: lab prepare {p.name})")


# --- commands -------------------------------------------------------------------------------------------------------


def need_raw(p: Paths) -> None:
    if not (p.raw / "meta/info.json").is_file():
        raise SystemExit(f"lab data: no recording at {p.raw}")


def cmd_show(reg: Registry, p: Paths, a) -> int:
    from drtc.data.record import check_recording

    need_raw(p)
    ds, results, _, _, _ = trim_results(p.raw)
    fps = ds.fps
    side = read_sidecar(p.raw)
    rotation = layouts_for(reg, p.name)
    issues = check_recording(p.raw, fps)
    by_ep: dict[int, list[str]] = {}
    for i in issues:
        if i.startswith("episode "):
            by_ep.setdefault(int(i.split()[1].rstrip(":")), []).append(i.split(" ", 2)[2])
    print(f"{p.raw_repo}: {len(results)} episodes, {int(ds.episodes['length'].sum()) / fps / 60:.1f} min, "
          f"task {task_of(p.raw)!r}")
    print(f"\n{'ep':>3}  {'raw s':>6}  {'kept s':>13}  {'trim':<8}  layout")
    for r in results:
        kept = f"{r.start / fps:5.1f}-{r.end / fps:5.1f}" if r.status == "trim" else "all"
        lay = side["layouts"].get(r.episode, "-")
        warn = ("  ! " + "; ".join(by_ep[r.episode])) if r.episode in by_ep else ""
        status = r.note or r.status
        print(f"{r.episode:>3}  {r.length / fps:6.1f}  {kept:>13}  {status:<8}  {lay}{warn}")
    if rotation:
        counts = {lay: 0 for lay in rotation}
        for lay in side["layouts"].values():
            counts[lay] = counts.get(lay, 0) + 1
        print("\nlayouts: " + ", ".join(f"{lay} {n}" for lay, n in counts.items()))
        missing = [r.episode for r in results if r.episode not in side["layouts"]]
        if missing:
            print(f"  without a layout: {fmt_ranges(missing)} (lab data label {p.name} <episodes> <layout number>)")
        print(f"  next to record: {next_layout(rotation, side['layouts'])}")
    ch = pending(p)
    print(f"\nnot pushed yet: {'; '.join(ch)} (lab data push {p.name})" if ch else "\nlocal copy = what was pushed last")
    if p.undo.exists():
        print(f"undo available: {(p.undo / 'change.txt').read_text()}")
    return 0


def fmt_ranges(xs: list[int]) -> str:
    out, xs = [], sorted(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(f"{xs[i]}" if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ", ".join(out)


def cmd_view(reg: Registry, p: Paths, a) -> int:
    need_raw(p)
    repo, root = (p.trimmed_repo, p.trimmed) if a.trimmed else (p.raw_repo, p.raw)
    cmd = [str(Path(sys.executable).with_name("lerobot-dataset-viz")), "--repo-id", repo, "--root", str(root),
           "--episode-index", str(a.episode)]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


def cmd_delete(reg: Registry, p: Paths, a) -> int:
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import delete_episodes

    need_raw(p)
    n = json.loads((p.raw / "meta/info.json").read_text())["total_episodes"]
    eps = parse_episodes(a.episodes, n)
    if len(eps) >= n:
        raise SystemExit(f"lab data: that deletes every episode; use lab data remove {p.name}")
    side = read_sidecar(p.raw)
    later = n - 1 - max(eps)
    print(f"delete episode(s) {fmt_ranges(eps)} of {n}" + (f"; the {later} after them are renumbered" if later else ""))
    if not a.yes and input("type 'yes' to delete: ").strip() != "yes":
        print("nothing deleted")
        return 1
    change = f"deleted episode(s) {fmt_ranges(eps)}"
    snapshot(p, change)
    new = p.raw.with_name(p.raw.name + ".new")
    if new.exists():
        shutil.rmtree(new)
    delete_episodes(LeRobotDataset(p.raw_repo, root=p.raw), eps, output_dir=new, repo_id=p.raw_repo)
    write_sidecar(new, remap(side, set(eps), n))
    shutil.rmtree(p.raw)  # the undo snapshot still holds its files
    new.rename(p.raw)
    add_pending(p, change)
    rebuild(p, quiet=True)
    print(f"{change}; {n - len(eps)} episodes left. Undo: lab data undo {p.name}; upload: lab data push {p.name}")
    return 0


def cmd_trim(reg: Registry, p: Paths, a) -> int:
    need_raw(p)
    from drtc.data import trim

    ds = trim.RawDataset(p.raw)
    fps, n = ds.fps, len(ds.episodes)
    ep = parse_episodes([str(a.episode)], n)[0]
    length = int(ds.episodes.loc[ds.episodes.episode_index == ep, "length"].iloc[0])
    side = read_sidecar(p.raw)
    if a.auto:
        if ep not in side["trim"]:
            print(f"episode {ep} already uses autotrim")
            return 0
        del side["trim"][ep]
        change = f"episode {ep}: back to autotrim"
    else:
        start = round((a.start or 0.0) * fps)
        end = length if a.end is None else round(a.end * fps)
        if not 0 <= start < end <= length:
            raise SystemExit(f"lab data: episode {ep} is {length / fps:.2f} s; need 0 <= start < end <= that")
        side["trim"][ep] = (start, end)
        change = f"episode {ep}: keep {start / fps:.2f}-{end / fps:.2f} s"
    snapshot(p, change)
    write_sidecar(p.raw, side)
    add_pending(p, change)
    rebuild(p, quiet=True)
    print(f"{change}. Undo: lab data undo {p.name}; upload: lab data push {p.name}")
    return 0


def cmd_label(reg: Registry, p: Paths, a) -> int:
    need_raw(p)
    rotation = layouts_for(reg, p.name)
    if not rotation:
        raise SystemExit(f"lab data: no layouts for {p.name} in lab/{LAYOUTS_FILE}")
    n = json.loads((p.raw / "meta/info.json").read_text())["total_episodes"]
    eps = parse_episodes(a.episodes, n)
    if not 1 <= a.layout <= len(rotation):
        raise SystemExit("lab data: layouts are numbered " + ", ".join(f"{i + 1} {lay}" for i, lay in enumerate(rotation)))
    side = read_sidecar(p.raw)
    for k, ep in enumerate(eps):
        # --rotate: the first gets the given layout, the next ones follow the rotation
        side["layouts"][ep] = rotation[(a.layout - 1 + (k if a.rotate else 0)) % len(rotation)]
    change = f"layouts of episode(s) {fmt_ranges(eps)}"
    snapshot(p, change)
    write_sidecar(p.raw, side)
    add_pending(p, change)
    for ep in eps:
        print(f"  {ep:>3}  {side['layouts'][ep]}")
    print(f"next to record: {next_layout(rotation, side['layouts'])}")
    return 0


def cmd_undo(reg: Registry, p: Paths, a) -> int:
    if not p.undo.exists():
        print(f"lab data: nothing to undo for {p.name} (one level; dropped at each push or recording)")
        return 1
    change = (p.undo / "change.txt").read_text()
    shutil.rmtree(p.raw)
    (p.undo / "raw").rename(p.raw)
    shutil.rmtree(p.undo)
    ch = pending(p)
    if ch and ch[-1] == change:
        ch = ch[:-1]
    _write_atomic(p.state / "pending.json", json.dumps(ch, indent=1))
    rebuild(p, quiet=True)
    print(f"undone: {change}")
    return 0


def cmd_push(reg: Registry, p: Paths, a) -> int:
    need_raw(p)
    publish(reg, p, a.held_out)
    return 0


def cmd_remove(reg: Registry, p: Paths, a) -> int:
    from drtc.data.record import REGISTRY_FILE

    local = [d for d in (p.raw, p.trimmed, p.rebuild, *p.variants(), p.state) if d.exists()]
    print("deletes:\n" + "\n".join(f"  {d}" for d in local))
    if a.hub:
        print(f"  https://huggingface.co/datasets/{p.raw_repo}\n  https://huggingface.co/datasets/{p.trimmed_repo}"
              "\n  (the Hub copies cannot be restored)")
    print(f"  registry entries {p.name}_raw, {p.name} (lab/datasets/{REGISTRY_FILE})")
    if input(f"type the name ({p.name}) to delete: ").strip() != p.name:
        print("nothing deleted")
        return 1
    for d in local:
        shutil.rmtree(d)
    if a.hub:
        from huggingface_hub import HfApi

        api = HfApi()
        for repo in (p.raw_repo, p.trimmed_repo):
            api.delete_repo(repo, repo_type="dataset", missing_ok=True)
            print(f"deleted https://huggingface.co/datasets/{repo}")
    path = reg.lab / "datasets" / REGISTRY_FILE
    if path.is_file():
        existing = yaml.safe_load(path.read_text()) or {}
        for k in (f"{p.name}_raw", p.name):
            existing.pop(k, None)
        path.write_text("# Written by lab record / lab data; review and commit.\n\n" + yaml.safe_dump(existing, sort_keys=False))
    print(f"{p.name} removed" + ("" if a.hub else " locally (the Hub copies stay; --hub deletes them too)"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab data", description=__doc__.split("\n\n")[0], allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="episodes, trims, layouts, warnings; what is not pushed yet")
    s.add_argument("name")
    s = sub.add_parser("view", help="open one episode in the Rerun viewer")
    s.add_argument("name")
    s.add_argument("episode", type=int)
    s.add_argument("--trimmed", action="store_true", help="the trimmed copy instead of the raw one")
    s = sub.add_parser("delete", help="delete episodes (e.g. 3, 5-9); later ones are renumbered")
    s.add_argument("name")
    s.add_argument("episodes", nargs="+")
    s.add_argument("--yes", action="store_true", help="no confirmation")
    s = sub.add_parser("trim", help="set one episode's kept range by hand (seconds), or --auto to go back")
    s.add_argument("name")
    s.add_argument("episode", type=int)
    s.add_argument("--start", type=float, help="seconds from the episode's start (default 0)")
    s.add_argument("--end", type=float, help="seconds from the episode's start (default: its end)")
    s.add_argument("--auto", action="store_true", help="drop the manual trim; autotrim again")
    s = sub.add_parser("label", help="set the layout of episodes (number from lab/layouts.yaml)")
    s.add_argument("name")
    s.add_argument("episodes", nargs="+", help="e.g. 0-10 or 4 7")
    s.add_argument("--layout", type=int, required=True, help="layout number, 1 = the first in layouts.yaml")
    s.add_argument("--rotate", action="store_true", help="the first episode gets --layout, the next ones follow "
                                                         "the rotation (as recorded)")
    s = sub.add_parser("undo", help="undo the last edit (one level, until the next push)")
    s.add_argument("name")
    s = sub.add_parser("push", help="upload both copies and update the registry")
    s.add_argument("name")
    s.add_argument("--held-out", type=int, help="held-out episodes (default: as registered, else 10%%)")
    s = sub.add_parser("remove", help="delete the whole dataset locally (--hub: also on the Hub)")
    s.add_argument("name")
    s.add_argument("--hub", action="store_true")
    a = ap.parse_args(argv)
    reg = Registry()
    p = paths(reg, a.name)
    return {"show": cmd_show, "view": cmd_view, "delete": cmd_delete, "trim": cmd_trim, "label": cmd_label,
            "undo": cmd_undo, "push": cmd_push, "remove": cmd_remove}[a.cmd](reg, p, a)


if __name__ == "__main__":
    sys.exit(main())
