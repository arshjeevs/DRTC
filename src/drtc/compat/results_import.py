"""Import the minor's robot-run results into the new results layout: `lab import-results <old results dir>`.

Copies (never moves) every run folder of the old results/<group>/<name>/<model>/<run_id>/ to
results/minor/<group>/<name>/<new model>/<run_id>/, with model names mapped by lab/renames.yaml, plus the other old
folders (offline/, benchmark/, summary/) as they are, and appends the old index rows to results/index.csv
(outcome "imported", robot "rig"). Already imported runs are skipped, so it can be re-run.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import yaml

from drtc.control.run import INDEX_FIELDS, append_index
from drtc.registry import find_lab_dir

RESULTS = Path("results") / "archive"  # imported minor results are not official results (lab/docs/EVALUATION.md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab import-results", description=__doc__.split("\n\n")[0])
    ap.add_argument("old", type=Path, help="the old results folder, e.g. ~/Projects/drtc/results")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    old = a.old.expanduser()
    index = old / "index.csv"
    if not index.is_file():
        print(f"lab import-results: no {index}", file=sys.stderr)
        return 1
    renames = yaml.safe_load((find_lab_dir() / "renames.yaml").read_text())["models"]
    dest_root = RESULTS / "minor"
    new_index = RESULTS / "index.csv"
    done = set()
    if new_index.is_file():
        done = {r["run_id"] for r in csv.DictReader(open(new_index)) if r.get("outcome") == "imported"}
    rows, copied, skipped = [], 0, 0
    for r in csv.DictReader(open(index)):
        if r["run_id"] in done:
            skipped += 1
            continue
        model = renames.get(r["model"], r["model"])
        src = old.parent / r["run_dir"] if not Path(r["run_dir"]).is_absolute() else Path(r["run_dir"])
        dst = dest_root / r["experiment"] / model / r["run_id"]
        if src.is_dir() and not dst.exists():
            if not a.dry_run:
                shutil.copytree(src, dst)
            copied += 1
        rows.append({**{k: r.get(k, "") for k in INDEX_FIELDS}, "model": model, "run_dir": str(dst),
                     "robot": "rig", "host": "minor", "outcome": "imported",
                     "note": (r.get("note", "") + f" [minor: {r['model']}]").strip()})
    extra = [p for p in old.iterdir() if p.is_dir() and p.name in ("offline", "benchmark", "summary")]
    print(f"{len(rows)} runs to import ({copied} folders to copy, {skipped} already imported); "
          f"also {', '.join(p.name for p in extra) or 'no other folders'}")
    if a.dry_run:
        return 0
    for p in extra:
        if not (dest_root / p.name).exists():
            shutil.copytree(p, dest_root / p.name)
    if rows:
        append_index(rows, new_index)
    print(f"imported into {dest_root} and {new_index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
