"""Where an eval's results live (lab/EVALUATION.md, "Layout"), and reading them back.

results/evals/<eval>/sessions/<YYYYMMDD-HHMM>-<machine>/
    session.json
    trials/<condition>/<model>/tNN/trial.json (+ the trial's raw files)

A session belongs to one machine and one sitting; nothing in it is written by another machine. A session that has
been pushed to the results store holds a `.pushed` marker (the commit it was pushed in).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path("results") / "evals"
PUSHED = ".pushed"


def eval_root(name: str, root: Path | None = None) -> Path:
    return (root or ROOT) / name


def sessions_dir(name: str, root: Path | None = None) -> Path:
    return eval_root(name, root) / "sessions"


def trial_dir(session: Path, condition: str, model: str, number: int) -> Path:
    return session / "trials" / condition / model / f"t{number:02d}"


@dataclass
class Trial:
    path: Path
    session: str
    condition: str
    model: str
    number: int
    info: dict = field(default_factory=dict)

    @property
    def void(self) -> bool:
        return bool(self.info.get("void"))

    @property
    def valid(self) -> bool:
        """Counts towards the plan: not voided, and labelled by the operator (s/f or the time limit)."""
        return not self.void and self.info.get("success") in (0, 1)


def sessions(name: str, root: Path | None = None) -> list[Path]:
    d = sessions_dir(name, root)
    return sorted(p for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def trials(name: str, root: Path | None = None) -> list[Trial]:
    out = []
    for s in sessions(name, root):
        for f in sorted((s / "trials").glob("*/*/t*/trial.json")):
            t = f.parent
            try:
                info = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                info = {"error": "unreadable trial.json"}
            out.append(Trial(t, s.name, t.parent.parent.name, t.parent.name, int(t.name[1:]), info))
    return out


def valid_counts(name: str, root: Path | None = None) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for t in trials(name, root):
        if t.valid:
            counts[(t.condition, t.model)] = counts.get((t.condition, t.model), 0) + 1
    return counts


def unpushed(name: str, root: Path | None = None) -> list[Path]:
    return [s for s in sessions(name, root) if not (s / PUSHED).is_file()]
