"""Run Phase 3 speed comparisons from tests/bench/variants.yaml with tests/bench/train_speed.sh.

    .venv/bin/python tests/bench/round.py [variant ...]      # default: every variant, in file order
    .venv/bin/python tests/bench/round.py --list              # what each variant runs

Each model of a variant runs as `LABEL=<variant> WORKERS=<workers> EXTRA="<flags>" train_speed.sh <model>`, one
after another, or all at once for a `parallel: true` variant (each with its own LABEL <variant>/<model>). At the end
it prints every result line, with the baseline line of the same model under it when logs/bench/baseline has one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def load(path: Path = HERE / "variants.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def extra_flags(flags: dict) -> str:
    from drtc.train import flag_value

    parts = []
    for k, v in (flags or {}).items():
        value = flag_value(v)
        if any(c.isspace() for c in value):
            raise SystemExit(f"flag {k}={value!r}: values with spaces cannot go through EXTRA")
        parts.append(f"--{k}={value}")
    return " ".join(parts)


def jobs(variant: str, spec: dict, workers: int = 8) -> list[tuple[str, dict]]:
    """(model, environment for train_speed.sh) per model of the variant."""
    out = []
    for model, m in (spec.get("models") or {}).items():
        m = m or {}
        label = f"{variant}/{model}" if spec.get("parallel") else variant
        out.append((model, {"LABEL": label, "WORKERS": str(m.get("workers", workers)),
                            "EXTRA": extra_flags(m.get("flags"))}))
    return out


def summary_line(label: str, model: str) -> str | None:
    f = REPO / "logs" / "bench" / label / f"speed_{model}.json"
    if not f.is_file():
        return None
    r = json.loads(f.read_text())
    return (f"step {r['step_s']:.3f} s, data {r.get('data_share', 0):.0%}, GPU {r.get('gpu_util_pct', '?')}%, "
            f"{r.get('smp/s', 0):.0f} samples/s, peak {r.get('peak_mem_gb')} GB, full run ≈ {r['projected_hours']} h")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    spec = load()
    names = a.variants or list(spec)
    unknown = [n for n in names if n not in spec]
    if unknown:
        print(f"unknown variants {unknown}; known: {list(spec)}", file=sys.stderr)
        return 1
    if a.list:
        for n in names:
            for model, env in jobs(n, spec[n]):
                print(f"{n}: {model}  workers {env['WORKERS']}  {env['EXTRA']}")
        return 0
    script = str(HERE / "train_speed.sh")
    for n in names:
        js = jobs(n, spec[n])
        print(f"### {n}: {', '.join(m for m, _ in js)}{' (at the same time)' if spec[n].get('parallel') else ''}",
              flush=True)
        procs = []
        for model, env in js:
            p = subprocess.Popen(["bash", script, model], cwd=REPO, env={**os.environ, **env})
            if spec[n].get("parallel"):
                procs.append(p)
            else:
                p.wait()
        for p in procs:
            p.wait()
    print("\n### results (baseline under each)")
    for n in names:
        for model, env in jobs(n, spec[n]):
            print(f"{n} {model}: {summary_line(env['LABEL'], model) or 'no result'}")
            base = summary_line("baseline", model)
            if base:
                print(f"{'':>{len(n)}} {'baseline':>{len(model)}}: {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
