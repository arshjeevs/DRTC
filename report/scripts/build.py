#!/usr/bin/env python3
"""Build the report's generated files from the result files: numbers, tables and figures.

Run from report/ with `make data` (or `../.venv/bin/python scripts/build.py`). Reads the laptop's copy of the results
store (results/: `lab eval report <eval> --pull` for robot evals and benches; the offline files) and the registry
(lab/), and writes report/generated/:

- numbers.tex      \\setR{key}{value} for every number the text quotes; \\R{key} prints it, or "(pending)"
- tab_*.tex        tables (booktabs), read by the chapters with \\gentable (missing: "pending")
- fig_*.pdf        figures for the report (colour only here, never in the text)
- slides/fig_*.png the same figures at slide size, for the presentation

Inputs that are missing are skipped, never invented: the report then shows "pending" where they would go. To add a
result later: pull it, add its numbers/table/figure below (one function each), run `make`.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from drtc.evals.metrics import fisher_p, mannwhitney_p, wilson  # noqa: E402  the official reports' statistics

HERE = Path(__file__).resolve().parents[1]  # report/
REPO = HERE.parent
RESULTS = Path(os.environ.get("DRTC_REPORT_RESULTS", REPO / "results"))  # override: tests
OUT = HERE / "generated"
SLIDES = OUT / "slides"

# --- what the report shows ------------------------------------------------------------------------------------------

MAIN_EVAL = "sort_one_drtc_main"  # the DRTC experiment
MODELS = [("smolvla_sort_one_v1_full", "smolvla", "SmolVLA"), ("groot_sort_one_v1_full", "groot", "GR00T N1.5")]
CONDS = [("baseline", "Baseline"), ("faults", "Baseline + faults"), ("drtc", "DRTC + faults"),
         ("smooth", "DRTC + smoothing + faults")]
MAIN_CONDS = {"baseline": "baseline", "baseline_faults": "faults", "drtc_faults": "drtc",
              "drtc_smooth_faults": "smooth"}
BENCH_CONDS = {"baseline": "baseline", "baseline_fault": "faults", "drtc_fault": "drtc",
               "drtc_smooth_fault": "smooth"}
FAULTS = [("spike", "Latency spike"), ("drops", "Dropped replies"), ("outage", "Outage"), ("mix", "Fault mix")]
FAULT_WINDOWS = [(2.0, 3.0, "spike"), (4.0, 5.0, "drops"), (6.0, 7.5, "outage")]  # the fault mix, s from the start
OFFLINE = [("so101_sort_one_v1", "sortone", "Sorting"), ("so101_cubebox_v0", "cubeboxvzero", "Cube in box v0"),
           ("so101_cubebox_v1", "cubeboxvone", "Cube in box v1")]
POLICY_NAMES = {"smolvla": "SmolVLA", "groot": "GR00T N1.5", "grootn17": "GR00T N1.7", "pi05": "$\\pi_{0.5}$",
                "molmoact2": "MolmoAct 2", "act": "ACT", "diffusion": "Diffusion Policy"}
POLICY_PLAIN = {"smolvla": "SmolVLA", "groot": "GR00T N1.5", "grootn17": "GR00T N1.7", "pi05": "π0.5",
                "molmoact2": "MolmoAct 2", "act": "ACT", "diffusion": "Diffusion Policy"}

# Categorical slots in fixed order (dataviz reference palette, validated on white: all checks pass; aqua and yellow
# are below 3:1 contrast, so every figure carries visible value labels and has a table twin in the report).
COLORS = {"baseline": "#2a78d6", "faults": "#eb6834", "drtc": "#1baf7a", "smooth": "#eda100"}
SINGLE = "#2a78d6"
INK, INK2, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"

STYLES = {
    "report": {"width": 6.0, "font": 8.0, "ext": "pdf", "out": OUT, "dot": 16, "lw": 1.4},
    "slides": {"width": 12.0, "font": 15.0, "ext": "png", "out": SLIDES, "dot": 60, "lw": 2.4},
}

NUM: dict[str, str] = {}


# --- helpers ----------------------------------------------------------------------------------------------------------

def setn(key: str, value: str) -> None:
    NUM[key] = value


def med(series) -> float:
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return float(s.median()) if len(s) else float("nan")


def q(series) -> tuple[float, float]:
    s = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return (float(s.quantile(0.25)), float(s.quantile(0.75))) if len(s) else (float("nan"), float("nan"))


def fmt(x: float, d: int = 1, unit: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{d}f}{unit}"


def pval(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}\\%" if n else "--"


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text)
    print(f"  {name}")


def style_axes(st: dict) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": st["font"], "axes.titlesize": st["font"],
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": st["font"],
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8, "xtick.color": INK2,
        "ytick.color": INK2, "xtick.labelsize": st["font"] * 0.92, "ytick.labelsize": st["font"] * 0.92,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": st["font"] * 0.95, "text.color": INK, "pdf.fonttype": 42, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save(fig, name: str, st: dict) -> None:
    st["out"].mkdir(parents=True, exist_ok=True)
    path = st["out"] / f"{name}.{st['ext']}"
    fig.savefig(path, dpi=160 if st["ext"] == "png" else None)
    plt.close(fig)
    print(f"  {path.relative_to(HERE)}")


def cond_legend(fig, st: dict, conds=CONDS, top: float = 1.02) -> None:
    handles = [Patch(facecolor=COLORS[c], edgecolor="none", label=f"{i + 1}  {name}")
               for i, (c, name) in enumerate(conds)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, top), ncol=len(handles),
               handlelength=1.0, columnspacing=1.4)


# --- the DRTC experiment ----------------------------------------------------------------------------------------------

def load_main():
    f = RESULTS / "evals" / MAIN_EVAL / "metrics.csv"
    if not f.is_file():
        print(f"  (no {f}: the DRTC experiment stays pending)")
        return None
    df = pd.read_csv(f)
    df["c"] = df["condition"].map(MAIN_CONDS)
    df["m"] = df["model"].map({full: key for full, key, _ in MODELS})
    return df.dropna(subset=["c", "m"])


def main_numbers(df) -> dict:
    summ = {}
    for (m, c), g in df.groupby(["m", "c"]):
        n, k, f = len(g), int(g["M1_success"].sum()), int(g["M1f_fault"].sum())
        lo, hi = wilson(k, n)
        s = {"n": n, "k": k, "f": f, "lo": lo, "hi": hi, "g": g}
        for col in ["M1t_time_to_success_s", "M2_stall_pct", "M3_stall_events", "M3_longest_stall_s",
                    "M4_turnaround_med_ms", "M4_turnaround_p95_ms", "M5_estimate_err_ms", "M6_server_ms",
                    "M7_jerk_k", "M8b_largest_step_deg"]:
            s[col] = med(g[col])
            s[col + "_iqr"] = q(g[col])
        s["net"] = med(g["M4_turnaround_med_ms"] - g["M6_server_ms"])
        summ[(m, c)] = s
        key = f"main/{m}/{c}"
        setn(f"{key}/M1", pct(k, n))
        setn(f"{key}/M1k", f"{k}/{n}")
        setn(f"{key}/M1ci", f"{100 * lo:.0f}--{100 * hi:.0f}\\%")
        setn(f"{key}/M1f", f"{f}/{n}")
        setn(f"{key}/M1t", fmt(s["M1t_time_to_success_s"], 1, "~s"))
        setn(f"{key}/M2", fmt(s["M2_stall_pct"], 1, "\\%"))
        setn(f"{key}/M3n", fmt(s["M3_stall_events"], 0))
        setn(f"{key}/M3long", fmt(s["M3_longest_stall_s"], 2, "~s"))
        setn(f"{key}/M4med", fmt(s["M4_turnaround_med_ms"], 0, "~ms"))
        setn(f"{key}/M4p95", fmt(s["M4_turnaround_p95_ms"], 0, "~ms"))
        setn(f"{key}/M5", fmt(s["M5_estimate_err_ms"], 0, "~ms"))
        setn(f"{key}/M6", fmt(s["M6_server_ms"], 0, "~ms"))
        setn(f"{key}/M7", fmt(s["M7_jerk_k"], 1))
        setn(f"{key}/M8b", fmt(s["M8b_largest_step_deg"], 1, "$^\\circ$"))
        setn(f"{key}/net", fmt(s["net"], 0, "~ms"))
    for _, m, _ in MODELS:
        for a, b in [("baseline", "faults"), ("faults", "drtc"), ("baseline", "drtc"), ("drtc", "smooth"),
                     ("baseline", "smooth")]:
            if (m, a) not in summ or (m, b) not in summ:
                continue
            sa, sb = summ[(m, a)], summ[(m, b)]
            setn(f"main/{m}/p/{a}-{b}/M1", pval(fisher_p(sa["k"], sa["n"], sb["k"], sb["n"])))
            for col, short in [("M2_stall_pct", "M2"), ("M3_longest_stall_s", "M3long"), ("M7_jerk_k", "M7"),
                               ("M8b_largest_step_deg", "M8b"), ("M4_turnaround_p95_ms", "M4p95"),
                               ("M5_estimate_err_ms", "M5")]:
                p = mannwhitney_p(sa["g"][col].dropna().tolist(), sb["g"][col].dropna().tolist())
                setn(f"main/{m}/p/{a}-{b}/{short}", pval(p))
        if (m, "drtc") in summ and (m, "smooth") in summ:
            r = summ[(m, "drtc")]["M7_jerk_k"] / summ[(m, "smooth")]["M7_jerk_k"]
            setn(f"main/{m}/jerkratio", fmt(r, 1))
    setn("main/n", str(len(df)))
    trials = RESULTS / "evals" / MAIN_EVAL / "trials.csv"
    if trials.is_file():
        t = pd.read_csv(trials)
        setn("main/notcounted", str(int((t["valid"] == 0).sum())))
        setn("main/sessions", str(t["session"].nunique()))
    return summ


def main_tables(summ: dict) -> None:
    rows = []
    for i, (c, name) in enumerate(CONDS):
        cells = [f"{i + 1}~~{name}"]
        for _, m, _ in MODELS:
            s = summ.get((m, c))
            cells += ([f"{s['k']}/{s['n']} ({pct(s['k'], s['n'])})", f"{100 * s['lo']:.0f}--{100 * s['hi']:.0f}\\%",
                       f"{s['f']}/{s['n']}"] if s else ["--"] * 3)
        rows.append(" & ".join(cells) + " \\\\")
    names = [n for _, _, n in MODELS]
    write("tab_main_success.tex", "\n".join([
        "\\begin{tabular}{l ccc ccc}", "\\toprule",
        f" & \\multicolumn{{3}}{{c}}{{{names[0]}}} & \\multicolumn{{3}}{{c}}{{{names[1]}}} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Condition & Success & 95\\% CI & Fault-ended & Success & 95\\% CI & Fault-ended \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", ""]))

    rows = []
    for _, m, mname in MODELS:
        for i, (c, _) in enumerate(CONDS):
            s = summ.get((m, c))
            if not s:
                continue
            rows.append(" & ".join([
                mname if i == 0 else "", str(i + 1), fmt(s["M2_stall_pct"], 1), fmt(s["M3_longest_stall_s"], 2),
                f"{fmt(s['M4_turnaround_med_ms'], 0)} / {fmt(s['M4_turnaround_p95_ms'], 0)}",
                fmt(s["M6_server_ms"], 0), fmt(s["M5_estimate_err_ms"], 0), fmt(s["M7_jerk_k"], 1),
                fmt(s["M8b_largest_step_deg"], 1), fmt(s["M1t_time_to_success_s"], 1)]) + " \\\\")
        rows.append("\\addlinespace")
    write("tab_main_motion.tex", "\n".join([
        "\\begin{tabular}{l c rrrrrrrr}", "\\toprule",
        "Model & Cond. & Stall & Longest & Turnaround & Inference & Estimate & Jerk & Largest & Time to \\\\",
        " & & (\\%) & stall (s) & med./p95 (ms) & (ms) & error (ms) & ($10^3\\,^\\circ$/s$^3$) & step ($^\\circ$) "
        "& success (s) \\\\", "\\midrule", *rows[:-1], "\\bottomrule", "\\end{tabular}", ""]))

    def cell(m, a, b, metric):
        return NUM.get(f"main/{m}/p/{a}-{b}/{metric}", "--")

    def rate(m, c):
        return NUM.get(f"main/{m}/{c}/M1", "--")

    def val(m, c, metric):
        return NUM.get(f"main/{m}/{c}/{metric}", "--")

    lines = []
    for a, b, label in [("baseline", "faults", "2 vs 1: faults vs.\\ normal network"),
                        ("faults", "drtc", "3 vs 2: DRTC vs.\\ baseline, under faults"),
                        ("baseline", "drtc", "3 vs 1: DRTC under faults vs.\\ normal network"),
                        ("drtc", "smooth", "4 vs 3: smoothing vs.\\ DRTC alone")]:
        cells = [label, "Success"]
        for _, m, _ in MODELS:
            cells.append(f"{rate(m, b)} vs.\\ {rate(m, a)}, {cell(m, a, b, 'M1')}")
        lines.append(" & ".join(cells) + " \\\\")
    for metric, label in [("M7", "Jerk"), ("M8b", "Largest step"), ("M2", "Stall fraction"),
                          ("M3long", "Longest stall")]:
        cells = ["", label]
        for _, m, _ in MODELS:
            cells.append(f"{val(m, 'smooth', metric)} vs.\\ {val(m, 'drtc', metric)}, {cell(m, 'drtc', 'smooth', metric)}")
        lines.append(" & ".join(cells) + " \\\\")
    write("tab_main_tests.tex", "\n".join([
        "\\begin{tabular}{l l l l}", "\\toprule",
        f"Comparison & Metric & {MODELS[0][2]} & {MODELS[1][2]} \\\\", "\\midrule",
        *lines[:4], "\\addlinespace", *lines[4:], "\\bottomrule", "\\end{tabular}", ""]))


def main_figures(df, summ: dict, st: dict) -> None:
    style_axes(st)
    w = st["width"]
    # Success with Wilson intervals
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.40), sharey=True)
    for ax, (_, m, mname) in zip(axes, MODELS):
        for i, (c, _) in enumerate(CONDS):
            s = summ.get((m, c))
            if not s:
                continue
            r = 100 * s["k"] / s["n"]
            ax.bar(i, r, width=0.72, color=COLORS[c], edgecolor="white", linewidth=1.5, zorder=2)
            ax.errorbar(i, r, yerr=[[r - 100 * s["lo"]], [100 * s["hi"] - r]], fmt="none", ecolor=INK2,
                        elinewidth=0.9, capsize=3, zorder=3)
            ax.text(i, 100 * s["hi"] + 2.5, f"{r:.0f}%", ha="center", va="bottom", color=INK,
                    fontsize=st["font"] * 0.95, fontweight="bold")
        ax.set_title(mname)
        ax.set_xticks(range(len(CONDS)), [str(i + 1) for i in range(len(CONDS))])
        ax.set_xlabel("Condition")
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Task success (%)")
    cond_legend(fig, st)
    save(fig, "fig_main_success", st)

    # Per-trial strip plots
    for col, ylabel, name in [("M7_jerk_k", "Jerk ($10^3$ deg/s$^3$)", "fig_main_jerk"),
                              ("M2_stall_pct", "Stall fraction (% of ticks)", "fig_main_stall")]:
        fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.40), sharey=True)
        rng = np.random.default_rng(0)
        for ax, (_, m, mname) in zip(axes, MODELS):
            for i, (c, _) in enumerate(CONDS):
                s = summ.get((m, c))
                if not s:
                    continue
                ys = s["g"][col].dropna().to_numpy()
                xs = i + rng.uniform(-0.17, 0.17, len(ys))
                ax.scatter(xs, ys, s=st["dot"], color=COLORS[c], edgecolor="white", linewidth=0.6, zorder=3)
                mid = float(np.median(ys)) if len(ys) else float("nan")
                ax.plot([i - 0.3, i + 0.3], [mid, mid], color=INK, lw=st["lw"], zorder=4)
                ax.text(i + 0.33, mid, f"{mid:.1f}", va="center", ha="left", color=INK, fontsize=st["font"] * 0.85)
            ax.set_title(mname)
            ax.set_xticks(range(len(CONDS)), [str(i + 1) for i in range(len(CONDS))])
            ax.set_xlim(-0.5, len(CONDS) - 0.2)
            ax.set_xlabel("Condition")
            ax.grid(axis="x", visible=False)
        top = max(float(np.nanmax(s["g"][col])) for s in summ.values())  # one scale for both panels
        axes[0].set_ylim(0, top * 1.08)
        axes[0].set_ylabel(ylabel)
        cond_legend(fig, st)
        save(fig, name, st)

    # One trial per condition over time: a commanded joint, with the fault windows
    fig = timeline(df, st)
    if fig is not None:
        save(fig, "fig_main_timeline", st)


def trial_dir(row) -> Path:
    cond = {v: k for k, v in MAIN_CONDS.items()}[row["c"]]
    return (RESULTS / "evals" / MAIN_EVAL / "sessions" / row["session"] / "trials" / cond / row["model"]
            / f"t{int(row['trial']):02d}")


def timeline(df, st: dict):
    m_full, m, mname = MODELS[0]
    picks = {}
    for c, _ in CONDS:
        g = df[(df["m"] == m) & (df["c"] == c)].copy()
        if g.empty:
            return None
        if c == "faults" and (g["M1f_fault"] == 1).any():
            g = g[g["M1f_fault"] == 1]  # the typical outcome under faults: the shipped system gives up
        g["d"] = (g["M7_jerk_k"] - g["M7_jerk_k"].median()).abs()
        picks[c] = g.sort_values("d").iloc[0]
    ticks = {c: pd.read_csv(trial_dir(r) / "ticks.csv") for c, r in picks.items()}
    joints = [j for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
              if f"cmd.{j}" in ticks["baseline"]]
    joint = max(joints, key=lambda j: np.nanmax(ticks["baseline"][f"cmd.{j}"]) - np.nanmin(ticks["baseline"][f"cmd.{j}"]))
    w = st["width"]
    fig, axes = plt.subplots(len(CONDS), 1, figsize=(w, w * 0.62), sharex=True, sharey=True)
    tmax = 10.0
    for ax, (c, name) in zip(axes, CONDS):
        t = ticks[c]
        y = t[f"cmd.{joint}"].ffill()  # the arm holds its last commanded position through a stall
        if c != "baseline":
            for a, b, label in FAULT_WINDOWS:
                ax.axvspan(a, b, color=BAND, zorder=0, lw=0)
                if c == "faults":
                    ax.text((a + b) / 2, 1.02, label, transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", color=INK2, fontsize=st["font"] * 0.85)
        ax.step(t["t"], y, where="post", color=COLORS[c], lw=st["lw"], zorder=3)
        stalled = t["action"] == 0
        if stalled.any():
            ax.scatter(t.loc[stalled, "t"], y[stalled], s=st["dot"] * 0.5, color=INK2, zorder=4, marker="|")
        if picks[c]["M1f_fault"] == 1:
            ax.scatter([t["t"].iloc[-1]], [y.iloc[-1]], marker="x", s=st["dot"] * 2.2, color=INK, zorder=5)
            ax.text(t["t"].iloc[-1] + 0.15, y.iloc[-1], "run ended by a fault", va="center", color=INK,
                    fontsize=st["font"] * 0.85)
        ax.set_ylabel(f"{CONDS.index((c, name)) + 1}", rotation=0, labelpad=12, va="center", color=INK,
                      fontweight="bold")
        ax.grid(axis="x", visible=False)
        tmax = max(tmax, min(float(t["t"].iloc[-1]), 14.0))
    axes[-1].set_xlim(0, tmax)
    axes[-1].set_xlabel("Time from the trial start (s)")
    fig.supylabel(f"{mname}: commanded {joint.replace('_', ' ')} (deg)", color=INK2, fontsize=st["font"] * 0.95,
                  x=-0.02)
    cond_legend(fig, st, top=1.0)
    fig.subplots_adjust(hspace=0.35)
    return fig


# --- the bench ----------------------------------------------------------------------------------------------------------

def load_bench():
    out = {}
    for fault, _ in FAULTS:
        f = RESULTS / "bench" / f"bench_main_{fault}" / "metrics.csv"
        if not f.is_file():
            print(f"  (no {f})")
            continue
        df = pd.read_csv(f)
        df["c"] = df["condition"].map(BENCH_CONDS)
        df["m"] = df["model"].map({full: key for full, key, _ in MODELS})
        out[fault] = df.dropna(subset=["c", "m"])
    return out


def bench_numbers(bench: dict) -> dict:
    summ = {}
    for fault, df in bench.items():
        for (m, c), g in df.groupby(["m", "c"]):
            s = {"n": len(g), "f": int(g["M1f_fault"].sum()), "M2": med(g["M2_stall_pct"]),
                 "M3long": med(g["M3_longest_stall_s"]), "M7": med(g["M7_jerk_k"]),
                 "M8b": med(g["M8b_largest_step_deg"])}
            summ[(fault, m, c)] = s
            key = f"bench/{fault}/{m}/{c}"
            setn(f"{key}/M1f", f"{s['f']}/{s['n']}")
            setn(f"{key}/M2", fmt(s["M2"], 1, "\\%"))
            setn(f"{key}/M3long", fmt(s["M3long"], 2, "~s"))
            setn(f"{key}/M7", fmt(s["M7"], 1))
            setn(f"{key}/M8b", fmt(s["M8b"], 1, "$^\\circ$"))
        setn(f"bench/{fault}/n", str(len(df)))
    return summ


def bench_table(summ: dict) -> None:
    rows = []
    for fault, fname in FAULTS:
        first = True
        for _, m, mname in MODELS:
            if not any(k[0] == fault and k[1] == m for k in summ):
                continue
            g = {c: summ.get((fault, m, c)) for c, _ in CONDS}
            ended = [f"{g[c]['f']}/{g[c]['n']}" if g[c] else "--" for c in ("faults", "drtc", "smooth")]
            stall = [fmt(g[c]["M2"], 1) if g[c] else "--" for c, _ in CONDS]
            jerk = [fmt(g[c]["M7"], 1) if g[c] else "--" for c in ("drtc", "smooth")]
            rows.append(" & ".join([fname if first else "", mname, *ended, *stall, *jerk]) + " \\\\")
            first = False
        rows.append("\\addlinespace")
    if not rows:
        return
    write("tab_bench.tex", "\n".join([
        "\\begin{tabular}{l l ccc rrrr rr}", "\\toprule",
        " & & \\multicolumn{3}{c}{Runs ended by a fault} & \\multicolumn{4}{c}{Stall fraction (\\%)} & "
        "\\multicolumn{2}{c}{Jerk ($10^3\\,^\\circ$/s$^3$)} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-11}",
        "Fault & Model & 2 & 3 & 4 & 1 & 2 & 3 & 4 & 3 & 4 \\\\", "\\midrule", *rows[:-1], "\\bottomrule",
        "\\end{tabular}", ""]))


def bench_figure(summ: dict, st: dict) -> None:
    faults = [(f, n) for f, n in FAULTS if any(k[0] == f for k in summ)]
    if not faults:
        return
    style_axes(st)
    w = st["width"]
    fig, axes = plt.subplots(2, len(faults), figsize=(w, w * 0.55), sharey="row")
    axes = np.atleast_2d(axes)
    width = 0.19
    for j, (fault, fname) in enumerate(faults):
        for row, (metric, ylabel) in enumerate([("f", "Runs ended\nby a fault (%)"), ("M7", "Jerk\n($10^3$ deg/s$^3$)")]):
            ax = axes[row, j]
            for mi, (_, m, mname) in enumerate(MODELS):
                for ci, (c, _) in enumerate(CONDS):
                    s = summ.get((fault, m, c))
                    if not s:
                        continue
                    v = 100 * s["f"] / s["n"] if metric == "f" else s["M7"]
                    x = mi + (ci - 1.5) * width
                    ax.bar(x, v, width=width, color=COLORS[c], edgecolor="white", linewidth=1.0, zorder=2)
                    if metric == "f" and c != "baseline":
                        ax.text(x, v + 2, f"{s['f']}", ha="center", va="bottom", fontsize=st["font"] * 0.75,
                                color=INK)
            ax.set_xticks(range(len(MODELS)), [n.split()[0] for _, _, n in MODELS])
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(fname)
                ax.set_ylim(0, 112)
            if j == 0:
                ax.set_ylabel(ylabel)
    cond_legend(fig, st)
    fig.subplots_adjust(hspace=0.45)
    save(fig, "fig_bench", st)


# --- offline ------------------------------------------------------------------------------------------------------------

def policy_of(model: str) -> str:
    return model.split("_")[0]


def load_offline() -> list[dict]:
    out = []
    for ds, dkey, dname in OFFLINE:
        for f in sorted((RESULTS / "offline" / ds).glob("*.json")) if (RESULTS / "offline" / ds).is_dir() else []:
            r = json.loads(f.read_text())
            root = r["checkpoints"].get("root", {})
            last = next((v for k, v in r["checkpoints"].items() if k.startswith("last")), None)
            m10 = root.get("M10", {})
            out.append({"ds": ds, "dkey": dkey, "dname": dname, "model": r["model"], "policy": policy_of(r["model"]),
                        "units": r.get("units", ""), "arm": root.get("M9_arm_mae"), "grip": root.get("M9_gripper_mae"),
                        "arm_last": (last or {}).get("M9_arm_mae"), "ms": m10.get("server_ms_median"),
                        "p95": m10.get("server_ms_p95"), "gb": m10.get("peak_gpu_gb"), "gpu": r.get("gpu"),
                        "pred": r.get("predictions"), "held": len(r.get("held_out_episodes", []))})
    order = list(POLICY_NAMES)
    out.sort(key=lambda r: ([d for d, _, _ in OFFLINE].index(r["ds"]), order.index(r["policy"])
                            if r["policy"] in order else 99))
    return out


def offline_outputs(rows: list[dict], sts: list[dict]) -> None:
    if not rows:
        return
    for r in rows:
        key = f"offline/{r['dkey']}/{r['policy']}"
        setn(f"{key}/M9arm", fmt(r["arm"], 2))
        setn(f"{key}/M9grip", fmt(r["grip"], 2))
        setn(f"{key}/M10", fmt(r["ms"], 0, "~ms"))
        setn(f"{key}/M10p95", fmt(r["p95"], 0, "~ms"))
        setn(f"{key}/M10gb", fmt(r["gb"], 1, "~GB"))
    setn("offline/gpu", rows[0]["gpu"] or "--")
    lines, last_ds = [], None
    for r in rows:
        if last_ds and r["ds"] != last_ds:
            lines.append("\\addlinespace")
        lines.append(" & ".join([
            r["dname"] if r["ds"] != last_ds else "", POLICY_NAMES.get(r["policy"], r["policy"]),
            f"{r['held']} / {r['pred']}", fmt(r["arm"], 2), fmt(r["arm_last"], 2), fmt(r["grip"], 2),
            fmt(r["ms"], 0), fmt(r["p95"], 0), fmt(r["gb"], 1)]) + " \\\\")
        last_ds = r["ds"]
    write("tab_offline.tex", "\n".join([
        "\\begin{tabular}{l l c rrr rrr}", "\\toprule",
        " & & Held-out & \\multicolumn{3}{c}{M9 held-out error} & \\multicolumn{3}{c}{M10 inference} \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
        "Dataset & Model & episodes / predictions & arm (served) & arm (last) & gripper & median (ms) & p95 (ms) & "
        "memory (GB) \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}", ""]))

    for st in sts:
        style_axes(st)
        w = st["width"]
        fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.42), gridspec_kw={"width_ratios": [1, 1.25]})
        sort = [r for r in rows if r["ds"] == OFFLINE[0][0] and r["arm"] is not None]
        ax = axes[0]
        ys = np.arange(len(sort))[::-1]
        for y, r in zip(ys, sort):
            ax.plot([0, r["arm"]], [y, y], color=GRID, lw=st["lw"], zorder=1)
            ax.scatter([r["arm"]], [y], s=st["dot"] * 2.5, color=SINGLE, edgecolor="white", linewidth=1, zorder=3)
            ax.text(r["arm"] + 0.08, y, f"{r['arm']:.2f}", va="center", fontsize=st["font"] * 0.85, color=INK)
        ax.set_yticks(ys, [POLICY_PLAIN.get(r["policy"], r["policy"]) for r in sort])
        ax.set_xlim(0, max([r["arm"] for r in sort] + [1]) * 1.25)
        ax.set_xlabel("Held-out arm error, sorting (deg)")
        ax.set_title("M9: action error")
        ax.grid(axis="y", visible=False)
        lab = [r for r in rows if r["ms"] is not None]
        ax = axes[1]
        ys = np.arange(len(lab))[::-1]
        for y, r in zip(ys, lab):
            ax.plot([r["ms"], r["p95"]], [y, y], color=AXIS, lw=st["lw"], zorder=1)
            ax.scatter([r["ms"]], [y], s=st["dot"] * 2.5, color=SINGLE, edgecolor="white", linewidth=1, zorder=3)
            ax.text(max(r["p95"], r["ms"]) + 6, y, f"{r['ms']:.0f}", va="center", fontsize=st["font"] * 0.85,
                    color=INK)
        ax.set_yticks(ys, [f"{POLICY_PLAIN.get(r['policy'], r['policy'])} ({r['dname']})" for r in lab])
        ax.set_xlim(0, max(r["p95"] for r in lab) * 1.2)
        ax.set_xlabel(f"Inference per chunk (ms), median to p95, {rows[0]['gpu'] or 'GPU'}")
        ax.set_title("M10: inference time")
        ax.grid(axis="y", visible=False)
        fig.subplots_adjust(wspace=0.75 if st["ext"] == "pdf" else 0.6)
        save(fig, "fig_offline", st)


# --- datasets (from the registry) -----------------------------------------------------------------------------------------

def datasets_table() -> None:
    from drtc.registry import load_registry

    reg = load_registry()
    rows = []
    for ds, _, dname in OFFLINE:
        d = reg.datasets.get(ds)
        if d is None:
            continue
        held = d.split.held_out if d.split else 0
        task = d.task.replace("&", "\\&")
        rows.append(" & ".join([f"\\texttt{{{ds.replace('_', chr(92) + '_')}}}", task, f"{d.episodes} ({d.episodes - held} + {held})",
                                f"{d.frames:,}".replace(",", "{,}"), "degrees" if d.use_degrees else "$-100$..$100$"]) + " \\\\")
        setn(f"data/{ds}/episodes", str(d.episodes))
        setn(f"data/{ds}/held", str(held))
        setn(f"data/{ds}/frames", f"{d.frames:,}".replace(",", "{,}"))
    write("tab_datasets.tex", "\n".join([
        "\\begin{tabularx}{\\textwidth}{l X r r l}", "\\toprule",
        "Dataset & Task instruction & Episodes (train + held out) & Frames & Joint units \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabularx}", ""]))


# --- main ----------------------------------------------------------------------------------------------------------------

def project_facts() -> None:
    """Facts that are not experiment results: the size of the test suite (counted from the code) and training-speed
    measurements made while setting up the trainings (2026-11-19, lab/docs/history/PLAN-2026-11-18-to-20.md, Phase 3)."""
    import re

    tests = sum(len(re.findall(r"^def test_", f.read_text(), re.M)) for f in (REPO / "tests").glob("test_*.py"))
    setn("software/tests", str(tests))
    setn("train/samples_per_s", "555")       # uint8 images, 8 workers, batch 64, so101_cubebox_v1 (laptop)
    setn("train/smolvla/h_before", "1.8~h")  # H200, SmolVLA recipe, without compile
    setn("train/smolvla/h_after", "1.1~h")   # with policy.compile_model


def main() -> int:
    os.chdir(REPO)  # the registry is found from the repository root
    OUT.mkdir(parents=True, exist_ok=True)
    print("report/generated:")
    datasets_table()
    df = load_main()
    if df is not None:
        summ = main_numbers(df)
        main_tables(summ)
        for st in STYLES.values():
            main_figures(df, summ, st)
    bench = load_bench()
    if bench:
        bsumm = bench_numbers(bench)
        bench_table(bsumm)
        for st in STYLES.values():
            bench_figure(bsumm, st)
    offline_outputs(load_offline(), list(STYLES.values()))
    import sys

    sys.path.insert(0, str(HERE / "scripts"))
    import v2_results

    for stale in [*OUT.glob("tab_v2_*.tex"), *OUT.glob("tab_base_*.tex"), OUT / "tab_e4.tex"]:
        stale.unlink(missing_ok=True)  # written again below only if their results exist
    v2_results.main_recovery(df, MODELS, CONDS, write, setn)
    v2_results.bench_v2(RESULTS, write, setn, POLICY_NAMES)
    v2_results.robot_v2(RESULTS, write, setn, POLICY_NAMES, wilson, fisher_p)
    project_facts()
    lines = ["% Generated by report/scripts/build.py from results/ -- do not edit; rebuild with `make`.",
             *[f"\\setR{{{k}}}{{{v}}}" for k, v in sorted(NUM.items())], ""]
    write("numbers.tex", "\n".join(lines))
    print(f"{len(NUM)} numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
