"""Summarise robot trials (results/) and inference benchmarks (results/benchmark/) into tables and figures.

Per trial (one metrics CSV, one row per control tick):
    stall_pct          % of ticks with an empty action queue, counted after the first chunk arrived
    stall_events       number of separate stalls (runs of empty-queue ticks)
    latency_ms         median / p95 measured round-trip latency (observation sent -> chunk received)
    estimate_error_ms  median (latency estimate - measured latency); > 0 = estimator is conservative
    chunk_l2           median disagreement between consecutive chunks where they overlap
Per experiment x model: trials, labelled trials, success rate with a Wilson 95% interval, and the median
of each trial metric over trials.

Writes results/summary/{summary.csv, summary.md, success.png, stalls.png, latency.png, benchmark.md}.
`--since` drops older trials (e.g. pre-validation runs); `--timeline <run_id>` plots one trial.

Usage:
    python -m lab.tools.analyze [--since 2026-11-15] [--model M ...] [--experiment E ...] [--timeline RUN_ID]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from lab.tools.config import RESULTS_DIR

SUMMARY_DIR = RESULTS_DIR / "summary"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate (well-behaved for small n and 0% / 100%)."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def trial_metrics(csv: Path) -> dict:
    df = pd.read_csv(csv)
    received = df.index[df["action_received"] == 1]
    running = df.loc[received[0]:] if len(received) else df.iloc[0:0]  # skip start-up before the first chunk
    stalls = running["stall"].astype(int)
    measured = df["measured_latency_ms"].dropna()
    at_arrival = df.loc[df["measured_latency_ms"].notna()]
    return {
        "stall_pct": 100 * stalls.mean() if len(stalls) else math.nan,
        "stall_events": int(((stalls.diff() == 1) | ((stalls.index == stalls.index.min()) & (stalls == 1))).sum()),
        "latency_ms_median": measured.median(),
        "latency_ms_p95": measured.quantile(0.95) if len(measured) else math.nan,
        "estimate_error_ms": (at_arrival["latency_estimate_ms"] - at_arrival["measured_latency_ms"]).median(),
        "chunk_l2": df["chunk_mean_l2"].dropna().median(),
        "chunks": len(measured),
    }


def load_trials(args) -> pd.DataFrame:
    index = pd.read_csv(RESULTS_DIR / "index.csv", dtype={"success": "string"})
    index["started_at"] = pd.to_datetime(index["started_at"])
    if args.since:
        index = index[index["started_at"] >= pd.Timestamp(args.since)]
    if args.model:
        index = index[index["model"].isin(args.model)]
    if args.experiment:
        index = index[index["experiment"].isin(args.experiment)]
    rows = []
    for row in index.itertuples():
        csv = RESULTS_DIR / row.experiment / row.model / row.run_id / f"{row.run_id}.csv"
        if not csv.is_file():
            continue
        rows.append({"run_id": row.run_id, "experiment": row.experiment, "model": row.model,
                     "success": row.success, "started_at": row.started_at, **trial_metrics(csv)})
    return pd.DataFrame(rows)


def summarise(trials: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (experiment, model), g in trials.groupby(["experiment", "model"]):
        labelled = g[g["success"].isin(["0", "1"])]
        k, n = int((labelled["success"] == "1").sum()), len(labelled)
        lo, hi = wilson(k, n)
        out.append({
            "experiment": experiment, "model": model, "trials": len(g), "labelled": n, "successes": k,
            "success_rate": k / n if n else math.nan, "ci_low": lo, "ci_high": hi,
            **{m: g[m].median() for m in ("stall_pct", "stall_events", "latency_ms_median", "latency_ms_p95",
                                          "estimate_error_ms", "chunk_l2")},
        })
    return pd.DataFrame(out)


def markdown(summary: pd.DataFrame) -> str:
    lines = ["| Experiment | Model | Success (95% CI) | Trials | Stall % | Stalls | Latency ms (median / p95) | Estimate error ms | Chunk L2 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in summary.itertuples():
        success = (f"{r.successes}/{r.labelled} = {r.success_rate:.0%} ({r.ci_low:.0%}–{r.ci_high:.0%})"
                   if r.labelled else "not labelled")
        lines.append(f"| {r.experiment} | {r.model} | {success} | {r.trials} | {r.stall_pct:.1f} | {r.stall_events:.0f} | "
                     f"{r.latency_ms_median:.0f} / {r.latency_ms_p95:.0f} | {r.estimate_error_ms:+.0f} | {r.chunk_l2:.2f} |")
    return "\n".join(lines) + "\n"


def plot_summary(summary: pd.DataFrame, trials: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    experiments = sorted(summary["experiment"].unique())
    models = sorted(summary["model"].unique())
    width = 0.8 / max(1, len(models))
    x = range(len(experiments))

    labelled = summary[summary["labelled"] > 0]
    if len(labelled):
        shown = sorted(labelled["model"].unique())  # models with labelled trials only
        bar = 0.8 / len(shown)
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(experiments)) + 3, 4))
        for i, model in enumerate(shown):
            s = labelled[labelled["model"] == model].set_index("experiment").reindex(experiments)
            pos = [j + (i - (len(shown) - 1) / 2) * bar for j in x]
            err = [s["success_rate"] - s["ci_low"], s["ci_high"] - s["success_rate"]]
            ax.bar(pos, s["success_rate"], bar, yerr=err, capsize=3, label=model)
        ax.set_xticks(list(x), [e.split("/")[-1] for e in experiments], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("success rate (95% Wilson CI)")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        written.append(SUMMARY_DIR / "success.png")
        fig.savefig(written[-1], dpi=150)
        plt.close(fig)

    for metric, label, name in (("stall_pct", "stall % of control ticks", "stalls.png"),
                                ("latency_ms_median", "median round-trip latency (ms)", "latency.png")):
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(experiments)), 4))
        data, positions, ticks = [], [], []
        for j, experiment in enumerate(experiments):
            for i, model in enumerate(models):
                values = trials[(trials["experiment"] == experiment) & (trials["model"] == model)][metric].dropna()
                if len(values):
                    data.append(values)
                    positions.append(j + (i - (len(models) - 1) / 2) * width)
        if data:
            ax.boxplot(data, positions=positions, widths=width * 0.9)
            ax.set_xticks(list(x), [e.split("/")[-1] for e in experiments], rotation=30, ha="right")
            ax.set_ylabel(label)
            ax.set_title(" | ".join(models), fontsize=8)
            fig.tight_layout()
            written.append(SUMMARY_DIR / name)
            fig.savefig(written[-1], dpi=150)
        plt.close(fig)
    return written


def plot_timeline(run_id: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matches = list(RESULTS_DIR.glob(f"**/{run_id}/{run_id}.csv"))
    if not matches:
        raise SystemExit(f"No trial {run_id} under {RESULTS_DIR}")
    df = pd.read_csv(matches[0])
    t = df["t"] - df["t"].iloc[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax1.plot(t, df["latency_estimate_ms"], label="latency estimate")
    arrived = df["measured_latency_ms"].notna()
    ax1.scatter(t[arrived], df.loc[arrived, "measured_latency_ms"], s=10, color="C1", label="measured round trip")
    ax1.set_ylabel("ms")
    ax1.legend(fontsize=8)
    ax2.plot(t, df["schedule_size"], label="actions queued")
    ax2.fill_between(t, 0, df["schedule_size"].max(), where=df["stall"] == 1, color="red", alpha=0.15, label="stall")
    ax2.set_ylabel("actions")
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=8)
    fig.suptitle(str(matches[0].relative_to(RESULTS_DIR).parent), fontsize=9)
    fig.tight_layout()
    out = SUMMARY_DIR / f"timeline_{run_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def benchmark_table() -> str | None:
    records = [json.loads(p.read_text()) for p in sorted((RESULTS_DIR / "benchmark").glob("*/*.json"))]
    if not records:
        return None
    lines = ["| Model | Device | Variant | Steps | Chunk ms (median / p90) | Peak GB | Params |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(records, key=lambda r: (r["model"], r["device_name"], r["variant"])):
        peak = f"{r['peak_memory_gb']:.2f}" if r.get("peak_memory_gb") is not None else "-"
        lines.append(f"| {r['model']} | {r['device_name']} ({r['host']}) | {r['dtype']}{' compile' if r['compile'] else ''} | "
                     f"{r.get('denoising_steps') or '-'} | {r['total_ms_median']:.0f} / {r['total_ms_p90']:.0f} | {peak} | "
                     f"{r['params_m']:.0f}M |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise robot trials and benchmarks")
    parser.add_argument("--since", help="only trials started on/after this date, e.g. 2026-11-15")
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--experiment", nargs="+", help="e.g. replication/baseline replication/spikes_jk")
    parser.add_argument("--timeline", metavar="RUN_ID", help="plot one trial's latency and action queue")
    args = parser.parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    if args.timeline:
        print(f"wrote {plot_timeline(args.timeline)}")
        return

    if (RESULTS_DIR / "index.csv").is_file():
        trials = load_trials(args)
        if len(trials):
            summary = summarise(trials)
            summary.to_csv(SUMMARY_DIR / "summary.csv", index=False)
            trials.to_csv(SUMMARY_DIR / "trials.csv", index=False)
            table = markdown(summary)
            (SUMMARY_DIR / "summary.md").write_text(table)
            print(table)
            for path in plot_summary(summary, trials):
                print(f"wrote {path}")
        else:
            print("No trials match the filters.")
    bench = benchmark_table()
    if bench:
        (SUMMARY_DIR / "benchmark.md").write_text(bench)
        print("\n" + bench)
    print(f"tables and figures in {SUMMARY_DIR}/")


if __name__ == "__main__":
    main()
