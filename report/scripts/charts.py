"""The report's data charts (report plan, stage 3): datasets, training, inference cost, baselines, the Pi's round trip
and E4.

    python report/scripts/charts.py --fetch    # once: Hub and wandb -> report/data/*.csv (small, committed)
    make -C report data                        # build.py calls charts(); every chart is rebuilt from the files

Every chart reads only report/data/ and results/, so the report builds offline. A chart whose data is missing is not
written, and the report shows its slot instead.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]  # report/
REPO = HERE.parent
DATA = HERE / "data"

DATASETS = [("so101_cubebox_v0", "Cube in box v0"), ("so101_cubebox_v1", "Cube in box v1"),
            ("so101_sort_one_v1", "Sorting")]
TASKS = [("sort_one", "Sorting"), ("cubebox_v1", "Cube in box v1")]
# Two task colours, validated with the dataviz palette checker on white (all checks pass, contrast >= 3:1).
TASK_COLORS = {"sort_one": "#2a78d6", "cubebox_v1": "#9c7a1e"}
POLICIES = ["act", "diffusion", "smolvla", "groot", "grootn17", "pi05", "molmoact2"]
E4_POLICIES = ["grootn17", "smolvla", "diffusion", "pi05"]
E4_CONDS = [("upstream_patient__none", "baseline", "1  Patient LeRobot"),
            ("upstream_patient__mix", "faults", "2  + faults"),
            ("drtc__mix", "drtc", "3  DRTC + faults"),
            ("drtc_stack_nortc__mix", "smooth", "4  DRTC + stack + faults")]
STEPS_STOPPED = {"pi05": 15000, "molmoact2": 30000}  # stopped early (Chapter 5)


def model_name(p: str, task: str) -> str:
    ds = task if task.endswith("_v1") else f"{task}_v1"  # sort_one -> sort_one_v1; cubebox_v1 stays
    return f"{p}_{ds}_lora" if p == "molmoact2" else f"{p}_{ds}_full"


# --- fetch (network): Hub metadata and wandb histories -> report/data --------------------------------------------------

def fetch() -> None:
    import yaml
    from huggingface_hub import HfApi, hf_hub_download

    DATA.mkdir(exist_ok=True)
    reg = {}
    for f in (REPO / "lab" / "datasets").glob("*.yaml"):
        reg.update(yaml.safe_load(f.read_text()) or {})
    api = HfApi()
    rows = []
    for ds, _ in DATASETS:
        for copy, name in (("trimmed", ds), ("raw", ds + "_raw")):
            d = reg[name]
            files = [f for f in api.list_repo_files(d["repo"], repo_type="dataset", revision=d["revision"])
                     if f.startswith("meta/episodes/") and f.endswith(".parquet")]
            for f in files:
                p = hf_hub_download(d["repo"], f, repo_type="dataset", revision=d["revision"])
                e = pd.read_parquet(p, columns=["episode_index", "length"])
                rows += [{"dataset": ds, "copy": copy, "episode": int(i), "frames": int(n)}
                         for i, n in zip(e["episode_index"], e["length"])]
    pd.DataFrame(rows).to_csv(DATA / "episode_lengths.csv", index=False)
    print(f"  data/episode_lengths.csv ({len(rows)} episodes)")

    params = []
    for t, _ in TASKS:
        for p in POLICIES:
            repo = f"orangemoon599/{model_name(p, t)}"
            try:
                meta = api.get_safetensors_metadata(repo)
                n = sum(meta.parameter_count.values())
            except Exception as e:  # noqa: BLE001
                print(f"  {repo}: no safetensors metadata ({type(e).__name__})")
                continue
            params.append({"model": model_name(p, t), "policy": p, "task": t, "params": n})
    pd.DataFrame(params).to_csv(DATA / "model_params.csv", index=False)
    print(f"  data/model_params.csv ({len(params)} models)")

    import wandb

    wapi = wandb.Api(timeout=60)
    runs = {r.name.rsplit("-", 1)[0]: r for r in wapi.runs(f"{wapi.default_entity}/drtc-lab", per_page=200)}
    curves = []
    for t, _ in TASKS:
        for p in POLICIES:
            m = model_name(p, t)
            r = runs.get(m)
            if r is None:
                print(f"  wandb: no run for {m}")
                continue
            keys = [k for k in r.summary.keys() if re.search(r"(^|/)loss$", k)]
            key = "train/loss" if "train/loss" in keys else (keys[0] if keys else None)
            if key is None:
                print(f"  wandb: {m} has no loss key")
                continue
            for row in r.scan_history(keys=["_step", key]):
                if row.get(key) is not None:
                    curves.append({"model": m, "policy": p, "task": t, "step": row["_step"], "loss": row[key]})
    pd.DataFrame(curves).to_csv(DATA / "loss_curves.csv", index=False)
    print(f"  data/loss_curves.csv ({len(curves)} points)")


# --- charts ------------------------------------------------------------------------------------------------------------

def turnarounds(eval_name: str) -> pd.DataFrame:
    """The client's periodic 'turnaround' status lines of every trial of an evaluation (ms)."""
    rows = []
    root = REPO / "results" / "evals" / eval_name / "sessions"
    for f in glob.glob(str(root / "*/trials/*/*/t*/rollout.log")):
        model = Path(f).parents[1].name
        for v in re.findall(r"turnaround ([0-9.]+) ms", Path(f).read_text(errors="ignore")):
            rows.append({"model": model, "ms": float(v)})
    return pd.DataFrame(rows)


def charts(st: dict, b) -> None:
    """b: the build module (its style, colours, names, save)."""
    import matplotlib.pyplot as plt

    b.style_axes(st)
    names = b.POLICY_PLAIN

    # C4.1 datasets: episode length before and after trimming
    f = DATA / "episode_lengths.csv"
    if f.is_file():
        df = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(st["width"], st["width"] * 0.42))
        pos, labels, ypos = 0, [], []
        for ds, dname in DATASETS:
            for copy, col in (("raw", b.MUTED), ("trimmed", b.SINGLE)):
                s = df[(df["dataset"] == ds) & (df["copy"] == copy)]["frames"] / 30.0
                if s.empty:
                    continue
                bp = ax.boxplot(s, positions=[pos], widths=0.55, vert=False, patch_artist=True, showfliers=True,
                                medianprops={"color": "white", "lw": 1.6},
                                boxprops={"facecolor": col, "edgecolor": col},
                                whiskerprops={"color": col}, capprops={"color": col},
                                flierprops={"marker": "o", "ms": 3, "mfc": col, "mec": "none"})
                ax.text(s.max() + 0.6, pos, f"median {s.median():.1f} s, {len(s)} episodes", va="center",
                        fontsize=st["font"] * 0.85, color=b.INK2)
                labels.append(f"{dname}, {copy}")
                ypos.append(pos)
                pos += 1
            pos += 0.6
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("episode length (s)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, ax.get_xlim()[1] + 6)
        b.save(fig, "fig_datasets", st)

    # C5.1 training loss
    f = DATA / "loss_curves.csv"
    if f.is_file():
        df = pd.read_csv(f)
        fig, axes = plt.subplots(1, 2, figsize=(st["width"], st["width"] * 0.42), sharey=True)
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa8"]
        for ax, (t, tname) in zip(axes, TASKS):
            for i, p in enumerate(POLICIES):
                g = df[(df.task == t) & (df.policy == p)].sort_values("step")
                if g.empty:
                    continue
                y = g["loss"].rolling(max(1, len(g) // 60), min_periods=1).median()
                ax.plot(g["step"] / 1000, y, color=palette[i], lw=st["lw"] * 0.8, label=names[p])
            ax.set_yscale("log")
            ax.set_title(tname)
            ax.set_xlabel("training step (thousands)")
        axes[0].set_ylabel("training loss (log scale)")
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1.4)
        b.save(fig, "fig_losscurves", st)

    # C5.2 size against server inference time (robot baselines, both tasks)
    fp = DATA / "model_params.csv"
    if fp.is_file():
        par = pd.read_csv(fp)
        pts = []
        for t, _ in TASKS:
            mf = REPO / "results" / "evals" / f"{t}_baseline" / "metrics.csv"
            if not mf.is_file():
                continue
            m = pd.read_csv(mf)
            for p in POLICIES:
                g = m[m.model == model_name(p, t)]["M6_server_ms"].dropna()
                pr = par[(par.policy == p) & (par.task == t)]["params"]
                if len(g) and len(pr):
                    pts.append((p, t, float(pr.iloc[0]) / 1e9, float(g.median())))
        if pts:
            fig, ax = plt.subplots(figsize=(st["width"] * 0.75, st["width"] * 0.45))
            for t, tname in TASKS:
                xs = [x for p, tt, x, y in pts if tt == t]
                ys = [y for p, tt, x, y in pts if tt == t]
                ax.scatter(xs, ys, s=st["dot"] * 2.2, color=TASK_COLORS[t], edgecolor="white", lw=0.8,
                           label=tname, zorder=3)
            for p in POLICIES:
                q = [(x, y) for pp, tt, x, y in pts if pp == p]
                if q:
                    x, y = np.mean([a for a, _ in q]), max(c for _, c in q)
                    ax.annotate(names[p], (x, y), xytext=(5, 4), textcoords="offset points",
                                fontsize=st["font"] * 0.85, color=b.INK2)
            ax.set_xscale("log")
            ax.set_xlabel("parameters (billions, log scale)")
            ax.set_ylabel("server inference time per chunk (ms)")
            ax.legend(loc="upper left")
            b.save(fig, "fig_sizespeed", st)

    # C9.1 baselines
    rows = []
    for t, tname in TASKS:
        mf = REPO / "results" / "evals" / f"{t}_baseline" / "metrics.csv"
        if mf.is_file():
            m = pd.read_csv(mf)
            for p in POLICIES:
                g = m[m.model == model_name(p, t)]
                if len(g):
                    k, n = int((g["M1_success"] == 1).sum()), len(g)
                    lo, hi = b.wilson(k, n)
                    rows.append({"p": p, "t": t, "k": k, "n": n, "r": k / n, "lo": lo, "hi": hi})
    if rows:
        r = pd.DataFrame(rows)
        order = r.groupby("p")["r"].mean().sort_values(ascending=False).index.tolist()
        fig, ax = plt.subplots(figsize=(st["width"], st["width"] * 0.42))
        w = 0.38
        for j, (t, tname) in enumerate(TASKS):
            for i, p in enumerate(order):
                g = r[(r.p == p) & (r.t == t)]
                if g.empty:
                    continue
                g = g.iloc[0]
                x = i + (j - 0.5) * (w + 0.03)
                ax.bar(x, 100 * g.r, width=w, color=TASK_COLORS[t], edgecolor="white", lw=1.0, zorder=2,
                       label=tname if i == 0 else None)
                ax.errorbar(x, 100 * g.r, yerr=[[100 * (g.r - g.lo)], [100 * (g.hi - g.r)]], fmt="none",
                            ecolor=b.INK2, elinewidth=0.8, capsize=2, zorder=3)
                ax.text(x, 100 * g.hi + 2, f"{g.k}/{g.n}", ha="center", fontsize=st["font"] * 0.8, color=b.INK2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([names[p].replace(" ", "\n", 1) + ("\n(broken)" if p == "molmoact2" else "") for p in order],
                           fontsize=st["font"] * 0.85)
        ax.set_ylabel("task success (%)")
        ax.set_ylim(0, 112)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", ncol=2)
        b.save(fig, "fig_baselines", st)

    # C10.1 reply turnaround from the Pi: WiFi (baselines) against cable (E4)
    wifi = pd.concat([turnarounds(f"{t}_baseline") for t, _ in TASKS], ignore_index=True)
    cable = pd.concat([turnarounds(f"{t}_drtc_v2") for t, _ in TASKS], ignore_index=True)
    if not wifi.empty:
        fig, ax = plt.subplots(figsize=(st["width"], st["width"] * 0.42))
        pol = [p for p in POLICIES if p != "molmoact2"]
        for j, (lab, d, col) in enumerate((("WiFi (baselines)", wifi, b.COLORS["faults"]),
                                           ("cable (E4)", cable, b.COLORS["baseline"]))):
            for i, p in enumerate(pol):
                s = d[d.model.str.startswith(p + "_")]["ms"] / 1000 if not d.empty else pd.Series(dtype=float)
                if s.empty:
                    continue
                y = i + (j - 0.5) * 0.36
                ax.boxplot(s, positions=[y], widths=0.3, vert=False, patch_artist=True, showfliers=False,
                           medianprops={"color": "white", "lw": 1.4}, boxprops={"facecolor": col, "edgecolor": col},
                           whiskerprops={"color": col}, capprops={"color": col})
            if not d.empty:
                ax.plot([], [], color=col, lw=6, label=lab)
        ax.axvline(0.25, color=b.INK2, lw=0.8, ls="--")
        ax.text(0.26, -0.75, "bench round trip (0.25 s)", fontsize=st["font"] * 0.8, color=b.INK2)
        ax.set_yticks(range(len(pol)))
        ax.set_yticklabels([names[p] for p in pol])
        ax.invert_yaxis()
        ax.set_xlabel("reply turnaround (s)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper right")
        b.save(fig, "fig_edgertt", st)

    # C9.2-C9.5 E4: success and motion per policy and condition, one row per task (written when E4 results exist)
    e4 = {t: REPO / "results" / "evals" / f"{t}_drtc_v2" / "metrics.csv" for t, _ in TASKS}
    have = {t: pd.read_csv(f) for t, f in e4.items() if f.is_file()}
    if have:
        fig, axes = plt.subplots(len(have), 1, figsize=(st["width"], st["width"] * 0.36 * len(have)), squeeze=False)
        for ax, (t, m) in zip(axes[:, 0], have.items()):
            for i, p in enumerate(E4_POLICIES):
                for j, (cond, ckey, _) in enumerate(E4_CONDS):
                    g = m[(m.model == model_name(p, t)) & (m.condition == cond)]
                    if g.empty:
                        continue
                    k, n = int((g["M1_success"] == 1).sum()), len(g)
                    x = i + (j - 1.5) * 0.2
                    ax.bar(x, 100 * k / n, width=0.18, color=b.COLORS[ckey], edgecolor="white", lw=0.8, zorder=2)
                    ax.text(x, 100 * k / n + 2, f"{k}", ha="center", fontsize=st["font"] * 0.7, color=b.INK2)
            ax.set_xticks(range(len(E4_POLICIES)))
            ax.set_xticklabels([names[p] for p in E4_POLICIES])
            ax.set_ylabel("task success (%)")
            ax.set_ylim(0, 112)
            ax.set_title(dict(TASKS)[t])
            ax.grid(axis="x", visible=False)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=b.COLORS[c], label=lab) for _, c, lab in E4_CONDS], loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1.0)
        b.save(fig, "fig_e4_success", st)

        cols = [("M2_stall_pct", "stall (%)"), ("M7_jerk_k", "jerk"), ("M11_recovery_outage_s", "recovery after the outage (s)")]
        fig, axes = plt.subplots(len(have), 3, figsize=(st["width"], st["width"] * 0.34 * len(have)), squeeze=False)
        for r_i, (t, m) in enumerate(have.items()):
            for c_i, (col, lab) in enumerate(cols):
                ax = axes[r_i, c_i]
                for i, p in enumerate(E4_POLICIES):
                    for j, (cond, ckey, _) in enumerate(E4_CONDS):
                        g = m[(m.model == model_name(p, t)) & (m.condition == cond)][col].dropna() \
                            if col in m else pd.Series(dtype=float)
                        if g.empty:
                            continue
                        x = i + (j - 1.5) * 0.2
                        ax.scatter([x], [g.median()], s=st["dot"], color=b.COLORS[ckey], zorder=3)
                ax.set_xticks(range(len(E4_POLICIES)))
                ax.set_xticklabels([names[p].replace(" ", "\n", 1) for p in E4_POLICIES], fontsize=st["font"] * 0.75)
                ax.set_title(f"{dict(TASKS)[t]}: {lab}", fontsize=st["font"] * 0.85)
        b.save(fig, "fig_e4_motion", st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch", action="store_true", help="download the chart data into report/data/")
    a = ap.parse_args()
    if a.fetch:
        fetch()
