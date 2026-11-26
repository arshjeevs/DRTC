"""Setup v2's results for the report (lab/PLAN.md): the first run's recovery (M11), the bench E1 + E2 and E3a, the
stack choice, E4 on the robot and the baselines. Every table is written only when its results exist; until then the
report shows its placeholder (\\gentable prints "pending"). Called from build.py."""

from __future__ import annotations

import math

import pandas as pd

DATASETS = [("sort_one", "Sorting"), ("cubebox_v1", "Cube in box v1")]
FAULTS = [("none", "none"), ("spike", "spike"), ("drops", "drops"), ("outage", "outage"), ("mix", "mix"),
          ("rise", "rise"), ("loss", "loss")]
REC = ["spike", "drops", "outage", "rise"]
ABL = [("a1_no_cooldown", "A1 no cooldown"), ("a2_upstream_estimator", "A2 upstream estimator"),
       ("a3_upstream_trigger", "A3 upstream trigger"), ("a4_half_chunk_cap", "A4 $H/2$ cooldown cap")]
METHODS = [("s_butterworth", "Butterworth"), ("s_zero_lag", "zero-lag"), ("s_ensemble", "ensembling"),
           ("s_rtc", "RTC"), ("s_adaptive", "adaptive playback"), ("s_ramp", "resume ramp"),
           ("s_butterworth_adaptive", "Butterworth + adaptive")]


def _med(s) -> float:
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    return float(s.median()) if len(s) else float("nan")


def _f(x, d=1) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _recovered(g: pd.DataFrame, faults) -> str:
    k = n = 0
    for f in faults:
        col = f"M11_recovered_{f}"
        if col in g:
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            k, n = k + int(v.sum()), n + len(v)
    return f"{k}/{n}" if n else "--"


def _rec_time(g: pd.DataFrame, faults) -> float:
    vals = []
    for f in faults:
        c, t = f"M11_recovered_{f}", f"M11_recovery_{f}_s"
        if c in g and t in g:
            ok = pd.to_numeric(g[c], errors="coerce") == 1
            vals += list(pd.to_numeric(g.loc[ok, t], errors="coerce").dropna())
    return _med(vals)


def _faults_of(fault: str):
    return {"mix": ["spike", "drops", "outage"]}.get(fault, [fault] if fault in REC else [])


def _policy(model: str, names: dict) -> str:
    return names.get(model.split("_")[0], model)


def main_recovery(df, models, conds, write, setn) -> None:
    """The first run (sort_one_drtc_main): recovery per fault of the mix, conditions 2-4."""
    if df is None or "M11_recovered_spike" not in df:
        return
    rows = []
    for c, cname in conds[1:]:
        for full, m, mname in models:
            g = df[(df["c"] == c) & (df["m"] == m)]
            if g.empty:
                continue
            cells = []
            for f in ("spike", "drops", "outage"):
                rec = _recovered(g, [f])
                t = _rec_time(g, [f])
                cells += [rec, _f(t, 2)]
                setn(f"main/{m}/{c}/M11/{f}", f"{rec}" + ("" if math.isnan(t) else f", {t:.2f}~s"))
            rows.append(" & ".join([cname, mname, *cells]) + " \\\\")
    write("tab_main_recovery.tex", "\n".join([
        "\\begin{tabular}{l l cr cr cr}", "\\toprule",
        " & & \\multicolumn{2}{c}{Spike} & \\multicolumn{2}{c}{Dropped replies} & \\multicolumn{2}{c}{Outage} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
        "Condition & Policy & recovered & s & recovered & s & recovered & s \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular}", ""]))


def bench_v2(results, write, setn, names) -> None:
    for ds, dsname in DATASETS:
        f = results / "bench" / f"bench_v2_drtc_{ds}" / "metrics.csv"
        if f.is_file():
            df = pd.read_csv(f)
            _e1(df, ds, write, setn, names)
            _e2(df, ds, write, setn)
        smooth = [results / "bench" / f"bench_v2_smooth_{ds}" / "metrics.csv",
                  results / "bench" / f"bench_v2_smooth_rtc_{ds}" / "metrics.csv"]
        frames = [pd.read_csv(p).assign(_src=p.parent.name) for p in smooth if p.is_file()]
        if frames:
            _e3a(frames, ds, write, names)
        stack = [results / "bench" / f"bench_v2_stack_{ds}" / "metrics.csv",
                 results / "bench" / f"bench_v2_stack_nortc_{ds}" / "metrics.csv"]
        frames = [_stack_frame(pd.read_csv(p)) for p in stack if p.is_file()]
        if frames:
            _e3b(pd.concat(frames, ignore_index=True), ds, write, setn, names)
    try:  # the stack, by the fixed rule
        from drtc.evals import stack

        seam, _ = stack.decide(stack.SEAM, results / "bench", only_models={"smolvla_", "groot_"})
        jitter, _ = stack.decide(stack.JITTER, results / "bench")
        if seam or jitter:
            parts = ["adaptive playback", *[{"rtc": "RTC", "ensemble": "temporal ensembling",
                                              "zero_lag": "zero-lag smoothing", "butterworth": "the Butterworth filter"}[p]
                                             for p in (seam, jitter) if p], "the resume ramp"]
            setn("v2/stack", ", ".join(parts[:-1]) + " and " + parts[-1])
    except Exception as e:  # noqa: BLE001 - no smoothing results yet
        print(f"  (stack not chosen yet: {type(e).__name__})")


def _e1(df, ds, write, setn, names) -> None:
    rows = []
    for fault, fname in FAULTS:
        first = True
        for model in sorted(df["model"].unique()):
            g = {s: df[(df["condition"] == f"{s}__{fault}") & (df["model"] == model)]
                 for s in ("upstream_patient", "upstream_shipped", "drtc")}
            if g["drtc"].empty:
                continue
            fs = _faults_of(fault)
            shipped = g["upstream_shipped"]
            ended = f"{int(shipped['M1f_fault'].sum())}/{len(shipped)}" if not shipped.empty else "--"
            m = model.split("_")[0]  # the policy (act, smolvla, groot, grootn17, ...): keys for the text
            setn(f"v2/{ds}/e1/{fault}/{m}/patient", _f(_med(g["upstream_patient"]["M2_stall_pct"])))
            setn(f"v2/{ds}/e1/{fault}/{m}/drtc", _f(_med(g["drtc"]["M2_stall_pct"])))
            setn(f"v2/{ds}/e1/{fault}/{m}/patientrec", _recovered(g["upstream_patient"], fs))
            setn(f"v2/{ds}/e1/{fault}/{m}/drtcrec", _recovered(g["drtc"], fs))
            setn(f"v2/{ds}/e1/{fault}/{m}/drtcrectime", _f(_rec_time(g["drtc"], fs), 2))
            setn(f"v2/{ds}/e1/{fault}/{m}/ended", ended)
            rows.append(" & ".join([
                fname if first else "", _policy(model, names),
                _f(_med(g["upstream_patient"]["M2_stall_pct"])), _f(_med(g["drtc"]["M2_stall_pct"])), ended,
                _recovered(g["upstream_patient"], fs), _recovered(g["drtc"], fs),
                _f(_rec_time(g["upstream_patient"], fs), 2), _f(_rec_time(g["drtc"], fs), 2)]) + " \\\\")
            first = False
        rows.append("\\addlinespace")
    if rows:
        setn(f"v2/{ds}/e1/n", str(len(df)))
        write(f"tab_v2_e1_{ds}.tex", "\n".join([
            "\\begin{tabular}{l l rr c cc rr}", "\\toprule",
            " & & \\multicolumn{2}{c}{Stall (\\%)} & Shipped & \\multicolumn{2}{c}{Recovered} & "
            "\\multicolumn{2}{c}{Recovery (s)} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
            "Fault & Policy & patient & \\drtc{} & ended & patient & \\drtc{} & patient & \\drtc{} \\\\", "\\midrule",
            *rows[:-1], "\\bottomrule", "\\end{tabular}", ""]))


def _e2(df, ds, write, setn) -> None:
    rows = []
    for a, aname in ABL:
        first = True
        for fault, fname in FAULTS:
            worse = n = 0
            d2 = []
            for model in df["model"].unique():
                ga = df[(df["condition"] == f"{a}__{fault}") & (df["model"] == model)]
                gd = df[(df["condition"] == f"drtc__{fault}") & (df["model"] == model)]
                if ga.empty or gd.empty:
                    continue
                x = _med(ga["M2_stall_pct"]) - _med(gd["M2_stall_pct"])
                d2.append(x)
                n += 1
                worse += int(x > 0)
            if n:
                setn(f"v2/{ds}/e2/{a}/{fault}/delta", _f(_med(d2), 1))
                setn(f"v2/{ds}/e2/{a}/{fault}/worse", f"{worse}/{n}")
                rows.append(" & ".join([aname if first else "", fname, f"{worse}/{n}", _f(_med(d2), 1)]) + " \\\\")
                first = False
        rows.append("\\addlinespace")
    if any("&" in r for r in rows):
        write(f"tab_v2_e2_{ds}.tex", "\n".join([
            "\\begin{tabular}{l l c r}", "\\toprule",
            "Ablation & Fault & Policies with more stalls & Median $\\Delta$ stall (pp) \\\\", "\\midrule",
            *rows[:-1], "\\bottomrule", "\\end{tabular}", ""]))


def _e3a(frames, ds, write, names) -> None:
    rows = []
    for meth, mname in METHODS:
        first = True
        for fr in frames:
            for model in sorted(fr["model"].unique()):
                g = fr[(fr["condition"] == f"{meth}__mix") & (fr["model"] == model)]
                ref = fr[(fr["condition"] == "drtc__mix") & (fr["model"] == model)]
                if g.empty or ref.empty:
                    continue
                jr = _med(g["M7_jerk_k"]) / _med(ref["M7_jerk_k"]) if _med(ref["M7_jerk_k"]) else float("nan")
                rows.append(" & ".join([mname if first else "", _policy(model, names), _f(jr, 2),
                                        _f(_med(g["M8b_largest_step_deg"]), 1),
                                        _f(_med(g.get("M12_plan_deviation_deg", pd.Series(dtype=float))), 2),
                                        _f(_med(g["M2_stall_pct"]), 1)]) + " \\\\")
                first = False
        rows.append("\\addlinespace")
    if any("&" in r for r in rows):
        write(f"tab_v2_e3a_{ds}.tex", "\n".join([
            "\\begin{tabular}{l l rrrr}", "\\toprule",
            "Method & Policy & Jerk vs \\drtc{} & Largest step ($^\\circ$) & Deviation ($^\\circ$) & Stall (\\%) \\\\",
            "\\midrule", *rows[:-1], "\\bottomrule", "\\end{tabular}", ""]))


PARTS = [("adaptive", "adaptive playback"), ("seam", "RTC / ensembling"), ("butterworth", "Butterworth"),
         ("ramp", "resume ramp")]


def _stack_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One E3b eval with uniform names: `stack`, `stack__none`, `minus_<part>`, `drtc`; `seam` = rtc or ensemble."""
    nortc = df["condition"].str.startswith("stack_nortc").any()
    def norm(c: str) -> str:
        c = c.replace("stack_nortc", "stack")
        if c == "stack__mix":
            return "stack"
        if c.startswith("stack_minus_"):
            part = c[len("stack_minus_"):].split("__")[0]
            return "minus_" + ("seam" if part in ("rtc", "ensemble") else part)
        return {"drtc__mix": "drtc"}.get(c, c)
    return df.assign(cond=df["condition"].map(norm), seam="ensembling" if nortc else "RTC")


def _e3b(df, ds, write, setn, names) -> None:
    """E3b: the final stack against DRTC alone (per policy), and what removing each part does."""
    mix = ["spike", "drops", "outage"]
    rows = []
    for model in sorted(df["model"].unique()):
        g = {c: df[(df["cond"] == c) & (df["model"] == model)] for c in ("drtc", "stack", "stack__none")}
        if g["drtc"].empty or g["stack"].empty:
            continue
        j = _med(g["stack"]["M7_jerk_k"]) / _med(g["drtc"]["M7_jerk_k"])
        seam = g["stack"]["seam"].iloc[0]
        pol = _policy(model, names)
        setn(f"v2/{ds}/e3b/{model.split('_')[0]}/jerkratio", f"{1 / j:.0f}" if j else "--")
        rows.append(" & ".join([
            pol, seam,
            _f(_med(g["drtc"]["M2_stall_pct"])), _f(_med(g["stack"]["M2_stall_pct"])),
            _f(_med(g["stack__none"]["M2_stall_pct"])),
            _f(j, 2), _f(_med(g["drtc"]["M8b_largest_step_deg"])), _f(_med(g["stack"]["M8b_largest_step_deg"])),
            _f(_med(g["stack"]["M12_plan_deviation_deg"]), 2),
            _f(_rec_time(g["drtc"], mix), 2), _f(_rec_time(g["stack"], mix), 2)]) + " \\\\")
    if rows:
        write(f"tab_v2_e3b_{ds}.tex", "\n".join([
            "\\begin{tabular}{l l rrr r rr r rr}", "\\toprule",
            " & & \\multicolumn{3}{c}{Stall (\\%)} & Jerk & \\multicolumn{2}{c}{Largest step ($^\\circ$)} & "
            "Deviation & \\multicolumn{2}{c}{Recovery (s)} \\\\",
            "\\cmidrule(lr){3-5}\\cmidrule(lr){7-8}\\cmidrule(lr){10-11}",
            "Policy & Seam & \\drtc{} & stack & stack, no faults & vs \\drtc{} & \\drtc{} & stack & ($^\\circ$) & "
            "\\drtc{} & stack \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""]))
    rows = []
    for part, pname in PARTS:
        first = True
        for seam in ("RTC", "ensembling"):
            jr, ds2, rec = [], [], []
            for model in sorted(df["model"].unique()):
                st = df[(df["cond"] == "stack") & (df["model"] == model) & (df["seam"] == seam)]
                mi = df[(df["cond"] == f"minus_{part}") & (df["model"] == model) & (df["seam"] == seam)]
                if st.empty or mi.empty:
                    continue
                jr.append(_med(mi["M7_jerk_k"]) / _med(st["M7_jerk_k"]))
                ds2.append(_med(mi["M2_stall_pct"]) - _med(st["M2_stall_pct"]))
                rec.append(_rec_time(mi, mix) - _rec_time(st, mix))
            if not jr:
                continue
            n = len(jr)
            rows.append(" & ".join([
                pname if first else "", f"{seam} ({n})",
                f"{sum(x > 1 for x in jr)}/{n}", _f(_med(jr), 2),
                f"{sum(x > 0 for x in ds2)}/{n}", _f(_med(ds2), 1), _f(_med(rec), 2)]) + " \\\\")
            first = False
        rows.append("\\addlinespace")
    if any("&" in r for r in rows):
        write(f"tab_v2_e3b_abl_{ds}.tex", "\n".join([
            "\\begin{tabular}{l l cr cr r}", "\\toprule",
            " & & \\multicolumn{2}{c}{Jerk without it} & \\multicolumn{2}{c}{Stall without it} & Recovery \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
            "Part removed & Policies & higher & $\\times$ (median) & higher & $\\Delta$ pp (median) & $\\Delta$ s (median) \\\\",
            "\\midrule", *rows[:-1], "\\bottomrule", "\\end{tabular}", ""]))


def robot_v2(results, write, setn, names, wilson, fisher_p) -> None:
    """E4 (sort_one_drtc_v2) and the baselines: success per condition / model with Wilson intervals."""
    plans = [("sort_one_drtc_v2", "e4"), ("cubebox_v1_drtc_v2", "e4_cubebox_v1"), ("sort_one_baseline", "base_sort_one"),
             ("cubebox_v1_baseline", "base_cubebox_v1")]
    for plan, key in plans:
        f = results / "evals" / plan / "metrics.csv"
        if not f.is_file():
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        rows = []
        for cond in df["condition"].unique():
            for model in sorted(df["model"].unique()):
                g = df[(df["condition"] == cond) & (df["model"] == model)]
                if g.empty:
                    continue
                k, n = int((g["M1_success"] == 1).sum()), len(g)
                lo, hi = wilson(k, n)
                rows.append(" & ".join([cond.replace("_", "\\_"), _policy(model, names), f"{k}/{n}",
                                        f"{100 * k / n:.0f}\\%", f"{100 * lo:.0f}--{100 * hi:.0f}\\%",
                                        f"{int(g['M1f_fault'].sum())}/{n}"]) + " \\\\")
                setn(f"{key}/{model.split('_')[0]}/{cond}/M1", f"{100 * k / n:.0f}\\%")
        write(f"tab_{key}.tex", "\n".join([
            "\\begin{tabular}{l l r r c c}", "\\toprule",
            "Condition & Policy & Successes & Rate & 95\\% interval & Ended by a fault \\\\", "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}", ""]))


def registry_tables(repo, write) -> None:
    """Appendix C: setup v2's systems x faults (from lab/experiments/v2) and every evaluation plan (lab/evals)."""
    import yaml

    v2 = repo / "lab" / "experiments" / "v2"
    names = sorted(p.stem for p in v2.glob("*.yaml"))
    systems: dict[str, set] = {}
    for n in names:
        sysname, _, fault = n.rpartition("__")
        systems.setdefault(sysname, set()).add(fault)
    faults = [f for f, _ in FAULTS]
    rows = []
    for sysname in systems:
        desc = (yaml.safe_load((v2 / f"{sysname}__{sorted(systems[sysname])[0]}.yaml").read_text()) or {}).get(
            "description", "")
        desc = desc.split(": ", 1)[-1].split("; fault:")[0]
        cells = ["$\\bullet$" if f in systems[sysname] else "" for f in faults]
        rows.append(" & ".join([f"\\texttt{{{sysname.replace('_', '\\_')}}}", *cells]) + " \\\\")
    write("tab_conditions.tex", "\n".join([
        "\\begin{tabular}{l " + "c" * len(faults) + "}", "\\toprule",
        "System & " + " & ".join(faults) + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""]))
    rows = []
    for p in sorted((repo / "lab" / "evals").glob("*.yaml")):
        d = yaml.safe_load(p.read_text()) or {}
        if not d.get("models"):
            continue
        kind = "bench" if d.get("robot") == "replay" else "robot"
        frozen = "yes" if d.get("frozen") and all(m in d["frozen"] for m in d["models"]) else "no"
        rows.append(" & ".join([f"\\texttt{{{p.stem.replace('_', '\\_')}}}", kind, str(len(d["models"])),
                                str(len(d.get("conditions", {}))), str(d.get("trials", "")),
                                str(len(d.get("primary", []))), frozen]) + " \\\\")
    write("tab_plans.tex", "\n".join([
        "\\begin{tabular}{l l r r r r c}", "\\toprule",
        "Plan & Kind & Models & Conditions & Trials & Primary & Frozen \\\\", "\\midrule", *rows, "\\bottomrule",
        "\\end{tabular}", ""]))
