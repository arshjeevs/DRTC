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


def _bold(x: str, on: bool) -> str:
    return f"\\textbf{{{x}}}" if on and x not in ("--", "") else x


def _k(rec: str) -> int:
    """'k/n' -> k (recovered faults); -1 when there is none."""
    try:
        return int(rec.split("/")[0])
    except (ValueError, IndexError):
        return -1


def _e1(df, ds, write, setn, names) -> None:
    """E1 as two compact tables, one row per policy: the stall under every fault type (patient against DRTC), and the
    recovery after each timed fault. The better value of each pair is in bold."""
    models = sorted(df["model"].unique())
    timed = [(f, n) for f, n in FAULTS if f not in ("none", "loss")]
    stall_rows, rec_rows = [], []
    for model in models:
        m = model.split("_")[0]  # the policy (act, smolvla, groot, grootn17, ...): keys for the text
        scells, rcells, any_cell = [_policy(model, names)], [_policy(model, names)], False
        for fault, _ in FAULTS:
            g = {s_: df[(df["condition"] == f"{s_}__{fault}") & (df["model"] == model)]
                 for s_ in ("upstream_patient", "upstream_shipped", "drtc")}
            if g["drtc"].empty:
                scells += ["--", "--"]
                continue
            any_cell = True
            fs = _faults_of(fault)
            shipped = g["upstream_shipped"]
            ended = f"{int(shipped['M1f_fault'].sum())}/{len(shipped)}" if not shipped.empty else "--"
            sp, sd = _med(g["upstream_patient"]["M2_stall_pct"]), _med(g["drtc"]["M2_stall_pct"])
            rp, rd = _recovered(g["upstream_patient"], fs), _recovered(g["drtc"], fs)
            setn(f"v2/{ds}/e1/{fault}/{m}/patient", _f(sp))
            setn(f"v2/{ds}/e1/{fault}/{m}/drtc", _f(sd))
            setn(f"v2/{ds}/e1/{fault}/{m}/patientrec", rp)
            setn(f"v2/{ds}/e1/{fault}/{m}/drtcrec", rd)
            setn(f"v2/{ds}/e1/{fault}/{m}/drtcrectime", _f(_rec_time(g["drtc"], fs), 2))
            setn(f"v2/{ds}/e1/{fault}/{m}/ended", ended)
            diff = (sp - sd) if sp == sp and sd == sd else 0.0
            scells += [_bold(_f(sp), diff <= -1.0), _bold(_f(sd), diff >= 1.0)]
            if (fault, fault) in [(f, f) for f, _ in timed]:
                kp, kd = _k(rp), _k(rd)
                rcells += [_bold(str(kp) if kp >= 0 else "--", kp > kd), _bold(str(kd) if kd >= 0 else "--", kd > kp),
                           _f(_rec_time(g["drtc"], fs), 2), ended.split("/")[0]]
        if any_cell:
            stall_rows.append(" & ".join(scells) + " \\\\")
            rec_rows.append(" & ".join(rcells) + " \\\\")
    if stall_rows:
        setn(f"v2/{ds}/e1/n", str(len(df)))
        heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in FAULTS)
        rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(FAULTS)))
        write(f"tab_v2_e1_{ds}.tex", "\n".join([
            "\\begin{tabular}{l " + "rr" * len(FAULTS) + "}", "\\toprule",
            "Fault & " + heads + " \\\\", rules,
            "Policy" + " & pat. & \\drtc{}" * len(FAULTS) + " \\\\", "\\midrule",
            *stall_rows, "\\bottomrule", "\\end{tabular}", ""]))
        heads = " & ".join(f"\\multicolumn{{4}}{{c}}{{{n}}}" for _, n in timed)
        rules = "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(timed)))
        write(f"tab_v2_e1rec_{ds}.tex", "\n".join([
            "\\begin{tabular}{l " + "cccc" * len(timed) + "}", "\\toprule",
            "Fault & " + heads + " \\\\", rules,
            "Policy" + " & pat. & \\drtc{} & s & ship." * len(timed) + " \\\\", "\\midrule",
            *rec_rows, "\\bottomrule", "\\end{tabular}", ""]))


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
                rows.append(" & ".join([aname if first else "", fname, f"{worse}/{n}", _bold(_f(_med(d2), 1), _med(d2) >= 1.0)]) + " \\\\")
                first = False
        rows.append("\\addlinespace")
    if any("&" in r for r in rows):
        write(f"tab_v2_e2_{ds}.tex", "\n".join([
            "\\begin{tabular}{l l c r}", "\\toprule",
            "Ablation & Fault & Policies with more stalls & Median $\\Delta$ stall (pp) \\\\", "\\midrule",
            *rows[:-1], "\\bottomrule", "\\end{tabular}", ""]))


def _e3a(frames, ds, write, names) -> None:
    """E3a as three compact tables, one row per policy and one column per method: the jerk relative to DRTC alone
    (bold below 1: smoother), the stall (bold: lower than DRTC alone by a point or more) and the deviation."""
    models = sorted({m for fr in frames for m in fr["model"].unique()})
    meths = [(meth, mname) for meth, mname in METHODS
             if any(not fr[fr["condition"] == f"{meth}__mix"].empty for fr in frames)]
    jr_rows, st_rows, dv_rows = [], [], []
    for model in models:
        jr, st, dv = [_policy(model, names)], [_policy(model, names)], [_policy(model, names)]
        ref_stall = None
        for fr in frames:
            ref = fr[(fr["condition"] == "drtc__mix") & (fr["model"] == model)]
            if not ref.empty and ref_stall is None:
                ref_stall = _med(ref["M2_stall_pct"])
        st.append(_f(ref_stall) if ref_stall is not None else "--")
        for meth, _ in meths:
            cell = None
            for fr in frames:
                g = fr[(fr["condition"] == f"{meth}__mix") & (fr["model"] == model)]
                ref = fr[(fr["condition"] == "drtc__mix") & (fr["model"] == model)]
                if not g.empty and not ref.empty:
                    cell = (g, ref)
                    break
            if cell is None:
                jr.append("--"); st.append("--"); dv.append("--")
                continue
            g, ref = cell
            r = _med(g["M7_jerk_k"]) / _med(ref["M7_jerk_k"]) if _med(ref["M7_jerk_k"]) else float("nan")
            sv = _med(g["M2_stall_pct"])
            jr.append(_bold(_f(r, 2), r < 1.0))
            st.append(_bold(_f(sv), ref_stall is not None and sv <= ref_stall - 1.0))
            dv.append(_f(_med(g.get("M12_plan_deviation_deg", pd.Series(dtype=float))), 2))
        jr_rows.append(" & ".join(jr) + " \\\\")
        st_rows.append(" & ".join(st) + " \\\\")
        dv_rows.append(" & ".join(dv) + " \\\\")
    if jr_rows:
        short = {"Butterworth": "Butterw.", "adaptive playback": "adaptive", "resume ramp": "ramp",
                 "Butterworth + adaptive": "Butterw.+adapt.", "ensembling": "ensembl."}
        mh = " & ".join(short.get(m, m) for _, m in meths)
        n = len(meths)
        def block(title, head, rows, extra=0):
            return ["\\begin{tabular}{l " + "r" * (n + extra) + "}", "\\toprule",
                    f"\\multicolumn{{{n + 1 + extra}}}{{l}}{{\\textit{{{title}}}}} \\\\", "\\midrule",
                    "Policy & " + head + " \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]
        write(f"tab_v2_e3a_{ds}.tex", "\n".join([
            "\\begin{tabular}{@{}c@{}}",
            *block("(a) Jerk relative to \\drtc{} alone (below 1: smoother)", mh, jr_rows), "\\\\[8pt]",
            *block("(b) Stall (\\% of the run)", "\\drtc{} alone & " + mh, st_rows, extra=1), "\\\\[8pt]",
            *block("(c) Deviation from the policy's plan ($^\\circ$)", mh, dv_rows), "\\end{tabular}", ""]))


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
            _f(_med(g["drtc"]["M2_stall_pct"])),
            _bold(_f(_med(g["stack"]["M2_stall_pct"])), _med(g["stack"]["M2_stall_pct"]) <= _med(g["drtc"]["M2_stall_pct"]) - 1.0),
            _f(_med(g["stack__none"]["M2_stall_pct"])),
            _bold(_f(j, 2), j < 1.0), _f(_med(g["drtc"]["M8b_largest_step_deg"])),
            _bold(_f(_med(g["stack"]["M8b_largest_step_deg"])),
                  _med(g["stack"]["M8b_largest_step_deg"]) < _med(g["drtc"]["M8b_largest_step_deg"])),
            _f(_med(g["stack"]["M12_plan_deviation_deg"]), 2),
            _f(_rec_time(g["drtc"], mix), 2),
            _bold(_f(_rec_time(g["stack"], mix), 2), _rec_time(g["stack"], mix) < _rec_time(g["drtc"], mix))]) + " \\\\")
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
                best = key.startswith("base_") and k == max(int((df[(df["condition"] == cond) & (df["model"] == mm)]
                                                                  ["M1_success"] == 1).sum()) for mm in df["model"].unique())
                rows.append(" & ".join([cond.replace("_", "\\_"), _policy(model, names), _bold(f"{k}/{n}", best),
                                        _bold(f"{100 * k / n:.0f}\\%", best), f"{100 * lo:.0f}--{100 * hi:.0f}\\%",
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
