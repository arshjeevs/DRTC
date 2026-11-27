"""What the results explorer shows, read from the local results (results/evals/<eval>/sessions, as pulled from the
results store). Nothing here writes to results/: every number is recomputed from the raw trial files with the same
code as `lab eval report` (drtc.evals.metrics), so the explorer and report.md always agree.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import yaml

from drtc.evals import metrics as M
from drtc.evals import spec as S
from drtc.evals import store

# The evals the explorer shows, in this order (the user, 2026-11-27: E4 and the two robot baselines only).
EVALS = {
    "cubebox_v1_drtc_v2": {
        "title": "DRTC on the robot",
        "tag": "E4 · cube in box",
        "summary": "Four policies, four conditions: patient LeRobot without faults, the same client under the fault "
                   "mix, DRTC under the fault mix, and DRTC with the smoothing stack under the fault mix.",
    },
    "cubebox_v1_baseline": {
        "title": "Baseline: cube in box",
        "tag": "Baseline · cube in box",
        "summary": "Every trained policy on the cube-in-box task with patient LeRobot on a normal network.",
        "notes": ["A first cube-in-box baseline run (70 trials) was discarded before analysis because its success "
                  "labels were entered by another person and are not reliable; it is archived unchanged in the "
                  "results store with the reason, and the evaluation was run again from scratch."],
    },
    "sort_one_baseline": {
        "title": "Baseline: sorting",
        "tag": "Baseline · sorting",
        "summary": "Every trained policy on the sorting task with patient LeRobot on a normal network.",
    },
}

# condition -> (label, short label, colour slot); slots follow the report's figures (baseline, faults, DRTC, stack)
CONDITIONS = {
    "upstream_patient__none": ("Patient LeRobot, normal network", "Patient · no faults", 1),
    "upstream_patient__mix": ("Patient LeRobot under the fault mix", "Patient · faults", 2),
    "drtc__mix": ("DRTC under the fault mix", "DRTC · faults", 3),
    "drtc_stack_nortc__mix": ("DRTC + smoothing stack under the fault mix", "DRTC + stack · faults", 4),
    "baseline": ("Patient LeRobot, normal network", "Baseline", 1),
}

POLICIES = {"smolvla": "SmolVLA", "groot": "GR00T N1.5", "grootn17": "GR00T N1.7", "pi05": "π0.5",
            "molmoact2": "MolmoAct 2", "act": "ACT", "diffusion": "Diffusion Policy"}

# The metrics the explorer offers (column, short name, unit, decimals, lower is better, plain explanation)
METRICS = [
    ("M2_stall_pct", "Stalled time", "%", 1, True,
     "Share of the run in which the arm had no new action to execute (it stood still waiting)."),
    ("M3_longest_stall_s", "Longest stall", "s", 2, True, "The longest single wait for actions in the run."),
    ("M3_stall_events", "Stall events", "", 0, True, "How many separate times the arm stopped and waited."),
    ("M1t_time_to_success_s", "Time to success", "s", 1, True,
     "From the first action to the operator's success key (successful runs only)."),
    ("M7_jerk_k", "Jerk", "10³ °/s³", 1, True,
     "How abrupt the commanded motion is while the arm moves (RMS of the third derivative); lower is smoother."),
    ("M8b_largest_step_deg", "Largest step", "°", 1, True, "The biggest jump of a joint between two control ticks."),
    ("M8a_switch_l2_deg", "Chunk switch jump", "°", 1, True,
     "How far the arm jumps where one chunk of actions hands over to the next."),
    ("M12_plan_deviation_deg", "Deviation from the plan", "°", 2, True,
     "How far smoothing moves the arm off the model's own planned actions."),
    ("M4_turnaround_med_ms", "Turnaround (median)", "ms", 0, True,
     "Time from sending an observation to receiving its chunk of actions."),
    ("M4_turnaround_p95_ms", "Turnaround (p95)", "ms", 0, True, "The slow end of the turnaround (95th percentile)."),
    ("M6_server_ms", "Server inference", "ms", 0, True, "GPU time per request on the server."),
    ("M5_estimate_abs_err_ms", "Latency estimate error", "ms", 0, True,
     "How far the client's latency estimate was from the real turnaround."),
]
FAULT_NAMES = {"spike": "Latency spike", "drops": "Lost replies", "outage": "Link down", "rise": "Rising latency"}


def policy_name(model: str) -> str:
    return POLICIES.get(model.split("_", 1)[0], model)


def outcome_of(info: dict, valid: bool) -> str:
    """success / failed (operator) / time_limit / fault (the system ended the run) / not_counted."""
    if not valid:
        return "not_counted"
    lab = store.label_of(info)
    return {"s": "success", "f": "failed", "time_limit": "time_limit", "fault": "fault"}.get(lab, "failed")


def _short(text: str, n: int = 150) -> str:
    text = " ".join(str(text).split()).split(" at /")[0]
    return text if len(text) <= n else text[: n - 1] + "…"


def _experiment(reg, exp: str) -> dict:
    p = Path("lab") / "experiments" / f"{exp}.yaml"
    return (yaml.safe_load(p.read_text()) or {}) if p.is_file() else {}


def describe_condition(exp: dict) -> dict:
    """What the condition runs, in plain words, from its experiment file."""
    drtc = exp.get("scheduler") == "drtc"
    client = "DRTC (early requests, latency estimate, cooldown re-send)" if drtc else \
        "Patient LeRobot (waits as long as the trial for a reply)"
    smooth = []
    if exp.get("playback_mode") == "adaptive":
        smooth.append("adaptive playback")
    if exp.get("temporal_ensemble"):
        smooth.append("temporal ensembling")
    if exp.get("action_filter_mode") not in (None, "none"):
        smooth.append(f"{exp['action_filter_mode']} filter")
    if exp.get("resume_ramp"):
        smooth.append("resume ramp")
    if exp.get("rtc_enabled"):
        smooth.append("guided RTC")
    faults = []
    for s in exp.get("spikes") or []:
        faults.append({"kind": "spike", "start": s["start_s"], "end": s["start_s"] + s["delay_ms"] / 1000,
                       "text": f"+{s['delay_ms'] / 1000:g} s latency spike at {s['start_s']:g} s"})
    for key, kind, word in (("drop_action", "drops", "replies lost"), ("disconnect", "outage", "link down"),
                            ("rise", "rise", "latency rising")):
        for w in exp.get(key) or []:
            faults.append({"kind": kind, "start": w["start_s"], "end": w["start_s"] + w["duration_s"],
                           "text": f"{w['duration_s']:g} s {word} from {w['start_s']:g} s"})
    shift = exp.get("fault_shift_s")
    return {"client": client, "drtc": drtc, "smoothing": smooth, "faults": faults,
            "shift": shift if isinstance(shift, list) else None}


# --- the key findings (the Results window's first view): the report's main claims, from the same raw trials ----------

E4 = "cubebox_v1_drtc_v2"
# executed actions per chunk at 30 Hz (the report's Table 2.1 and Figure 11.1)
CHUNK_ACTIONS = {"act": 100, "smolvla": 50, "pi05": 50, "diffusion": 24, "groot": 16, "grootn17": 16}
BASELINES = ("sort_one_baseline", "cubebox_v1_baseline")


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


def reply_times(names=BASELINES) -> dict[str, float]:
    """Median reply turnaround (s) per policy over the robot baselines, from the client's 'turnaround N ms' lines in
    every trial's rollout.log (as the report's Figure 11.1)."""
    import re

    ms: dict[str, list[float]] = {}
    for name in names:
        for f in (store.ROOT / name / "sessions").glob("*/trials/*/*/t*/rollout.log"):
            pol = f.parents[1].name.split("_", 1)[0]
            ms.setdefault(pol, []).extend(float(v) for v in re.findall(r"turnaround ([0-9.]+) ms",
                                                                        f.read_text(errors="ignore")))
    return {p: statistics.median(v) / 1000 for p, v in ms.items() if v}


def highlights(evals: list[dict]) -> dict | None:
    e = next((x for x in evals if x.get("name") == E4 and not x.get("missing")), None)
    if e is None:
        return None
    by = {c: [t for t in e["trials"] if t["valid"] and t["condition"] == c] for c in (c["id"] for c in e["conditions"])}
    P, F, DR, ST = "upstream_patient__none", "upstream_patient__mix", "drtc__mix", "drtc_stack_nortc__mix"
    if not all(by.get(c) for c in (P, F, DR, ST)):
        return None

    def rec(ts, f):
        h = [t for t in ts if t["metrics"].get(f"M11_recovered_{f}") is not None]
        return {"k": sum(1 for t in h if t["metrics"][f"M11_recovered_{f}"] == 1), "n": len(h)}

    def med(ts, col):
        return _median(t["metrics"].get(col) for t in ts)

    policies = []
    for m in e["models"]:
        g = {c: [t for t in by[c] if t["model"] == m["id"]] for c in (P, F, DR, ST)}
        test = next((t for t in e["tests"] if t["condition"] == DR and t["reference"] == F and t["model"] == m["id"]), {})
        policies.append({"model": m["id"], "name": m["name"],
                         "k": {c: sum(1 for t in g[c] if t["outcome"] == "success") for c in g},
                         "n": {c: len(g[c]) for c in g}, "p_holm": test.get("p_holm"),
                         "significant": test.get("significant", False),
                         "jerk": {c: med(g[c], "M7_jerk_k") for c in (DR, ST)},
                         "deviation": med(g[ST], "M12_plan_deviation_deg"),
                         "stall_nofault": med(g[P], "M2_stall_pct")})
    replies = reply_times()
    chunk = [{"policy": p, "name": POLICIES[p], "chunk_s": n / 30, "reply_s": replies[p]}
             for p, n in CHUNK_ACTIONS.items() if p in replies]
    chunk.sort(key=lambda r: r["chunk_s"] - r["reply_s"], reverse=True)
    pooled = {p["condition"]: p for p in e["pooled"]}
    ptest = next((t for t in e["pooled_tests"] if t["condition"] == DR and t["reference"] == F), {})
    jd, js = med(by[DR], "M7_jerk_k"), med(by[ST], "M7_jerk_k")
    return {
        "eval": E4, "conditions": [P, F, DR, ST], "ladder": [pooled[c] for c in (P, F, DR, ST)],
        "pooled_p": ptest.get("p"),
        "ended_by_client": {c: sum(1 for t in by[c] if t["outcome"] == "fault") for c in (F, DR, ST)},
        "recovery": [{"fault": f, **{c: rec(by[c], f) for c in (F, DR, ST)}} for f in ("spike", "drops", "outage")],
        "jerk": {"drtc": jd, "stack": js, "ratio": jd / js if jd and js else None},
        "policies": policies, "chunk": chunk,
        "baselines": [{"name": x["name"], "title": x["title"],
                       "range": [min(g["rate"] for g in x["groups"] if g["n"]), max(g["rate"] for g in x["groups"] if g["n"])],
                       "policies": len(x["models"])} for x in evals if x.get("name") in BASELINES and not x.get("missing")],
    }


def build(reg, names=None) -> dict:
    """The whole bundle: every eval with its plan, groups, comparisons and trials."""
    out = {"evals": [], "metrics": [{"col": c, "name": n, "unit": u, "dec": d, "lower": lo, "help": h}
                                    for c, n, u, d, lo, h in METRICS],
           "faults": FAULT_NAMES}
    for name in names or EVALS:
        try:
            out["evals"].append(build_eval(reg, name))
        except FileNotFoundError as e:
            out["evals"].append({"name": name, "missing": str(e), **EVALS.get(name, {})})
    out["highlights"] = highlights(out["evals"])
    return out


def build_eval(reg, name: str) -> dict:
    sp = S.load(reg, name)
    root = S.results_root(sp)
    if not store.sessions_dir(name, root).is_dir():
        raise FileNotFoundError(f"no local results for {name}: run `lab eval pull {name}`")
    rig = sp.robot == "rig"
    trials = store.trials(name, root)
    conds = []
    for c, exp in sp.conditions.items():
        label, short, slot = CONDITIONS.get(c, (c, c, 1))
        conds.append({"id": c, "label": label, "short": short, "slot": slot, "experiment": exp,
                      **describe_condition(_experiment(reg, exp))})
    models = []
    for m in sp.models:
        r = reg.resolve(m)
        models.append({"id": m, "name": policy_name(m), "hub": r.hub, "commit": sp.frozen.get(m)})

    rows, groups = [], {}
    for t in trials:
        i = t.info
        met = M.trial_metrics(t.path) if t.valid else {}
        oc = outcome_of(i, t.valid)
        vid = (i.get("video") or {}).get("files") or {}
        frames = (i.get("video") or {}).get("frames") or {}
        tid = f"{t.session}/{t.condition}/{t.model}/t{t.number:02d}"
        rows.append({
            "id": tid, "session": t.session, "condition": t.condition, "model": t.model, "trial": t.number,
            "round": i.get("round"), "valid": t.valid, "outcome": oc, "label": store.label_of(i), "key": i.get("key"),
            "void": i.get("void"), "error": i.get("error"), "run_outcome": i.get("outcome"),
            "time_to_success_s": met.get("M1t_time_to_success_s"), "key_time_s": i.get("key_time_s"),
            "first_action_s": i.get("first_action_s"), "started_at": i.get("started_at"),
            "machine": i.get("machine"), "server": i.get("server"), "server_gpu": i.get("server_gpu"),
            "revision": i.get("revision"), "cadence_hz": i.get("cadence_hz"),
            "cameras": [c for c in ("scene", "wrist") if c in vid and (t.path / vid[c]).is_file()]
            + [c for c in vid if c not in ("scene", "wrist") and (t.path / vid[c]).is_file()],
            "frames": frames,
            "metrics": {k: v for k, v in met.items() if v is not None},
        })
        if t.valid:
            groups.setdefault((t.condition, t.model), []).append(rows[-1])

    gout = []
    for c in sp.conditions:
        for m in sp.models:
            g = groups.get((c, m), [])
            n = len(g)
            k = sum(1 for r in g if r["outcome"] == "success")
            lo, hi = M.wilson(k, n) if n else (None, None)
            stats = {}
            for col, *_ in METRICS:
                v = M.median_iqr(r["metrics"].get(col) for r in g)
                if v is not None:
                    stats[col] = list(v)
            rec = {}
            for f in M.RECOVERY_FAULTS:
                have = [r for r in g if r["metrics"].get(f"M11_recovered_{f}") is not None]
                if have:
                    kk = sum(1 for r in have if r["metrics"][f"M11_recovered_{f}"] == 1)
                    v = M.median_iqr(r["metrics"].get(f"M11_recovery_{f}_s") for r in have)
                    rec[f] = {"k": kk, "n": len(have), "time": list(v) if v else None}
            gout.append({"condition": c, "model": m, "n": n, "k": k, "rate": k / n if n else None, "lo": lo, "hi": hi,
                         "outcomes": {o: sum(1 for r in g if r["outcome"] == o)
                                      for o in ("success", "failed", "time_limit", "fault")},
                         "stats": stats, "recovery": rec})

    pooled = []
    for c in sp.conditions:
        g = [r for m in sp.models for r in groups.get((c, m), [])]
        k = sum(1 for r in g if r["outcome"] == "success")
        lo, hi = M.wilson(k, len(g)) if g else (None, None)
        pooled.append({"condition": c, "k": k, "n": len(g), "rate": k / len(g) if g else None, "lo": lo, "hi": hi,
                       "fault": sum(1 for r in g if r["outcome"] == "fault")})

    # primary comparisons as `lab eval report` tests them (success: Fisher's exact test; Holm over the eval)
    pairs = S.comparison_pairs(sp)
    primary = {tuple(x.strip() for x in p.split(" vs ")) for p in sp.primary}
    tests = []
    for c, ref in [p for p in pairs if p in primary]:
        for m in sp.models:
            a, b = groups.get((c, m), []), groups.get((ref, m), [])
            if not a or not b or not rig:
                continue
            ka = sum(1 for r in a if r["outcome"] == "success")
            kb = sum(1 for r in b if r["outcome"] == "success")
            tests.append({"condition": c, "reference": ref, "model": m, "metric": "M1_success", "ka": ka,
                          "na": len(a), "kb": kb, "nb": len(b), "p": M.fisher_p(ka, len(a), kb, len(b)),
                          "better": ka / len(a) > kb / len(b)})
    for t_, pa in zip(tests, M.holm([t_["p"] for t_ in tests])):
        t_["p_holm"] = pa
        t_["significant"] = bool(t_["better"] and pa is not None and pa < 0.05)
    # the same comparisons pooled over the models (exploratory: not part of the Holm family)
    pooled_tests = []
    for c, ref in [p for p in pairs if p in primary]:
        a = next((p for p in pooled if p["condition"] == c), None)
        b = next((p for p in pooled if p["condition"] == ref), None)
        if a and b and a["n"] and b["n"]:
            pooled_tests.append({"condition": c, "reference": ref, "ka": a["k"], "na": a["n"], "kb": b["k"],
                                 "nb": b["n"], "p": M.fisher_p(a["k"], a["n"], b["k"], b["n"])})

    sessions = []
    for s in sorted({t.session for t in trials}):
        ts = [t for t in trials if t.session == s]
        sessions.append({"id": s, "machine": ts[0].info.get("machine"), "server": ts[0].info.get("server"),
                         "server_gpu": ts[0].info.get("server_gpu"), "trials": len(ts),
                         "valid": sum(1 for t in ts if t.valid),
                         "first": min((t.info.get("started_at") or "") for t in ts),
                         "last": max((t.info.get("started_at") or "") for t in ts)})
    not_counted = [{"id": r["id"], "why": _short(r["void"] or f"{r['label'] or 'no label'}: {r['run_outcome'] or ''} "
                                                 f"{r['error'] or ''}")} for r in rows if not r["valid"]]
    info = EVALS.get(name, {})
    return {"name": name, "title": info.get("title", name), "tag": info.get("tag", ""),
            "summary": info.get("summary", sp.description), "notes": info.get("notes", []),
            "description": sp.description, "task": S.task_of(reg, sp), "success_rule": sp.success,
            "time_limit_s": sp.time_limit_s, "planned_trials": sp.trials, "robot": sp.robot,
            "conditions": conds, "models": models, "groups": gout, "pooled": pooled, "tests": tests,
            "pooled_tests": pooled_tests, "sessions": sessions, "not_counted": not_counted, "trials": rows}


# --- the chosen videos (the Trials window shows only these once a choice is saved) ------------------------------------

SELECTION = Path("lab") / "explorer" / "videos.yaml"
SELECTION_HEADER = ("# Written by `lab explore` (Trials & videos -> Choose videos -> Save): the trials whose videos the\n"
                    "# explorer shows, per eval. Once this file exists, the Trials window shows only these trials.\n"
                    "# Delete an eval's list (or the file) to show every trial again.\n")


def load_selection(path: Path = SELECTION) -> dict[str, list[str]] | None:
    """eval -> trial ids, in the saved order; None when nothing has been saved yet."""
    if not path.is_file():
        return None
    d = yaml.safe_load(path.read_text()) or {}
    return {str(k): [str(x) for x in (v or [])] for k, v in d.items()}


def save_selection(sel: dict[str, list[str]], known: set[tuple[str, str]], path: Path = SELECTION) -> dict:
    """Keep only known trials (eval, id), drop empty evals, write atomically. Returns what was written."""
    clean = {}
    for name, ids in sel.items():
        keep = list(dict.fromkeys(i for i in ids if (name, i) in known))
        if keep:
            clean[name] = keep
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(SELECTION_HEADER + (yaml.safe_dump(clean, sort_keys=False, allow_unicode=True) if clean else "{}\n"))
    tmp.replace(path)
    return clean


# --- one trial in full (the trial window) -----------------------------------------------------------------------------

TICK_COLS = ["t", "action", "queue", "pending", "latency_est", "playback_rate"]


def _read_csv(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _flt(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def trial_detail(tdir: Path) -> dict:
    """Everything the timeline needs: ticks (queue, joints), requests, faults and DRTC's re-sends."""
    info = json.loads((tdir / "trial.json").read_text())
    ticks = _read_csv(tdir / "ticks.csv")
    joints = [k[4:] for k in (ticks[0].keys() if ticks else []) if k.startswith("cmd.")]
    tk = {c: [_flt(r.get(c)) for r in ticks] for c in TICK_COLS}
    tk["cmd"] = {j: [_flt(r.get(f"cmd.{j}")) for r in ticks] for j in joints}
    tk["meas"] = {j: [_flt(r.get(f"meas.{j}")) for r in ticks] for j in joints}
    chunks = [{"sent": _flt(r.get("t_sent")), "received": _flt(r.get("t_received")),
               "accepted": r.get("accepted") == "1", "turnaround": _flt(r.get("turnaround_s")),
               "estimate": _flt(r.get("estimate_at_send_s")), "server": _flt(r.get("server_total_s")),
               "steps": _flt(r.get("steps"))} for r in _read_csv(tdir / "chunks.csv")]
    windows = [{"kind": k, "start": a, "end": b} for k, (a, b) in M.fault_ends(tdir).items()]
    events = [{"t": _flt(r.get("t")), "event": r.get("fault"), "value": r.get("value")}
              for r in _read_csv(tdir / "faults.csv")]
    events += [{"t": _flt(r.get("t")), "event": r.get("action"), "value": None}
               for r in _read_csv(tdir / "recovery.csv")]
    events.sort(key=lambda e: e["t"] if e["t"] is not None else 0)
    cfg = {}
    cp = tdir / "config.yaml"
    if cp.is_file():
        c = yaml.safe_load(cp.read_text()) or {}
        cfg = {k: c.get(k) for k in ("scheduler", "estimator", "cooldown", "actions_per_chunk", "s_min",
                                     "playback_mode", "action_filter_mode", "temporal_ensemble", "resume_ramp",
                                     "rtc_enabled", "fault_shift_applied_s", "fps", "client_host", "server_host", "gpu")
               if k in c}
    # Video time -> trial time: one frame per robot observation, from the first observation to the end of the run
    # (the last tick); encoded at 30 fps while the loop ran at cadence_hz, so the clip is stretched by 30 / cadence.
    t_end = next((v for v in reversed(tk["t"]) if v is not None), None)
    frames = max(((info.get("video") or {}).get("frames") or {}).values(), default=0)
    hz = info.get("cadence_hz") or 30.0
    video = None
    if frames and t_end is not None:
        span = frames / hz
        video = {"t0": t_end - span, "scale": span / (frames / 30.0)}
    return {"info": {k: v for k, v in info.items() if k not in ("command",)}, "ticks": tk, "joints": joints,
            "chunks": chunks, "windows": windows, "events": events, "config": cfg, "video": video, "t_end": t_end}
