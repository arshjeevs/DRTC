"""Writes the setup v2 condition files (lab/PLAN.md, "DRTC + SMOOTHING EXPERIMENT SETUP v2", 2026-11-22).

    .venv/bin/python lab/experiments/v2/_make.py                      # bench (v2/) and robot (v2_robot/) conditions
    .venv/bin/python lab/experiments/v2/_make.py --stack adaptive,rtc,zero_lag,ramp   # + the final stack (E3b, E4)
    .venv/bin/python lab/experiments/v2/_make.py --evals                               # also the v2 eval files

`--evals` (re)writes the setup v2 eval files in lab/evals/ with their comparisons and primary comparisons (metrics v3);
a file that is already frozen is left alone (an eval with trials only grows: lab/WORKFLOW.md).

Every condition is <system>__<fault>.yaml. The files are generated so the systems and faults stay identical
everywhere; edit this script, not the files, and run it again (it overwrites only files it writes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parent / "v2_robot"

# The network of the bench: the measured network part of the robot's round trip (sort_one_drtc_main, ~260 ms).
BENCH_NET = {"latency": {"base_ms": 250, "jitter_ms": 20, "seed": 0}}
SHIFT = {"fault_shift_s": [0.0, 4.0]}  # every timed fault moved by a seeded 0-4 s per trial number

FAULTS = {
    "none": {},
    "spike": {"spikes": [{"start_s": 2.0, "delay_ms": 1000}]},
    "drops": {"drop_action": [{"start_s": 4.0, "duration_s": 1.0}]},
    "outage": {"disconnect": [{"start_s": 6.0, "duration_s": 1.5}]},
    "mix": {"spikes": [{"start_s": 2.0, "delay_ms": 1000}], "drop_action": [{"start_s": 4.0, "duration_s": 1.0}],
            "disconnect": [{"start_s": 6.0, "duration_s": 1.5}]},
    "rise": {"rise": [{"start_s": 2.0, "duration_s": 5.0, "delay_ms": 400}]},
    "loss": {"loss": {"p": 0.1, "seed": 0}},
}
ALL = list(FAULTS)

# Patient (2026-11-22): the starvation grace (unset = the whole trial) and the request deadline both last the whole
# trial, so no default ends a run; upstream as shipped keeps LeRobot's 1 s grace and 5 s request deadline.
PATIENT = {"request_deadline": "trial"}
UPSTREAM = {"scheduler": "upstream", "estimator": "upstream", "cooldown": False, **PATIENT}
DRTC = {"scheduler": "drtc", "estimator": "jk", "cooldown": True, "cooldown_cap": "none", **PATIENT}
SHIPPED = {"starvation_grace_s": 1.0, "request_deadline": "lerobot"}  # LeRobot's defaults

# system -> (description, settings, faults it runs under on the bench)
SYSTEMS = {
    # E1
    "upstream_patient": ("upstream LeRobot, starvation grace and request deadline = the whole trial", UPSTREAM, ALL),
    "upstream_shipped": ("upstream LeRobot as it ships (1 s starvation grace, 5 s request deadline)",
                         {**UPSTREAM, **SHIPPED}, ALL[1:]),
    "drtc": ("DRTC (our version), no smoothing", DRTC, ALL),
    # E2: DRTC minus one part
    "a1_no_cooldown": ("DRTC without the cooldown re-request", {**DRTC, "cooldown": False}, ALL),
    "a2_upstream_estimator": ("DRTC with upstream's estimator (max of the last 100 replies)",
                              {**DRTC, "estimator": "upstream"}, ALL),
    "a3_upstream_trigger": ("DRTC with upstream's trigger (refill at max(0.5 s, turnaround))",
                            {**DRTC, "scheduler": "upstream"}, ALL),
    "a4_half_chunk_cap": ("DRTC with its original H/2 cooldown cap", {**DRTC, "cooldown_cap": "half_chunk"}, ALL),
    "a5_smin_half": ("DRTC with s_min x 0.5", {**DRTC, "s_min_scale": 0.5}, ["none", "mix"]),
    "a5_smin_x1_5": ("DRTC with s_min x 1.5", {**DRTC, "s_min_scale": 1.5}, ["none", "mix"]),
}

# E3a: each smoothing method alone, on DRTC (+ the old condition 4)
METHODS = {
    "butterworth": {"action_filter_mode": "butterworth"},
    "adaptive": {"playback_mode": "adaptive"},
    "rtc": {"rtc_enabled": True},
    "ensemble": {"temporal_ensemble": True},
    "zero_lag": {"zero_lag": True},
    "ramp": {"resume_ramp": True},
}
for name, s in METHODS.items():
    SYSTEMS[f"s_{name}"] = (f"DRTC + {name}", {**DRTC, **s}, ["mix"])
SYSTEMS["s_butterworth_adaptive"] = ("DRTC + Butterworth + adaptive playback (the old condition 4)",
                                     {**DRTC, **METHODS["butterworth"], **METHODS["adaptive"]}, ["mix"])

# The robot (E4 and the baselines): the real network, no simulated latency
ROBOT_SYSTEMS = {
    "upstream_patient": (SYSTEMS["upstream_patient"][0], UPSTREAM, ["none", "mix"]),
    "drtc": (SYSTEMS["drtc"][0], DRTC, ["mix"]),
}


def write(path: Path, desc: str, settings: dict, fault: str, bench: bool) -> None:
    body = {"description": f"Setup v2 ({'bench' if bench else 'robot'}): {desc}; fault: {fault}",
            "model": "smolvla_sort_one_v1_full",  # every eval names its own models
            "action_filter_mode": "none",  # _defaults has the minor's Butterworth on; v2 turns it on only by name
            **settings, **FAULTS[fault]}
    if fault != "none" and fault != "loss":
        body.update(SHIFT)
    if bench:
        body.update(BENCH_NET)
    path.write_text("# Generated by lab/experiments/v2/_make.py: edit the script, not this file.\n"
                    + yaml.safe_dump(body, sort_keys=False))


def stack_settings(parts: list[str]) -> dict:
    out = dict(DRTC)
    for p in parts:
        out.update(METHODS[p])
    return out


EVALS = HERE.parent.parent / "evals"
DATASETS = {
    "sort_one": ("so101_sort_one_v1", ["smolvla_sort_one_v1_full", "groot_sort_one_v1_full", "grootn17_sort_one_v1_full",
                                       "act_sort_one_v1_full", "diffusion_sort_one_v1_full"]),
    "cubebox_v1": ("so101_cubebox_v1", ["smolvla_cubebox_v1_full", "groot_cubebox_v1_full", "grootn17_cubebox_v1_full",
                                        "act_cubebox_v1_full", "diffusion_cubebox_v1_full"]),
}
RTC_MODEL = ("smolvla_", "groot_")  # GR00T N1.7 (grootn17_): relative actions, LeRobot turns its RTC prefix off
ABLATIONS = ["a1_no_cooldown", "a2_upstream_estimator", "a3_upstream_trigger", "a4_half_chunk_cap"]


def write_eval(name: str, header: str, body: dict) -> None:
    path = EVALS / f"{name}.yaml"
    if path.is_file() and (yaml.safe_load(path.read_text()) or {}).get("frozen"):
        print(f"  {name}: frozen, left alone")
        return
    path.write_text(header + "# Generated by lab/experiments/v2/_make.py --evals.\n"
                    + yaml.safe_dump(body, sort_keys=False, width=120, allow_unicode=True))


def bench_evals(stack_parts: list[str] | None) -> None:
    base = {"success": "n/a (replay robot)", "time_limit_s": 14, "trials": 8, "seed": 0}
    for ds, (dsid, models) in DATASETS.items():
        conds, cmp, prim = {}, [], []
        for f in ALL:
            conds[f"upstream_patient__{f}"] = f"v2/upstream_patient__{f}"
        for f in ALL[1:]:
            conds[f"upstream_shipped__{f}"] = f"v2/upstream_shipped__{f}"
        for f in ALL:
            conds[f"drtc__{f}"] = f"v2/drtc__{f}"
        for a in ABLATIONS:
            for f in ALL:
                conds[f"{a}__{f}"] = f"v2/{a}__{f}"
        for a in ("a5_smin_half", "a5_smin_x1_5"):
            for f in ("none", "mix"):
                conds[f"{a}__{f}"] = f"v2/{a}__{f}"
        for f in ALL[1:]:
            prim.append(f"drtc__{f} vs upstream_patient__{f}")  # E1's claim, per fault: tested
            cmp += [f"drtc__{f} vs upstream_patient__{f}", f"upstream_shipped__{f} vs upstream_patient__{f}"]
        cmp.append("drtc__none vs upstream_patient__none")
        for a in ABLATIONS:
            cmp += [f"{a}__{f} vs drtc__{f}" for f in ALL]
        for a in ("a5_smin_half", "a5_smin_x1_5"):
            cmp += [f"{a}__{f} vs drtc__{f}" for f in ("none", "mix")]
        write_eval(f"bench_v2_drtc_{ds}",
                   f"# Setup v2 bench E1 + E2 on {dsid} (lab/PLAN.md, 2026-11-22): DRTC against upstream (patient and as\n"
                   "# shipped) and the DRTC ablations A1-A5, under every fault type, in one eval so every comparison is\n"
                   "# inside it. Replay robot on the held-out episodes, simulated network 250 +- 20 ms; M2-M8, M11, M12.\n"
                   "# 14 s per run: the latest fault (the outage, shifted by up to 4 s) ends at 11.5 s. Primary: DRTC\n"
                   "# against upstream (patient) per fault (stalls, recovery); the rest is exploratory (METRICS.md v3).\n",
                   {"description": f"Bench v2 E1 + E2, {ds}: DRTC vs upstream and the DRTC ablations, all fault types "
                                   f"(replay robot)", **base, "models": models, "conditions": conds,
                    "comparisons": cmp, "primary": prim, "video": False, "robot": "replay"})
        smooth = {"drtc__mix": "v2/drtc__mix"}
        for m in ["butterworth", "adaptive", "ensemble", "zero_lag", "ramp", "butterworth_adaptive"]:
            smooth[f"s_{m}__mix"] = f"v2/s_{m}__mix"
        write_eval(f"bench_v2_smooth_{ds}",
                   f"# Setup v2 bench E3a on {dsid}: each smoothing method alone on DRTC + the fault mix (and the old\n"
                   "# condition 4). RTC is in bench_v2_smooth_rtc_<dataset> (flow-matching models). The final stack is\n"
                   "# chosen from these by `python -m drtc.evals.stack` (the rule fixed 2026-11-22, before this ran).\n",
                   {"description": f"Bench v2 E3a, {ds}: smoothing methods alone on DRTC + mix (replay robot)", **base,
                    "models": models, "conditions": smooth,
                    "comparisons": [f"{c} vs drtc__mix" for c in smooth if c != "drtc__mix"], "video": False,
                    "robot": "replay"})
        write_eval(f"bench_v2_smooth_rtc_{ds}",
                   f"# Setup v2 bench E3a, the RTC cell, on {dsid}: SmolVLA and GR00T N1.5 only (ACT and Diffusion have\n"
                   "# no RTC; GR00T N1.7 is trained with relative actions, for which LeRobot switches its RTC prefix off).\n"
                   "# With its own DRTC reference so the comparison stays inside this eval. Warm up first (WORKFLOW).\n",
                   {"description": f"Bench v2 E3a, {ds}: RTC on DRTC + mix (replay robot, flow-matching models)", **base,
                    "models": [m for m in models if m.startswith(RTC_MODEL)],
                    "conditions": {"drtc__mix": "v2/drtc__mix", "s_rtc__mix": "v2/s_rtc__mix"},
                    "comparisons": ["s_rtc__mix vs drtc__mix"], "video": False, "robot": "replay"})
        if stack_parts:
            groups = [("all", models)] if "rtc" not in stack_parts else [
                ("rtc", [m for m in models if m.startswith(RTC_MODEL)]),
                ("nortc", [m for m in models if not m.startswith(RTC_MODEL)])]
            for group, ms in groups:
                parts = stack_parts if group != "nortc" else [("ensemble" if p == "rtc" else p) for p in stack_parts]
                tag = "stack" if parts == stack_parts else "stack_nortc"
                conds = {"drtc__mix": "v2/drtc__mix", f"{tag}__none": f"v2/{tag}__none", f"{tag}__mix": f"v2/{tag}__mix"}
                conds.update({f"{tag}_minus_{p}__mix": f"v2/{tag}_minus_{p}__mix" for p in parts})
                write_eval(f"bench_v2_{tag}_{ds}",
                           f"# Setup v2 bench E3b on {dsid}: the final stack ({' + '.join(parts)}), without faults, and\n"
                           "# without each of its parts, on DRTC + the fault mix.\n",
                           {"description": f"Bench v2 E3b, {ds}: the final stack and its ablations (replay robot)",
                            **base, "models": ms, "conditions": conds,
                            "comparisons": [f"{tag}__mix vs drtc__mix"] + [f"{tag}_minus_{p}__mix vs {tag}__mix"
                                                                          for p in parts],
                            "video": False, "robot": "replay"})
    if stack_parts:
        conds = {"upstream_patient__none": "v2_robot/upstream_patient__none",
                 "upstream_patient__mix": "v2_robot/upstream_patient__mix", "drtc__mix": "v2_robot/drtc__mix",
                 "drtc_stack__mix": "v2_robot/drtc_stack__mix"}
        cmp = ["upstream_patient__mix vs upstream_patient__none", "drtc__mix vs upstream_patient__mix",
               "drtc_stack__mix vs drtc__mix"]
        write_eval("sort_one_drtc_v2",
                   f"# Setup v2 E4 (lab/PLAN.md): the DRTC experiment on the robot, sort_one, SmolVLA + GR00T N1.5, the\n"
                   f"# final stack ({' + '.join(stack_parts)}). Grows by `trials` (lab/WORKFLOW.md, Growing an eval).\n",
                   {"description": "DRTC v2, sort_one - upstream patient, + mix, DRTC + mix, DRTC + stack + mix",
                    "success": "The cube is fully inside the square of its colour and the gripper has let go of it, "
                               "before the time limit.", "time_limit_s": 30, "trials": 10, "seed": 0,
                    "models": ["smolvla_sort_one_v1_full", "groot_sort_one_v1_full"], "conditions": conds,
                    "comparisons": cmp, "primary": list(cmp), "video": True})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", help="the final stack chosen from E3a, e.g. adaptive,rtc,zero_lag,ramp (E3b + E4)")
    ap.add_argument("--evals", action="store_true", help="also (re)write the v2 eval files (not frozen ones)")
    a = ap.parse_args()
    systems = dict(SYSTEMS)
    robot = dict(ROBOT_SYSTEMS)
    if a.stack:
        parts = [p.strip() for p in a.stack.split(",") if p.strip()]
        unknown = [p for p in parts if p not in METHODS]
        if unknown:
            raise SystemExit(f"unknown methods {unknown}; known: {', '.join(METHODS)}")
        systems["stack"] = (f"DRTC + the final stack ({' + '.join(parts)})", stack_settings(parts), ["none", "mix"])
        for p in parts:
            rest = [q for q in parts if q != p]
            systems[f"stack_minus_{p}"] = (f"DRTC + the final stack without {p}", stack_settings(rest), ["mix"])
        robot["drtc_stack"] = (systems["stack"][0], stack_settings(parts), ["mix"])
        if "rtc" in parts:  # the same stack for models without RTC: ensembling as the seam method
            nr = [("ensemble" if p == "rtc" else p) for p in parts]
            systems["stack_nortc"] = (f"DRTC + the final stack, no RTC ({' + '.join(nr)})", stack_settings(nr),
                                      ["none", "mix"])
            for p in nr:
                rest = [q for q in nr if q != p]
                systems[f"stack_nortc_minus_{p}"] = (f"DRTC + the final stack (no RTC) without {p}",
                                                     stack_settings(rest), ["mix"])
    n = 0
    for name, (desc, settings, faults) in systems.items():
        for f in faults:
            write(HERE / f"{name}__{f}.yaml", desc, settings, f, bench=True)
            n += 1
    ROBOT.mkdir(exist_ok=True)
    for name, (desc, settings, faults) in robot.items():
        for f in faults:
            write(ROBOT / f"{name}__{f}.yaml", desc, settings, f, bench=False)
            n += 1
    print(f"wrote {n} condition files in {HERE.name}/ and {ROBOT.name}/")
    if a.evals:
        bench_evals(parts if a.stack else None)
        print("wrote the v2 eval files (lab/evals/)")


if __name__ == "__main__":
    main()
