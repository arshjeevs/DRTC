"""Setup v2 (lab/PLAN.md, 2026-11-22): the new faults, the fault shift, the s_min ablation, the smoothing methods,
the append-only growth of evals."""

import queue as _q
import time as _t
from pathlib import Path

import numpy as np

from drtc.registry import Registry


# --- faults ----------------------------------------------------------------------------------------------------------


def test_rise_delays_replies_only_inside_its_window():
    from drtc.control.faults import FaultInjector, FaultPlan, _ReplyChannel

    plan = FaultPlan.from_dict({"rise": [{"start_s": 0.0, "duration_s": 0.3, "delay_ms": 200}]})
    assert plan and plan.rise_delay(0.1) == 0.2 and plan.rise_delay(0.5) == 0.0
    inj = FaultInjector(plan)
    inj.start()
    inner = _q.Queue()
    ch = _ReplyChannel(inner, inj)
    inner.put(b"a")
    t0 = _t.monotonic()
    assert ch.get(timeout=1.0) == b"a" and _t.monotonic() - t0 >= 0.19  # inside the window: delayed
    _t.sleep(0.35)
    inner.put(b"b")
    t1 = _t.monotonic()
    assert ch.get(timeout=1.0) == b"b" and _t.monotonic() - t1 < 0.1  # after it: on time


def test_loss_drops_about_p_of_the_replies_and_is_seeded():
    from drtc.control.faults import FaultInjector, FaultPlan, _ReplyChannel

    def lost(seed):
        inj = FaultInjector(FaultPlan.from_dict({"loss": {"p": 0.1, "seed": seed}}))
        inj.start()
        inner = _q.Queue()
        ch = _ReplyChannel(inner, inj)
        out = []
        for i in range(2000):
            inner.put(i)
            try:
                ch.get(timeout=0.01)
                out.append(0)
            except _q.Empty:
                out.append(1)
        return out

    a, b = lost(3), lost(3)
    assert a == b  # the same seed loses the same replies
    assert 0.07 < sum(a) / len(a) < 0.13
    assert lost(4) != a


def test_fault_shift_is_the_same_for_a_trial_number_and_moves_every_timed_fault():
    from drtc.control.run import apply_faults, fault_shift

    c = {"fault_shift_s": [0.0, 4.0], "spikes": [{"start_s": 2.0, "delay_ms": 1000}],
         "drop_action": [{"start_s": 4.0, "duration_s": 1.0}], "disconnect": [{"start_s": 6.0, "duration_s": 1.5}],
         "rise": [{"start_s": 2.0, "duration_s": 5.0, "delay_ms": 400}], "loss": {"p": 0.1, "seed": 0},
         "latency": {"base_ms": 250, "jitter_ms": 20, "seed": 0}}
    shifts = [fault_shift(c, k) for k in range(1, 21)]
    assert all(0.0 <= s <= 4.0 for s in shifts) and len(set(shifts)) > 10
    assert fault_shift(c, 5) == fault_shift(dict(c), 5)  # depends on the trial number only
    f = apply_faults(dict(c), 5)
    s = fault_shift(c, 5)
    assert f["spikes"][0]["start_s"] == round(2.0 + s, 4) and f["disconnect"][0]["start_s"] == round(6.0 + s, 4)
    assert f["rise"][0]["start_s"] == round(2.0 + s, 4)
    assert f["latency"] == c["latency"] and f["loss"]["seed"] == 5  # not timed: not shifted; loss seeded per trial
    assert fault_shift({"spikes": c["spikes"]}, 5) == 0.0  # no shift unless asked for


def test_s_min_scale_follows_each_models_own_s_min():
    from drtc.control.run import scaled_s_min

    assert scaled_s_min({"s_min": 20}) == 20
    assert scaled_s_min({"s_min": 20, "s_min_scale": 0.5, "actions_per_chunk": 50}) == 10
    assert scaled_s_min({"s_min": 8, "s_min_scale": 1.5, "actions_per_chunk": 16}) == 12
    assert scaled_s_min({"s_min": 8, "s_min_scale": 3.0, "actions_per_chunk": 16}) == 15  # stays below the chunk


# --- smoothing -------------------------------------------------------------------------------------------------------


def test_temporal_ensemble_averages_every_prediction_of_a_step():
    from drtc.control.smoothing import TemporalEnsemble

    te = TemporalEnsemble(m=0.0)  # m = 0: a plain average
    first = te.apply(0, np.zeros((4, 1)))
    te.commit()
    assert np.allclose(first, 0.0)
    second = te.apply(2, np.full((4, 1), 6.0))  # steps 2, 3 were predicted before (0), 4, 5 are new
    te.commit()
    assert np.allclose(second[:, 0], [3.0, 3.0, 6.0, 6.0])
    third = te.apply(3, np.full((2, 1), 9.0))  # step 3: 0, 6, 9; step 4: 6, 9
    assert np.allclose(third[:, 0], [5.0, 7.5])
    te2 = TemporalEnsemble(m=1.0)  # m > 0: the older prediction weighs more (ACT)
    te2.apply(0, np.zeros((2, 1)))
    te2.commit()
    assert te2.apply(0, np.ones((2, 1)))[0, 0] < 0.5


def test_temporal_ensemble_forgets_a_chunk_that_was_not_accepted():
    from drtc.control.smoothing import TemporalEnsemble

    te = TemporalEnsemble(m=0.0)
    te.apply(0, np.zeros((4, 1)))
    te.commit()
    te.apply(0, np.full((4, 1), 100.0))  # rejected: no commit
    assert np.allclose(te.apply(0, np.full((4, 1), 2.0))[:, 0], 1.0)


def test_zero_lag_removes_jitter_without_lag():
    from scipy.signal import butter, sosfilt

    from drtc.control.smoothing import zero_lag

    t = np.arange(60) / 30.0
    clean = np.sin(2 * np.pi * 0.5 * t)[:, None] * 20.0  # a 0.5 Hz motion, 20 deg
    noisy = clean + np.random.default_rng(0).normal(0, 1.0, clean.shape)
    out = zero_lag(noisy[:15], noisy[15:], cutoff_hz=3.0, fps=30.0)
    assert out.shape == noisy[15:].shape
    err_noisy = np.abs(noisy[15:] - clean[15:]).mean()
    err_out = np.abs(out - clean[15:]).mean()
    assert err_out < 0.6 * err_noisy  # the jitter is gone
    causal = sosfilt(butter(2, 3.0 / 15.0, output="sos"), noisy, axis=0)[15:]
    assert err_out < np.abs(causal - clean[15:]).mean()  # and it follows the motion better than the causal filter
    assert np.allclose(zero_lag(None, noisy[:2], 3.0, 30.0), noisy[:2])  # too short to filter: unchanged


def test_resume_ramp_eases_from_the_held_pose_after_a_stall_only():
    from drtc.control.smoothing import ResumeRamp

    now = [0.0]
    r = ResumeRamp(steps=5, min_stall_s=0.1, clock=lambda: now[0])
    for i in range(3):  # normal ticks: passed through
        now[0] += 1 / 30
        assert r.apply(np.array([float(i)]))[0] == float(i)
    now[0] += 1.0  # a 1 s stall, then the new chunk starts 10 away
    out = []
    for _ in range(6):
        out.append(float(r.apply(np.array([12.0]))[0]))
        now[0] += 1 / 30
    assert out[0] < 3.0 and out == sorted(out) and out[-1] == 12.0  # eased, never past the target, then the stream
    assert r.ramps == 1


def test_soft_stop_slows_down_before_the_queue_runs_dry():
    from drtc.control.playback import AdaptivePlayback

    q = [float(i) for i in range(12)]
    pb = AdaptivePlayback(adaptive=False, soft_stop_steps=8, soft_stop_floor=0.15)
    rates = []
    for _ in range(40):
        pb.next(lambda: q.pop(0) if q else None, queued=len(q), ticks_until_chunk=0)
        rates.append(pb.rate)
    assert rates[0] == 1.0 and min(rates) < 0.5  # full speed, then slower over the last actions
    q = [float(i) for i in range(100)]
    pb = AdaptivePlayback(adaptive=False, soft_stop_steps=8)
    out = [pb.next(lambda: q.pop(0) if q else None, queued=len(q), ticks_until_chunk=0) for _ in range(5)]
    assert out == [0.0, 1.0, 2.0, 3.0, 4.0]  # with a full queue: exactly fixed playback


def test_every_v2_condition_parses_for_its_models():
    import draccus

    from drtc.control.engine import DRTCInferenceConfig
    from drtc.control.run import build

    reg = Registry()
    root = reg.lab / "experiments"
    exps = sorted(str(p.relative_to(root).with_suffix("")) for g in ("v2", "v2_robot") for p in (root / g).glob("*.yaml"))
    assert len(exps) >= 60
    for exp in exps:
        models = ["groot_sort_one_v1_full", "smolvla_cubebox_v1_full"] if "rtc" in exp else \
            ["act_sort_one_v1_full", "diffusion_cubebox_v1_full", "grootn17_sort_one_v1_full"]
        for m in models:
            cmd, composed, notes = build(reg, exp, m, "replay", 0, "tcp/127.0.0.1:1", 14, Path("/tmp/x"), "d", trial=3)
            assert not [n for n in notes if not n.startswith("rtc_")], (exp, notes)  # rtc_*: the models' old hints
            args = [x.replace("--inference.", "--") for x in cmd if x.startswith("--inference.") and "type=" not in x]
            draccus.parse(DRTCInferenceConfig, args=args)
            if exp.endswith("__none"):
                assert "fault_shift_applied_s" not in composed or composed["fault_shift_applied_s"] == 0.0


# --- growing evals ---------------------------------------------------------------------------------------------------


def test_a_plan_may_only_grow(tmp_path):
    from drtc.evals.run import plan_change_problem, plan_core

    base = ("description: d\nsuccess: s\ntime_limit_s: 30\ntrials: 10\nseed: 0\nmodels: [a, b]\n"
            "conditions: {c: v2_robot/upstream_patient__none}\nfrozen: {a: '1', b: '2'}\n")
    p = tmp_path / "e.yaml"
    p.write_text(base)
    old = plan_core(p)
    p.write_text(base.replace("trials: 10", "trials: 20").replace("[a, b]", "[a, b, c2]")
                 .replace("b: '2'}", "b: '2', c2: '3'}"))
    assert plan_change_problem(old, plan_core(p)) is None  # more trials, a model added: fine
    p.write_text(base.replace("trials: 10", "trials: 5"))
    assert "trials went down" in plan_change_problem(old, plan_core(p))
    p.write_text(base.replace("b: '2'", "b: '9'"))
    assert "model b" in plan_change_problem(old, plan_core(p))
    p.write_text(base.replace("time_limit_s: 30", "time_limit_s: 20"))
    assert "changed" in plan_change_problem(old, plan_core(p))


def test_the_resume_ramp_has_no_soft_stop():
    """Dropped 2026-11-22 (the local check): the ramp condition must not slow the queue down before a stall."""
    from drtc.control.run import build

    cmd, _, _ = build(Registry(), "v2/s_ramp__mix", "smolvla_sort_one_v1_full", "replay", 0, "tcp/127.0.0.1:1", 14,
                      Path("/tmp/x"), "d", trial=1)
    assert "--inference.resume_ramp=True" in cmd and "--inference.soft_stop_steps=0" in cmd
    assert "--inference.playback_mode=fixed" in cmd


def test_patient_systems_never_end_a_run_on_a_lost_reply_and_shipped_keeps_lerobots_defaults():
    from drtc.control.run import build

    reg = Registry()
    for exp in ("v2/upstream_patient__drops", "v2/drtc__drops", "v2/a1_no_cooldown__loss", "v2_robot/upstream_patient__mix"):
        cmd, _, _ = build(reg, exp, "act_sort_one_v1_full", "replay", 0, "tcp/127.0.0.1:1", 14, Path("/tmp/x"), "d")
        assert "--inference.action_timeout_s=14.0" in cmd and "--inference.action_starvation_grace_s=14.0" in cmd, exp
    cmd, _, _ = build(reg, "v2/upstream_shipped__drops", "act_sort_one_v1_full", "replay", 0, "tcp/127.0.0.1:1", 14,
                      Path("/tmp/x"), "d")
    assert not any(x.startswith("--inference.action_timeout_s") for x in cmd)
    assert "--inference.action_starvation_grace_s=1.0" in cmd


def test_a_run_the_system_ended_is_a_fault_whatever_key_was_pressed():
    from drtc.evals import store
    from drtc.evals.metrics import trial_metrics

    ended = {"outcome": "exit 1", "first_action_s": 0.4, "label": "f", "success": 0, "robot": "rig"}
    assert store.ended_by_system(ended) and store.label_of(ended) == "fault"
    assert not store.ended_by_system({"outcome": "exit 1", "first_action_s": None})  # before the first action: setup
    assert store.label_of({"outcome": "completed", "first_action_s": 0.4, "label": "f"}) == "f"
    import json, tempfile
    d = Path(tempfile.mkdtemp())
    (d / "trial.json").write_text(json.dumps(ended))
    m = trial_metrics(d)
    assert m["M1f_fault"] == 1 and m["M1_success"] == 0


def test_jerk_counts_only_while_the_arm_moves(tmp_path):
    """Metrics v3, M7: a long hold adds no zero-jerk samples; the stop and the restart still count."""
    import csv, json

    from drtc.evals.metrics import trial_metrics

    def run(name, hold_ticks):
        d = tmp_path / name
        d.mkdir()
        (d / "trial.json").write_text(json.dumps({"robot": "replay", "outcome": "completed", "first_action_s": 0.0}))
        cols = ["t", "action"] + [f"cmd.j{i}" for i in range(5)]
        rows, t, x = [], 0.0, 0.0
        rng = np.random.default_rng(0)
        for k in range(60):  # 2 s of shaky motion, a hold, 2 s more
            if k == 30:
                t += hold_ticks / 30.0
            x += 0.5 + rng.normal(0, 0.3)
            rows.append([round(t, 4), 1] + [x] * 5)
            t += 1 / 30.0
        rows.append([round(t, 4), 0] + [""] * 5)
        with open(d / "ticks.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        return trial_metrics(d)

    short, long_ = run("short", 15), run("long", 300)  # a 0.5 s and a 10 s hold
    assert long_["M2_stall_pct"] > short["M2_stall_pct"]
    assert abs(long_["M7_jerk_k"] - short["M7_jerk_k"]) < 0.05 * short["M7_jerk_k"]  # the hold does not dilute it
