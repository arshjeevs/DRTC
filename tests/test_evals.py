"""Eval files, rounds and the results layout (lab/EVALUATION.md, lab/METRICS.md rules 1-3)."""

import json

import yaml

from drtc.evals import spec as S
from drtc.evals import store
from drtc.registry import Registry


def make_spec(**kw):
    base = dict(description="t", success="done", time_limit_s=30, trials=3, models=["a", "b", "c"],
                conditions={"baseline": "replication/baseline"})
    base.update(kw)
    return S.EvalSpec.model_validate(base)


def test_rounds_cover_every_pair_and_are_seeded():
    sp = make_spec()
    n, pairs = S.next_round(sp, {})
    assert n == 1 and sorted(pairs) == sorted(sp.pairs())
    assert S.next_round(sp, {}) == (n, pairs)  # the same order on every machine
    assert S.next_round(make_spec(seed=1), {})[1] != pairs or len(pairs) < 3


def test_round_waits_for_the_pair_that_is_behind():
    sp = make_spec()
    counts = {("baseline", "a"): 2, ("baseline", "b"): 1, ("baseline", "c"): 2}
    assert S.next_round(sp, counts) == (2, [("baseline", "b")])
    full = {p: 3 for p in sp.pairs()}
    assert S.next_round(sp, full) == (0, [])


def test_write_frozen_replaces_only_the_block(tmp_path):
    path = tmp_path / "e.yaml"
    path.write_text("# keep me\ndescription: x  # and me\nmodels: [a]\n")
    S.write_frozen(path, {"a": "111"})
    S.write_frozen(path, {"a": "222"})
    text = path.read_text()
    assert "# keep me" in text and "# and me" in text
    assert text.count("frozen:") == 1 and text.count(S.FROZEN_HEADER) == 1
    assert yaml.safe_load(text)["frozen"] == {"a": "222"}


def test_voided_and_unlabelled_trials_do_not_count(tmp_path):
    sess = tmp_path / "e" / "sessions" / "20261120-1000-laptop"
    for n, info in enumerate([{"success": 1}, {"success": 0}, {"success": 1, "void": "cube off its mark"}, {}], 1):
        d = store.trial_dir(sess, "baseline", "a", n)
        d.mkdir(parents=True)
        (d / "trial.json").write_text(json.dumps(info))
    assert store.valid_counts("e", root=tmp_path) == {("baseline", "a"): 2}
    assert [s.name for s in store.unpushed("e", root=tmp_path)] == [sess.name]
    (sess / store.PUSHED).write_text("abc")
    assert store.unpushed("e", root=tmp_path) == []


def test_repo_evals_are_valid():
    reg = Registry()
    for path in sorted(S.evals_dir(reg).glob("*.yaml")):
        sp = S.load(reg, path.stem)
        assert S.check(reg, sp) == [], path.name


def _write_trial(d, ticks, chunks, info=None):
    import csv as _csv

    d.mkdir(parents=True)
    (d / "trial.json").write_text(json.dumps(info or {"success": 1, "time_to_success_s": 4.0}))
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    with open(d / "ticks.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["t", "action"] + [f"cmd.{n}" for n in names])
        for i, (a, x) in enumerate(ticks):
            w.writerow([i / 30, a] + ([x] * 5 + [50] if a else [""] * 6))
    with open(d / "chunks.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["turnaround_s", "estimate_at_send_s", "server_total_s", "overlap_steps", "switch_l2"])
        w.writerows(chunks)


def test_trial_metrics_follow_metrics_v1(tmp_path):
    from drtc.evals import metrics as M

    # 2 start-up ticks (not counted), then 10 moving ticks with a 3-tick stall in the middle
    ticks = [(0, None)] * 2 + [(1, float(i)) for i in range(4)] + [(0, None)] * 3 + [(1, float(i)) for i in range(4, 7)]
    chunks = [(0.10, "", 0.05, 0, ""), (0.12, 0.15, 0.06, 20, 2.0), (0.20, 0.14, 0.07, 18, 4.0)]
    _write_trial(tmp_path / "t", ticks, chunks)
    m = M.trial_metrics(tmp_path / "t")
    assert m["M1_success"] == 1 and m["M1t_time_to_success_s"] == 4.0
    assert m["M2_stall_pct"] == 30.0 and m["M3_stall_events"] == 1 and abs(m["M3_longest_stall_s"] - 0.1) < 1e-9
    assert abs(m["M4_turnaround_med_ms"] - 120) < 1e-6
    assert abs(m["M5_estimate_err_ms"] - (-1.5 * 10)) < 1e-6  # median of +30 and -60 ms
    assert abs(m["M6_server_ms"] - 60) < 1e-6
    assert m["M8a_switch_l2_deg"] == 3.0  # switches with an overlap only
    assert m["M8b_largest_step_deg"] == 1.0  # held through the stall, so no jump
    assert m["M7_jerk_k"] > 0


def test_wilson_and_tests():
    from drtc.evals import metrics as M

    lo, hi = M.wilson(14, 20)
    assert abs(lo - 0.481) < 0.002 and abs(hi - 0.855) < 0.002
    assert M.fisher_p(20, 20, 0, 20) < 1e-6
    assert M.mannwhitney_p([1], [2, 3]) is None


def test_void_marks_the_session_for_a_new_push(tmp_path):
    from drtc.evals.run import void_trial

    sess = tmp_path / "e" / "sessions" / "s1"
    d = store.trial_dir(sess, "baseline", "a", 1)
    d.mkdir(parents=True)
    (d / "trial.json").write_text(json.dumps({"success": 1}))
    (sess / store.PUSHED).write_text("abc")
    void_trial(d, "cube off its mark")
    assert json.loads((d / "trial.json").read_text())["void"] == "cube off its mark"
    assert not (sess / store.PUSHED).exists()


def test_stalls_count_ticks_that_were_never_logged(tmp_path):
    """While the arm holds, the control loop does not ask for actions: those ticks have no row (2026-11-20)."""
    from drtc.evals import metrics as M

    d = tmp_path / "t"
    _write_trial(d, [], [])
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    rows = [(i / 30, 1) for i in range(10)] + [(40 / 30, 1)] + [(60 / 30, 0)]  # a 1 s hold, then a stall to the end
    with open(d / "ticks.csv", "w") as f:
        f.write("t,action," + ",".join(f"cmd.{n}" for n in names) + "\n")
        for t, a in rows:
            f.write(f"{t},{a}," + ",".join(["1.0"] * 6 if a else [""] * 6) + "\n")
    m = M.trial_metrics(d)
    assert m["M3_stall_events"] == 2
    assert abs(m["M3_longest_stall_s"] - 30 / 30) < 1e-6  # ticks 10..39 had no action
    assert abs(m["M2_stall_pct"] - 100 * (30 + 19) / 60) < 1e-6  # and 41..59 up to the end


def test_cooldown_is_never_shorter_than_the_latency():
    from types import SimpleNamespace

    from drtc.control.engine import DRTCEngine

    rt = SimpleNamespace(turnaround=0.7, _completed=3, interval=1 / 30, horizon=16)
    eng = SimpleNamespace(runtime=rt, config=SimpleNamespace(s_min=8, epsilon=1, cooldown_cap="none",
                                                             startup_timeout_s=10.0))
    assert DRTCEngine.cooldown_s(eng) >= 0.7  # option b (2026-11-20)
    eng.config.cooldown_cap = "half_chunk"
    assert DRTCEngine.cooldown_s(eng) < 0.7  # DRTC's original cap: shorter than the reply
    rt._completed = 0
    assert DRTCEngine.cooldown_s(eng) == 10.0  # before the first reply: the startup timeout
    rt._completed, eng.config.cooldown_cap = 3, "none"
    rt.recent = [0.5, 0.9, 0.6]
    assert DRTCEngine.cooldown_s(eng) >= 0.9  # covers the slowest recent reply, not only the estimate


def test_request_delay_never_exceeds_the_chunk(monkeypatch):
    """'Invalid RTC delay' (2026-11-20): GR00T's 16-step chunk with a 0.56 s (17-step) turnaround."""
    from dataclasses import dataclass

    import lerobot.inference.execution as ex

    from drtc.control.engine import DRTCRuntime

    @dataclass(frozen=True)
    class Req:
        delay: int
        request_id: str = "r"
        submitted_at: float = 0.0

    monkeypatch.setattr(ex.ChunkRuntime, "begin", lambda self, obs: Req(17))
    rt = object.__new__(DRTCRuntime)
    rt.max_delay, rt.sent, rt._completed, rt.pending = 16, {}, 0, None
    out = rt.begin(None)
    assert out.delay == 16 and rt.pending.delay == 16
    rt.max_delay = 20
    assert rt.begin(None).delay == 17  # within the chunk: unchanged


def test_latency_fault_delays_replies_without_blocking():
    import queue as _q
    import time as _t

    from drtc.control.faults import FaultInjector, FaultPlan, _ReplyChannel

    inner = _q.Queue()
    inj = FaultInjector(FaultPlan.from_dict({"latency": {"base_ms": 200, "jitter_ms": 0}}))
    inj.start()
    ch = _ReplyChannel(inner, inj)
    inner.put(b"a")
    inner.put(b"b")
    t0 = _t.monotonic()
    try:
        ch.get(timeout=0.05)  # not there yet: a short wait times out, as on a slow network
        raise AssertionError("delivered too early")
    except _q.Empty:
        pass
    assert ch.get(timeout=1.0) == b"a" and _t.monotonic() - t0 >= 0.19
    assert ch.get(timeout=1.0) == b"b"  # order kept
    assert any(e[1] == "latency" for e in inj.events)


def test_adaptive_playback_is_fixed_with_a_full_queue_and_slows_when_short():
    from drtc.control.playback import AdaptivePlayback

    q = [float(i) for i in range(100)]
    pb = AdaptivePlayback(min_rate=0.5, margin_steps=4, slew=0.05)
    out = [pb.next(lambda: q.pop(0) if q else None, queued=len(q), ticks_until_chunk=5) for _ in range(5)]
    assert out == [0.0, 1.0, 2.0, 3.0, 4.0]  # full queue: one action per tick, exactly fixed playback

    q = [float(i) for i in range(6)]
    pb = AdaptivePlayback(min_rate=0.5, margin_steps=4, slew=0.5)
    out = [pb.next(lambda: q.pop(0) if q else None, queued=len(q), ticks_until_chunk=20) for _ in range(10)]
    moved = [x for x in out if x is not None]
    assert moved == sorted(moved) and any(x != int(x) for x in moved)  # slower, interpolated, never backwards
    assert len(moved) > 6  # the 6 actions last longer than 6 ticks
