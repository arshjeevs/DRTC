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
