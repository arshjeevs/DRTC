"""The results explorer (lab explore): byte ranges for the videos, the plain-words condition summary, the trial
timeline read from a trial's raw files, and the numbers it shows against lab eval report's code."""

import json

import pytest

from drtc.explorer import data as D
from drtc.explorer.server import parse_range


def test_parse_range():
    assert parse_range(None, 1000) is None
    assert parse_range("bytes=0-99", 1000) == (0, 99)
    assert parse_range("bytes=900-", 1000) == (900, 999)
    assert parse_range("bytes=-100", 1000) == (900, 999)
    assert parse_range("bytes=950-5000", 1000) == (950, 999)
    assert parse_range("bytes=1000-", 1000) is None
    assert parse_range("bytes=-", 1000) is None
    assert parse_range("items=0-1", 1000) is None


def test_describe_condition():
    c = D.describe_condition({"scheduler": "drtc", "playback_mode": "adaptive", "temporal_ensemble": True,
                              "action_filter_mode": "butterworth", "resume_ramp": True,
                              "spikes": [{"start_s": 2.0, "delay_ms": 1000}],
                              "drop_action": [{"start_s": 4.0, "duration_s": 1.0}],
                              "disconnect": [{"start_s": 6.0, "duration_s": 1.5}], "fault_shift_s": [0.0, 4.0]})
    assert c["drtc"] and c["smoothing"] == ["adaptive playback", "temporal ensembling", "butterworth filter",
                                            "resume ramp"]
    assert [(f["kind"], f["start"], f["end"]) for f in c["faults"]] == [("spike", 2.0, 3.0), ("drops", 4.0, 5.0),
                                                                        ("outage", 6.0, 7.5)]
    assert c["shift"] == [0.0, 4.0]
    p = D.describe_condition({"scheduler": "upstream"})
    assert not p["drtc"] and p["faults"] == [] and p["smoothing"] == []


def test_trial_detail(tmp_path):
    t = tmp_path / "t01"
    t.mkdir()
    (t / "trial.json").write_text(json.dumps({"cadence_hz": 30.0, "video": {"frames": {"scene": 90}},
                                              "first_action_s": 0.5, "command": ["x"]}))
    (t / "ticks.csv").write_text("t,action,queue,pending,latency_est,accepted,last_turnaround,horizon,playback_rate,"
                                 "cmd.shoulder_pan,meas.shoulder_pan\n0.5,1,10,0,,1,,24,1.0,1.0,1.1\n"
                                 "1.0,1,9,0,,1,,24,1.0,2.0,2.1\n3.0,0,0,0,,0,,24,,,\n")
    (t / "chunks.csv").write_text("t_sent,t_received,accepted,turnaround_s,estimate_at_send_s,server_total_s,steps\n"
                                  "0.1,0.5,1,0.4,,0.2,24\n")
    (t / "faults.csv").write_text("t,fault,value\n1.2,drop_action,\n")
    (t / "config.yaml").write_text("scheduler: drtc\ndrop_action:\n- {start_s: 1.0, duration_s: 1.0}\n"
                                   "fault_shift_applied_s: 0.5\n")
    d = D.trial_detail(t)
    assert d["joints"] == ["shoulder_pan"] and d["ticks"]["cmd"]["shoulder_pan"][:2] == [1.0, 2.0]
    assert d["windows"] == [{"kind": "drops", "start": 1.5, "end": 2.5}]
    assert d["events"][0]["event"] == "drop_action" and d["chunks"][0]["accepted"]
    assert "command" not in d["info"] and d["config"]["fault_shift_applied_s"] == 0.5
    # 90 frames at 30 Hz end at the last tick (3.0 s): the clip starts at 0.0 s and is not stretched
    assert d["video"]["t0"] == pytest.approx(0.0) and d["video"]["scale"] == pytest.approx(1.0)


def test_outcome_of():
    assert D.outcome_of({"label": "s", "outcome": "completed"}, True) == "success"
    assert D.outcome_of({"label": "f", "outcome": "exit 1", "first_action_s": 0.7}, True) == "fault"
    assert D.outcome_of({"label": "time_limit", "outcome": "completed"}, True) == "time_limit"
    assert D.outcome_of({"label": "s"}, False) == "not_counted"


def test_selection_round_trip(tmp_path):
    f = tmp_path / "videos.yaml"
    assert D.load_selection(f) is None  # nothing saved: every trial is shown
    known = {("e4", "s/c/m/t01"), ("e4", "s/c/m/t02"), ("base", "s/b/m/t01")}
    saved = D.save_selection({"e4": ["s/c/m/t02", "s/c/m/t01", "s/c/m/t02", "bogus"], "base": [], "other": ["x"]},
                             known, f)
    assert saved == {"e4": ["s/c/m/t02", "s/c/m/t01"]}  # order kept, duplicates, unknown ids and empty evals dropped
    assert D.load_selection(f) == saved
    assert f.read_text().startswith("# Written by `lab explore`")
    assert D.save_selection({}, known, f) == {} and D.load_selection(f) == {}  # an empty choice shows nothing
