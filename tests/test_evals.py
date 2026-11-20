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
