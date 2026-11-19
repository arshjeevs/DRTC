import pytest

from drtc.data.edit import fmt_ranges, next_layout, parse_episodes, read_sidecar, remap, write_sidecar

ROT = ["a", "b", "c", "d"]


def test_next_layout_follows_the_rotation_when_balanced():
    assert next_layout(ROT, {}) == "a"
    assert next_layout(ROT, {0: "a"}) == "b"
    assert next_layout(ROT, {0: "a", 1: "b", 2: "c", 3: "d"}) == "a"


def test_next_layout_fills_what_a_deletion_left_missing():
    # a, b, c, d, a, b recorded; then the 2nd "a" deleted (renumbered): a, b, c, d, b
    assert next_layout(ROT, {0: "a", 1: "b", 2: "c", 3: "d", 4: "b"}) == "c"  # a, c, d tied at 1; after b
    assert next_layout(ROT, {0: "c", 1: "c"}) == "d"  # least recorded first, in rotation order after the last


def test_remap_after_deleting_episodes():
    d = {"layouts": {0: "a", 1: "b", 2: "c", 3: "d"}, "trim": {3: (5, 90), 1: (0, 10)}}
    assert remap(d, {1}, 4) == {"layouts": {0: "a", 1: "c", 2: "d"}, "trim": {2: (5, 90)}}


def test_parse_and_format_episode_ranges():
    assert parse_episodes(["3", "5-7,9"], 10) == [3, 5, 6, 7, 9]
    assert fmt_ranges([3, 5, 6, 7, 9]) == "3, 5-7, 9"
    with pytest.raises(SystemExit):
        parse_episodes(["10"], 10)


def test_sidecar_roundtrip_and_atomic_write(tmp_path):
    import os

    (tmp_path / "meta").mkdir()
    write_sidecar(tmp_path, {"layouts": {2: "b"}, "trim": {1: (3, 40)}})
    link = tmp_path / "snapshot.json"
    os.link(tmp_path / "meta/drtc.json", link)  # as the undo snapshot holds it
    write_sidecar(tmp_path, {"layouts": {2: "c"}, "trim": {}})
    assert read_sidecar(tmp_path) == {"layouts": {2: "c"}, "trim": {}}
    assert '"b"' in link.read_text()  # the snapshot's copy is untouched


def test_record_prompts_store_each_episodes_layout_and_keep_it_on_rerecord(tmp_path, monkeypatch, capsys):
    """lerobot-record's loop, reduced to its calls: say, (re-record), reset, save."""
    import types

    from lerobot.datasets import LeRobotDataset

    from drtc.data.record import install_layout_prompts

    events = {"rerecord_episode": False}
    fake = types.SimpleNamespace(log_say=lambda *a, **k: None, init_keyboard_listener=lambda: (None, events))
    monkeypatch.setattr(LeRobotDataset, "save_episode", lambda self: setattr(self, "num_episodes", self.num_episodes + 1))
    install_layout_prompts(fake, ROT, {0: "a"})  # one episode already recorded
    ds = types.SimpleNamespace(num_episodes=1, root=tmp_path)
    (tmp_path / "meta").mkdir()
    fake.init_keyboard_listener()
    fake.log_say("Recording episode 1")
    events["rerecord_episode"] = True
    fake.log_say("Reset the environment")
    events["rerecord_episode"] = False
    fake.log_say("Recording episode 1")
    fake.log_say("Reset the environment")
    LeRobotDataset.save_episode(ds)
    fake.log_say("Recording episode 2")
    LeRobotDataset.save_episode(ds)
    out = capsys.readouterr().out
    assert "re-record episode 1: same layout, b" in out and "set up episode 2: c" in out
    assert read_sidecar(tmp_path)["layouts"] == {0: "a", 1: "b", 2: "c"}


def test_prune_old_revisions_keeps_the_current_and_ones_in_use(tmp_path, monkeypatch):
    import mmap

    from drtc.data import prepare

    monkeypatch.setattr(prepare, "variant_root", lambda repo, rev, need: tmp_path / repo / f"{rev[:10]}_x")
    repo = tmp_path / "acc/ds"
    for name in ("aaaaaaaaaa_frames", "aaaaaaaaaa_rel", "bbbbbbbbbb_frames", "cccccccccc_frames"):
        (repo / name).mkdir(parents=True)
        (repo / name / "f.bin").write_bytes(b"\0" * 4096)
    with open(repo / "cccccccccc_frames" / "f.bin", "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ):
        prepare.prune_old_revisions("acc/ds", "aaaaaaaaaa1234")  # a training memory-maps c's frames
    assert sorted(d.name for d in repo.iterdir()) == ["aaaaaaaaaa_frames", "aaaaaaaaaa_rel", "cccccccccc_frames"]
