import pytest

from drtc.data.edit import fmt_ranges, parse_episodes, read_sidecar, remap, write_sidecar


def test_remap_after_deleting_episodes():
    d = {"trim": {3: (5, 90), 1: (0, 10)}}
    assert remap(d, {1}, 4) == {"trim": {2: (5, 90)}}


def test_parse_and_format_episode_ranges():
    assert parse_episodes(["3", "5-7,9"], 10) == [3, 5, 6, 7, 9]
    assert fmt_ranges([3, 5, 6, 7, 9]) == "3, 5-7, 9"
    with pytest.raises(SystemExit):
        parse_episodes(["10"], 10)


def test_sidecar_roundtrip_and_atomic_write(tmp_path):
    import os

    (tmp_path / "meta").mkdir()
    write_sidecar(tmp_path, {"trim": {1: (3, 40)}})
    link = tmp_path / "snapshot.json"
    os.link(tmp_path / "meta/drtc.json", link)  # as the undo snapshot holds it
    write_sidecar(tmp_path, {"trim": {}})
    assert read_sidecar(tmp_path) == {"trim": {}}
    assert "40" in link.read_text()  # the snapshot's copy is untouched


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
