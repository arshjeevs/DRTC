"""The new trimming (drtc.data.trim, LeRobot 0.7) must reproduce the trimmed dataset made in the old setup.

so101_cubebox_v1 was made on 2026-11-17 from so101_cubebox_v1_raw by lab/tools/autotrim.py + the
lerobot-dataset-editor; its crops are in so101_cubebox_v1/_editor/autotrim.json. Here:
- detection on the raw dataset finds the same noise floor and the same crops;
- applying them gives the same data rows, episode windows and metadata, and matching statistics.
Skipped where the datasets are not local.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drtc.data.trim import Params, RawDataset, apply, detect

H = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot")) / "orangemoon599"
RAW, OLD = H / "so101_cubebox_v1_raw", H / "so101_cubebox_v1"
pytestmark = pytest.mark.skipif(not (RAW / "meta/info.json").is_file() or not (OLD / "_editor/autotrim.json").is_file(),
                                reason="needs so101_cubebox_v1_raw and so101_cubebox_v1 locally")


@pytest.fixture(scope="module")
def detected():
    ds = RawDataset(RAW)
    results, noise, floor = detect(ds, Params())
    return ds, results, noise, floor


@pytest.fixture(scope="module")
def trimmed(detected, tmp_path_factory):
    ds, results, _, _ = detected
    out = tmp_path_factory.mktemp("trim") / "so101_cubebox_v1"
    apply(ds, results, out, quantiles="weighted_mean")  # the old setup's convention, to compare like with like
    return out


def test_same_crops_as_the_old_setup(detected):
    _, results, noise, floor = detected
    old = json.loads((OLD / "_editor/autotrim.json").read_text())
    assert noise == pytest.approx(old["noise"], rel=1e-9)
    assert floor == pytest.approx(old["floor"], rel=1e-9)
    assert [(r.episode, r.start, r.end, r.status) for r in results] == \
        [(e["episode"], e["start"], e["end"], e["status"]) for e in old["episodes"]]


def _episodes(root: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))],
                     ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def test_same_data_rows(trimmed):
    for f in sorted((OLD / "data").rglob("*.parquet")):
        old = pd.read_parquet(f).sort_values("index").reset_index(drop=True)
        new = pd.read_parquet(trimmed / f.relative_to(OLD)).sort_values("index").reset_index(drop=True)
        assert list(new.columns) == list(old.columns)
        for c in old.columns:
            a, b = old[c].to_numpy(), new[c].to_numpy()
            if a.dtype == object:
                a, b = np.stack(a), np.stack(b)
            np.testing.assert_array_equal(b, a, err_msg=c)


def test_same_episode_windows(trimmed):
    old, new = _episodes(OLD), _episodes(trimmed)
    for c in old.columns:
        if c.startswith("stats/"):
            continue
        if c.endswith("_timestamp"):
            np.testing.assert_allclose(new[c].to_numpy(float), old[c].to_numpy(float), atol=1e-6, err_msg=c)
        elif c == "tasks":
            assert [list(x) for x in new[c]] == [list(x) for x in old[c]]
        else:
            np.testing.assert_array_equal(new[c].to_numpy(), old[c].to_numpy(), err_msg=c)


def test_matching_statistics(trimmed):
    old = json.loads((OLD / "meta/stats.json").read_text())
    new = json.loads((trimmed / "meta/stats.json").read_text())
    assert set(new) == set(old)
    for feat in old:
        for stat in old[feat]:
            a, b = np.asarray(old[feat][stat], float), np.asarray(new[feat][stat], float)
            np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-6, err_msg=f"{feat}/{stat}")


def test_episode_statistics_are_exact(trimmed):
    """Per-episode numeric stats equal the values computed directly from the trimmed frames (float64).

    Not compared with the old editor's: it computed std in float32 and is off by up to ~10% for joints with a small
    spread around a large mean (e.g. the wrist roll: std 0.15 around -54); the new values are the exact ones.
    """
    eps = _episodes(trimmed).set_index("episode_index")
    data = pd.concat([pd.read_parquet(f) for f in sorted((trimmed / "data").rglob("*.parquet"))])
    for key in ("action", "observation.state"):
        for ep, d in data.groupby("episode_index"):
            x = np.stack(d[key].to_numpy()).astype(np.float64)
            for stat, value in (("min", x.min(0)), ("max", x.max(0)), ("mean", x.mean(0)), ("std", x.std(0))):
                np.testing.assert_allclose(np.asarray(eps.loc[ep, f"stats/{key}/{stat}"], float), value,
                                           rtol=1e-5, atol=1e-5, err_msg=f"episode {ep} {key}/{stat}")


def test_info_and_videos(trimmed):
    old_info = json.loads((OLD / "meta/info.json").read_text())
    new_info = json.loads((trimmed / "meta/info.json").read_text())
    for k in ("total_episodes", "total_frames", "fps", "features", "codebase_version"):
        assert new_info[k] == old_info[k], k
    import filecmp

    for f in (RAW / "videos").rglob("*.mp4"):
        new = trimmed / f.relative_to(RAW)
        if os.stat(f).st_dev == os.stat(new).st_dev:  # same filesystem: must be a hardlink (no extra disk)
            assert os.path.samefile(f, new), f"{f.name} is not a hardlink of the raw video"
        else:  # across filesystems a copy is made: byte-identical
            assert filecmp.cmp(f, new, shallow=False), f"{f.name} differs from the raw video"
