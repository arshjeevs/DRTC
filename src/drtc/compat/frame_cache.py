"""Training reads cached frames from a prepared variant (drtc.data.prepare, recipe `data: {frames: [H, W]}`).

LeRobot 0.7's DatasetReader decodes each sample's camera frames from the dataset's videos (`_query_videos`). A
reader whose root holds drtc_frames/ (a variant `lab train` made) becomes a CachedDatasetReader, which returns the
cached frames instead: same keys, same layout ((3, H, W) for one timestamp, (T, 3, H, W) for several), uint8 like
training's own reads, only already resized to the policy's input size. Every other root decodes as before.

lerobot-train starts its DataLoader workers with `spawn` (fresh interpreters that import modules again), so a patch
of LeRobot's class in the training process would not reach them; until 2026-11-19 it did not, and training decoded
the full-size videos despite the cache. The reader's own class travels with the pickled dataset instead: a spawned
worker unpickles a CachedDatasetReader, importing this module. Lookup: row = the episode's dataset_from_index +
round(timestamp * fps); build_frames checked that the dataset's timestamps are frame_index / fps. Upstream code is
not edited.
"""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.datasets.dataset_reader import DatasetReader

FRAMES_DIR = "drtc_frames"
_open: dict[str, tuple[dict, dict] | None] = {}  # per process (each DataLoader worker opens its own memmaps)


def cache_for(root) -> tuple[dict, dict] | None:
    key = str(root)
    if key not in _open:
        meta_file = Path(root) / FRAMES_DIR / "meta.json"
        if not meta_file.is_file():
            _open[key] = None
        else:
            import numpy as np

            meta = json.loads(meta_file.read_text())
            arrays = {k: np.load(Path(root) / FRAMES_DIR / f"{k}.npy", mmap_mode="r") for k in meta["keys"]}
            _open[key] = (meta, arrays)
    return _open[key]


def read_frames(cache: tuple[dict, dict], query_timestamps: dict[str, list[float]], from_index: int,
                tolerance_s: float, return_uint8: bool = True) -> dict:
    import numpy as np
    import torch

    meta, arrays = cache
    fps = meta["fps"]
    out = {}
    for key, ts in query_timestamps.items():
        if key not in arrays:
            raise KeyError(f"camera {key!r} is not in the frame cache ({sorted(arrays)})")
        pos = [t * fps for t in ts]
        rows = [from_index + round(p) for p in pos]
        worst = max(abs(p - round(p)) for p in pos) / fps
        if worst > max(tolerance_s, 1e-6):
            raise ValueError(f"{key}: timestamps {ts} are {worst:.2e} s off the frame grid (tolerance {tolerance_s})")
        frames = torch.from_numpy(np.ascontiguousarray(arrays[key][rows]))
        if not return_uint8:
            frames = frames.to(torch.float32) / 255.0
        out[key] = frames.squeeze(0)
    return out


class CachedDatasetReader(DatasetReader):
    """A DatasetReader over a root with drtc_frames/: frames come from the cache, everything else as upstream."""

    def _query_videos(self, query_timestamps, ep_idx):
        cache = cache_for(self.root)
        if cache is None:  # the cache was removed: decode as upstream
            return super()._query_videos(query_timestamps, ep_idx)
        ep = self._meta.episodes[ep_idx]
        return read_frames(cache, query_timestamps, int(ep["dataset_from_index"]), self._tolerance_s,
                           self._return_uint8)


def install() -> None:
    """Every DatasetReader made from now on over a cached root is a CachedDatasetReader (and stays one in spawned
    DataLoader workers)."""
    if getattr(DatasetReader, "_drtc_frame_cache", False):
        return
    original_init = DatasetReader.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if type(self) is DatasetReader and cache_for(self.root) is not None:
            self.__class__ = CachedDatasetReader

    DatasetReader.__init__ = __init__
    DatasetReader._drtc_frame_cache = True
