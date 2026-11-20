from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.robots import Robot

from .config_replay import ReplayRobotConfig

# Decoded episodes, shared by every replay robot in this process (lab eval run runs all its trials in one process):
# (repo, revision, episode) -> list of (state, {camera: HWC uint8}). Decoding AV1 frames on every control tick took
# 73 ms on the shared H200 (2026-11-20), so the loop ran at 12 Hz instead of 30 Hz; decoded once before the trial, a
# tick only copies a frame. About 0.7 GB per 240-frame episode of two 800x600 cameras; the oldest is dropped beyond
# _CACHE_MAX.
from collections import OrderedDict

_CACHE: "OrderedDict[tuple, list]" = OrderedDict()
_CACHE_MAX = 8


def preload(repo_id: str, revision: str | None, episode: int, root=None) -> int:
    """Decode one episode into the shared cache ahead of time (lab eval run does it before the first trial)."""
    from lerobot.datasets import LeRobotDataset

    key = (repo_id, revision, episode)
    if key in _CACHE:
        return len(_CACHE[key])
    ds = LeRobotDataset(repo_id, root=root, revision=revision, episodes=[episode])
    cams = [k.removeprefix("observation.images.") for k, f in ds.meta.features.items() if f["dtype"] == "video"]
    frames = []
    for i in range(len(ds)):
        item = ds[i]
        frames.append((item["observation.state"].numpy().astype(np.float64),
                       {c: (item[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                        for c in cams}))
    _CACHE[key] = frames
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return len(frames)


class ReplayRobot(Robot):
    config_class = ReplayRobotConfig
    name = "replay"
    # The wrapper keeps re-sending the last target while waiting for actions; for a replay that is exactly a hold.
    supports_position_hold = True

    def __init__(self, config: ReplayRobotConfig):
        super().__init__(config)
        self.config = config
        self.ds = None
        self.t0: float | None = None
        self.state: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.commands: list[dict] = []
        self.observations = 0
        from lerobot.datasets import LeRobotDatasetMetadata

        self.meta = LeRobotDatasetMetadata(config.repo_id, root=config.root, revision=config.revision)
        names = self.meta.features["observation.state"]["names"]
        self.joints = list(names)
        self.cams = {k.removeprefix("observation.images."): tuple(f["shape"])
                     for k, f in self.meta.features.items() if f["dtype"] == "video"}

    # --- features -------------------------------------------------------------------------------------------

    @property
    def observation_features(self) -> dict:
        return {**{j: float for j in self.joints}, **{k: tuple(s) for k, s in self.cams.items()}}

    @property
    def action_features(self) -> dict:
        return {j: float for j in self.joints}

    # --- lifecycle ------------------------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self.ds is not None

    def connect(self, calibrate: bool = True) -> None:
        from lerobot.datasets import LeRobotDataset

        c = self.config
        self.ds = LeRobotDataset(c.repo_id, root=c.root, revision=c.revision, episodes=[c.episode])
        self.length = len(self.ds)
        self.frames = None
        if c.preload:
            key = (c.repo_id, c.revision, c.episode)
            if key not in _CACHE:
                _CACHE[key] = [self._decode(self.ds[i]) for i in range(self.length)]
                while len(_CACHE) > _CACHE_MAX:
                    _CACHE.popitem(last=False)
            _CACHE.move_to_end(key)
            self.frames = _CACHE[key]
        first = self.ds[0]
        self.state = first["observation.state"].numpy().astype(np.float64)
        self.target = self.state.copy()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if self.config.log_path and self.commands:
            Path(self.config.log_path).write_text("\n".join(json.dumps(c) for c in self.commands) + "\n")
        self.ds = None

    # --- I/O ------------------------------------------------------------------------------------------------

    def _frame_index(self) -> int:
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        i = int((now - self.t0) * self.meta.fps)
        if i >= self.length:
            i = i % self.length if self.config.loop else self.length - 1
        return i

    def _decode(self, item) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        state = item["observation.state"].numpy().astype(np.float64)
        images = {key: (item[f"observation.images.{key}"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                  for key in self.cams}  # C,H,W float in [0, 1] -> H,W,C uint8, as a camera delivers it
        return state, images

    def get_observation(self) -> dict[str, Any]:
        i = self._frame_index()
        state, images = self.frames[i] if self.frames is not None else self._decode(self.ds[i])
        self.observations += 1
        if self.config.state == "dataset":
            self.state = state
        else:  # first-order lag toward the last commanded target
            self.state = self.state + self.config.follow_alpha * (self.target - self.state)
        obs: dict[str, Any] = {j: float(v) for j, v in zip(self.joints, self.state)}
        for key in self.cams:
            obs[key] = images[key].copy()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.target = np.array([float(action[j]) for j in self.joints])
        self.commands.append({"t": time.monotonic(), "frame": self._frame_index(),
                              "action": [float(action[j]) for j in self.joints]})
        return {j: float(action[j]) for j in self.joints}
