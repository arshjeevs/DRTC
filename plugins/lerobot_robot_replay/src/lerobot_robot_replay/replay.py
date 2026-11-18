from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.robots import Robot

from .config_replay import ReplayRobotConfig


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

    def get_observation(self) -> dict[str, Any]:
        i = self._frame_index()
        item = self.ds[i]
        self.observations += 1
        if self.config.state == "dataset":
            self.state = item["observation.state"].numpy().astype(np.float64)
        else:  # first-order lag toward the last commanded target
            self.state = self.state + self.config.follow_alpha * (self.target - self.state)
        obs: dict[str, Any] = {j: float(v) for j, v in zip(self.joints, self.state)}
        for key in self.cams:
            img = item[f"observation.images.{key}"]  # C,H,W float in [0, 1]
            obs[key] = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.target = np.array([float(action[j]) for j in self.joints])
        self.commands.append({"t": time.monotonic(), "frame": self._frame_index(),
                              "action": [float(action[j]) for j in self.joints]})
        return {j: float(action[j]) for j in self.joints}
