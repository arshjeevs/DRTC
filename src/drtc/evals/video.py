"""Trial video (lab/docs/EVALUATION.md): every camera of the rig, as the robot computer saw it, one MP4 per camera.

Frames come from the engine's observation hook (drtc.control.engine.DRTCEngine.observers) and are encoded in a
background thread (H.264, libx264 'veryfast'), so the control loop never waits: when the encoder falls behind, frames
are dropped and counted (`dropped` in trial.json), never queued without bound.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np


class TrialVideo:
    def __init__(self, out_dir: Path, cameras: dict[str, str], fps: int = 30, crf: int = 28, preset: str = "veryfast"):
        """cameras: observation key -> file name (the camera's role, e.g. wrist, scene)."""
        self.out_dir, self.cameras, self.fps, self.crf, self.preset = out_dir, cameras, fps, crf, preset
        self.q: queue.Queue = queue.Queue(maxsize=fps * 2)
        self.frames = {k: 0 for k in cameras}
        self.dropped = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="trial-video", daemon=True)
        self._t0: float | None = None

    def start(self) -> "TrialVideo":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        return self

    def observe(self, obs: dict) -> None:
        """Called on the control thread with every robot observation; never blocks."""
        if self._stop.is_set():
            return
        frames = {k: obs[k] for k in self.cameras if isinstance(obs.get(k), np.ndarray)}
        if not frames:
            return
        try:
            self.q.put_nowait((time.monotonic(), frames))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        import av

        containers, streams = {}, {}
        try:
            while not (self._stop.is_set() and self.q.empty()):
                try:
                    t, frames = self.q.get(timeout=0.1)
                except queue.Empty:
                    continue
                for key, img in frames.items():
                    if key not in containers:
                        h, w = img.shape[:2]
                        c = av.open(str(self.out_dir / f"{self.cameras[key]}.mp4"), mode="w")
                        s = c.add_stream("libx264", rate=self.fps)
                        s.width, s.height, s.pix_fmt = w - w % 2, h - h % 2, "yuv420p"
                        s.options = {"crf": str(self.crf), "preset": self.preset}
                        containers[key], streams[key] = c, s
                    s = streams[key]
                    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img[: s.height, : s.width, :3]),
                                                       format="rgb24")
                    for packet in s.encode(frame):
                        containers[key].mux(packet)
                    self.frames[key] += 1
        except Exception as e:  # noqa: BLE001 - a video problem never stops a trial; it is recorded
            self.error = f"{type(e).__name__}: {e}"
        finally:
            for key, c in containers.items():
                try:
                    for packet in streams[key].encode():
                        c.mux(packet)
                    c.close()
                except Exception:  # noqa: BLE001
                    pass

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=30)
        return {"files": {self.cameras[k]: f"video/{self.cameras[k]}.mp4" for k in self.cameras if self.frames[k]},
                "frames": {self.cameras[k]: n for k, n in self.frames.items()}, "dropped": self.dropped,
                "codec": f"libx264 crf {self.crf} {self.preset}", "error": self.error}
