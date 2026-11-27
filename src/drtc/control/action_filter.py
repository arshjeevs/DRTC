"""Action filter applied to every action before it goes to the robot (the minor's client default).

Ported unchanged from the minor's DRTC client (lerobot/async_inference/utils/action_filter.py, tag minor-final):
a causal Butterworth low-pass per joint (default order 2, cutoff 3 Hz at the control rate), then a gain on the
step-to-step change (default 1.4) to compensate the filter's amplitude loss. The filter keeps its state across chunk
boundaries (that is what smooths the seam between chunks); it is reset when motion is invalidated.
"""

from __future__ import annotations

import numpy as np


class ButterworthFilter:
    def __init__(self, cutoff: float, fps: float, order: int = 2, gain: float = 1.4):
        if cutoff <= 0 or fps <= 0:
            raise ValueError("cutoff and fps must be positive")
        self.cutoff, self.fps, self.order, self.gain = cutoff, fps, order, gain
        self._sos = None
        self._zi = None
        self._prev: np.ndarray | None = None

    def _init(self, action: np.ndarray) -> None:
        from scipy.signal import butter, sosfilt_zi

        normalized = min(max(self.cutoff / (self.fps / 2.0), 0.01), 0.99)
        self._sos = butter(self.order, normalized, btype="low", output="sos")
        zi = sosfilt_zi(self._sos)
        self._zi = np.array([zi * a for a in action])

    def apply(self, action: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt

        action = np.asarray(action, dtype=np.float64)
        if self._sos is None:  # the first action passes through and seeds the filter state
            self._init(action)
            self._prev = action.copy()
            return action
        filtered = np.empty_like(action)
        for j in range(len(action)):
            out, self._zi[j] = sosfilt(self._sos, [action[j]], zi=self._zi[j])
            filtered[j] = out[0]
        if self.gain != 1.0 and self._prev is not None:
            filtered = self._prev + (filtered - self._prev) * self.gain
        self._prev = filtered.copy()
        return filtered

    def reset(self) -> None:
        self._sos = self._zi = self._prev = None
