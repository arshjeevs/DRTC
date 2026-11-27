"""Smoothing methods of setup v2 (lab/PLAN.md, 2026-11-22), next to adaptive playback (playback.py) and the
Butterworth filter (action_filter.py). Each one targets one source of non-smooth motion:

- TemporalEnsemble (seams, every model): when a chunk is accepted, each queued step becomes the weighted average of
  every chunk that predicted that step, ACT's temporal ensembling (Zhao et al. 2023): w_i = exp(-m * i), i = 0 for
  the oldest prediction. With m > 0 the older predictions weigh more (ACT's default m = 0.01: almost an equal
  average). Kept per step of the runtime's queue cursor, so it follows exactly what was executed.
- zero_lag (jitter, no lag): when a chunk is accepted, the recent executed actions plus the new queued future are
  smoothed forward and backward (zero-phase Butterworth, scipy sosfiltfilt) and the future part is queued. A causal
  filter (the Butterworth filter) only sees the past and lags; here the future is known (it is in the queue), so
  smoothing in both directions removes the jitter without delaying the motion. Redone at every accepted chunk.
- ResumeRamp (the stops and jumps of faults): after a stall (no action for at least `min_stall_s`), the first actions
  are eased from the held pose into the policy's stream with a minimum-jerk blend
  w(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 over `steps` actions (zero speed and acceleration at the start, the stream's
  speed at the end), instead of jumping to the new chunk's first action. A soft stop before the stall was its other
  half (AdaptivePlayback soft_stop_steps) and is off since 2026-11-22: in the local check (SmolVLA, the fault mix)
  it kept the queue from running dry, so a reply made from a 1.3 s old observation was accepted and the next fresh
  chunk jumped (largest step 38-42 deg against 18 deg; stalls 23% against 15%). Without it the ramp lowered the jerk
  (42 against 54-60) and the largest step (14 against 16-18 deg).
"""

from __future__ import annotations

import math
import time

import numpy as np


class TemporalEnsemble:
    def __init__(self, m: float = 0.01):
        self.m = m
        self.state: dict[int, tuple[np.ndarray, float, int]] = {}  # cursor step -> (weighted sum, weight sum, count)
        self._pending: tuple[int, dict] | None = None

    def reset(self) -> None:
        self.state.clear()
        self._pending = None

    def apply(self, cursor: int, future: np.ndarray) -> np.ndarray:
        """The ensembled future for the queue steps cursor, cursor + 1, ... (call commit() if it was accepted)."""
        out = np.empty_like(future, dtype=np.float64)
        pending = {}
        for k, a in enumerate(np.asarray(future, dtype=np.float64)):
            s, w, n = self.state.get(cursor + k, (None, 0.0, 0))
            weight = math.exp(-self.m * n)  # the n-th prediction of this step (0 = the first, oldest)
            s = a * weight if s is None else s + a * weight
            w, n = w + weight, n + 1
            pending[cursor + k] = (s, w, n)
            out[k] = s / w
        self._pending = (cursor, pending)
        return out

    def commit(self) -> None:
        if self._pending is None:
            return
        _, pending = self._pending
        # steps before the cursor are executed and steps beyond the new chunk are no longer queued: keep only the
        # new chunk's steps
        self.state = pending
        self._pending = None


def zero_lag(history: np.ndarray | None, future: np.ndarray, cutoff_hz: float, fps: float,
             order: int = 2) -> np.ndarray:
    """Zero-phase low-pass of history + future along time; returns the smoothed future."""
    from scipy.signal import butter, sosfiltfilt

    future = np.asarray(future, dtype=np.float64)
    h = 0 if history is None or len(history) == 0 else len(history)
    x = future if h == 0 else np.concatenate([np.asarray(history, dtype=np.float64), future], axis=0)
    if len(x) < 4 or len(future) == 0:
        return future
    sos = butter(order, min(max(cutoff_hz / (fps / 2.0), 0.01), 0.99), btype="low", output="sos")
    # scipy's default pad length (3 * (2 * sections + 1) samples) cannot exceed the signal
    padlen = min(3 * (2 * len(sos) + 1), len(x) - 1)
    y = sosfiltfilt(sos, x, axis=0, padlen=padlen)
    return np.ascontiguousarray(y[h:])  # sosfiltfilt returns a reversed view (negative strides): torch refuses it


def min_jerk(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


class ResumeRamp:
    def __init__(self, steps: int = 10, min_stall_s: float = 0.1, clock=time.monotonic):
        self.steps, self.min_stall_s, self.clock = max(1, int(steps)), float(min_stall_s), clock
        self.held: np.ndarray | None = None  # the last position sent
        self.last_t: float | None = None
        self.start: np.ndarray | None = None
        self.k: int | None = None
        self.ramps = 0  # how many resumes were eased (for the logs)

    def apply(self, action: np.ndarray | None) -> np.ndarray | None:
        now = self.clock()
        if action is None:
            return None
        action = np.asarray(action, dtype=np.float64)
        if self.held is not None and self.last_t is not None and now - self.last_t >= self.min_stall_s:
            self.start, self.k = self.held.copy(), 0
            self.ramps += 1
        out = action
        if self.k is not None:
            self.k += 1
            tau = self.k / self.steps
            if tau >= 1.0:
                self.k = self.start = None
            else:
                out = self.start + min_jerk(tau) * (action - self.start)
        self.held, self.last_t = out.copy(), now
        return out
