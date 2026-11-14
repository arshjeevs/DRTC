"""Latency-adaptive playback for the DRTC robot client (lab addition).

With fixed playback the client executes one scheduled action per control tick. When a chunk is short
compared with the round-trip time (GR00T: 16 actions, ~8-9 ticks of latency), the queue runs dry before
the next chunk arrives and the robot stops and starts: jerky motion and stall ticks.

Adaptive playback advances a fractional action phase by a rate r <= 1 per tick and sends the action
interpolated between the two scheduled actions around the phase. The rate follows how long the queued
actions must last:

    r_target = clip(queued / (ticks_until_next_chunk + margin), min_rate, 1)

with ticks_until_next_chunk from DRTC's own latency estimate (minus the ticks since the in-flight
request was sent). The rate changes by at most `slew` per tick, so speed changes are gradual. With a
full queue r = 1 and the phase stays on whole steps: exactly the fixed behaviour. The schedule is keyed
by action step, so actions used up during the latency cost fewer steps when slowed down, and a larger
part of every new chunk is still ahead of the phase when it arrives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PlaybackStep:
    step: int  # scheduled action step at or before the phase (the client's action_step)
    action: np.ndarray  # action to send this tick
    src_control_step: int
    chunk_start_step: int
    advanced: float  # phase advance this tick (the playback rate actually used)


class AdaptivePlayback:
    def __init__(self, min_rate: float = 0.5, margin_steps: float = 4.0, slew: float = 0.05):
        self.min_rate = min_rate
        self.margin = margin_steps
        self.slew = slew
        self.rate = 1.0
        self.phase: float | None = None
        self._base: tuple[int, np.ndarray, int, int] | None = None  # (step, action, src, chunk_start)

    def target_rate(self, queued: int, ticks_until_chunk: float) -> float:
        need = max(ticks_until_chunk, 0.0) + self.margin
        if need <= 0 or queued >= need:
            return 1.0
        return max(self.min_rate, queued / need)

    def next(self, schedule, ticks_until_chunk: float) -> PlaybackStep | None:
        """Advance one tick through `schedule` (an ActionSchedule). None: nothing to move to (stall)."""
        if self._base is None:
            first = schedule.pop_front()
            if first is None:
                return None
            self._base = first
            self.phase = float(first[0])
            return PlaybackStep(first[0], first[1], first[2], first[3], 1.0)

        target = self.target_rate(schedule.get_size(), ticks_until_chunk)
        self.rate += min(max(target - self.rate, -self.slew), self.slew)
        previous = self.phase
        phase = previous + self.rate

        # Step over every scheduled action the phase has reached.
        while (front := schedule.peek_front()) is not None and front[0] <= math.floor(phase + 1e-9):
            self._base = schedule.pop_front()
        step, action, src, chunk_start = self._base
        front = schedule.peek_front()
        if front is None:
            phase = min(phase, float(step))  # nothing further scheduled: hold at the last action
        self.phase = phase
        if phase <= previous:
            return None
        if front is None or phase <= step:
            out = action
        else:
            frac = (phase - step) / (front[0] - step)
            out = action + frac * (np.asarray(front[1]) - action)
        return PlaybackStep(step, out, src, chunk_start, phase - previous)
