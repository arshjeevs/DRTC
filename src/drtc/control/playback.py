"""Smoothing method A: latency-adaptive playback (ported from the minor's async_inference/playback.py, 2026-11-20).

With fixed playback the robot executes one action per control tick. When a chunk is short compared with the round
trip (GR00T: 16 actions against 0.5-0.7 s over WiFi), the queue runs dry before the next chunk arrives and the arm
stops and starts. Adaptive playback advances a fractional phase by a rate r <= 1 per tick and sends the position
interpolated between the two actions around the phase:

    r_target = clip(queued / (ticks_until_next_chunk + margin), min_rate, 1)

with ticks_until_next_chunk from DRTC's latency estimate minus the ticks since the request in flight was sent. The
rate changes by at most `slew` per tick. With a full queue r = 1: exactly fixed playback. The runtime's queue (and
its aligned cursor) advances only when the phase passes a whole action, so new chunks are aligned to what was
actually executed, and slowed-down actions leave more of every new chunk ahead.
"""

from __future__ import annotations


class AdaptivePlayback:
    def __init__(self, min_rate: float = 0.5, margin_steps: float = 4.0, slew: float = 0.05):
        self.min_rate, self.margin, self.slew = min_rate, margin_steps, slew
        self.rate = 1.0
        self.frac = 0.0
        self.a0 = None  # the action at the phase's whole step
        self.a1 = None  # the next action (already taken from the queue)

    def reset(self) -> None:
        self.rate, self.frac, self.a0, self.a1 = 1.0, 0.0, None, None

    def target_rate(self, queued: int, ticks_until_chunk: float) -> float:
        need = max(ticks_until_chunk, 0.0) + self.margin
        if need <= 0 or queued >= need:
            return 1.0
        return max(self.min_rate, queued / need)

    def next(self, pop, queued: int, ticks_until_chunk: float):
        """One tick. `pop()` takes the next action from the runtime (None when there is none). Returns the position
        to send, or None (nothing to move to: a stall)."""
        if self.a0 is None:
            self.a0 = pop()
            if self.a0 is None:
                return None
            self.frac, self.rate = 0.0, 1.0
            return self.a0
        target = self.target_rate(queued + (1 if self.a1 is not None else 0), ticks_until_chunk)
        self.rate += min(max(target - self.rate, -self.slew), self.slew)
        self.frac += self.rate
        while self.frac >= 1.0 - 1e-9:
            if self.a1 is None:
                self.a1 = pop()
                if self.a1 is None:  # nothing further: hold at the last action
                    self.frac = 0.0
                    return None
            self.a0, self.a1 = self.a1, None
            self.frac -= 1.0
        if self.frac <= 1e-9:
            return self.a0
        if self.a1 is None:
            self.a1 = pop()
            if self.a1 is None:  # the next action is not there yet: a stall, counted like fixed playback's
                self.frac = 0.0
                return None
        return self.a0 + self.frac * (self.a1 - self.a0)
