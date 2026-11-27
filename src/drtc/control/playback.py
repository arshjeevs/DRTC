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

Soft stop (setup v2, 2026-11-22; dropped from the resume-ramp method the same day, off by default: see
drtc.control.smoothing): with soft_stop_steps > 0,
when at most that many actions are left the target rate also falls with them, max(floor, left / soft_stop_steps),
so the arm slows down over its last actions instead of stopping at full speed when the queue runs dry (a fault, a
late reply). The rate may then fall by up to `slew_down` per tick (default: max(slew, 0.15)), fast enough to slow
down within those few actions; it rises again by `slew` when a chunk arrives. With `adaptive` False only the soft
stop sets the rate (the resume ramp alone, without adaptive playback).
"""

from __future__ import annotations


class AdaptivePlayback:
    def __init__(self, min_rate: float = 0.5, margin_steps: float = 4.0, slew: float = 0.05, *, adaptive: bool = True,
                 soft_stop_steps: int = 0, soft_stop_floor: float = 0.15, slew_down: float | None = None):
        self.min_rate, self.margin, self.slew = min_rate, margin_steps, slew
        self.adaptive, self.soft_steps, self.soft_floor = adaptive, int(soft_stop_steps), soft_stop_floor
        self.slew_down = slew_down if slew_down is not None else (max(slew, 0.15) if self.soft_steps else slew)
        self.rate = 1.0
        self.frac = 0.0
        self.a0 = None  # the action at the phase's whole step
        self.a1 = None  # the next action (already taken from the queue)

    def reset(self) -> None:
        self.rate, self.frac, self.a0, self.a1 = 1.0, 0.0, None, None

    def target_rate(self, queued: int, ticks_until_chunk: float) -> float:
        rate = 1.0
        if self.adaptive:
            need = max(ticks_until_chunk, 0.0) + self.margin
            if need > 0 and queued < need:
                rate = max(self.min_rate, queued / need)
        if self.soft_steps and queued <= self.soft_steps:
            rate = min(rate, max(self.soft_floor, queued / self.soft_steps))
        return rate

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
        self.rate += min(max(target - self.rate, -self.slew_down), self.slew)
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
