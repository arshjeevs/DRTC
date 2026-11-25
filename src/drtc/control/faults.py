"""Network fault injection at the remote-inference transport (the 10 replication conditions of the minor).

Faults are injected where a real network would cause them, so upstream's own logic responds (deadlines, stale-reply
discarding, starvation recovery, faults):
- outgoing observations (the client's publish on the "/obs" key): drop_obs, dup_obs, reorder_obs;
- incoming replies (the client's reply channel): drop_action, dup_action, reorder_action, spikes (a one-time extra
  delay before a reply is delivered, like a slow inference);
- disconnect: both directions dropped;
- rise (setup v2, 2026-11-22): extra delay on every reply that arrives inside a window, {start_s, duration_s,
  delay_ms}, on top of the steady `latency` (a network that gets slower and stays slower for a while);
- loss (setup v2): each reply is lost with probability p, {p, seed[, start_s, duration_s]} (default: the whole run).
Each fault is a list of windows {start_s, duration_s} (spikes: {start_s, delay_ms}) on the run's clock, as in
lab/experiments/replication. Every injected fault is recorded (faults.csv).

With upstream's capacity-one runtime (one request in flight) a duplicate reply is rejected by its identity checks,
and a held-back message has no later one to swap with until a reply or the request deadline; lost messages end in
the client's request deadline (action_timeout_s), which upstream treats as a terminal fault.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty

KINDS = ("drop_obs", "drop_action", "dup_obs", "dup_action", "reorder_obs", "reorder_action", "disconnect")


@dataclass
class FaultPlan:
    windows: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    spikes: list[list] = field(default_factory=list)  # [start_s, delay_s, used]
    # a steady network delay on every reply: (base_s, jitter_s, seed); jitter is uniform in [0, jitter]
    latency: tuple[float, float, int] | None = None
    rises: list[tuple[float, float, float]] = field(default_factory=list)  # (start_s, duration_s, delay_s)
    # (p, seed, start_s, duration_s); duration None = until the end of the run
    loss: tuple[float, int, float, float | None] | None = None

    @classmethod
    def from_dict(cls, spec: dict) -> "FaultPlan":
        windows = {k: [(float(w["start_s"]), float(w["duration_s"])) for w in spec.get(k) or []] for k in KINDS}
        spikes = [[float(s["start_s"]), float(s["delay_ms"]) / 1000.0, False] for s in spec.get("spikes") or []]
        lat = spec.get("latency")
        latency = None
        if lat:
            latency = (float(lat.get("base_ms", 0)) / 1000.0, float(lat.get("jitter_ms", 0)) / 1000.0,
                       int(lat.get("seed", 0)))
        rises = [(float(r["start_s"]), float(r["duration_s"]), float(r["delay_ms"]) / 1000.0)
                 for r in spec.get("rise") or []]
        lo = spec.get("loss")
        loss = None
        if lo and float(lo.get("p", 0)) > 0:
            dur = lo.get("duration_s")
            loss = (float(lo["p"]), int(lo.get("seed", 0)), float(lo.get("start_s", 0.0)),
                    None if dur is None else float(dur))
        return cls({k: v for k, v in windows.items() if v}, spikes, latency, rises, loss)

    def __bool__(self) -> bool:
        return bool(self.windows or self.spikes or self.latency or self.rises or self.loss)

    def rise_delay(self, t: float) -> float:
        """The extra reply delay of the rise windows active at t (0 outside them)."""
        return sum(d for start, dur, d in self.rises if start <= t < start + dur)

    def loss_active(self, t: float) -> bool:
        if self.loss is None:
            return False
        _, _, start, dur = self.loss
        return t >= start and (dur is None or t < start + dur)

    def active(self, kind: str, t: float) -> bool:
        return any(start <= t < start + dur for start, dur in self.windows.get(kind, ()))


class _ReplyChannel:
    """Wraps the client's reply channel: drop, duplicate, hold back or delay incoming replies."""

    def __init__(self, inner, injector: "FaultInjector"):
        self._inner, self._inj = inner, injector
        self._stash: deque = deque()  # duplicates and released held-back replies, delivered first
        self._held = None
        self._delayed: deque = deque()  # (release time, payload): the steady network delay (latency)
        lat = injector.plan.latency
        # the delayed path (a reply is held until its release time) serves the steady latency and the rise windows
        self._rng = random.Random(lat[2]) if lat else (random.Random(0) if injector.plan.rises else None)
        loss = injector.plan.loss
        self._loss_rng = random.Random(f"loss-{loss[1]}") if loss else None

    def __getattr__(self, name):  # oversized, dropped, ... pass through
        return getattr(self._inner, name)

    def get(self, timeout=None):
        if self._stash:
            return self._stash.popleft()
        if self._rng is not None:
            payload = self._delayed_get(timeout)
        else:
            payload = self._inner.get(timeout=timeout)  # raises Empty like the real channel
        return self._faults(payload)

    def _delayed_get(self, timeout):
        """Hold every reply until its release time, never blocking longer than the caller's timeout: like a slow
        network, a late reply is just not there yet (so the cooldown sees it as late)."""
        base, jitter, _ = self._inj.plan.latency or (0.0, 0.0, 0)
        deadline = time.monotonic() + (timeout if timeout is not None else 1e9)
        while True:
            now = time.monotonic()
            if self._delayed and self._delayed[0][0] <= now:
                return self._delayed.popleft()[1]
            try:
                wait = max(0.0, min(deadline, self._delayed[0][0] if self._delayed else deadline) - now)
                payload = self._inner.get(timeout=max(wait, 0.001))
                release = now + base + self._rng.uniform(0.0, jitter) + self._inj.plan.rise_delay(self._inj.now())
                if self._delayed:
                    release = max(release, self._delayed[-1][0])  # a network keeps order
                self._delayed.append((release, payload))
                self._inj.record("latency", self._inj.now(), round(release - now, 4))
            except Empty:
                pass
            if time.monotonic() >= deadline:
                raise Empty

    def _faults(self, payload):
        inj = self._inj
        t = inj.now()
        if inj.plan.active("disconnect", t) or inj.plan.active("drop_action", t):
            inj.record("drop_action" if not inj.plan.active("disconnect", t) else "disconnect_reply", t)
            raise Empty
        if self._loss_rng is not None and inj.plan.loss_active(t) and self._loss_rng.random() < inj.plan.loss[0]:
            inj.record("loss", t)
            raise Empty
        spike = inj.due_spike(t)
        if spike is not None:
            inj.record("spike", t, spike)
            time.sleep(spike)
        if inj.plan.active("reorder_action", t):
            if self._held is None:
                self._held = payload
                inj.record("reorder_action_hold", t)
                raise Empty
            held, self._held = self._held, None
            self._stash.append(held)
            inj.record("reorder_action_swap", t)
        elif self._held is not None:  # window over: release what was held
            self._stash.append(self._held)
            self._held = None
        if inj.plan.active("dup_action", t):
            self._stash.append(payload)
            inj.record("dup_action", t)
        return payload


class FaultInjector:
    def __init__(self, plan: FaultPlan):
        self.plan = plan
        self.t0: float | None = None
        self.events: list[tuple[float, str, float | None]] = []
        self._lock = threading.Lock()
        self._held_obs = None

    def start(self) -> None:
        self.t0 = time.monotonic()

    def now(self) -> float:
        return 0.0 if self.t0 is None else time.monotonic() - self.t0

    def record(self, kind: str, t: float, value: float | None = None) -> None:
        with self._lock:
            self.events.append((round(t, 4), kind, value))

    def due_spike(self, t: float) -> float | None:
        with self._lock:
            for s in self.plan.spikes:
                if not s[2] and t >= s[0]:
                    s[2] = True
                    return s[1]
        return None

    # Live faults (the demo console's buttons), from another thread while the run goes on. The readers iterate the
    # plan's lists without the lock; appending to a list or adding a dict key is atomic in CPython, so they see the
    # new fault from their next check on.
    def spike_now(self, delay_s: float = 1.0) -> float:
        t = self.now()
        with self._lock:
            self.plan.spikes.append([t, float(delay_s), False])
        return t

    def drop_replies_now(self, duration_s: float = 1.0) -> float:
        return self._window_now("drop_action", duration_s)

    def outage_now(self, duration_s: float = 1.5) -> float:
        return self._window_now("disconnect", duration_s)

    def _window_now(self, kind: str, duration_s: float) -> float:
        t = self.now()
        with self._lock:
            self.plan.windows.setdefault(kind, []).append((t, float(duration_s)))
        return t

    def install(self, client) -> None:
        """Wrap the client's transport publish (observations) and its reply channel (actions)."""
        transport = client.transport
        publish = transport.publish

        def faulty_publish(key, payload, *args, **kwargs):
            if not str(key).endswith("/obs"):
                return publish(key, payload, *args, **kwargs)
            t = self.now()
            if self.plan.active("disconnect", t) or self.plan.active("drop_obs", t):
                self.record("drop_obs" if not self.plan.active("disconnect", t) else "disconnect_obs", t)
                return None
            if self.plan.active("reorder_obs", t):
                if self._held_obs is None:
                    self._held_obs = (key, payload)
                    self.record("reorder_obs_hold", t)
                    return None
                held, self._held_obs = self._held_obs, None
                publish(key, payload, *args, **kwargs)
                self.record("reorder_obs_swap", t)
                return publish(held[0], held[1], *args, **kwargs)
            if self._held_obs is not None:
                held, self._held_obs = self._held_obs, None
                publish(held[0], held[1], *args, **kwargs)
            result = publish(key, payload, *args, **kwargs)
            if self.plan.active("dup_obs", t):
                publish(key, payload, *args, **kwargs)
                self.record("dup_obs", t)
            return result

        transport.publish = faulty_publish
        client._actions = _ReplyChannel(client._actions, self)
