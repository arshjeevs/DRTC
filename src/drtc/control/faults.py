"""Network fault injection at the remote-inference transport (the 10 replication conditions of the minor).

Faults are injected where a real network would cause them, so upstream's own logic responds (deadlines, stale-reply
discarding, starvation recovery, faults):
- outgoing observations (the client's publish on the "/obs" key): drop_obs, dup_obs, reorder_obs;
- incoming replies (the client's reply channel): drop_action, dup_action, reorder_action, spikes (a one-time extra
  delay before a reply is delivered, like a slow inference);
- disconnect: both directions dropped.
Each fault is a list of windows {start_s, duration_s} (spikes: {start_s, delay_ms}) on the run's clock, as in
lab/experiments/replication. Every injected fault is recorded (faults.csv).

With upstream's capacity-one runtime (one request in flight) a duplicate reply is rejected by its identity checks,
and a held-back message has no later one to swap with until a reply or the request deadline; lost messages end in
the client's request deadline (action_timeout_s), which upstream treats as a terminal fault.
"""

from __future__ import annotations

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

    @classmethod
    def from_dict(cls, spec: dict) -> "FaultPlan":
        windows = {k: [(float(w["start_s"]), float(w["duration_s"])) for w in spec.get(k) or []] for k in KINDS}
        spikes = [[float(s["start_s"]), float(s["delay_ms"]) / 1000.0, False] for s in spec.get("spikes") or []]
        return cls({k: v for k, v in windows.items() if v}, spikes)

    def __bool__(self) -> bool:
        return bool(self.windows or self.spikes)

    def active(self, kind: str, t: float) -> bool:
        return any(start <= t < start + dur for start, dur in self.windows.get(kind, ()))


class _ReplyChannel:
    """Wraps the client's reply channel: drop, duplicate, hold back or delay incoming replies."""

    def __init__(self, inner, injector: "FaultInjector"):
        self._inner, self._inj = inner, injector
        self._stash: deque = deque()  # duplicates and released held-back replies, delivered first
        self._held = None

    def __getattr__(self, name):  # oversized, dropped, ... pass through
        return getattr(self._inner, name)

    def get(self, timeout=None):
        if self._stash:
            return self._stash.popleft()
        payload = self._inner.get(timeout=timeout)  # raises Empty like the real channel
        inj = self._inj
        t = inj.now()
        if inj.plan.active("disconnect", t) or inj.plan.active("drop_action", t):
            inj.record("drop_action" if not inj.plan.active("disconnect", t) else "disconnect_reply", t)
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
        for s in self.plan.spikes:
            if not s[2] and t >= s[0]:
                s[2] = True
                return s[1]
        return None

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
