"""DRTC scheduling and per-tick metrics on LeRobot 0.7 remote inference: `--inference.type=drtc`.

DRTCInferenceConfig extends upstream's RemoteInferenceConfig, so rollout builds it through the same remote path
(admission, handshake, hold, starvation recovery, faults); lab run swaps in DRTCEngine for it. Only the scheduling
inside upstream's ChunkRuntime changes (DRTCRuntime), and every upstream safety gate stays: one request in flight,
observation freshness, generations, deadlines, latched faults.

Scheduler (lab/PLAN.md, decision 2026-11-18, option a: DRTC's trigger and estimators on upstream's runtime):
- "drtc":     request when the queued actions are <= H - s_min (H: the last accepted chunk's length): Jack Vial's
              DRTC trigger. With one request in flight the pending gate plays the role of DRTC's cooldown.
- "upstream": request when queued playback <= max(refill_seconds, turnaround + one action interval).
Estimator (the turnaround used by the upstream rule and by RTC's delay estimate):
- "upstream": max of the last 100 steady-state turnarounds (upstream's own);
- "jk":       Jacobson-Karels, smoothed mean + k * smoothed deviation (RFC 6298 alpha/beta), as in DRTC;
- "max_last_10": max of the last 10 turnarounds, as in DRTC's comparison estimator;
- "fixed":    a constant (fixed_latency_s).
Cooldown re-request (option b, decided 2026-11-18): with cooldown on, a request unanswered for its cooldown
(latency estimate in steps, capped at H/2, plus epsilon steps; as in DRTC) is given up and a fresh observation is
sent (freshest wins). Inside upstream's server rules: a BUSY reply (the old inference still running) backs off one
action interval; a STALE "duplicate" reply for the request in flight (a duplicated message) is ignored. Every
recovery action is logged (recovery.csv).
Metrics: one row per control tick (queue length, whether an action was available, request pending, latency
estimate, accepted chunks, last turnaround), written as CSV when the engine stops.
"""

from __future__ import annotations

import csv
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lerobot.inference.execution import ChunkRuntime
from lerobot.inference.factory import InferenceEngineConfig, RemoteInferenceConfig
from lerobot.remote_inference.engine import RemoteInferenceEngine


# --- latency estimators ----------------------------------------------------------------------------------------------


class JKEstimator:
    """Jacobson-Karels: l_hat = smoothed RTT + k * smoothed |error| (DRTC's default)."""

    def __init__(self, alpha: float = 0.125, beta: float = 0.25, k: float = 1.5):
        self.alpha, self.beta, self.k = alpha, beta, k
        self.mean: float | None = None
        self.dev = 0.0

    def update(self, rtt: float) -> None:
        if self.mean is None:
            self.mean, self.dev = rtt, 0.0
            return
        err = rtt - self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * rtt
        self.dev = (1 - self.beta) * self.dev + self.beta * abs(err)

    def estimate(self) -> float | None:
        return None if self.mean is None else self.mean + self.k * self.dev


class MaxWindowEstimator:
    def __init__(self, window: int = 10):
        self.buf: deque[float] = deque(maxlen=window)

    def update(self, rtt: float) -> None:
        self.buf.append(rtt)

    def estimate(self) -> float | None:
        return max(self.buf) if self.buf else None


class FixedEstimator:
    def __init__(self, value: float):
        self.value = value

    def update(self, rtt: float) -> None:
        pass

    def estimate(self) -> float | None:
        return self.value


# --- runtime ---------------------------------------------------------------------------------------------------------


class DRTCRuntime(ChunkRuntime):
    def __init__(self, *, scheduler: str, estimator, s_min: int, **kwargs):
        super().__init__(**kwargs)
        self.scheduler = scheduler
        self.estimator = estimator  # None = upstream's own window
        self.s_min = s_min
        self.horizon: int | None = None  # H: the last accepted chunk's length
        self.last_turnaround: float | None = None
        self.not_before = 0.0  # backoff after a BUSY reply

    def abandon(self, request_id: str, backoff: float = 0.0) -> bool:
        """Give up the request in flight (its late reply is then discarded as stale)."""
        with self.lock:
            if self.pending is None or self.pending.request_id != request_id:
                return False
            self.pending = None
            # Upstream's aligned mode refuses another request until the cursor advances past the last request's
            # anchor (anti-spam); after giving one up, the next fresh observation must be allowed at once.
            self._last_anchor = None
            if backoff:
                self.not_before = self.clock() + backoff
            return True

    def should_request(self, *, task_version: int | None = None) -> bool:
        if self.clock() < self.not_before:
            return False
        return super().should_request(task_version=task_version)

    @property
    def turnaround(self) -> float:
        if self.estimator is None:
            return super().turnaround
        est = self.estimator.estimate()
        return super().turnaround if est is None else est

    @property
    def effective_refill(self) -> float:
        if self.scheduler == "drtc" and self.horizon is not None:
            return max(self.horizon - self.s_min, 1) * self.interval
        return super().effective_refill

    def accept(self, request, chunk, *, task_version: int) -> bool:
        completed = self._completed
        submitted = request.submitted_at
        accepted = super().accept(request, chunk, task_version=task_version)
        if self._completed > completed:  # super measured this request's turnaround
            rtt = self.clock() - submitted
            self.last_turnaround = rtt
            if self.estimator is not None:
                self.estimator.update(rtt)
        if accepted:
            self.horizon = int(chunk.execution_steps)
        return accepted


# --- config and engine -----------------------------------------------------------------------------------------------


@InferenceEngineConfig.register_subclass("drtc")
@dataclass
class DRTCInferenceConfig(RemoteInferenceConfig):
    scheduler: Literal["drtc", "upstream"] = "drtc"
    estimator: Literal["jk", "max_last_10", "fixed", "upstream"] = "jk"
    s_min: int = 20
    latency_k: float = 1.5
    latency_alpha: float = 0.125
    latency_beta: float = 0.25
    fixed_latency_s: float = 0.1
    metrics_path: str | None = None  # per-tick CSV, written when the engine stops
    faults_json: str | None = None  # network faults to inject (drtc.control.faults), JSON
    cooldown: bool = True  # option b: give up an unanswered request after its cooldown and send a fresh one
    epsilon: int = 1  # cooldown margin, action steps
    # Action filter on every action before it goes to the robot (the minor client's default: butterworth).
    action_filter: Literal["none", "butterworth"] = "none"
    filter_cutoff_hz: float = 3.0
    filter_order: int = 2
    filter_gain: float = 1.4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.s_min < 1:
            raise ValueError("s_min must be at least 1")


def make_estimator(cfg: DRTCInferenceConfig):
    if cfg.estimator == "jk":
        return JKEstimator(cfg.latency_alpha, cfg.latency_beta, cfg.latency_k)
    if cfg.estimator == "max_last_10":
        return MaxWindowEstimator(10)
    if cfg.estimator == "fixed":
        return FixedEstimator(cfg.fixed_latency_s)
    return None


TICK_FIELDS = ["t", "action", "queue", "pending", "latency_est", "accepted", "last_turnaround", "horizon"]


class DRTCEngine(RemoteInferenceEngine):
    """Upstream's remote engine with the DRTC runtime and per-tick metrics (for DRTCInferenceConfig)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        if isinstance(cfg, DRTCInferenceConfig):
            old = self.runtime
            self.runtime = DRTCRuntime(
                scheduler=cfg.scheduler, estimator=make_estimator(cfg), s_min=cfg.s_min,
                mode=old.mode, action_interval=old.interval, refill_seconds=old.refill_seconds,
                max_observation_age_s=old.max_age, action_timeout_s=old.action_timeout,
                startup_timeout_s=old.startup_timeout, action_starvation_grace_s=old.starvation_grace,
                training_max_delay=old.training_max_delay, chunk_merge=old.chunk_merge,
                blend_steps=old.blend_steps, blend_weight=old.blend_weight, blend_indices=old.blend_indices,
            )
            self.latency = self.runtime.latency
        self._ticks: list[list] = []
        self._t0 = time.monotonic()
        self.action_filter = None
        if isinstance(cfg, DRTCInferenceConfig) and cfg.action_filter == "butterworth":
            from drtc.control.action_filter import ButterworthFilter

            self.action_filter = ButterworthFilter(cfg.filter_cutoff_hz, 1.0 / self.runtime.interval,
                                                   cfg.filter_order, cfg.filter_gain)
        self.faults = None
        spec = getattr(cfg, "faults_json", None)
        if spec:
            import yaml

            from drtc.control.faults import FaultInjector, FaultPlan

            # The CLI parser (draccus) may hand the JSON over re-printed as a Python dict; YAML reads both.
            plan = FaultPlan.from_dict(spec if isinstance(spec, dict) else yaml.safe_load(spec))
            if plan:
                self.faults = FaultInjector(plan)
                self.faults.install(self.client)

        self.recovery: list[tuple[float, str, str]] = []
        if isinstance(cfg, DRTCInferenceConfig) and cfg.cooldown:
            self.client._exchange = self._cooldown_exchange(self.client._exchange)

    def cooldown_s(self) -> float:
        """DRTC: latency estimate in steps (capped at H/2) + epsilon steps."""
        rt = self.runtime
        est = rt.turnaround if rt._completed else None
        if est is None or not math.isfinite(est):
            # Before the first measurement (cold start is slow), seed with s_min + epsilon steps, as DRTC does.
            return (self.config.s_min + self.config.epsilon) * rt.interval
        steps = max(1, math.ceil(est / rt.interval))
        if getattr(rt, "horizon", None):
            steps = min(steps, max(1, rt.horizon // 2))
        return max(2, steps + self.config.epsilon) * rt.interval

    def _note(self, kind: str, request_id: str) -> None:
        self.recovery.append((round(time.monotonic() - self._t0, 4), kind, request_id))

    def _cooldown_exchange(self, original):
        """The client's observation exchange with DRTC's cooldown re-request (other exchanges unchanged)."""
        import lerobot.remote_inference.client as client_mod
        from queue import Empty

        from lerobot.remote_inference.client import RequestCancelled
        from lerobot.transport.protocol import decode_reply
        from lerobot.transport.protocol.envelope import ErrorCode, MessageType, ProtocolError

        engine = self

        def exchange(request, suffix, channel, timeout, cancelled):
            if suffix != "/obs":
                return original(request, suffix, channel, timeout, cancelled)
            client = engine.client
            if cancelled():
                raise RequestCancelled()
            if not client.present:
                raise ConnectionError("Server presence lost; session cannot recover")
            started = time.monotonic()
            payload = client_mod.encode_message(request)
            if cancelled():
                raise RequestCancelled()
            client.transport.publish(client._key + suffix, payload)
            cooldown = engine.cooldown_s()
            while time.monotonic() - started <= timeout:
                if cancelled():
                    raise RequestCancelled()
                if channel.oversized or channel.dropped:
                    raise ProtocolError(ErrorCode.MALFORMED, "Inference reply channel overflow or oversized message")
                try:
                    payload = channel.get(timeout=min(0.02, timeout))
                except Empty:
                    if time.monotonic() - started > cooldown:
                        if engine.runtime.abandon(request.request_id):
                            engine._note("cooldown_resend", request.request_id)
                        raise RequestCancelled()
                    continue
                response = decode_reply(payload, request, discard_stale=True)
                if response is None:
                    continue
                if response.message_type is MessageType.ERROR:
                    code, message = response.body.get("code"), str(response.body.get("message", ""))
                    if code == ErrorCode.STALE.value and "Duplicate" in message:
                        engine._note("duplicate_ignored", request.request_id)
                        continue  # a duplicated copy of this request; its real reply is still coming
                    if code == ErrorCode.BUSY.value:
                        engine.runtime.abandon(request.request_id, backoff=engine.runtime.interval)
                        engine._note("busy_backoff", request.request_id)
                        raise RequestCancelled()
                return response
            raise TimeoutError("Remote inference request deadline exceeded")

        return exchange

    def start(self) -> None:
        self._t0 = time.monotonic()
        if self.faults is not None:
            self.faults.start()  # fault windows are on the run's clock, from here
        super().start()

    def get_action(self, obs_frame):
        action = super().get_action(obs_frame)
        if action is not None and self.action_filter is not None:
            import torch

            filtered = self.action_filter.apply(action.detach().cpu().numpy())
            action = torch.as_tensor(filtered, dtype=action.dtype, device=action.device)
        rt = self.runtime
        est = rt.turnaround if rt._completed else None
        self._ticks.append([
            round(time.monotonic() - self._t0, 4), int(action is not None), rt.queue.qsize(), int(rt.pending is not None),
            None if est is None or not math.isfinite(est) else round(est, 4), rt._completed,
            None if getattr(rt, "last_turnaround", None) is None else round(rt.last_turnaround, 4),
            getattr(rt, "horizon", None),
        ])
        return action

    def _invalidate(self, operation: str) -> None:
        if self.action_filter is not None:
            self.action_filter.reset()  # motion restarts from a held pose: do not smooth across it
        super()._invalidate(operation)

    def stop(self) -> None:
        try:
            super().stop()
        finally:
            path = getattr(self.config, "metrics_path", None)
            if path and self._ticks:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(TICK_FIELDS)
                    w.writerows(self._ticks)
            if path and self.recovery:
                with open(Path(path).with_name("recovery.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "action", "request_id"])
                    w.writerows(self.recovery)
            if path and self.faults is not None:
                with open(Path(path).with_name("faults.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "fault", "value"])
                    w.writerows(self.faults.events)


def install() -> None:
    """Make rollout build DRTCEngine for remote sessions (it behaves as upstream's for a plain remote config)."""
    import lerobot.rollout.context as context

    context.RemoteInferenceEngine = DRTCEngine
