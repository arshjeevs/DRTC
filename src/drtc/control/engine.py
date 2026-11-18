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

    def start(self) -> None:
        self._t0 = time.monotonic()
        if self.faults is not None:
            self.faults.start()  # fault windows are on the run's clock, from here
        super().start()

    def get_action(self, obs_frame):
        action = super().get_action(obs_frame)
        rt = self.runtime
        est = rt.turnaround if rt._completed else None
        self._ticks.append([
            round(time.monotonic() - self._t0, 4), int(action is not None), rt.queue.qsize(), int(rt.pending is not None),
            None if est is None or not math.isfinite(est) else round(est, 4), rt._completed,
            None if getattr(rt, "last_turnaround", None) is None else round(rt.last_turnaround, 4),
            getattr(rt, "horizon", None),
        ])
        return action

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
            if path and self.faults is not None:
                with open(Path(path).with_name("faults.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "fault", "value"])
                    w.writerows(self.faults.events)


def install() -> None:
    """Make rollout build DRTCEngine for remote sessions (it behaves as upstream's for a plain remote config)."""
    import lerobot.rollout.context as context

    context.RemoteInferenceEngine = DRTCEngine
