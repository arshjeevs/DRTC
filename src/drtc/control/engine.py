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
Metrics (lab/METRICS.md v1), written as CSV when the engine stops:
- ticks.csv, one row per action request of the control loop (while the arm holds through a stall the loop does not
  ask, so stalls are the gaps between executed actions; the last row marks the end of the run): queue length, whether an action was available, request pending, latency
  estimate, accepted chunks, last turnaround, chunk length; the commanded joints (cmd.<joint>) and the latest measured
  joints (meas.<joint>), in the model's dataset units;
- chunks.csv, one row per chunk reply: when its request was sent and its reply arrived, turnaround, the latency
  estimate at send (M5), the server's own timings (M6), the chunk's length, and at each switch to a new chunk the
  overlap with the queued actions and their mean arm-joint distance (M8a, DRTC's chunk_mean_l2).
"""

from __future__ import annotations

import csv
import math
import time
from collections import deque

import numpy as np
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
        self.sent: dict[str, tuple[float, float | None]] = {}  # request_id -> (sent at, latency estimate then)
        self.max_delay: int | None = None  # the server's prediction steps (set by the engine from the handshake)
        self.recent: deque[float] = deque(maxlen=10)  # the last measured turnarounds (the cooldown covers them)
        self.chunk_log: list[list] = []  # CHUNK_FIELDS rows, monotonic times
        # setup v2 smoothing at each accepted chunk (drtc.control.smoothing), set by the engine
        self.ensemble = None  # TemporalEnsemble
        self.zero_lag: tuple[float, int] | None = None  # (cutoff_hz, history length)
        self.history: deque = deque(maxlen=15)  # the last executed actions (zero-lag smoothing's past)

    def invalidate(self, *, held: bool = False, preserve_starvation: bool = False) -> int:
        # motion restarts from a held pose: no ensembling or smoothing across it (the resume ramp eases the restart)
        if getattr(self, "ensemble", None) is not None:
            self.ensemble.reset()
        if getattr(self, "history", None) is not None:
            self.history.clear()
        return super().invalidate(held=held, preserve_starvation=preserve_starvation)

    def pop(self):
        item = super().pop()
        if item is not None and self.zero_lag is not None:
            self.history.append(item[0].detach().cpu().float().numpy().copy())
        return item

    def _accept_aligned(self, request, actions, provenance, now):
        if self.ensemble is None and self.zero_lag is None:
            return super()._accept_aligned(request, actions, provenance, now)
        import torch

        snap = self.queue.snapshot()
        anchor = request.observation.action_cursor
        trim = None if anchor is None else snap.cursor - anchor
        if trim is None or trim < 0 or trim >= len(actions):
            return super()._accept_aligned(request, actions, provenance, now)  # upstream rejects it the same way
        future = actions[trim:].detach().cpu().double().numpy()
        if self.ensemble is not None:
            future = self.ensemble.apply(snap.cursor, future)
        if self.zero_lag is not None:
            from drtc.control.smoothing import zero_lag

            hist = np.stack(list(self.history)) if self.history else None
            future = zero_lag(hist, future, self.zero_lag[0], 1.0 / self.interval)
        smoothed = actions.clone()
        smoothed[trim:] = torch.as_tensor(future, dtype=actions.dtype, device=actions.device)
        accepted = super()._accept_aligned(request, smoothed, provenance, now)
        if accepted and self.ensemble is not None:
            self.ensemble.commit()
        return accepted

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

    def begin(self, observation):
        request = super().begin(observation)
        if request is not None and self.max_delay is not None and request.delay > self.max_delay:
            # Upstream estimates the delay as turnaround / step time, unbounded, and the server rejects a delay longer
            # than the model's chunk ("Invalid RTC delay"): with GR00T's 16-step chunk any turnaround above 0.53 s
            # (a remote GPU over WiFi, 2026-11-20) ended the trial. In chunk mode the value is only checked, so it is
            # capped at the chunk length.
            from dataclasses import replace

            request = replace(request, delay=self.max_delay)
            self.pending = request
        if request is not None:
            est = self.turnaround if self._completed else None
            self.sent[request.request_id] = (request.submitted_at,
                                             est if est is not None and math.isfinite(est) else None)
        return request

    def accept(self, request, chunk, *, task_version: int) -> bool:
        completed = self._completed
        submitted = request.submitted_at
        overlap, l2 = 0, None
        try:  # the switch, measured on the queue before the new chunk replaces its future (read only)
            snap = self.queue.snapshot()
            anchor = request.observation.action_cursor
            if anchor is not None and snap.canonical_actions is not None:
                trim = snap.cursor - anchor
                if 0 <= trim < len(chunk.canonical_actions):
                    overlap, l2 = switch_disagreement(snap.canonical_actions, chunk.canonical_actions[trim:])
        except Exception:  # noqa: BLE001 - a metric must never stop the robot
            pass
        accepted = super().accept(request, chunk, task_version=task_version)
        rtt = None
        if self._completed > completed:  # super measured this request's turnaround
            rtt = self.clock() - submitted
            self.last_turnaround = rtt
            self.recent.append(rtt)
            if self.estimator is not None:
                self.estimator.update(rtt)
        if accepted:
            self.horizon = int(chunk.execution_steps)
        sd = getattr(chunk, "server_durations", None) or {}
        sent_at, est = self.sent.pop(request.request_id, (submitted, None))
        parts = [sd.get(k) for k in ("preprocessing", "policy", "postprocessing")]
        total = sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None
        self.chunk_log.append([sent_at, self.clock(), request.request_id, int(bool(accepted)), rtt, est, *parts, total,
                               int(chunk.execution_steps), overlap, l2])
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
    cooldown_cap: Literal["none", "half_chunk"] = "none"  # half_chunk: DRTC's original H/2 cap (see cooldown_s)
    # Action filter on every action before it goes to the robot (the minor client's default: butterworth).
    action_filter: Literal["none", "butterworth"] = "none"
    filter_cutoff_hz: float = 3.0
    filter_order: int = 2
    filter_gain: float = 1.4
    # Smoothing method A (drtc.control.playback): adaptive playback while few actions are queued.
    playback_mode: Literal["fixed", "adaptive"] = "fixed"
    playback_min_rate: float = 0.5
    playback_margin_steps: float = 4.0
    playback_rate_slew: float = 0.05
    # Setup v2 smoothing (drtc.control.smoothing; lab/PLAN.md 2026-11-22)
    temporal_ensemble: bool = False  # seams: ACT's temporal ensembling of the queued steps (aligned merge)
    ensemble_m: float = 0.01
    zero_lag: bool = False  # jitter: zero-phase smoothing of recent actions + the queued future (aligned merge)
    zero_lag_cutoff_hz: float = 3.0
    zero_lag_history: int = 15
    resume_ramp: bool = False  # faults: the minimum-jerk resume ramp after a stall
    resume_ramp_steps: int = 10
    resume_min_stall_s: float = 0.1
    # the soft stop before a stall (playback.py): off; dropped from the ramp method 2026-11-22 (it kept a queue alive
    # long enough to accept a reply made from an old observation, and the next fresh chunk then jumped)
    soft_stop_steps: int = 0
    soft_stop_floor: float = 0.15

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


TICK_FIELDS = ["t", "action", "queue", "pending", "latency_est", "accepted", "last_turnaround", "horizon",
               "playback_rate"]
CHUNK_FIELDS = ["t_sent", "t_received", "request_id", "accepted", "turnaround_s", "estimate_at_send_s",
                "server_preprocessing_s", "server_policy_s", "server_postprocessing_s", "server_total_s", "steps",
                "overlap_steps", "switch_l2"]
ARM_JOINTS = 5  # SO101 action order: five arm joints in degrees, then the gripper (lab/METRICS.md "Terms")


def switch_disagreement(previous, future) -> tuple[int, float | None]:
    """Overlap steps and mean arm-joint Euclidean distance between the queued and the new chunk's actions (M8a)."""
    if previous is None or future is None:
        return 0, None
    n = min(len(previous), len(future))
    if n == 0:
        return 0, None
    import torch

    d = torch.linalg.vector_norm(previous[:n, :ARM_JOINTS].float() - future[:n, :ARM_JOINTS].float(), dim=1)
    return n, float(d.mean())


# Callables(engine) run when a DRTCEngine is built: lab eval run uses it to reach the trial's engine (its start time,
# the video observer). Empty for lab run.
ON_ENGINE: list = []


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
        self._joint_names: list[str] = []  # from the first observation (keys ending in .pos)
        self._measured: list[float] | None = None
        self.observers: list = []  # callables(obs dict) on every robot observation (e.g. the trial video)
        self.action_filter = None
        if isinstance(cfg, DRTCInferenceConfig) and cfg.action_filter == "butterworth":
            from drtc.control.action_filter import ButterworthFilter

            self.action_filter = ButterworthFilter(cfg.filter_cutoff_hz, 1.0 / self.runtime.interval,
                                                   cfg.filter_order, cfg.filter_gain)
        self.playback = None
        v2 = isinstance(cfg, DRTCInferenceConfig)
        soft = cfg.soft_stop_steps if v2 and cfg.resume_ramp else 0
        if v2 and (cfg.playback_mode == "adaptive" or soft):
            from drtc.control.playback import AdaptivePlayback

            self.playback = AdaptivePlayback(cfg.playback_min_rate, cfg.playback_margin_steps, cfg.playback_rate_slew,
                                             adaptive=cfg.playback_mode == "adaptive",
                                             soft_stop_steps=soft,
                                             soft_stop_floor=cfg.soft_stop_floor)
        self.ramp = None
        if v2 and cfg.resume_ramp:
            from drtc.control.smoothing import ResumeRamp

            self.ramp = ResumeRamp(cfg.resume_ramp_steps, cfg.resume_min_stall_s)
        if v2 and isinstance(self.runtime, DRTCRuntime):
            if cfg.temporal_ensemble or cfg.zero_lag:
                if self.runtime.chunk_merge != "aligned":
                    raise ValueError("temporal_ensemble and zero_lag work on the aligned merge (not with RTC's append)")
            if cfg.temporal_ensemble:
                from drtc.control.smoothing import TemporalEnsemble

                self.runtime.ensemble = TemporalEnsemble(cfg.ensemble_m)
            if cfg.zero_lag:
                self.runtime.zero_lag = (cfg.zero_lag_cutoff_hz, cfg.zero_lag_history)
                self.runtime.history = deque(maxlen=max(1, cfg.zero_lag_history))
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
        if isinstance(getattr(self, "runtime", None), DRTCRuntime):
            caps = getattr(self.client, "capabilities", None)
            self.runtime.max_delay = getattr(caps, "prediction_steps", None)
        for fn in ON_ENGINE:
            fn(self)

    def cooldown_s(self) -> float:
        """DRTC: latency estimate in steps + epsilon steps.

        DRTC caps the estimate at H/2 (half the last chunk). With a latency above H/2 (GR00T's 16-action chunk is
        0.53 s; a remote GPU over WiFi answers in 0.4-0.7 s, measured 2026-11-20) that cooldown is shorter than every
        reply, so every request is given up and resent and the arm never gets actions. Decided with the user
        (2026-11-20, option b): the cooldown is never shorter than the measured latency (no cap). The original cap
        stays available as `cooldown_cap: half_chunk` in an experiment, for Phase 8.
        """
        rt = self.runtime
        est = rt.turnaround if rt._completed else None
        if est is None or not math.isfinite(est):
            # Before the first reply there is no latency to adapt to. DRTC seeds the cooldown with s_min + epsilon
            # steps, which on a link slower than that (a remote GPU over WiFi: ~0.4 s, 2026-11-20) gives up every
            # first request before it can be answered, so no reply is ever measured and the robot never starts.
            # Until the first reply, wait as long as the startup timeout allows (upstream's cold start); from then
            # on the cooldown follows the measured latency as in DRTC.
            return max((self.config.s_min + self.config.epsilon) * rt.interval,
                       float(getattr(self.config, "startup_timeout_s", 10.0)))
        # Option b, made literal (2026-11-20, a GR00T trial): the smoothed estimate is close to the typical reply,
        # and with replies at 0.52-0.59 s and a 0.6-0.69 s cooldown, every slow reply was given up; the server, still
        # computing it, made the next one slower still, until nothing got through. So the cooldown covers the slowest
        # of the last 10 measured replies as well.
        recent = getattr(rt, "recent", None)
        if recent:
            est = max(est, max(recent))
        steps = max(1, math.ceil(est / rt.interval))
        if getattr(rt, "horizon", None) and getattr(self.config, "cooldown_cap", "none") == "half_chunk":
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
                try:
                    response = decode_reply(payload, request, discard_stale=True)
                except ProtocolError as e:  # decode_reply raises the server's error replies
                    code = getattr(e.code, "value", e.code)
                    if code == ErrorCode.STALE.value and "Duplicate" in str(e):
                        engine._note("duplicate_ignored", request.request_id)
                        continue  # a duplicated copy of this request; its real reply is still coming
                    if code == ErrorCode.BUSY.value:
                        engine.runtime.abandon(request.request_id, backoff=engine.runtime.interval)
                        engine._note("busy_backoff", request.request_id)
                        raise RequestCancelled() from None
                    raise
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

    def notify_observation(self, obs: dict) -> None:
        try:
            if not self._joint_names:
                self._joint_names = [k[:-4] for k in obs if k.endswith(".pos")]
            self._measured = [float(obs[f"{j}.pos"]) for j in self._joint_names]
        except Exception:  # noqa: BLE001 - metrics never stop the robot
            pass
        for fn in self.observers:
            try:
                fn(obs)
            except Exception:  # noqa: BLE001
                pass
        super().notify_observation(obs)

    def _ticks_until_chunk(self) -> float:
        rt = self.runtime
        est = rt.turnaround if rt._completed else None
        if est is None or not math.isfinite(est):
            return 0.0
        pending = rt.pending
        since = (time.monotonic() - pending.submitted_at) if pending is not None else 0.0
        return max(est - since, 0.0) / rt.interval

    def get_action(self, obs_frame):
        if self.playback is not None:
            action = self.playback.next(lambda: super(DRTCEngine, self).get_action(obs_frame),
                                        self.runtime.queue.qsize(), self._ticks_until_chunk())
        else:
            action = super().get_action(obs_frame)
        if action is not None and self.ramp is not None:
            import torch

            eased = self.ramp.apply(action.detach().cpu().double().numpy())
            action = torch.as_tensor(eased, dtype=action.dtype, device=action.device)
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
            1.0 if self.playback is None else round(self.playback.rate, 3),
            None if action is None else [round(float(x), 4) for x in action.detach().cpu().flatten().tolist()],
            None if self._measured is None else [round(x, 4) for x in self._measured],
        ])
        return action

    def _invalidate(self, operation: str) -> None:
        if self.action_filter is not None:
            self.action_filter.reset()  # motion restarts from a held pose: do not smooth across it
        if self.playback is not None:
            self.playback.reset()
        super()._invalidate(operation)

    def stop(self) -> None:
        try:
            super().stop()
        finally:
            path = getattr(self.config, "metrics_path", None)
            if self._ticks:  # the end of the run: stalls up to here count (lab/METRICS.md M2, M3)
                rt = self.runtime
                self._ticks.append([round(time.monotonic() - self._t0, 4), 0, rt.queue.qsize(), int(rt.pending is not None),
                                    None, rt._completed, None, getattr(rt, "horizon", None), None, None, None])
            if path and self._ticks:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                n = len(self._joint_names) or max((len(r[-2] or r[-1] or []) for r in self._ticks), default=0)
                names = self._joint_names or [f"j{i}" for i in range(n)]
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(TICK_FIELDS + [f"cmd.{j}" for j in names] + [f"meas.{j}" for j in names])
                    for r in self._ticks:
                        cmd, meas = r[-2], r[-1]
                        w.writerow(r[:-2] + (cmd or [None] * n)[:n] + (meas or [None] * n)[:n])
            rt = self.runtime
            if path and getattr(rt, "chunk_log", None):
                with open(Path(path).with_name("chunks.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(CHUNK_FIELDS)
                    for r in rt.chunk_log:
                        w.writerow([round(r[0] - self._t0, 4), round(r[1] - self._t0, 4), *r[2:]])
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
