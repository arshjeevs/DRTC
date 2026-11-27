// DRTC simulator: a simplified model of the robot client's scheduling, LeRobot 0.7 as shipped / patient vs DRTC.
//
// It follows the real rules (src/drtc/control/engine.py, LeRobot 0.7 inference/execution.py, src/drtc/control/faults.py,
// lab/experiments/v2/): one request in flight, the trigger, the latency estimates, DRTC's cooldown with its startup
// budget and the slowest-recent cover, the BUSY backoff, the starvation path (the queue runs dry: the request in flight
// is cancelled and one fresh request is sent), the starvation grace, the request and startup deadlines, the aligned
// merge (the actions executed while the request was in flight are skipped), and the bench's fault shapes.
// Simplified: one server that computes one request at a time with a fixed time per policy (+/-10%), a fixed upload
// time, the bench's reply latency (250 ms + 0-20 ms), no adaptive playback or smoothing, no RTC. Numbers are
// indicative; the bench measures the real code.
(function (root) {
  "use strict";

  const DT = 1 / 30;          // the control loop, 30 Hz
  const STEP = 1 / 600;       // event resolution of the simulation
  const UPLOAD = 0.02;        // observation upload + transfer overhead (bench M4 - M6 - network, ~20-30 ms)
  const NET = 0.25, JITTER = 0.02;   // bench reply latency: 250 ms + U(0, 20 ms) (faults.py latency)
  const REFILL = 0.5;         // LeRobot refill_seconds
  const STARTUP = 10.0;       // LeRobot startup_timeout_s (initial action deadline)

  // Policies on the bench (H200): chunk length H, s_min, median server time (bench M6, fault-free).
  const POLICIES = {
    act:       { name: "ACT",         H: 100, sMin: 20, server: 0.074 },
    smolvla:   { name: "SmolVLA",     H: 50,  sMin: 20, server: 0.041 },
    pi05:      { name: "π0.5",        H: 50,  sMin: 20, server: 0.470 },
    molmoact2: { name: "MolmoAct 2",  H: 30,  sMin: 20, server: 0.339 },
    diffusion: { name: "Diffusion",   H: 24,  sMin: 8,  server: 0.195 },
    groot:     { name: "GR00T N1.5",  H: 16,  sMin: 8,  server: 0.268 },
    grootn17:  { name: "GR00T N1.7",  H: 16,  sMin: 8,  server: 0.326 },
  };

  // Clients. upstream: LeRobot's scheduler; shipped = its 1 s starvation grace and 5 s request deadline, patient =
  // both raised to the whole trial (setup v2's baseline). drtc: our DRTC; the a* flags are the E2 ablations.
  const CLIENTS = {
    shipped: { name: "LeRobot as shipped", trigger: "upstream", estimator: "upstream", cooldown: false, grace: 1.0, deadline: 5.0 },
    patient: { name: "LeRobot, patient",   trigger: "upstream", estimator: "upstream", cooldown: false, grace: null, deadline: null },
    drtc:    { name: "DRTC",               trigger: "drtc",     estimator: "jk",       cooldown: true,  grace: null, deadline: null, cap: "none" },
  };

  // Faults as on the bench (lab/experiments/v2/_make.py), before the per-trial shift.
  const FAULTS = {
    none:   [],
    spike:  [{ kind: "spike", start: 2.0, delay: 1.0 }],
    drops:  [{ kind: "drops", start: 4.0, dur: 1.0 }],
    outage: [{ kind: "outage", start: 6.0, dur: 1.5 }],
    mix:    [{ kind: "spike", start: 2.0, delay: 1.0 }, { kind: "drops", start: 4.0, dur: 1.0 },
             { kind: "outage", start: 6.0, dur: 1.5 }],
    rise:   [{ kind: "rise", start: 2.0, dur: 5.0, delay: 0.4 }],
    loss:   [{ kind: "loss", p: 0.1 }],
  };

  function rng(seed) {   // mulberry32
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class JK {   // Jacobson-Karels, engine.py JKEstimator
    constructor() { this.mean = null; this.dev = 0; }
    update(r) {
      if (this.mean === null) { this.mean = r; this.dev = 0; return; }
      const err = r - this.mean;
      this.mean = 0.875 * this.mean + 0.125 * r;
      this.dev = 0.75 * this.dev + 0.25 * Math.abs(err);
    }
    estimate() { return this.mean === null ? null : this.mean + 1.5 * this.dev; }
  }

  // One client lane: its own server and network, the same faults and seed as the other lanes.
  class Lane {
    constructor(opts) {
      const p = POLICIES[opts.policy], c = Object.assign({}, CLIENTS[opts.client], opts.overrides || {});
      this.p = p; this.c = c; this.opts = opts;
      this.H0 = p.H;
      this.sMin = Math.max(1, Math.min(p.H - 1, Math.round(p.sMin * (opts.sMinScale || 1))));
      this.server = p.server * (opts.serverScale || 1);
      this.net = opts.net === undefined ? NET : opts.net;
      this.duration = opts.duration || 14;
      this.faults = (opts.faults || []).map(f => Object.assign({}, f, f.start !== undefined ? { start: f.start + (opts.shift || 0) } : {}));
      this.rand = rng(opts.seed || 1);
      this.lossRand = rng((opts.seed || 1) * 7919 + 13);
      this.t = 0; this.nextTick = 0; this.tick = 0;
      this.queue = 0; this.cursor = 0; this.H = null;
      this.hasExecuted = false; this.startedAt = 0; this.ended = null;
      this.pending = null; this.lastAnchor = null; this.notBefore = 0; this.generation = 0;
      this.held = false; this.starveDeadline = null; this.refetchAt = null;
      this.completed = 0; this.initialTurn = 0; this.turns = []; this.recent = []; this.jk = new JK();
      this.serverBusyUntil = 0;
      this.inflight = [];     // server work and replies on the way: {id, gen, at, kind}
      this.channelFree = 0;   // until then the client's worker sleeps inside the spike (time.sleep in faults.py)
      this.lastRelease = 0; this.starveReqDeadline = null;
      this.spikeUsed = {};
      this.nextId = 1;
      // records
      this.ticks = [];        // per control tick: {t, queue, moved, pending}
      this.events = [];       // {t, kind, id}
      this.requests = {};     // id -> {sent, fate, replyAt}
      this.moving = 0; this.counted = 0; this.everMoved = false;
    }

    fault(kind, t) { return this.faults.some(f => f.kind === kind && t >= f.start && t < f.start + f.dur); }
    rise(t) { return this.faults.filter(f => f.kind === "rise" && t >= f.start && t < f.start + f.dur).reduce((s, f) => s + f.delay, 0); }
    log(kind, id, extra) { this.events.push(Object.assign({ t: this.t, kind, id }, extra || {})); }

    turnaround() {   // the estimate the trigger and the cooldown use
      if (this.c.estimator === "jk") {
        const e = this.jk.estimate();
        if (e !== null) return e;
      }
      return this.turns.length ? Math.max(...this.turns) : this.initialTurn;   // LeRobot: max of the last 100
    }

    cooldown() {   // engine.py cooldown_s
      if (!this.completed) return Math.max((this.sMin + 1) * DT, STARTUP);
      let est = this.turnaround();
      if (this.recent.length) est = Math.max(est, Math.max(...this.recent));
      let steps = Math.max(1, Math.ceil(est / DT - 1e-9));
      if (this.c.cap === "half_chunk" && this.H) steps = Math.min(steps, Math.max(1, Math.floor(this.H / 2)));
      return Math.max(2, steps + 1) * DT;
    }

    refill() {   // the playback threshold (s): request when queued playback <= this
      if (this.c.trigger === "drtc" && this.H !== null) return Math.max(this.H - this.sMin, 1) * DT;
      return Math.max(REFILL, this.turnaround() + DT);
    }

    shouldRequest(anchor) {
      if (this.ended || this.held || this.pending || this.t < this.notBefore) return false;
      if (this.lastAnchor !== null && anchor <= this.lastAnchor) return false;   // aligned: the cursor must advance
      return this.queue * DT <= this.refill() + 1e-9;
    }

    send(anchor) {
      const id = this.nextId++;
      this.pending = { id, sent: this.t, gen: this.generation, anchor, cooldown: this.c.cooldown ? this.cooldown() : null };
      this.lastAnchor = anchor;
      this.requests[id] = { sent: this.t, fate: "pending" };
      this.log("send", id);
      const arrive = this.t + UPLOAD;
      if (this.fault("outage", this.t)) { this.requests[id].fate = "lost"; this.log("lost_request", id); return; }
      this.inflight.push({ id, gen: this.generation, at: arrive, kind: "arrive" });
    }

    abandon(reason, backoff) {
      const id = this.pending.id;
      this.requests[id].fate = this.requests[id].fate === "pending" ? reason : this.requests[id].fate;
      this.requests[id].endAt = this.t;
      this.log(reason, id);
      this.pending = null; this.lastAnchor = null;
      if (backoff) this.notBefore = this.t + backoff;
    }

    serverStep(ev) {   // a request reaches the server
      if (ev.kind === "arrive") {
        if (this.t < this.serverBusyUntil - 1e-9) {   // still computing an earlier request: BUSY
          this.deliver({ id: ev.id, gen: ev.gen, busy: true }, this.t + 0.001);
          return;
        }
        const s = this.server * (0.9 + 0.2 * this.rand());
        this.serverBusyUntil = this.t + s;
        this.deliver({ id: ev.id, gen: ev.gen, busy: false }, this.t + s);
      }
    }

    deliver(reply, producedAt) {   // the network carries the reply: latency, then the faults at its arrival
      let release = producedAt + this.net + JITTER * this.rand() + this.rise(producedAt);
      release = Math.max(release, this.lastRelease);   // a network keeps order (faults.py _delayed_get)
      this.lastRelease = release;
      this.inflight.push(Object.assign({ at: release, kind: "reply" }, reply));
    }

    receive(ev) {   // a reply reaches the client (faults.py _ReplyChannel._faults order)
      const t = this.t;
      if (this.fault("outage", t) || this.fault("drops", t)) { this.markLost(ev); return; }
      const loss = this.faults.find(f => f.kind === "loss");
      if (loss && this.lossRand() < loss.p) { this.markLost(ev); return; }
      const sp = this.faults.find((f, i) => f.kind === "spike" && !this.spikeUsed[i] && t >= f.start);
      if (sp) {   // one-time hold of this reply; the channel is blocked meanwhile
        this.spikeUsed[this.faults.indexOf(sp)] = true;
        this.channelFree = t + sp.delay;
        ev.at = t + sp.delay; ev.spiked = true;
        this.inflight.push(ev);
        this.log("spike", ev.id);
        return;
      }
      this.handle(ev);
    }

    markLost(ev) {
      if (this.requests[ev.id] && !ev.busy) this.requests[ev.id].lostReply = true;
      this.log("lost_reply", ev.id);
    }

    handle(ev) {
      const pend = this.pending;
      if (!pend || pend.id !== ev.id || ev.gen !== this.generation) { this.log("stale", ev.id); return; }   // discarded
      if (ev.busy) {
        if (this.c.cooldown) { this.abandon("busy", DT); return; }
        this.log("busy", ev.id); return;
      }
      const rtt = this.t - pend.sent;
      this.completed += 1;
      if (this.completed === 1) this.initialTurn = rtt; else { this.turns.push(rtt); if (this.turns.length > 100) this.turns.shift(); }
      this.recent.push(rtt); if (this.recent.length > 10) this.recent.shift();
      if (this.c.estimator === "jk") this.jk.update(rtt);
      this.pending = null;
      const trim = this.cursor - pend.anchor;
      if (trim < 0 || trim >= this.H0) { this.requests[ev.id].fate = "too_late"; this.requests[ev.id].endAt = this.t; this.log("too_late", ev.id); return; }
      this.queue = this.H0 - trim; this.H = this.H0;
      this.requests[ev.id].fate = "accepted"; this.requests[ev.id].replyAt = this.t; this.requests[ev.id].endAt = this.t;
      this.log("accept", ev.id, { trim });
      if (this.held || this.starveDeadline !== null) { this.held = false; this.starveDeadline = null; this.log("resume", ev.id); }
    }

    controlTick() {
      const t = this.t;
      // deadlines (execution.py check_deadlines; the request and startup deadlines pause while holding)
      if (this.starveReqDeadline !== null && t > this.starveReqDeadline) return this.end("request deadline while holding");
      if (this.starveDeadline !== null && t > this.starveDeadline) return this.end("starvation grace expired");
      if (!this.held) {
        if (this.pending && this.c.deadline !== null && t - this.pending.sent > this.c.deadline) return this.end("request deadline");
        if (!this.hasExecuted && t - this.startedAt > STARTUP) return this.end("no action for 10 s (startup deadline)");
      }
      // while a spike holds a reply, the worker thread sleeps: no cooldown, no new request, no hold release
      const blocked = t < this.channelFree - 1e-12;
      // the cooldown (DRTC): give up an unanswered request and send a fresh observation
      if (!blocked && this.pending && this.pending.cooldown !== null && t - this.pending.sent > this.pending.cooldown) this.abandon("replaced");
      // after a starvation hold, one fresh request once the server acknowledged the invalidation (release_hold)
      if (!blocked && this.held && this.refetchAt !== null && t >= this.refetchAt) {
        this.held = false; this.refetchAt = null; this.startedAt = t; this.starveReqDeadline = null;
      }
      const anchor = this.cursor;   // this tick's observation, taken before the action is executed
      // execute one action, or stall
      let moved = false;
      if (this.queue > 0) {
        this.queue -= 1; this.cursor += 1; this.hasExecuted = true; moved = true;
      } else if (this.hasExecuted) {   // the queue ran dry: starvation (execution.py _wait_for_actions)
        this.hasExecuted = false; this.startedAt = t;
        this.generation += 1;   // invalidate: the request in flight is cancelled, its late reply discarded
        if (this.pending && this.c.deadline !== null) this.starveReqDeadline = this.pending.sent + this.c.deadline;
        if (this.pending) { const r = this.requests[this.pending.id]; if (r.fate === "pending") r.fate = "cancelled"; r.endAt = t; this.log("cancelled", this.pending.id); }
        this.pending = null; this.lastAnchor = null; this.held = true;
        this.refetchAt = Math.max(t + DT, this.serverBusyUntil);   // hold acknowledged, the server invalidated
        if (this.c.grace !== null) this.starveDeadline = t + this.c.grace;
        this.log("starved", null);
      }
      // the worker polls every 2 ms: a request goes out right after the action that crossed the trigger level
      if (!blocked && this.shouldRequest(anchor)) this.send(anchor);
      if (moved) this.everMoved = true;
      if (this.everMoved) { this.counted += 1; if (moved) this.moving += 1; }   // stall share from the first action on
      this.ticks.push({ t, queue: this.queue, moved, pending: !!this.pending, held: this.held, thr: this.refill() / DT });
    }

    end(reason) {
      this.ended = { t: this.t, reason };
      for (const r of Object.values(this.requests)) if (r.endAt === undefined) r.endAt = this.t;
      this.log("ended", null, { reason });
    }

    // advance the simulation to time T
    advance(T) {
      T = Math.min(T, this.duration);
      while (this.t < T - 1e-12 && !this.ended) {
        const next = Math.min(this.t + STEP, T);
        this.t = next;
        // network and server events due by now (in time order)
        this.inflight.sort((a, b) => a.at - b.at);
        while (this.inflight.length && this.inflight[0].at <= this.t + 1e-12) {
          const ev = this.inflight.shift();
          if (ev.kind === "arrive") this.serverStep(ev);
          else if (ev.kind === "reply") {
            if (!ev.spiked && this.t < this.channelFree - 1e-12) { ev.at = this.channelFree; this.inflight.push(ev); continue; }   // read after the sleep
            if (ev.spiked) this.handle(ev); else this.receive(ev);
          }
          if (this.ended) break;
        }
        if (this.t >= this.nextTick - 1e-12) { this.controlTick(); this.tick += 1; this.nextTick = this.tick * DT; }
      }
    }

    stats() {
      const stall = this.counted ? 1 - this.moving / this.counted : 0;
      return { stallPct: 100 * stall, ended: this.ended, recoveries: this.recoveries() };
    }

    recoveries() {   // per timed fault: seconds from the fault's end until the arm moves again (if it was stalled)
      const out = [];
      for (const f of this.faults) {
        if (f.start === undefined) continue;
        const end = f.start + (f.dur !== undefined ? f.dur : f.delay);
        if (end > this.duration) continue;
        const after = this.ticks.filter(k => k.t >= f.start);
        if (!after.length) continue;
        const stalledAtEnd = this.ticks.filter(k => k.t >= f.start && k.t <= end).some(k => !k.moved);
        const mv = this.ticks.find(k => k.t >= end && k.moved);
        out.push({ kind: f.kind, end, stalled: stalledAtEnd, recovery: mv ? Math.max(0, mv.t - end) : null });
      }
      return out;
    }
  }

  function runTrial(opts) {
    const lane = new Lane(opts);
    lane.advance(lane.duration);
    return lane;
  }

  const api = { DT, POLICIES, CLIENTS, FAULTS, Lane, runTrial };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.DRTCSim = api;
})(typeof window !== "undefined" ? window : globalThis);
