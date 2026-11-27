// The Trials window: pick any trial from the matrix, watch both cameras in sync, and read the run underneath on one
// clock: the faults, the requests and replies, DRTC's re-sends, the action queue, when the arm stood still, and the
// commanded joint. Every number comes from the trial's own raw files.
"use strict";

let B = null, E = null, T = null, D = null;  // bundle, eval, trial row, trial detail
const F = { conditions: new Set(), models: new Set(), outcomes: new Set(), videoOnly: false };
let videos = [], joint = null, rate = 1, playing = false, clockT = 0, virtualStart = null, raf = null;

const EVENT_TEXT = {
  spike: "Latency spike starts", drop_action: "A reply is lost", disconnect_obs: "Link down: a request is lost",
  rise: "Latency starts rising", cooldown_resend: "DRTC re-sends (cooldown)", resend: "Client re-sends",
};

const SHORT_FAULT = { spike: "spike", drops: "lost", outage: "down", rise: "rise" };
const MATRIX_NAME = { 1: "Patient", 2: "Faults", 3: "DRTC", 4: "Stack" };

document.getElementById("top").replaceWith(topbar("trials"));

loadData().then((b) => {
  B = b;
  fromHash();
}).catch((err) => {
  document.getElementById("side").replaceChildren(el("div", { class: "panel empty", text: `Could not read the results: ${err.message}` }));
});
addEventListener("hashchange", () => { if (B && !hashWriting) fromHash(); });

let hashWriting = false;
function writeHash() {
  hashWriting = true;
  hash.set({ eval: E.name, condition: [...F.conditions].join(","), model: [...F.models].join(","),
    outcome: [...F.outcomes].join(","), video: F.videoOnly ? "1" : "", id: T ? T.id : "" });
  setTimeout(() => { hashWriting = false; }, 0);
}

function fromHash() {
  const h = hash.get();
  E = B.evals.find((e) => e.name === h.eval && !e.missing) || B.evals.find((e) => !e.missing);
  F.conditions = new Set((h.condition || "").split(",").filter((c) => E.cond[c]));
  F.models = new Set((h.model || "").split(",").filter((m) => E.model[m]));
  F.outcomes = new Set((h.outcome || "").split(",").filter((o) => OUTCOMES[o]));
  F.videoOnly = h.video === "1";
  const t = E.trials.find((x) => x.id === h.id);
  renderSide();
  openTrial(t || ordered().find((x) => x.cameras.length) || ordered()[0] || null, false);
}

// --- filters and the trial matrix -------------------------------------------------------------------------------------
const condOrder = () => E.conditions.map((c) => c.id);
function visibleCond(c) { return !F.conditions.size || F.conditions.has(c); }
function visibleModel(m) { return !F.models.size || F.models.has(m); }
function matches(t) {
  return visibleCond(t.condition) && visibleModel(t.model) && (!F.outcomes.size || F.outcomes.has(t.outcome)) &&
    (!F.videoOnly || t.cameras.length);
}
function ordered() {
  const ci = Object.fromEntries(condOrder().map((c, i) => [c, i]));
  const mi = Object.fromEntries(E.models.map((m, i) => [m.id, i]));
  return E.trials.filter(matches).sort((a, b) => (mi[a.model] - mi[b.model]) || (ci[a.condition] - ci[b.condition]) ||
    (a.trial - b.trial) || a.session.localeCompare(b.session));
}

function toggle(set, v) { if (set.has(v)) set.delete(v); else set.add(v); }

function renderSide() {
  const side = document.getElementById("side");
  const evalSel = el("select", { class: "input", "aria-label": "Evaluation", onchange: (e) => {
    E = B.evals.find((x) => x.name === e.target.value); F.conditions.clear(); F.models.clear(); F.outcomes.clear();
    renderSide(); openTrial(ordered().find((x) => x.cameras.length) || ordered()[0], true);
  } }, B.evals.filter((e) => !e.missing).map((e) => el("option", { value: e.name, selected: e.name === E.name ? true : null, text: `${e.title} (${e.tag})` })));
  const chipRow = (items, set, label) => el("div", { class: "row" }, items.map(([id, text, cls]) =>
    el("button", { class: `chip ${cls || ""}`, type: "button", "aria-pressed": set.has(id) ? "true" : "false",
      onclick: () => { toggle(set, id); renderSide(); refreshCurrent(); } }, cls ? el("span", { class: "sw" }) : null, text)));
  const filters = el("div", { class: "panel filters" },
    el("label", { class: "t", text: "Evaluation" }), evalSel,
    E.conditions.length > 1 ? [el("label", { class: "t", text: "Condition" }),
      chipRow(E.conditions.map((c) => [c.id, c.short, `s${c.slot}`]), F.conditions)] : null,
    el("label", { class: "t", text: "Policy" }), chipRow(E.models.map((m) => [m.id, m.name]), F.models),
    el("label", { class: "t", text: "Ending" }), chipRow(["success", "failed", "time_limit", "fault", "not_counted"]
      .filter((o) => E.trials.some((t) => t.outcome === o)).map((o) => [o, `${OUTCOMES[o].icon} ${OUTCOMES[o].short || OUTCOMES[o].label}`]), F.outcomes),
    el("div", { class: "row" }, el("button", { class: "chip", type: "button", "aria-pressed": F.videoOnly ? "true" : "false",
      onclick: () => { F.videoOnly = !F.videoOnly; renderSide(); refreshCurrent(); }, text: "With video only" }),
    (F.conditions.size || F.models.size || F.outcomes.size || F.videoOnly) ? el("button", { class: "chip", type: "button",
      onclick: () => { F.conditions.clear(); F.models.clear(); F.outcomes.clear(); F.videoOnly = false; renderSide(); refreshCurrent(); }, text: "Clear filters" }) : null));

  const n = ordered().length;
  const matrix = el("div", { class: "panel matrix" },
    el("div", { class: "section-head" }, el("h3", { text: `Trials (${n} shown)` }),
      el("span", { class: "small muted" }, el("kbd", { text: "←" }), " ", el("kbd", { text: "→" }), " previous / next")),
    E.models.filter((m) => visibleModel(m.id)).map((m) => el("div", { class: "mrow" },
      el("div", { class: "name", text: m.name }),
      el("div", { class: "cells" }, E.conditions.filter((c) => visibleCond(c.id)).map((c) => {
        const ts = E.trials.filter((t) => t.model === m.id && t.condition === c.id).sort((a, b) => a.trial - b.trial || a.session.localeCompare(b.session));
        return el("div", { class: "cell" },
          el("span", { class: `cl s${c.slot}`, title: c.label }, el("span", { class: "sw" }), `${E.conditions.length > 1 ? MATRIX_NAME[c.slot] + " " : ""}${ts.filter((t) => t.outcome === "success").length}/${ts.filter((t) => t.valid).length}`),
          el("div", { class: "sq" }, ts.map((t) => {
            const b = el("button", { type: "button", class: `${t.outcome}${t.cameras.length ? "" : " novideo"}${matches(t) ? "" : " dim"}`,
              "aria-current": T && T.id === t.id ? "true" : "false", "aria-label": `Trial ${t.trial}, ${OUTCOMES[t.outcome].label}`,
              onclick: () => openTrial(t, true), text: OUTCOMES[t.outcome].icon });
            return hoverable(b, () => [tipRow(`Trial ${t.trial}`, `${m.name} · ${c.short}`),
              tipRow(OUTCOMES[t.outcome].label, t.time_to_success_s ? `in ${fmt.sec(t.time_to_success_s)}` : ""),
              el("div", { class: "k small", text: t.cameras.length ? `video: ${t.cameras.join(" + ")}` : "no video" })]);
          })));
      })))),
    el("div", { class: "legend small" }, ["success", "failed", "time_limit", "fault", "not_counted"].map((o) =>
      el("span", {}, el("span", { class: "sw", style: { background: OUTCOMES[o].color } }), OUTCOMES[o].short || OUTCOMES[o].label)),
      el("span", {}, "• = no video")));
  side.replaceChildren(filters, matrix);
}

function refreshCurrent() {
  if (T && !matches(T)) { const o = ordered(); if (o.length) return openTrial(o.find((x) => x.cameras.length) || o[0], true); }
  writeHash();
}

addEventListener("keydown", (e) => {
  if (!E || /INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName)) return;
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const o = ordered(); const i = o.findIndex((x) => x.id === T?.id);
    const n = o[(i + (e.key === "ArrowRight" ? 1 : -1) + o.length) % o.length];
    if (n) { e.preventDefault(); openTrial(n, true); }
  } else if (e.key === " " && document.activeElement?.tagName !== "BUTTON") { e.preventDefault(); togglePlay(); }
});

// --- one trial ----------------------------------------------------------------------------------------------------------
async function openTrial(t, push) {
  stop();
  T = t;
  const player = document.getElementById("player");
  if (!t) { player.replaceChildren(el("div", { class: "panel empty", text: "No trial matches the filters." })); return; }
  writeHash(push);
  document.querySelectorAll(".sq button").forEach((b) => b.setAttribute("aria-current", "false"));
  renderSide();
  player.replaceChildren(el("div", { class: "panel empty", text: "Loading the trial…" }));
  const r = await fetch(`/api/trial?eval=${encodeURIComponent(E.name)}&id=${encodeURIComponent(t.id)}`);
  if (T !== t) return;
  D = r.ok ? await r.json() : null;
  if (!D) { player.replaceChildren(el("div", { class: "panel empty", text: "Could not read this trial." })); return; }
  const cmd = D.ticks.cmd || {};
  if (!joint || !D.joints.includes(joint)) {
    joint = D.joints.slice(0, 5).map((j) => [j, range(cmd[j])]).sort((a, b) => b[1] - a[1])[0]?.[0] || D.joints[0];
  }
  clockT = D.ticks.t.find((x) => x !== null) ?? 0;
  renderPlayer();
}

function range(xs) { const v = (xs || []).filter((x) => x !== null); return v.length ? Math.max(...v) - Math.min(...v) : 0; }

function renderPlayer() {
  const c = E.cond[T.condition], m = E.model[T.model];
  const o = ordered(), i = o.findIndex((x) => x.id === T.id);
  const nav = (d) => el("button", { class: "iconbtn", type: "button", disabled: o.length < 2 ? true : null,
    onclick: () => openTrial(o[(i + d + o.length) % o.length], true), text: d < 0 ? "← Previous" : "Next →" });
  const head = el("div", { class: "panel phead" },
    el("span", { class: `chip s${c.slot}`, style: { cursor: "default" } }, el("span", { class: "sw" }), c.label),
    el("h1", { text: `${m.name} · trial ${T.trial}` }),
    outcomeBadge(T.outcome),
    T.time_to_success_s ? el("span", { class: "muted", text: `done in ${fmt.sec(T.time_to_success_s)}` }) : null,
    el("span", { class: "spacer" }), nav(-1), nav(1),
    el("a", { class: "iconbtn", style: { textDecoration: "none" }, href: `/#eval=${encodeURIComponent(E.name)}`, text: "Results ↗" }));

  videos = [];
  const vwrap = el("div", { class: `videos ${T.cameras.length < 2 ? "one" : ""}` });
  if (T.cameras.length) {
    for (const cam of T.cameras) {
      const v = el("video", { muted: true, playsinline: true, preload: "auto", src: `/video/${encodeURIComponent(E.name)}/${T.id}/${cam}.mp4` });
      v.muted = true;
      v.addEventListener("loadedmetadata", () => seek(clockT));
      v.addEventListener("ended", () => { if (videos[0] === v) stop(); });
      videos.push(v);
      vwrap.append(el("div", { class: "vbox" }, v, el("span", { class: "cap", text: cam === "scene" ? "Scene camera" : cam === "wrist" ? "Wrist camera" : cam }),
        el("span", { class: "fault-flag", text: "" }), el("span", { class: "stall-flag", text: "Arm waiting for actions" })));
    }
  } else {
    vwrap.append(el("div", { class: "vbox novideo-box", text: "No video was recorded for this trial; the timeline below still plays." }));
  }

  const playBtn = el("button", { class: "playbtn", id: "playbtn", type: "button", onclick: togglePlay, text: "▶ Play" });
  const speeds = el("div", { class: "segmented", role: "group", "aria-label": "Speed" }, [0.25, 0.5, 1, 2].map((s) =>
    el("button", { type: "button", "aria-pressed": s === rate ? "true" : "false", onclick: (e) => {
      rate = s; for (const v of videos) v.playbackRate = s;
      e.target.parentNode.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === e.target ? "true" : "false"));
      if (playing && !videos.length) { virtualStart = { wall: performance.now(), t: clockT }; }
    }, text: `${s}×` })));
  const controls = el("div", { class: "controls" }, playBtn,
    el("button", { class: "iconbtn", type: "button", onclick: () => { stop(); seek(D.ticks.t.find((x) => x !== null) ?? 0); }, text: "⏮ Start" }),
    el("span", { class: "clock", id: "clock", text: "" }), speeds,
    el("span", { class: "small muted", text: "Click or drag on the timeline to jump; the videos follow." }));

  const tl = el("div", { class: "timeline", id: "timeline" });
  const jointChips = el("div", { class: "joint-chips" }, el("span", { class: "small muted", text: "Joint:" }), D.joints.map((j) =>
    el("button", { class: "chip", type: "button", "aria-pressed": j === joint ? "true" : "false", onclick: () => { joint = j; renderPlayer(); }, text: j.replace("_", " ") })),
    el("span", { class: "legend small", style: { marginLeft: "8px" } },
      el("span", {}, el("span", { class: "sw line", style: { background: "var(--ink)" } }), "commanded"),
      el("span", {}, el("span", { class: "sw line", style: { background: "repeating-linear-gradient(90deg, var(--ink-2) 0 4px, transparent 4px 7px)" } }), "measured")));

  const timelinePanel = el("div", { class: "panel" },
    el("div", { class: "section-head" }, el("div", {}, el("h2", { text: "What happened, on one clock" }),
      el("p", { class: "lede small", text: "Orange bands: the network faults. Requests: each bar is one request from sending to its reply; ✕ a lost request or reply; ▲ DRTC asking again. Queue: actions waiting on the robot. Arm: red where it stood still with nothing to do." })),
      jointChips),
    tl);

  const player = document.getElementById("player");
  player.replaceChildren(head, el("div", { class: "panel", style: { display: "grid", gap: "10px" } }, vwrap, controls), timelinePanel,
    el("div", { class: "grid2" }, metricsCard(), factsCard()));
  requestAnimationFrame(() => { drawTimeline(); seek(clockT); });
}

// --- the timeline -----------------------------------------------------------------------------------------------------
let TL = null;  // geometry of the drawn timeline
addEventListener("resize", () => { if (D) { drawTimeline(); paintPlayhead(); } });

function stalls() {
  const t = D.ticks.t, a = D.ticks.action, dt = 1 / 30, out = [];
  const ex = t.filter((x, i) => x !== null && a[i] === 1);
  for (let i = 0; i < ex.length - 1; i++) if (ex[i + 1] - (ex[i] + dt) > 0.5 * dt) out.push([ex[i] + dt, ex[i + 1]]);
  if (ex.length && D.t_end - (ex[ex.length - 1] + dt) > 0.5 * dt) out.push([ex[ex.length - 1] + dt, D.t_end]);
  return out;
}

function drawTimeline() {
  const box = document.getElementById("timeline"); if (!box) return;
  const W = Math.max(300, box.clientWidth), L = W < 500 ? 70 : 92, R = 12;
  const lanes = { head: [0, 22], req: [26, 70], queue: [78, 136], arm: [144, 162], joint: [172, 280], axis: [280, 304] };
  const H = lanes.axis[1];
  const tmax = Math.max(D.t_end || 1, ...(D.windows.map((w) => w.end))) + 0.3;
  const x = (t) => L + (t / tmax) * (W - L - R);
  const s = sv("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "Trial timeline" });
  // fault bands across every lane
  for (const w of D.windows) {
    s.append(sv("rect", { x: x(w.start), y: lanes.head[1] - 2, width: Math.max(2, x(w.end) - x(w.start)), height: lanes.joint[1] - lanes.head[1] + 2, fill: "var(--fault-band)" }));
    s.append(sv("text", { x: (x(w.start) + x(w.end)) / 2, y: 15, "text-anchor": "middle", class: "value-label", style: "fill:var(--fault-band-ink)", text: x(w.end) - x(w.start) < 70 ? (SHORT_FAULT[w.kind] || w.kind) : (B.faults[w.kind] || w.kind) }));
  }
  const laneLabel = (txt, [a, b]) => s.append(sv("text", { x: 0, y: (a + b) / 2 + 4, class: "lane-label", text: txt }));
  laneLabel("Requests", lanes.req); laneLabel("Action queue", lanes.queue); laneLabel("Arm", lanes.arm);
  laneLabel(joint ? joint.replace("_", " ") : "Joint", lanes.joint);
  for (const k of ["req", "queue", "arm", "joint"]) s.append(sv("line", { x1: L, x2: W - R, y1: lanes[k][1], y2: lanes[k][1], class: "gridline" }));

  // requests: greedy rows so overlapping requests stack
  const rows = [], rh = 9;
  const chunks = D.chunks.filter((c) => c.sent !== null);
  for (const c of chunks) {
    const end = c.received ?? D.t_end;
    let r = rows.findIndex((e) => e < c.sent - 0.02); if (r < 0) { rows.push(0); r = rows.length - 1; }
    rows[r] = end;
    const y0 = lanes.req[0] + 4 + (r % 4) * (rh + 2);
    const g = sv("g", {});
    g.append(sv("rect", { x: x(c.sent), y: y0, width: Math.max(2, x(end) - x(c.sent)), height: rh, rx: 3,
      fill: c.received === null ? "none" : c.accepted ? "var(--s1)" : "var(--ink-3)", stroke: c.received === null ? "var(--ink-3)" : "none", "stroke-dasharray": c.received === null ? "3 2" : null }));
    g.append(sv("rect", { class: "hit", x: x(c.sent) - 2, y: y0 - 2, width: Math.max(6, x(end) - x(c.sent) + 4), height: rh + 4 }));
    s.append(hoverable(g, () => [tipRow(c.turnaround !== null ? `${Math.round(c.turnaround * 1000)} ms` : "no reply", "request → reply", "var(--s1)"),
      el("div", { class: "k small", text: `sent ${fmt.num(c.sent, 2)} s${c.received !== null ? `, reply ${fmt.num(c.received, 2)} s` : ""}${c.server ? `, server ${Math.round(c.server * 1000)} ms` : ""}${c.accepted ? "" : " · not used"}` })]));
  }
  for (const ev of D.events) {
    if (ev.t === null) continue;
    const X = x(ev.t);
    if (ev.event === "cooldown_resend" || ev.event === "resend") {
      const p = sv("path", { d: `M${X},${lanes.req[1] - 12} l5,9 h-10 Z`, fill: "var(--s3)", stroke: "var(--surface)", "stroke-width": 1 });
      s.append(hoverable(p, () => [tipRow(`${fmt.num(ev.t, 2)} s`, EVENT_TEXT[ev.event] || ev.event, "var(--s3)")]));
    } else if (ev.event === "drop_action" || ev.event === "disconnect_obs") {
      const y = lanes.req[1] - 8;
      const g = sv("g", {}, sv("path", { d: `M${X - 4},${y - 4} l8,8 M${X + 4},${y - 4} l-8,8`, stroke: "var(--critical)", "stroke-width": 2, "stroke-linecap": "round" }),
        sv("rect", { class: "hit", x: X - 7, y: y - 7, width: 14, height: 14 }));
      s.append(hoverable(g, () => [tipRow(`${fmt.num(ev.t, 2)} s`, EVENT_TEXT[ev.event], "var(--critical)")]));
    }
  }

  // action queue (step line, held through stalls)
  const t = D.ticks.t, q = D.ticks.queue;
  const qmax = Math.max(1, ...q.filter((v) => v !== null));
  const yq = (v) => lanes.queue[1] - (v / qmax) * (lanes.queue[1] - lanes.queue[0] - 4);
  let d = "", prev = null;
  t.forEach((ti, i) => {
    if (ti === null || q[i] === null) return;
    d += prev === null ? `M${x(ti)},${yq(q[i])}` : ` H${x(ti)} V${yq(q[i])}`;
    prev = q[i];
  });
  if (prev !== null) d += ` H${x(D.t_end)}`;
  s.append(sv("path", { d: d + ` V${lanes.queue[1]} H${x(t.find((v) => v !== null))} Z`, fill: "var(--s1)", "fill-opacity": 0.16 }));
  s.append(sv("path", { d, fill: "none", stroke: "var(--s1)", "stroke-width": 2 }));
  s.append(sv("text", { x: L - 6, y: lanes.queue[0] + 10, "text-anchor": "end", class: "tick-label", style: "font-size:10px", text: qmax }));
  s.append(sv("text", { x: L - 6, y: lanes.queue[1] - 1, "text-anchor": "end", class: "tick-label", style: "font-size:10px", text: 0 }));

  // arm: moving vs standing still
  const first = t.find((v) => v !== null) ?? 0;
  s.append(sv("rect", { x: x(first), y: lanes.arm[0], width: Math.max(0, x(D.t_end) - x(first)), height: lanes.arm[1] - lanes.arm[0], rx: 3, fill: "var(--good)", "fill-opacity": 0.35 }));
  const st = stalls();
  for (const [a, b] of st) {
    const r = sv("rect", { x: x(a), y: lanes.arm[0], width: Math.max(1.5, x(b) - x(a)), height: lanes.arm[1] - lanes.arm[0], fill: "var(--stall)" });
    s.append(hoverable(r, () => [tipRow(`${fmt.num(b - a, 2)} s`, "the arm stood still (no new action)", "var(--critical)"),
      el("div", { class: "k small", text: `${fmt.num(a, 2)} – ${fmt.num(b, 2)} s` })]));
  }

  // the commanded joint (solid) and where it really was (dashed)
  const cmd = (D.ticks.cmd[joint] || []), meas = (D.ticks.meas[joint] || []);
  const all = [...cmd, ...meas].filter((v) => v !== null);
  if (all.length) {
    let lo = Math.min(...all), hi = Math.max(...all); if (hi - lo < 1) { hi += 0.5; lo -= 0.5; }
    const yj = (v) => lanes.joint[1] - 6 - ((v - lo) / (hi - lo)) * (lanes.joint[1] - lanes.joint[0] - 12);
    const line = (vals) => { let p = ""; let last = null; t.forEach((ti, i) => { if (ti === null || vals[i] === null) return;
      p += last === null ? `M${x(ti)},${yj(vals[i])}` : (ti - last > 0.08 ? ` H${x(ti)} V${yj(vals[i])}` : ` L${x(ti)},${yj(vals[i])}`); last = ti; }); return p; };
    s.append(sv("path", { d: line(meas), fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
    s.append(sv("path", { d: line(cmd), fill: "none", stroke: "var(--ink)", "stroke-width": 2 }));
    s.append(sv("text", { x: L - 6, y: lanes.joint[0] + 12, "text-anchor": "end", class: "tick-label", style: "font-size:10px", text: `${Math.round(hi)}°` }));
    s.append(sv("text", { x: L - 6, y: lanes.joint[1] - 4, "text-anchor": "end", class: "tick-label", style: "font-size:10px", text: `${Math.round(lo)}°` }));
  }

  // axis and moments
  const { ticks } = niceTicks(tmax, Math.max(4, Math.floor((W - L) / 90)));
  for (const v of ticks) {
    if (v > tmax) continue;
    s.append(sv("line", { x1: x(v), x2: x(v), y1: lanes.axis[0], y2: lanes.axis[0] + 4, class: "axisline" }));
    s.append(sv("text", { x: x(v), y: lanes.axis[0] + 17, "text-anchor": "middle", class: "tick-label", text: `${v} s` }));
  }
  s.append(sv("line", { x1: L, x2: W - R, y1: lanes.axis[0], y2: lanes.axis[0], class: "axisline" }));
  const moment = (tt, label, color) => {
    if (tt === null || tt === undefined) return;
    s.append(sv("line", { x1: x(tt), x2: x(tt), y1: lanes.req[0], y2: lanes.joint[1], stroke: color, "stroke-width": 1.5, "stroke-dasharray": "2 3" }));
    const right = x(tt) > W - R - 110;
    s.append(sv("text", { x: x(tt) + (right ? -5 : 5), y: lanes.joint[0] + 12, "text-anchor": right ? "end" : "start", class: "tick-label", style: `font-size:11px;font-weight:600;fill:${color}`, text: label }));
  };
  moment(D.info.first_action_s, "first action", "var(--ink-2)");
  if (D.info.key_time_s && T.outcome !== "fault") moment(D.info.key_time_s, T.outcome === "success" ? "✓ success" : T.outcome === "failed" ? "✗ operator: failed" : "end", T.outcome === "success" ? "var(--good-text)" : "var(--critical)");

  // playhead, hover and seeking
  const ph = sv("g", { id: "playhead", style: "pointer-events:none" },
    sv("line", { y1: lanes.head[1] - 2, y2: lanes.joint[1], stroke: "var(--ink)", "stroke-width": 2 }),
    sv("circle", { cy: lanes.head[1] - 2, r: 5, fill: "var(--ink)" }));
  const hover = sv("line", { y1: lanes.req[0], y2: lanes.joint[1], stroke: "var(--ink-3)", "stroke-width": 1, style: "pointer-events:none", visibility: "hidden" });
  s.append(hover, ph);
  const tOf = (evt) => { const r = s.getBoundingClientRect(); const px = ((evt.clientX - r.left) / r.width) * W; return Math.max(0, Math.min(tmax, ((px - L) / (W - L - R)) * tmax)); };
  let dragging = false;
  s.addEventListener("pointerdown", (e) => { if (e.target.closest(".hit")) return; dragging = true; s.setPointerCapture(e.pointerId); seek(tOf(e)); });
  s.addEventListener("pointermove", (e) => {
    const tt = tOf(e);
    if (dragging) { seek(tt); return; }
    if (e.target.closest(".hit") || e.target.tagName === "path" && e.target.getAttribute("fill") === "var(--s3)") return;
    hover.setAttribute("x1", x(tt)); hover.setAttribute("x2", x(tt)); hover.setAttribute("visibility", "visible");
    tip.show(e, readout(tt));
  });
  s.addEventListener("pointerup", () => { dragging = false; });
  s.addEventListener("pointerleave", () => { hover.setAttribute("visibility", "hidden"); tip.hide(); });
  box.replaceChildren(s);
  TL = { x, tmax, st };
}

function stateAt(tt) {
  const t = D.ticks.t;
  let i = -1;
  for (let k = 0; k < t.length; k++) { if (t[k] !== null && t[k] <= tt) i = k; else if (t[k] > tt) break; }
  const fault = D.windows.find((w) => tt >= w.start && tt <= w.end);
  const stalled = TL?.st.some(([a, b]) => tt >= a && tt <= b) || false;
  const inflight = D.chunks.filter((c) => c.sent !== null && c.sent <= tt && (c.received === null || c.received > tt)).length;
  return { i, fault, stalled, inflight, started: i >= 0 };
}

function readout(tt) {
  const s = stateAt(tt);
  const q = s.i >= 0 ? D.ticks.queue[s.i] : null;
  const c = s.i >= 0 ? D.ticks.cmd[joint]?.[s.i] : null;
  return [tipRow(`${fmt.num(tt, 2)} s`, s.fault ? B.faults[s.fault.kind] : "", s.fault ? "var(--serious)" : null),
    s.started ? tipRow(s.stalled ? "standing still" : "moving", "arm", s.stalled ? "var(--critical)" : "var(--good)") : tipRow("not started", "arm"),
    q !== null ? tipRow(`${s.stalled ? 0 : q}`, "actions in the queue", "var(--s1)") : null,
    tipRow(`${s.inflight}`, s.inflight === 1 ? "request waiting for its reply" : "requests waiting for replies"),
    c !== null && c !== undefined ? tipRow(`${fmt.num(c, 1)}°`, `${joint.replace("_", " ")} commanded`, "var(--ink)") : null];
}

// --- the clock: video time <-> trial time ------------------------------------------------------------------------------
const toVideo = (tt) => (D.video ? (tt - D.video.t0) / D.video.scale : tt);
const toTrial = (vt) => (D.video ? D.video.t0 + vt * D.video.scale : vt);

function seek(tt) {
  clockT = Math.max(0, Math.min(tt, TL ? TL.tmax : tt));
  for (const v of videos) { if (v.readyState >= 1) v.currentTime = Math.max(0, Math.min(v.duration || 1e9, toVideo(clockT))); }
  if (playing && !videos.length) virtualStart = { wall: performance.now(), t: clockT };
  paintPlayhead();
}

function paintPlayhead() {
  if (!TL) return;
  const ph = document.getElementById("playhead");
  if (ph) ph.setAttribute("transform", `translate(${TL.x(clockT)},0)`);
  const clock = document.getElementById("clock");
  if (clock) clock.textContent = `${clockT.toFixed(2)} s / ${(D.t_end || 0).toFixed(1)} s`;
  const s = stateAt(clockT);
  document.querySelectorAll(".fault-flag").forEach((f) => { f.style.display = s.fault ? "block" : "none"; f.textContent = s.fault ? `⚡ ${B.faults[s.fault.kind]}` : ""; });
  document.querySelectorAll(".stall-flag").forEach((f) => { f.style.display = s.stalled ? "block" : "none"; });
}

function togglePlay() { if (playing) stop(); else play(); }

function play() {
  if (!D) return;
  if (TL && clockT >= TL.tmax - 0.05) seek(D.ticks.t.find((x) => x !== null) ?? 0);
  playing = true;
  const b = document.getElementById("playbtn"); if (b) b.textContent = "❚❚ Pause";
  if (videos.length) {
    for (const v of videos) { v.playbackRate = rate; v.play().catch(() => {}); }
  } else virtualStart = { wall: performance.now(), t: clockT };
  const step = () => {
    if (!playing) return;
    if (videos.length) {
      const master = videos[0];
      clockT = toTrial(master.currentTime);
      for (const v of videos.slice(1)) if (Math.abs(v.currentTime - master.currentTime) > 0.08) v.currentTime = master.currentTime;
      if (master.ended) { stop(); return; }
    } else {
      clockT = virtualStart.t + ((performance.now() - virtualStart.wall) / 1000) * rate;
      if (clockT >= TL.tmax) { clockT = TL.tmax; stop(); }
    }
    paintPlayhead();
    raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
}

function stop() {
  playing = false;
  if (raf) cancelAnimationFrame(raf);
  for (const v of videos) v.pause();
  const b = document.getElementById("playbtn"); if (b) b.textContent = "▶ Play";
}

// --- numbers of this trial -------------------------------------------------------------------------------------------------
function metricsCard() {
  const g = E.group(T.condition, T.model);
  const cols = ["M2_stall_pct", "M3_longest_stall_s", "M3_stall_events", "M1t_time_to_success_s", "M7_jerk_k", "M8b_largest_step_deg", "M4_turnaround_med_ms", "M6_server_ms"];
  const rows = cols.map((c) => B.metric[c]).filter((m) => m && (T.metrics[m.col] !== undefined || g?.stats[m.col])).map((m) => {
    const v = T.metrics[m.col], st = g?.stats[m.col];
    return el("tr", {}, el("td", { text: m.name }), el("td", { class: "n", style: { fontWeight: 700 }, text: v !== undefined ? `${fmt.num(v, m.dec)} ${m.unit}` : "–" }),
      el("td", { class: "n muted", text: st ? `${fmt.num(st[0], m.dec)} [${fmt.num(st[1], m.dec)}–${fmt.num(st[2], m.dec)}]` : "–" }));
  });
  const rec = ["spike", "drops", "outage", "rise"].filter((f) => T.metrics[`M11_recovered_${f}`] !== undefined).map((f) => {
    const ok = T.metrics[`M11_recovered_${f}`] === 1, s = T.metrics[`M11_recovery_${f}_s`];
    const gr = g?.recovery[f];
    return el("tr", {}, el("td", { text: `Recovered after the ${B.faults[f].toLowerCase()}` }),
      el("td", { class: "n", style: { fontWeight: 700 }, text: ok ? (s ? `yes, ${fmt.num(s, 2)} s` : "never stopped") : "no" }),
      el("td", { class: "n muted", text: gr ? `${gr.k}/${gr.n} runs` : "–" }));
  });
  return el("div", { class: "panel" }, el("h3", { text: "This trial against its group" }),
    el("p", { class: "small muted", text: `The group: the ${g?.n ?? 0} counted trials of ${E.model[T.model].name} under “${E.cond[T.condition].short}” (median [middle half]).` }),
    el("div", { class: "tbl", style: { marginTop: "8px" } }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { text: "Measure" }), el("th", { class: "n", text: "This trial" }), el("th", { class: "n", text: "Group" }))),
      el("tbody", {}, rows, rec))));
}

function factsCard() {
  const i = D.info, c = D.config;
  const events = [];
  if (i.first_action_s) events.push({ t: i.first_action_s, text: "First action reaches the arm" });
  for (const w of D.windows) events.push({ t: w.start, text: `${B.faults[w.kind] || w.kind} begins (until ${fmt.num(w.end, 2)} s)` });
  for (const e of D.events) if (e.event !== "spike") events.push({ t: e.t, text: EVENT_TEXT[e.event] || e.event });
  if (i.key_time_s) events.push({ t: i.key_time_s, text: T.outcome === "success" ? "Operator: success" : T.outcome === "failed" ? "Operator: failed" : `Run ends (${OUTCOMES[T.outcome].label.toLowerCase()})` });
  events.sort((a, b) => a.t - b.t);
  const client = c.scheduler === "drtc" ? "DRTC" : c.scheduler ? "patient LeRobot" : "–";
  return el("div", { class: "panel" }, el("h3", { text: "The run" }),
    el("dl", { class: "kv", style: { marginTop: "8px" } },
      el("dt", { text: "Ending" }), el("dd", {}, outcomeBadge(T.outcome), T.void ? ` (void: ${T.void})` : ""),
      el("dt", { text: "Client" }), el("dd", { text: `${client}${c.playback_mode === "adaptive" ? " + smoothing stack" : ""}, chunk ${c.actions_per_chunk ?? "–"} actions` }),
      el("dt", { text: "Faults shifted by" }), el("dd", { text: c.fault_shift_applied_s !== undefined && D.windows.length ? `${fmt.num(c.fault_shift_applied_s, 2)} s` : "no faults" }),
      el("dt", { text: "Robot → server" }), el("dd", { text: `${i.machine} → ${i.server_gpu || i.server}` }),
      el("dt", { text: "Started" }), el("dd", { text: `${fmt.time(i.started_at)} · session ${T.session} · round ${T.round ?? "–"}` }),
      el("dt", { text: "Model commit" }), el("dd", { class: "small", style: { fontFamily: "var(--mono)" }, text: (i.revision || "–").slice(0, 12) }),
      el("dt", { text: "Control loop" }), el("dd", { text: i.cadence_hz ? `${fmt.num(i.cadence_hz, 1)} Hz` : "–" })),
    el("h3", { style: { marginTop: "14px" }, text: "Events (click to jump)" }),
    el("div", { class: "tbl" }, el("table", { class: "events" }, el("tbody", {}, events.map((e) =>
      el("tr", { onclick: () => seek(e.t - 0.3) }, el("td", { class: "n", text: `${fmt.num(e.t, 2)} s` }), el("td", { text: e.text })))))));
}
