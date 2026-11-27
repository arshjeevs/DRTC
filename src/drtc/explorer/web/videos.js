// The Videos page: every chosen video (lab/explorer/videos.yaml), one after another. Scene and wrist camera in sync,
// a caption per trial, live flags while a fault is on or the arm waits, and a strip of the trial's faults and stalls.
"use strict";

let B = null, LIST = [], idx = 0, rate = 1, loop = true, playing = false, raf = null, nextT = null;
let vids = [], cur = null;   // the videos on screen and the current trial's detail
const details = new Map();

document.getElementById("top").replaceWith(topbar("videos"));

loadData().then((b) => {
  B = b;
  const sel = b.selection || {};
  for (const e of b.evals) {
    if (e.missing) continue;
    for (const id of sel[e.name] || []) {
      const t = e.trials.find((x) => x.id === id);
      if (t && t.cameras.length) LIST.push({ e, t });
    }
  }
  if (!LIST.length) return empty();
  build();
  const h = hash.get();
  const start = Math.max(0, LIST.findIndex((x) => x.t.id === h.id && x.e.name === h.eval));
  load(start, true);
}).catch((err) => {
  document.getElementById("app").replaceChildren(el("div", { class: "empty", text: `Could not read the results: ${err.message}` }));
});

function empty() {
  document.getElementById("app").replaceChildren(el("div", { class: "panel", style: { display: "grid", gap: "8px", maxWidth: "640px" } },
    el("h2", { text: "No videos chosen yet" }),
    el("p", { class: "muted", text: B.editable
      ? "Choose them in Trials & videos: press “☆ Choose videos”, mark the trials with “☆ Show this video”, then Save. This page then plays them one after another."
      : "The saved choice is empty; start lab explore without --present to choose videos." }),
    B.editable ? el("div", {}, el("a", { class: "iconbtn", style: { textDecoration: "none", display: "inline-block" }, href: "/trials", text: "Go to Trials & videos →" })) : null));
}

// --- the page --------------------------------------------------------------------------------------------------------
let stage, cams, caption, strip, clock, playBtn, list;

function build() {
  cams = el("div", { class: "cams" });
  caption = el("div", { class: "caption" });
  strip = el("div", { class: "strip", title: "Click to jump", onclick: (e) => {
    const m = vids[0]; if (!m || !m.duration) return;
    const r = strip.getBoundingClientRect();
    seekVideo(((e.clientX - r.left) / r.width) * m.duration);
  } });
  clock = el("span", { class: "clock" });
  playBtn = el("button", { class: "main", type: "button", onclick: toggle, text: "❚❚ Pause" });
  const speeds = [0.5, 1, 2].map((s) => el("button", { type: "button", "aria-pressed": s === rate ? "true" : "false", onclick: (e) => {
    rate = s; for (const v of vids) v.playbackRate = s;
    e.target.parentNode.querySelectorAll("button[data-speed]").forEach((b) => b.setAttribute("aria-pressed", b === e.target ? "true" : "false"));
  }, "data-speed": s, text: `${s}×` }));
  const loopBtn = el("button", { type: "button", "aria-pressed": "true", onclick: (e) => { loop = !loop; e.target.setAttribute("aria-pressed", loop ? "true" : "false"); }, text: "Loop" });
  stage = el("div", { class: "stage" }, caption, cams, strip,
    el("div", { class: "strip-legend" },
      el("span", {}, el("span", { class: "sw", style: { background: "rgba(241, 138, 89, 0.7)" } }), "network fault"),
      el("span", {}, el("span", { class: "sw", style: { background: "rgba(220, 60, 61, 0.9)" } }), "arm waiting for actions")),
    el("div", { class: "ctrl" },
      el("button", { type: "button", onclick: () => go(-1), text: "⏮ Previous" }), playBtn,
      el("button", { type: "button", onclick: () => go(1), text: "Next ⏭" }), clock,
      el("span", { style: { flex: "1" } }), ...speeds, loopBtn,
      el("button", { type: "button", onclick: fullscreen, text: "⛶ Full screen" })));
  list = el("div", { class: "list" }, LIST.map((x, i) => el("button", { class: "item", type: "button", onclick: () => load(i, true) },
    el("span", { class: "n", text: `${i + 1} / ${LIST.length} · ${x.e.tag}` }),
    el("span", { class: "t", text: `${x.e.model[x.t.model].name} · trial ${x.t.trial}` }),
    el("span", { class: "c" }, el("span", { class: `sw s${x.e.cond[x.t.condition].slot}` }), x.e.cond[x.t.condition].short),
    el("span", {}, outcomeBadge(x.t.outcome)))));
  document.getElementById("app").replaceChildren(stage,
    el("div", { class: "section-head", style: { marginTop: "18px" } }, el("h2", { text: `The chosen videos (${LIST.length})` }),
      el("span", { class: "small muted" }, el("kbd", { text: "space" }), " play / pause · ", el("kbd", { text: "←" }), " ", el("kbd", { text: "→" }), " previous / next · ", el("kbd", { text: "F" }), " full screen")),
    list);
}

function load(i, autoplay) {
  clearTimeout(nextT);
  stopLoop();
  idx = (i + LIST.length) % LIST.length;
  const { e, t } = LIST[idx];
  hash.set({ eval: e.name, id: t.id });
  const c = e.cond[t.condition];
  caption.replaceChildren(
    el("span", { class: "title", text: `${e.model[t.model].name} · trial ${t.trial}` }),
    el("span", { class: "cond" }, el("span", { class: `sw s${c.slot}` }), c.label),
    outcomeBadge(t.outcome),
    t.time_to_success_s ? el("span", { style: { color: "#d6d6d6" }, text: `done in ${fmt.sec(t.time_to_success_s)}` }) : null,
    el("span", { class: "count", text: `${idx + 1} / ${LIST.length}` }),
    el("a", { href: trialsLink(e.name, { id: t.id }), style: { color: "#9cc4f5", fontSize: "13px" }, text: "details ↗" }));
  vids = [];
  cams.className = `cams ${t.cameras.length < 2 ? "one" : ""}`;
  cams.replaceChildren(...t.cameras.map((cam) => {
    const v = el("video", { muted: true, playsinline: true, preload: "auto", src: `/video/${encodeURIComponent(e.name)}/${t.id}/${cam}.mp4` });
    v.muted = true; v.playbackRate = rate;
    vids.push(v);
    return el("div", { class: "cam" }, v, el("span", { class: "cap", text: cam === "scene" ? "Scene camera" : cam === "wrist" ? "Wrist camera" : cam }),
      el("span", { class: "flag fault" }), el("span", { class: "flag stall", text: "Arm waiting for actions" }));
  }));
  vids[0].addEventListener("ended", onEnded);
  vids[0].addEventListener("loadedmetadata", drawStrip);
  list.querySelectorAll(".item").forEach((b, k) => b.setAttribute("aria-current", k === idx ? "true" : "false"));
  list.children[idx]?.scrollIntoView({ block: "nearest" });
  cur = null;
  const key = `${e.name}|${t.id}`;
  (details.has(key) ? Promise.resolve(details.get(key)) : fetch(`/api/trial?eval=${encodeURIComponent(e.name)}&id=${encodeURIComponent(t.id)}`)
    .then((r) => (r.ok ? r.json() : null)).then((d) => { details.set(key, d); return d; }))
    .then((d) => { if (d && !d.stalls) d.stalls = stallsOf(d); if (LIST[idx].t === t) { cur = d; drawStrip(); } });
  if (autoplay) play(); else { playing = false; playBtn.textContent = "▶ Play"; }
}

function play() {
  playing = true; playBtn.textContent = "❚❚ Pause";
  for (const v of vids) { v.playbackRate = rate; v.play().catch(() => {}); }
  stopLoop();
  const step = () => { tick(); raf = requestAnimationFrame(step); };
  raf = requestAnimationFrame(step);
}
function pause() { playing = false; playBtn.textContent = "▶ Play"; for (const v of vids) v.pause(); clearTimeout(nextT); }
function toggle() { if (playing) pause(); else { if (vids[0]?.ended) seekVideo(0); play(); } }
function stopLoop() { if (raf) cancelAnimationFrame(raf); raf = null; }
function go(d) { load(idx + d, true); }

function onEnded() {
  if (!playing) return;
  if (idx === LIST.length - 1 && !loop) { pause(); return; }
  nextT = setTimeout(() => load(idx + 1, true), 1200);  // a short breath between trials
}

function seekVideo(vt) { for (const v of vids) v.currentTime = vt; tick(); }

function fullscreen() {
  if (document.fullscreenElement) document.exitFullscreen(); else stage.requestFullscreen?.().catch(() => {});
}

addEventListener("keydown", (e) => {
  if (!LIST.length || /INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName)) return;
  if (e.key === " ") { e.preventDefault(); toggle(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
  else if (e.key === "f" || e.key === "F") fullscreen();
});

// --- trial time, faults and stalls --------------------------------------------------------------------------------------
const toTrial = (vt) => (cur?.video ? cur.video.t0 + vt * cur.video.scale : vt);
const toVideo = (tt) => (cur?.video ? (tt - cur.video.t0) / cur.video.scale : tt);

function stallsOf(d) {
  const t = d.ticks.t, a = d.ticks.action, dt = 1 / 30, out = [];
  const ex = t.filter((x, i) => x !== null && a[i] === 1);
  for (let i = 0; i < ex.length - 1; i++) if (ex[i + 1] - (ex[i] + dt) > 0.5 * dt) out.push([ex[i] + dt, ex[i + 1]]);
  if (ex.length && d.t_end - (ex[ex.length - 1] + dt) > 0.5 * dt) out.push([ex[ex.length - 1] + dt, d.t_end]);
  return out;
}

function drawStrip() {
  const m = vids[0];
  const ph = el("i", { class: "ph", style: { left: "0%" } });
  const parts = [];
  if (cur && m && m.duration) {
    const pct = (tt) => `${Math.max(0, Math.min(100, (100 * toVideo(tt)) / m.duration))}%`;
    const w = (a, b) => `${Math.max(0.3, (100 * (toVideo(b) - toVideo(a))) / m.duration)}%`;
    for (const f of cur.windows) parts.push(el("i", { class: "fb", style: { left: pct(f.start), width: w(f.start, f.end) }, title: B.faults[f.kind] }));
    for (const [a, b] of cur.stalls) parts.push(el("i", { class: "st", style: { left: pct(a), width: w(a, b) } }));
  }
  strip.replaceChildren(...parts, ph);
}

function tick() {
  const m = vids[0];
  if (!m) return;
  for (const v of vids.slice(1)) if (Math.abs(v.currentTime - m.currentTime) > 0.08) v.currentTime = m.currentTime;
  const ph = strip.querySelector(".ph");
  if (ph && m.duration) ph.style.left = `${(100 * m.currentTime) / m.duration}%`;
  const tt = toTrial(m.currentTime);
  clock.textContent = cur ? `${tt.toFixed(1)} s` : `${m.currentTime.toFixed(1)} s`;
  const fault = cur?.windows.find((w) => tt >= w.start && tt <= w.end);
  const stalled = !!cur?.stalls?.some(([a, b]) => tt >= a && tt <= b);
  stage.querySelectorAll(".flag.fault").forEach((f) => { f.style.display = fault ? "block" : "none"; f.textContent = fault ? `⚡ ${B.faults[fault.kind]}` : ""; });
  stage.querySelectorAll(".flag.stall").forEach((f) => { f.style.display = stalled ? "block" : "none"; });
}
