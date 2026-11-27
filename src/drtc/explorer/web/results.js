// The Results window: the whole picture of one eval (setup, success, the tests, how runs ended, timing and
// smoothness, recovery), every number from the raw trials. Every mark opens its trials in the Trials window.
"use strict";

let B = null;           // the bundle
let E = null;           // the eval shown
let METRIC = "M2_stall_pct";

document.getElementById("top").replaceWith(topbar("results"));

loadData().then((b) => {
  B = b;
  const h = hash.get();
  if (h.metric && B.metric[h.metric]) METRIC = h.metric;
  const known = h.eval && (B.evals.some((e) => e.name === h.eval) || (h.eval === "key" && B.highlights));
  select(known ? h.eval : B.highlights ? "key" : B.evals[0].name);
}).catch((err) => {
  document.getElementById("app").replaceChildren(el("div", { class: "empty", text: `Could not read the results: ${err.message}` }));
});

let resizeT = null;
addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => E && render(), 150); });
addEventListener("popstate", () => { const h = hash.get(); if (B && h.eval && h.eval !== E.name) select(h.eval, true); });

const KEY = { name: "key", key: true };  // the key findings view

function select(name, fromHistory) {
  E = name === "key" ? KEY : B.evals.find((e) => e.name === name);
  if (!fromHistory) hash.set({ eval: name, metric: METRIC });
  render();
}

function render() {
  const app = document.getElementById("app");
  const scroll = scrollY;
  app.replaceChildren(evalSwitcher());
  if (E.key) { app.append(...keyFindings(B)); scrollTo(0, scroll); return; }
  if (E.missing) { app.append(el("div", { class: "panel empty", text: E.missing })); return; }
  const multi = E.conditions.length > 1;
  app.append(setupPanel(), multi ? conditionsPanel() : null, headline(), successPanel(),
    E.tests.length ? testsPanel() : null, outcomesPanel(), metricsPanel(),
    hasRecovery() ? recoveryPanel() : null, fullTablePanel(), sessionsPanel());
  scrollTo(0, scroll);
}

const multiCond = () => E.conditions.length > 1;
const condName = (c) => E.cond[c]?.short || c;
const modelName = (m) => E.model[m]?.name || m;

// --- the three evals ---------------------------------------------------------------------------------------------
function evalSwitcher() {
  const key = B.highlights ? el("button", { class: "evalcard keycard", role: "tab", type: "button", "aria-selected": E.key ? "true" : "false",
    onclick: () => select("key") }, el("span", { class: "tag", text: "Start here" }), el("span", { class: "t", text: "Key findings" }),
  el("span", { class: "h", text: B.highlights.ladder.map((p) => fmt.pct(p.rate)).join(" → ") })) : null;
  return el("div", { class: "evals", role: "tablist", "aria-label": "Evaluations" }, key, B.evals.map((e) => {
    let head = "";
    if (!e.missing) {
      head = e.pooled.length > 1
        ? e.pooled.map((p) => `${p.k}/${p.n}`).join(" → ") + " successes"
        : `${e.pooled[0].k}/${e.pooled[0].n} successes · ${e.models.length} policies`;
    }
    return el("button", {
      class: "evalcard", role: "tab", type: "button", "aria-selected": e.name === E.name ? "true" : "false",
      onclick: () => select(e.name),
    }, el("span", { class: "tag", text: e.tag }), el("span", { class: "t", text: e.title }),
    el("span", { class: "h", text: e.missing ? "no local results" : head }));
  }));
}

function setupPanel() {
  const counted = E.trials.filter((t) => t.valid).length;
  const withVideo = E.trials.filter((t) => t.cameras.length).length;
  const sess = E.sessions;
  const where = [...new Set(sess.map((s) => `${s.machine === "pi" ? "Raspberry Pi 5" : s.machine} → ${s.server_gpu || s.server} server`))].join(", ");
  const dates = sess.length ? `${fmt.time(sess[0].first)} – ${fmt.time(sess[sess.length - 1].last).slice(11)}` : "–";
  return el("section", { class: "section" }, el("div", { class: "panel" },
    el("header", {}, el("h1", { text: E.title }), el("p", { class: "lede", text: E.summary })),
    el("div", { class: "grid2", style: { marginTop: "14px" } },
      el("dl", { class: "kv" },
        el("dt", { text: "Task" }), el("dd", { text: E.task }),
        el("dt", { text: "Success rule" }), el("dd", { text: E.success_rule }),
        el("dt", { text: "Plan" }), el("dd", { text: `${E.models.length} ${E.models.length === 1 ? "policy" : "policies"} × ${E.conditions.length} condition${E.conditions.length > 1 ? "s" : ""} × ${E.planned_trials} trials, ${E.time_limit_s} s limit per trial` })),
      el("dl", { class: "kv" },
        el("dt", { text: "Robot → server" }), el("dd", { text: where }),
        el("dt", { text: "Recorded" }), el("dd", { text: `${dates} (${sess.length} session${sess.length > 1 ? "s" : ""})` }),
        el("dt", { text: "Trials" }), el("dd", { text: `${counted} counted, ${E.trials.length - counted} not counted, ${withVideo} with video` }),
        el("dt", { text: "Policies" }), el("dd", { text: E.models.map((m) => m.name).join(", ") }))),
    E.notes.length ? el("div", { style: { marginTop: "12px", display: "grid", gap: "8px" } }, E.notes.map((n) => el("p", { class: "note", text: n }))) : null));
}

function conditionsPanel() {
  const cards = [];
  E.conditions.forEach((c, i) => {
    cards.push(el("div", { class: `cond s${c.slot}` },
      el("h3", { text: c.label }),
      el("ul", {},
        el("li", { text: c.client }),
        el("li", { text: c.smoothing.length ? `Smoothing: ${c.smoothing.join(", ")}` : "No smoothing" }),
        el("li", { text: c.faults.length ? `Faults: ${c.faults.map((f) => B.faults[f.kind].toLowerCase()).join(", ")}` : "Normal network (no faults)" }))));
  });
  const mix = E.conditions.find((c) => c.faults.length);
  return el("section", { class: "section" }, el("div", { class: "panel" },
    el("div", { class: "section-head" }, el("div", {}, el("h2", { text: "The conditions" }),
      el("p", { class: "lede", text: "Each policy runs the same task under every condition; only the client and the network change." }))),
    el("div", { class: "conds", style: { marginTop: "12px" } }, cards),
    mix ? faultSchedule(mix) : null));
}

function faultSchedule(c) {
  const box = el("div", { style: { marginTop: "16px" } });
  const W = Math.max(290, Math.min(1180, (document.getElementById("app").clientWidth || 900) - 44));
  const H = 74, L = 8, R = 8, top = 24, h = 24;
  const tmax = 12, x = (t) => L + (t / tmax) * (W - L - R);
  const s = sv("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": "Fault mix schedule: " + c.faults.map((f) => f.text).join("; ") });
  s.append(sv("line", { x1: x(0), x2: x(tmax), y1: top + h, y2: top + h, class: "axisline" }));
  for (let t = 0; t <= tmax; t += 2) {
    s.append(sv("line", { x1: x(t), x2: x(t), y1: top + h, y2: top + h + 4, class: "axisline" }));
    s.append(sv("text", { x: x(t), y: top + h + 17, "text-anchor": "middle", class: "tick-label", text: `${t} s` }));
  }
  for (const f of c.faults) {
    const g = sv("g", {});
    g.append(sv("rect", { x: x(f.start), y: top, width: Math.max(2, x(f.end) - x(f.start)), height: h, rx: 4, fill: "var(--fault-band)", stroke: "var(--serious)" }));
    g.append(sv("text", { x: (x(f.start) + x(f.end)) / 2, y: top - 7, "text-anchor": "middle", class: "value-label", text: B.faults[f.kind] }));
    s.append(hoverable(g, () => [tipRow(f.text)]));
  }
  const shift = c.shift ? ` The whole pattern starts ${c.shift[0]}–${c.shift[1]} s later in each trial (a random shift, the same for every condition and trial number), so a policy cannot learn when the faults come.` : "";
  box.append(el("h3", { text: "The fault mix" }),
    el("p", { class: "muted small", text: `Times from the start of a trial: ${c.faults.map((f) => f.text).join(", ")}.${shift}` }), s);
  return box;
}

// --- headline tiles ----------------------------------------------------------------------------------------------
function headline() {
  const tiles = E.pooled.map((p) => {
    const c = E.cond[p.condition];
    return el("button", { class: `tile s${c.slot}`, type: "button", onclick: () => { location.href = trialsLink(E.name, { condition: p.condition }); } },
      el("span", { class: "lbl" }, el("span", { class: "sw" }), multiCond() ? c.short : "All policies"),
      el("span", { class: "big" }, `${p.k}`, el("small", { text: ` / ${p.n}` })),
      el("span", { class: "sub", text: `${fmt.pct(p.rate)} success · 95% CI ${fmt.ci(p.lo, p.hi)}` }),
      el("span", { class: "sub", text: p.fault ? `${p.fault} run${p.fault > 1 ? "s" : ""} ended by the fault` : "no run ended by a fault" }),
      el("span", { class: "bar" }, el("i", { style: { width: `${100 * (p.rate || 0)}%` } })));
  });
  if (!multiCond()) {
    const best = [...E.groups].sort((a, b) => (b.rate ?? -1) - (a.rate ?? -1))[0];
    if (best) tiles.push(el("div", { class: "tile s1", style: { cursor: "default" } },
      el("span", { class: "lbl", text: "Best policy" }), el("span", { class: "big", style: { fontSize: "26px" }, text: modelName(best.model) }),
      el("span", { class: "sub", text: `${best.k}/${best.n} · ${fmt.pct(best.rate)} success` })));
  }
  return el("section", { class: "section" },
    el("div", { class: "section-head" }, el("h2", { text: multiCond() ? "Successes, all policies together" : "Overall" }),
      el("span", { class: "muted small", text: "Click a tile to watch its trials." })),
    el("div", { class: "tiles" }, tiles));
}

// --- success per policy -----------------------------------------------------------------------------------------------
function successPanel() {
  const panel = el("div", { class: "panel" });
  const head = el("div", { class: "section-head" },
    el("div", {}, el("h2", { text: "Success per policy" }),
      el("p", { class: "lede", text: "Bars: successes out of the counted trials; thin lines: 95% Wilson interval. Click a bar for its trials." })),
    multiCond() ? legend() : null);
  panel.append(head);
  const box = el("div", { style: { marginTop: "10px" } });
  panel.append(box);
  requestAnimationFrame(() => box.append(multiCond() ? groupedBars(box.clientWidth) : rankedBars(box.clientWidth)));
  return el("section", { class: "section" }, panel);
}

function legend() {
  return el("div", { class: "legend" }, E.conditions.map((c) => el("span", { class: `s${c.slot}` }, el("span", { class: "sw" }), c.short)));
}

function groupTip(g) {
  const o = g.outcomes;
  return [tipRow(`${g.k}/${g.n}`, `${modelName(g.model)} · ${condName(g.condition)}`, slotColor(E.cond[g.condition].slot)),
    tipRow(fmt.pct(g.rate), `success, 95% CI ${fmt.ci(g.lo, g.hi)}`),
    el("div", { class: "k small", text: `✓ ${o.success}  ✗ ${o.failed}  ⏱ ${o.time_limit}  ⚡ ${o.fault}` })];
}

function groupedBars(width) {
  const W = Math.max(290, width), H = 300, L = 44, R = 8, T = 24, Bm = 46;
  const s = sv("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "Success per policy and condition" });
  const y = (v) => T + (1 - v) * (H - T - Bm);
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    s.append(sv("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: v ? "gridline" : "axisline" }));
    s.append(sv("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", class: "tick-label", text: `${v * 100}%` }));
  }
  const nm = E.models.length, nc = E.conditions.length;
  const gw = (W - L - R) / nm, pad = Math.min(28, gw * 0.14), bw = Math.min(54, (gw - 2 * pad - (nc - 1) * 2) / nc);
  E.models.forEach((m, mi) => {
    const gx = L + mi * gw + (gw - (nc * bw + (nc - 1) * 2)) / 2;
    s.append(sv("text", { x: L + mi * gw + gw / 2, y: H - Bm + 22, "text-anchor": "middle", class: "cat-label", text: m.name }));
    E.conditions.forEach((c, ci) => {
      const g = E.group(c.id, m.id); if (!g || !g.n) return;
      const x0 = gx + ci * (bw + 2), top = y(g.rate), base = y(0), hgt = Math.max(0, base - top);
      const grp = sv("g", { style: "cursor:pointer", onclick: () => { location.href = trialsLink(E.name, { condition: c.id, model: m.id }); } });
      const r = Math.min(4, hgt / 2, bw / 2);
      grp.append(sv("path", { class: "mark", fill: slotColor(c.slot), d: hgt > 0
        ? `M${x0},${base} V${top + r} Q${x0},${top} ${x0 + r},${top} H${x0 + bw - r} Q${x0 + bw},${top} ${x0 + bw},${top + r} V${base} Z`
        : `M${x0},${base - 1.5} H${x0 + bw} V${base} H${x0} Z` }));
      const cx = x0 + bw / 2;
      grp.append(sv("line", { x1: cx, x2: cx, y1: y(g.hi), y2: y(g.lo), stroke: "var(--ink)", "stroke-opacity": 0.45, "stroke-width": 1.5 }));
      grp.append(sv("text", { x: cx, y: Math.min(top, y(g.hi)) - 6, "text-anchor": "middle", class: "value-label", text: `${g.k}/${g.n}` }));
      grp.append(sv("rect", { class: "hit", x: x0 - 1, y: T, width: bw + 2, height: H - T - Bm }));
      s.append(hoverable(grp, () => groupTip(g)));
    });
  });
  return s;
}

function rankedBars(width) {
  const rows = [...E.groups].filter((g) => g.n).sort((a, b) => b.rate - a.rate);
  const W = Math.max(290, width), rowH = 34, L = 140, R = 110, T = 8, H = T + rows.length * rowH + 26;
  const s = sv("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "Success per policy" });
  const x = (v) => L + v * (W - L - R), yb = T + rows.length * rowH;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    s.append(sv("line", { x1: x(v), x2: x(v), y1: T, y2: yb, class: v ? "gridline" : "axisline" }));
    s.append(sv("text", { x: x(v), y: yb + 18, "text-anchor": "middle", class: "tick-label", text: `${v * 100}%` }));
  }
  rows.forEach((g, i) => {
    const yc = T + i * rowH + rowH / 2, bh = 18, w = x(g.rate) - x(0);
    const grp = sv("g", { style: "cursor:pointer", onclick: () => { location.href = trialsLink(E.name, { model: g.model }); } });
    grp.append(sv("text", { x: L - 10, y: yc + 4, "text-anchor": "end", class: "cat-label", text: modelName(g.model) }));
    const r = Math.min(4, w / 2);
    grp.append(sv("path", { class: "mark", fill: "var(--s1)", d: w > 0
      ? `M${x(0)},${yc - bh / 2} H${x(0) + w - r} Q${x(0) + w},${yc - bh / 2} ${x(0) + w},${yc - bh / 2 + r} V${yc + bh / 2 - r} Q${x(0) + w},${yc + bh / 2} ${x(0) + w - r},${yc + bh / 2} H${x(0)} Z`
      : `M${x(0)},${yc - bh / 2} h1.5 v${bh} h-1.5 Z` }));
    grp.append(sv("line", { x1: x(g.lo), x2: x(g.hi), y1: yc, y2: yc, stroke: "var(--ink)", "stroke-opacity": 0.45, "stroke-width": 1.5 }));
    grp.append(sv("text", { x: Math.max(x(g.rate), x(g.hi)) + 8, y: yc + 4, class: "value-label", text: `${g.k}/${g.n} · ${fmt.pct(g.rate)}` }));
    grp.append(sv("rect", { class: "hit", x: 0, y: yc - rowH / 2, width: W, height: rowH }));
    s.append(hoverable(grp, () => groupTip(g)));
  });
  return s;
}

// --- the tests -----------------------------------------------------------------------------------------------------
function testsPanel() {
  const n = E.tests.length;
  const rows = E.tests.map((t) => el("tr", { class: "click", onclick: () => { location.href = trialsLink(E.name, { model: t.model }); } },
    el("td", {}, compareCell(t.condition, t.reference)),
    el("td", { text: modelName(t.model) }),
    el("td", { class: "n", text: `${t.ka}/${t.na} vs ${t.kb}/${t.nb}` }),
    el("td", { class: "n", text: fmt.p(t.p) }),
    el("td", { class: "n", text: fmt.p(t.p_holm) }),
    el("td", {}, verdict(t))));
  const pooled = E.pooled_tests.map((t) => el("tr", {},
    el("td", {}, compareCell(t.condition, t.reference)), el("td", { class: "muted", text: "all policies pooled" }),
    el("td", { class: "n", text: `${t.ka}/${t.na} vs ${t.kb}/${t.nb}` }), el("td", { class: "n", text: fmt.p(t.p) }),
    el("td", { class: "n muted", text: "–" }), el("td", {}, el("span", { class: "badge ns", text: "exploratory" }))));
  return el("section", { class: "section" }, el("div", { class: "panel" },
    el("h2", { text: "Is the difference real? The planned tests" }),
    el("p", { class: "lede", text: `The three comparisons were named before the trials ran. Each is tested per policy with Fisher's exact test (10 against 10 trials), and the ${n} p-values are Holm-corrected together, so a "significant" result survives the correction for testing ${n} times. The pooled rows (all policies together) are exploratory.` }),
    el("div", { class: "tbl", style: { marginTop: "10px" } }, el("table", {},
      el("thead", {}, el("tr", {}, ["Comparison", "Policy"].map((h) => el("th", { text: h })), ["Successes", "p", "p (Holm)"].map((h) => el("th", { class: "n", text: h })), el("th", { text: "Result" }))),
      el("tbody", {}, rows, pooled)))));
}

function compareCell(a, b) {
  return el("span", { style: { display: "inline-flex", gap: "6px", alignItems: "center", flexWrap: "wrap" } },
    el("span", { class: `s${E.cond[a].slot}`, style: { display: "inline-flex", gap: "5px", alignItems: "center" } }, el("span", { class: "sw" }), condName(a)),
    el("span", { class: "muted", text: "vs" }),
    el("span", { class: `s${E.cond[b].slot}`, style: { display: "inline-flex", gap: "5px", alignItems: "center" } }, el("span", { class: "sw" }), condName(b)));
}

function verdict(t) {
  if (t.significant) return el("span", { class: "badge sig", text: "✓ better, significant" });
  const dir = t.ka / t.na > t.kb / t.nb ? "better" : t.ka / t.na < t.kb / t.nb ? "worse" : "same";
  return el("span", { class: "badge ns", text: `${dir}, not significant` });
}

// --- how the runs ended -----------------------------------------------------------------------------------------------
function outcomesPanel() {
  const panel = el("div", { class: "panel" },
    el("div", { class: "section-head" },
      el("div", {}, el("h2", { text: "How the runs ended" }),
        el("p", { class: "lede", text: "Every counted trial by its ending: the task done, failed (the operator stopped a wrong attempt), out of time, or ended by the client itself when the network fault starved it. Click a row for its trials." })),
      el("div", { class: "legend" }, ["success", "failed", "time_limit", "fault"].map((o) =>
        el("span", {}, el("span", { class: "sw", style: { background: OUTCOMES[o].color } }), `${OUTCOMES[o].icon} ${OUTCOMES[o].short || OUTCOMES[o].label}`)))));
  const box = el("div", { style: { marginTop: "10px" } });
  panel.append(box);
  requestAnimationFrame(() => box.append(outcomeBars(box.clientWidth)));
  return el("section", { class: "section" }, panel);
}

function outcomeBars(width) {
  const rows = [];
  if (multiCond()) for (const c of E.conditions) for (const m of E.models) rows.push(E.group(c.id, m.id));
  else rows.push(...[...E.groups].sort((a, b) => b.rate - a.rate));
  const valid = rows.filter((g) => g && g.n);
  const multi = multiCond();
  const W = Math.max(290, width), rowH = 26, L = multi ? 270 : 140, R = 10, T = 4;
  let H = T + valid.length * rowH + 8;
  const s = sv("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "How the runs ended" });
  const maxN = Math.max(...valid.map((g) => g.n));
  const x = (v) => L + (v / maxN) * (W - L - R);
  let prevCond = null;
  valid.forEach((g, i) => {
    const yc = T + i * rowH + rowH / 2, bh = 18;
    const grp = sv("g", { style: "cursor:pointer", onclick: () => { location.href = trialsLink(E.name, multi ? { condition: g.condition, model: g.model } : { model: g.model }); } });
    if (multi && g.condition !== prevCond) {
      if (prevCond) s.append(sv("line", { x1: 0, x2: W, y1: yc - rowH / 2, y2: yc - rowH / 2, class: "gridline" }));
      prevCond = g.condition;
    }
    const c = E.cond[g.condition];
    if (multi) grp.append(sv("rect", { x: 0, y: yc - 5, width: 10, height: 10, rx: 3, fill: slotColor(c.slot) }));
    grp.append(sv("text", { x: multi ? 16 : L - 10, y: yc + 4, "text-anchor": multi ? "start" : "end", class: "cat-label",
      text: multi ? `${modelName(g.model)} · ${c.short}` : modelName(g.model) }));
    let acc = 0;
    for (const o of ["success", "failed", "time_limit", "fault"]) {
      const k = g.outcomes[o]; if (!k) continue;
      const x0 = x(acc) + (acc ? 1 : 0), x1 = x(acc + k) - 1;
      grp.append(sv("rect", { x: x0, y: yc - bh / 2, width: Math.max(1, x1 - x0), height: bh, rx: 3, fill: OUTCOMES[o].color }));
      if (x1 - x0 > 16) grp.append(sv("text", { x: (x0 + x1) / 2, y: yc + 4, "text-anchor": "middle", class: "value-label",
        style: `fill:${o === "time_limit" || o === "fault" ? "#0b0b0b" : "#ffffff"}`, text: k }));
      acc += k;
    }
    grp.append(sv("rect", { class: "hit", x: 0, y: yc - rowH / 2, width: W, height: rowH }));
    s.append(hoverable(grp, () => groupTip(g)));
  });
  return s;
}

// --- timing and smoothness ------------------------------------------------------------------------------------------
function metricsPanel() {
  const available = B.metrics.filter((m) => E.groups.some((g) => g.stats[m.col]));
  if (!available.some((m) => m.col === METRIC)) METRIC = available[0]?.col;
  const chips = el("div", { class: "metric-chips", role: "group", "aria-label": "Metric" }, available.map((m) =>
    el("button", { class: "chip", type: "button", "aria-pressed": m.col === METRIC ? "true" : "false",
      onclick: () => { METRIC = m.col; hash.set({ eval: E.name, metric: METRIC }); render(); }, text: m.name })));
  const m = B.metric[METRIC];
  const panel = el("div", { class: "panel" },
    el("div", { class: "section-head" }, el("div", {}, el("h2", { text: "Timing and smoothness" }),
      el("p", { class: "lede", text: "Pick a measure. Dot: the median over the counted trials; bar: the middle half (25th to 75th percentile)." })),
      multiCond() ? legend() : null),
    el("div", { style: { marginTop: "10px" } }, chips),
    el("p", { class: "small", style: { marginTop: "10px" } }, el("strong", { text: `${m.name}${m.unit ? ` (${m.unit})` : ""}: ` }),
      el("span", { class: "muted", text: `${m.help} ${m.lower ? "Lower is better." : ""}` })));
  const box = el("div", { style: { marginTop: "8px" } });
  panel.append(box);
  requestAnimationFrame(() => box.append(rangeChart(box.clientWidth, m)));
  return el("section", { class: "section" }, panel);
}

function rangeChart(width, m) {
  const multi = multiCond();
  const W = Math.max(290, width), H = 280, L = 52, R = 8, T = 22, Bm = 46;
  const vals = E.groups.flatMap((g) => g.stats[m.col] || []);
  const { max, ticks } = niceTicks(Math.max(...vals, 0) * 1.05);
  const s = sv("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `${m.name} per policy` });
  const y = (v) => T + (1 - v / max) * (H - T - Bm);
  for (const v of ticks) {
    s.append(sv("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: v ? "gridline" : "axisline" }));
    s.append(sv("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", class: "tick-label", text: `${+v.toFixed(2)}` }));
  }
  s.append(sv("text", { x: 4, y: 12, class: "tick-label", text: m.unit || "" }));
  const models = multi ? E.models : [...E.models].sort((a, b) => ((E.group(E.conditions[0].id, a.id)?.stats[m.col] || [Infinity])[0]) - ((E.group(E.conditions[0].id, b.id)?.stats[m.col] || [Infinity])[0]));
  const nm = models.length, nc = E.conditions.length, gw = (W - L - R) / nm, slot = Math.min(46, (gw * 0.8) / nc);
  models.forEach((md, mi) => {
    const gx = L + mi * gw + (gw - nc * slot) / 2;
    s.append(sv("text", { x: L + mi * gw + gw / 2, y: H - Bm + 22, "text-anchor": "middle", class: "cat-label", text: md.name }));
    E.conditions.forEach((c, ci) => {
      const g = E.group(c.id, md.id); const st = g?.stats[m.col]; if (!st) return;
      const cx = gx + ci * slot + slot / 2, col = slotColor(c.slot);
      const grp = sv("g", { style: "cursor:pointer", onclick: () => { location.href = trialsLink(E.name, { condition: c.id, model: md.id }); } });
      grp.append(sv("rect", { x: cx - 3, y: y(st[2]), width: 6, height: Math.max(2, y(st[1]) - y(st[2])), rx: 3, fill: col, "fill-opacity": 0.35 }));
      grp.append(sv("circle", { class: "mark", cx, cy: y(st[0]), r: 6, fill: col, stroke: "var(--surface)", "stroke-width": 2 }));
      grp.append(sv("text", { x: cx, y: y(st[2]) - 7, "text-anchor": "middle", class: "value-label", style: "font-size:11px", text: fmt.num(st[0], m.dec) }));
      grp.append(sv("rect", { class: "hit", x: cx - slot / 2, y: T, width: slot, height: H - T - Bm }));
      s.append(hoverable(grp, () => [tipRow(`${fmt.num(st[0], m.dec)} ${m.unit}`, `median · ${md.name} · ${condName(c.id)}`, col),
        tipRow(`${fmt.num(st[1], m.dec)} – ${fmt.num(st[2], m.dec)}`, "middle half (IQR)"), el("div", { class: "k small", text: `${g.n} trials` })]));
    });
  });
  return s;
}

// --- recovery -------------------------------------------------------------------------------------------------------
function hasRecovery() { return E.groups.some((g) => Object.keys(g.recovery).length); }

function recoveryPanel() {
  const faults = ["spike", "drops", "outage", "rise"].filter((f) => E.groups.some((g) => g.recovery[f]));
  const conds = E.conditions.filter((c) => E.groups.some((g) => g.condition === c.id && Object.keys(g.recovery).length));
  const head1 = el("tr", {}, el("th", { text: "" }), faults.map((f) => el("th", { colspan: conds.length, style: { textAlign: "center" }, text: B.faults[f] })));
  const head2 = el("tr", {}, el("th", { text: "Policy" }), faults.map(() => conds.map((c) =>
    el("th", { style: { textAlign: "center" } }, el("span", { class: `s${c.slot}`, style: { display: "inline-flex", gap: "5px", alignItems: "center" } }, el("span", { class: "sw" }), c.short.replace(" · faults", ""))))));
  const body = E.models.map((m) => el("tr", {}, el("td", { class: "cat", text: m.name }), faults.map((f) => conds.map((c) => {
    const r = E.group(c.id, m.id)?.recovery[f];
    if (!r) return el("td", { class: "heat muted", text: "–" });
    const share = r.k / r.n;
    const td = el("td", { class: "heat", style: { background: `color-mix(in srgb, var(--s1) ${Math.round(8 + 42 * share)}%, transparent)` } },
      el("div", { style: { fontWeight: 700 }, text: `${r.k}/${r.n}` }),
      el("div", { class: "small muted", text: r.time ? `${fmt.num(r.time[0], 1)} s` : "never" }));
    return hoverable(td, () => [tipRow(`${r.k} of ${r.n}`, `runs moved again after the ${B.faults[f].toLowerCase()}`),
      r.time ? tipRow(`${fmt.num(r.time[0], 2)} s`, "median time until the arm moved again") : null,
      el("div", { class: "k small", text: `${m.name} · ${c.label}` })]);
  }))));
  return el("section", { class: "section" }, el("div", { class: "panel" },
    el("h2", { text: "Did the arm recover after each fault?" }),
    el("p", { class: "lede", text: "For every fault of the mix: in how many runs the arm moved again after it ended (darker = more runs), and the median seconds it took (0 s: the arm never stopped). A run the client ended counts as not recovered." }),
    el("div", { class: "tbl", style: { marginTop: "10px" } }, el("table", {}, el("thead", {}, head1, head2), el("tbody", {}, body)))));
}

// --- every number --------------------------------------------------------------------------------------------------
function fullTablePanel() {
  const cols = B.metrics.filter((m) => E.groups.some((g) => g.stats[m.col]));
  const rows = E.groups.filter((g) => g.n).map((g) => el("tr", { class: "click", onclick: () => { location.href = trialsLink(E.name, { condition: g.condition, model: g.model }); } },
    multiCond() ? el("td", {}, el("span", { class: `s${E.cond[g.condition].slot}`, style: { display: "inline-flex", gap: "5px", alignItems: "center" } }, el("span", { class: "sw" }), condName(g.condition))) : null,
    el("td", { text: modelName(g.model) }),
    el("td", { class: "n", text: `${g.k}/${g.n}` }),
    cols.map((m) => { const v = g.stats[m.col]; return el("td", { class: "n", text: v ? `${fmt.num(v[0], m.dec)} [${fmt.num(v[1], m.dec)}–${fmt.num(v[2], m.dec)}]` : "–" }); })));
  return el("section", { class: "section" }, el("details", { class: "panel" },
    el("summary", { text: "Every number (median [IQR] per policy and condition)" }),
    el("div", { class: "tbl", style: { marginTop: "10px" } }, el("table", {},
      el("thead", {}, el("tr", {}, multiCond() ? el("th", { text: "Condition" }) : null, el("th", { text: "Policy" }), el("th", { class: "n", text: "Success" }),
        cols.map((m) => el("th", { class: "n", text: `${m.name}${m.unit ? ` (${m.unit})` : ""}` })))),
      el("tbody", {}, rows)))));
}

function sessionsPanel() {
  return el("section", { class: "section" }, el("details", { class: "panel" },
    el("summary", { text: `Sessions and trials not counted (${E.not_counted.length})` }),
    el("div", { class: "grid2", style: { marginTop: "12px" } },
      el("div", { class: "tbl" }, el("table", {},
        el("thead", {}, el("tr", {}, ["Session", "Robot → server"].map((h) => el("th", { text: h })), el("th", { class: "n", text: "Trials (counted)" }))),
        el("tbody", {}, E.sessions.map((s) => el("tr", {}, el("td", { text: s.id }), el("td", { text: `${s.machine} → ${s.server_gpu || s.server}` }), el("td", { class: "n", text: `${s.trials} (${s.valid})` })))))),
      el("div", { class: "tbl" }, E.not_counted.length ? el("table", {},
        el("thead", {}, el("tr", {}, el("th", { text: "Trial" }), el("th", { text: "Why not counted" }))),
        el("tbody", {}, E.not_counted.map((n) => el("tr", { class: "click", onclick: () => { location.href = trialsLink(E.name, { id: n.id }); } },
          el("td", { class: "small", text: n.id.split("/").slice(1).join(" / ") }), el("td", { class: "small muted", text: n.why })))))
        : el("p", { class: "muted", text: "Every recorded trial counts." })))));
}
