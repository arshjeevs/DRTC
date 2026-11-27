// Shared helpers of the Results and Trials windows: DOM and SVG builders (text always via textContent), the
// tooltip, formatting, theme, the data bundle and the URL hash.
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(n.style, v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) if (c !== null && c !== undefined && c !== false) n.append(c instanceof Node ? c : String(c));
  return n;
}

function sv(tag, attrs, ...kids) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const c of kids.flat(Infinity)) if (c) n.append(c);
  return n;
}

// --- formatting ------------------------------------------------------------------------------------------------
const fmt = {
  pct: (x, d = 0) => (x === null || x === undefined ? "–" : `${(100 * x).toFixed(d)}%`),
  num: (x, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d)),
  p: (x) => (x === null || x === undefined ? "–" : x < 0.001 ? "< 0.001" : x.toFixed(3)),
  ci: (lo, hi) => (lo === null || lo === undefined ? "" : `${Math.round(100 * lo)}–${Math.round(100 * hi)}%`),
  time: (iso) => (iso ? iso.replace("T", " ").slice(0, 16) : "–"),
  sec: (x, d = 1) => (x === null || x === undefined ? "–" : `${Number(x).toFixed(d)} s`),
};

const OUTCOMES = {
  success: { label: "Success", icon: "✓", color: "var(--good)" },
  failed: { label: "Failed", icon: "✗", color: "var(--critical)" },
  time_limit: { label: "Time limit", icon: "⏱", color: "var(--warning)" },
  fault: { label: "Ended by the network fault", short: "Ended by fault", icon: "⚡", color: "var(--serious)" },
  not_counted: { label: "Not counted", icon: "–", color: "var(--ink-3)" },
};

function outcomeBadge(o) {
  const x = OUTCOMES[o] || OUTCOMES.failed;
  return el("span", { class: `badge ${o}` }, el("span", { "aria-hidden": "true", text: x.icon }), x.short || x.label);
}

const slotColor = (slot) => `var(--s${slot})`;

// --- tooltip ---------------------------------------------------------------------------------------------------
const tip = {
  node: null,
  show(evt, ...rows) {
    if (!this.node) { this.node = el("div", { class: "tip", role: "tooltip" }); document.body.append(this.node); }
    this.node.replaceChildren(...rows.flat().filter(Boolean));
    this.node.style.display = "block";
    this.move(evt);
  },
  move(evt) {
    if (!this.node || this.node.style.display === "none") return;
    const r = this.node.getBoundingClientRect();
    let x, y;
    if (evt && evt.clientX !== undefined && evt.type !== "focus") { x = evt.clientX + 14; y = evt.clientY + 14; }
    else { const b = evt.target.getBoundingClientRect(); x = b.right + 8; y = b.top; }
    if (x + r.width > innerWidth - 8) x = Math.max(8, x - r.width - 28);
    if (y + r.height > innerHeight - 8) y = Math.max(8, innerHeight - r.height - 8);
    this.node.style.left = `${x}px`; this.node.style.top = `${y}px`;
  },
  hide() { if (this.node) this.node.style.display = "none"; },
};
function tipRow(value, label, color) {
  return el("div", { class: "row" },
    color ? el("span", { class: "sw line", style: { background: color } }) : null,
    el("span", { class: "v", text: value }), label ? el("span", { class: "k", text: label }) : null);
}
function hoverable(node, rowsFn) {
  node.setAttribute("tabindex", "0");
  const show = (e) => tip.show(e, rowsFn());
  node.addEventListener("pointerenter", show);
  node.addEventListener("pointermove", (e) => tip.move(e));
  node.addEventListener("pointerleave", () => tip.hide());
  node.addEventListener("focus", show);
  node.addEventListener("blur", () => tip.hide());
  return node;
}

// --- theme -----------------------------------------------------------------------------------------------------
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private window */ } },
};
function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  else document.documentElement.removeAttribute("data-theme");
}
applyTheme(store.get("explorer.theme"));
function themeButton() {
  const names = { auto: "Theme: auto", light: "Theme: light", dark: "Theme: dark" };
  const b = el("button", { class: "iconbtn", type: "button" });
  const cur = () => store.get("explorer.theme") || "auto";
  b.textContent = names[cur()];
  b.addEventListener("click", () => {
    const next = { auto: "light", light: "dark", dark: "auto" }[cur()];
    store.set("explorer.theme", next); applyTheme(next); b.textContent = names[next];
    window.dispatchEvent(new Event("themechange"));
  });
  return b;
}

function topbar(page) {
  const nav = el("nav", { class: "nav", "aria-label": "Windows" },
    el("a", { href: "/", "aria-current": page === "results" ? "page" : null, text: "Results" }),
    el("a", { href: "/trials", "aria-current": page === "trials" ? "page" : null, text: "Trials & videos" }));
  return el("header", { class: "topbar" },
    el("div", { class: "brand" }, "DRTC ", el("span", { text: "results explorer" })), nav,
    el("div", { class: "spacer" }), themeButton());
}

// --- data and URL --------------------------------------------------------------------------------------------------
async function loadData(reload) {
  const r = await fetch(reload ? "/api/data?reload=1" : "/api/data");
  if (!r.ok) throw new Error(`data: HTTP ${r.status}`);
  const b = await r.json();
  for (const e of b.evals) {
    if (e.missing) continue;
    e.cond = Object.fromEntries(e.conditions.map((c) => [c.id, c]));
    e.model = Object.fromEntries(e.models.map((m) => [m.id, m]));
    e.group = (c, m) => e.groups.find((g) => g.condition === c && g.model === m);
  }
  b.metric = Object.fromEntries(b.metrics.map((m) => [m.col, m]));
  return b;
}

const hash = {
  get() { return Object.fromEntries(new URLSearchParams(location.hash.slice(1))); },
  set(obj, push) {
    const s = new URLSearchParams(Object.entries(obj).filter(([, v]) => v !== null && v !== undefined && v !== ""));
    const h = `#${s.toString()}`;
    if (push) history.pushState(null, "", h); else history.replaceState(null, "", h);
  },
};

function trialsLink(evalName, params) {
  const s = new URLSearchParams({ eval: evalName, ...params });
  return `/trials#${s.toString()}`;
}

// a "nice" axis maximum and ticks
function niceTicks(max, n = 4) {
  if (!(max > 0)) return { max: 1, ticks: [0, 1] };
  const raw = max / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => s >= raw);
  const top = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = 0; v <= top + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return { max: top, ticks };
}
