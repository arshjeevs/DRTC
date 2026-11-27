// Key findings: the Results window's first view. The report's main claims, each with its number, a chart from the
// raw trials (B.highlights, built in drtc/explorer/data.py) and a link to watch the trials behind it.
"use strict";

function keyFindings(B) {
  const H = B.highlights, E4 = B.evals.find((e) => e.name === H.eval);
  const [P, F, DR, ST] = H.conditions;
  const col = (c) => slotColor(E4.cond[c].slot);
  const short = (c) => E4.cond[c].short;
  const pct = (k, n) => (n ? Math.round((100 * k) / n) : 0);
  const shown = H.ladder.map((p) => Math.round(100 * p.rate + 1e-9));  // as fmt.pct shows them
  const watch = (text, params) => el("a", { class: "watch", href: trialsLink(H.eval, params) }, el("span", { "aria-hidden": "true", text: "▶" }), text);
  const gr = H.policies.find((p) => p.model.startsWith("grootn17_")) || H.policies[0];
  const mix = E4.cond[F].faults;

  // --- hero ---------------------------------------------------------------------------------------------------------
  const ladderBox = el("div", {});
  const hero = el("section", { class: "section" }, el("div", { class: "panel hero" },
    el("span", { class: "eyebrow", text: "The main result · SO-101 arm, Raspberry Pi 5 → GPU server over WiFi · 160 robot trials" }),
    el("h1", {}, "Network faults cut task success from ", el("em", { text: `${shown[0]}%` }), " to ", el("em", { text: `${shown[1]}%` }),
      ". DRTC brought it back to ", el("em", { text: `${shown[2]}%` }), ", and the smoothing stack kept it at ", el("em", { text: `${shown[3]}%` }),
      ` with ${fmt.num(H.jerk.ratio, 1)}× smoother motion.`),
    el("p", { class: "faultline", text: `Task: put the green cube in the box. The faults, in every trial: ${mix.map((f) => f.text.replace(/ at .*| from .*/, "")).join(", ")}, inside about six seconds. Four policies × ten trials in each condition; every trial recorded on video.` }),
    ladderBox,
    el("div", {}, el("h3", { style: { marginBottom: "8px" }, text: "What is new in this work" }), el("div", { class: "new" },
      el("div", {}, el("strong", { text: "DRTC inside LeRobot 0.7" }), "DRTC's scheduling ported into LeRobot's own remote-inference runtime, with four fixes found on slow links and short chunks."),
      el("div", {}, el("strong", { text: "A real low-cost arm, real faults" }), "An SO-101 driven by a Raspberry Pi 5, the policy on a remote GPU, network faults injected on purpose and the same in every condition."),
      el("div", {}, el("strong", { text: "Seven policies, their authors' recipes" }), "ACT, Diffusion Policy, SmolVLA, GR00T N1.5 and N1.7, π0.5 and MolmoAct 2, each trained as published."),
      el("div", {}, el("strong", { text: "A fair test, fixed in advance" }), "The baseline client waits as long as DRTC; plans, faults and the tests were frozen before the first trial.")))));
  requestAnimationFrame(() => ladderBox.append(ladder(ladderBox.clientWidth)));

  function ladder(width) {
    const W = Math.max(290, width), H_ = 300, L = 8, R = 8, T = 44, Bm = 58;
    const s = sv("svg", { class: "chart", viewBox: `0 0 ${W} ${H_}`, width: W, height: H_, role: "img", "aria-label": `Success: ${H.ladder.map((p, i) => `${short(H.conditions[i])} ${shown[i]}%`).join(", ")}` });
    const y = (v) => T + (1 - v) * (H_ - T - Bm);
    s.append(sv("line", { x1: L, x2: W - R, y1: y(0), y2: y(0), class: "axisline" }));
    const n = H.ladder.length, slot = (W - L - R) / n, bw = Math.min(110, slot * 0.46);
    const cx = (i) => L + slot * i + slot / 2;
    H.ladder.forEach((p, i) => {
      const c = H.conditions[i], x0 = cx(i) - bw / 2, top = y(p.rate), r = 5;
      const g = sv("g", { style: "cursor:pointer", onclick: () => { location.href = trialsLink(H.eval, { condition: c }); } });
      g.append(sv("path", { class: "mark", fill: col(c), d: `M${x0},${y(0)} V${top + r} Q${x0},${top} ${x0 + r},${top} H${x0 + bw - r} Q${x0 + bw},${top} ${x0 + bw},${top + r} V${y(0)} Z` }));
      g.append(sv("line", { x1: cx(i), x2: cx(i), y1: y(p.hi), y2: y(p.lo), stroke: "var(--ink)", "stroke-opacity": 0.4, "stroke-width": 1.5 }));
      g.append(sv("text", { x: cx(i), y: Math.min(top, y(p.hi)) - 10, "text-anchor": "middle", class: "value-label", style: "font:700 26px var(--display);fill:var(--ink)", text: `${shown[i]}%` }));
      g.append(sv("text", { x: cx(i), y: y(0) + 20, "text-anchor": "middle", class: "cat-label", text: short(c) }));
      g.append(sv("text", { x: cx(i), y: y(0) + 37, "text-anchor": "middle", class: "tick-label", text: `${p.k}/${p.n} trials` }));
      g.append(sv("rect", { class: "hit", x: cx(i) - slot / 2, y: 0, width: slot, height: H_ }));
      s.append(hoverable(g, () => [tipRow(`${p.k}/${p.n}`, short(c), col(c)), tipRow(fmt.pct(p.rate), `success, 95% CI ${fmt.ci(p.lo, p.hi)}`),
        el("div", { class: "k small", text: "Click to watch these trials" })]));
      if (i) {  // the step from the previous condition
        const d = shown[i] - shown[i - 1], xm = (cx(i - 1) + cx(i)) / 2;
        const lab = i === 1 ? "faults" : i === 2 ? "DRTC" : "smoothing";
        s.append(sv("text", { x: xm, y: 16, "text-anchor": "middle", class: "value-label", style: `font-size:15px;fill:${d < 0 ? "var(--critical)" : "var(--good-text)"}`, text: `${d > 0 ? "+" : "−"}${Math.abs(d)} pts` }));
        s.append(sv("text", { x: xm, y: 32, "text-anchor": "middle", class: "tick-label", text: i === 2 && H.pooled_p !== null ? `${lab} (p ${H.pooled_p < 0.001 ? "< 0.001" : "= " + H.pooled_p.toFixed(3)})` : lab }));
      }
    });
    return s;
  }

  // --- one finding: text on the left, its chart on the right ---------------------------------------------------------
  function finding(n, title, stat, statSmall, why, honest, chartFn, links) {
    const box = el("div", {});
    requestAnimationFrame(() => box.append(chartFn(box.clientWidth)));
    return el("section", { class: "section" }, el("div", { class: "panel finding" },
      el("div", {}, el("span", { class: "num", text: `Finding ${n}` }), el("h2", { text: title }),
        el("div", { class: "stat" }, stat, " ", el("small", { text: statSmall })),
        el("p", { class: "why", text: why }), honest ? el("p", { class: "honest", text: honest }) : null, el("div", {}, links)),
      box));
  }

  const ended = H.ended_by_client;
  const rec = (f) => H.recovery.find((r) => r.fault === f);
  const out = rec("outage");
  const jr = H.policies.map((p) => p.jerk[DR] / p.jerk[ST]).filter((x) => x > 0);
  const dev = H.policies.map((p) => p.deviation).filter((x) => x !== null);
  const kSum = (c) => H.policies.reduce((a, p) => a + p.k[c], 0), nSum = (c) => H.policies.reduce((a, p) => a + p.n[c], 0);
  const gr17 = H.chunk.find((c) => c.policy === "grootn17"), sm = H.policies.find((p) => p.model.startsWith("smolvla_"));
  const grootReplies = H.chunk.filter((c) => c.policy.startsWith("groot")).map((c) => c.reply_s);

  const f1 = finding(1, "Faults break remote inference, even for a client that waits",
    `${ended[F]} of ${nSum(F)}`, "runs were ended by the client itself",
    `LeRobot's patient client waits the whole trial for a reply, but it stops replacing lost replies (it re-sends at most once), so its queue never refills and the run ends. With DRTC, ${ended[DR]} of ${nSum(DR)} runs ended this way; with the smoothing stack, ${ended[ST]}.`,
    null,
    (w) => hbars(w, [F, DR, ST].map((c) => ({ label: short(c), vals: [{ v: ended[c] / nSum(c), color: col(c), text: `${ended[c]}/${nSum(c)}`, tip: `${ended[c]} of ${nSum(c)} runs ended by the client` }], link: trialsLink(H.eval, { condition: c, outcome: "fault" }) })), [], { max: 1, axis: (v) => `${Math.round(v * 100)}%` }),
    [watch(`${gr.name} under faults, waiting client`, { condition: F, model: gr.model }), watch("the same trials with DRTC", { condition: DR, model: gr.model })]);

  const f2 = finding(2, "DRTC asks again, so the arm starts moving again",
    `${out[DR].k} of ${out[DR].n}`, `runs moved again after the link went down, against ${out[F].k} of ${out[F].n} with the waiting client`,
    "When a reply is late or lost, DRTC's cooldown sends a fresh request instead of waiting for the lost one. For every fault of the mix the arm moved again in almost every run; the waiting client recovered from the latency spike but rarely from lost replies or the link going down.",
    "Recovered = the arm moved again after the fault ended (a run the client ended counts as not recovered).",
    (w) => hbars(w, ["spike", "drops", "outage"].map((f) => ({ label: B.faults[f], vals: [F, DR, ST].map((c) => ({ v: rec(f)[c].k / rec(f)[c].n, color: col(c), text: `${rec(f)[c].k}/${rec(f)[c].n}`, tip: `${short(c)}: ${rec(f)[c].k} of ${rec(f)[c].n} runs moved again after the ${B.faults[f].toLowerCase()}` })) })),
      [F, DR, ST].map((c) => ({ name: short(c), color: col(c) })), { max: 1, axis: (v) => `${Math.round(v * 100)}%` }),
    [watch("DRTC through the link down", { condition: DR, model: gr.model })]);

  const f3 = finding(3, "Every policy wins success back, the short-chunk ones the most",
    `${gr.k[F]} → ${gr.k[DR]}`, `of ${gr.n[DR]} for ${gr.name}: significant even after correcting for all 12 tests (Holm p = ${fmt.p(gr.p_holm)})`,
    "Successes per policy with the waiting client under faults, then with DRTC, next to the no-fault level. The policies whose chunks last under a second (GR00T N1.7, Diffusion Policy) lost almost everything and gained the most; π0.5, whose long chunk rides out short faults, lost and gained the least.",
    `Ten trials per cell, so only ${gr.name}'s gain is significant on its own. All four together: ${kSum(F)}/${nSum(F)} → ${kSum(DR)}/${nSum(DR)}${H.pooled_p !== null ? ` (p ${H.pooled_p < 0.001 ? "< 0.001" : "= " + H.pooled_p.toFixed(3)}, exploratory)` : ""}.`,
    (w) => hbars(w, H.policies.map((p) => ({ label: p.name, link: trialsLink(H.eval, { model: p.model }),
      vals: [P, F, DR].map((c) => ({ v: p.k[c] / p.n[c], color: col(c), text: `${p.k[c]}/${p.n[c]}`, tip: `${p.name}, ${short(c)}: ${p.k[c]} of ${p.n[c]} successes` })) })),
    [P, F, DR].map((c) => ({ name: short(c), color: col(c) })), { max: 1, axis: (v) => `${Math.round(v * 100)}%` }),
    [watch(`${gr.name} with DRTC`, { condition: DR, model: gr.model })]);

  const f4 = finding(4, `${fmt.num(H.jerk.ratio, 1)}× smoother motion, with no loss in success`,
    `${Math.round(H.jerk.drtc)} → ${Math.round(H.jerk.stack)}`, "jerk (10³ °/s³), median of all runs: DRTC alone → DRTC + smoothing stack",
    `The stack is adaptive playback, temporal ensembling, a Butterworth filter and a resume ramp, chosen on the bench by a rule fixed before it ran. Jerk fell ${fmt.num(Math.min(...jr), 1)} to ${fmt.num(Math.max(...jr), 1)}× for every policy, success went from ${kSum(DR)}/${nSum(DR)} to ${kSum(ST)}/${nSum(ST)}, and the arm stayed within ${fmt.num(Math.min(...dev), 1)}–${fmt.num(Math.max(...dev), 1)}° of the policy's own plan.`,
    null,
    (w) => hbars(w, H.policies.map((p) => ({ label: p.name, link: trialsLink(H.eval, { condition: ST, model: p.model }),
      vals: [DR, ST].map((c, i) => ({ v: p.jerk[c], color: col(c), text: i ? `${fmt.num(p.jerk[c], 0)}  (${fmt.num(p.jerk[DR] / p.jerk[ST], 1)}× lower)` : fmt.num(p.jerk[c], 0), tip: `${p.name}, ${short(c)}: median jerk ${fmt.num(p.jerk[c], 1)} ×10³ °/s³` })) })),
    [DR, ST].map((c) => ({ name: short(c), color: col(c) })), { axis: (v) => `${v}` , unit: "jerk, 10³ °/s³" }),
    [watch("the smoothing stack", { condition: ST, model: gr.model }), watch("DRTC alone", { condition: DR, model: gr.model })]);

  const f5 = finding(5, "The limit: a chunk shorter than the round trip stalls anyway",
    `${Math.round(gr.stall_nofault)}%`, `of the time ${gr.name}'s arm stood still with no faults at all (${sm ? `${sm.name}: ${Math.round(sm.stall_nofault)}%` : ""})`,
    `The arm keeps moving only if the next chunk arrives before the current one runs out. ${gr17 ? `GR00T's 16-action chunk lasts ${fmt.num(gr17.chunk_s, 2)} s, but a reply took ${grootReplies.map((r) => fmt.num(r, 2)).join("–")} s` : ""}, so its queue empties before every reply whatever the scheduler does. The remedy is the playback and the link (adaptive playback, the cable), not DRTC.`,
    "Reply time: the median turnaround of the robot baselines (Pi over WiFi).",
    (w) => hbars(w, H.chunk.map((c) => ({ label: c.name, vals: [
      { v: c.chunk_s, color: "var(--s1)", text: `${fmt.num(c.chunk_s, 2)} s`, tip: `${c.name}: one chunk lasts ${fmt.num(c.chunk_s, 2)} s` },
      { v: c.reply_s, color: "var(--s2)", text: `${fmt.num(c.reply_s, 2)} s  ${c.chunk_s - c.reply_s > 0.05 ? `(${fmt.num(c.chunk_s - c.reply_s, 2)} s spare)` : c.chunk_s - c.reply_s < -0.05 ? `(${fmt.num(c.reply_s - c.chunk_s, 2)} s short)` : "(about even)"}`, tip: `${c.name}: one reply takes ${fmt.num(c.reply_s, 2)} s` }] })),
    [{ name: "one chunk lasts", color: "var(--s1)" }, { name: "one reply takes", color: "var(--s2)" }], { axis: (v) => `${v} s` }),
    [watch(`${gr.name}, no faults`, { condition: P, model: gr.model })]);

  const base = el("section", { class: "section" }, el("div", { class: "panel", style: { display: "grid", gap: "8px" } },
    el("h2", { text: "Before any of this: the baselines" }),
    el("p", { class: "lede", text: `With the waiting client on a normal network, the seven policies reached ${H.baselines.map((b) => `${fmt.pct(b.range[0])}–${fmt.pct(b.range[1])} on ${b.title.replace("Baseline: ", "")}`).join(" and ")}. The DRTC experiment uses the four strongest on cube in box.` }),
    el("div", {}, H.baselines.map((b) => el("button", { class: "watch", type: "button", onclick: () => select(b.name), text: `${b.title} →` })))));

  return [hero, f1, f2, f3, f4, f5, base];
}

// horizontal bars, grouped: groups [{label, link?, vals: [{v, color, text, tip}]}], series [{name, color}] for the legend.
// Narrow (a phone): each group's label sits above its bars, the legend wraps, fewer ticks.
function hbars(width, groups, series, opt) {
  const W = Math.max(290, width), narrow = W < 480, R = 8, bh = 15, gap = 4, ggap = narrow ? 10 : 16, lab = narrow ? 18 : 0;
  const L = narrow ? 8 : 136;
  // legend rows
  const leg = [];
  let lx = L, ly = 4;
  for (const se of series) {
    const w = 16 + se.name.length * 6.6 + 16;
    if (lx + w > W - R && lx > L) { lx = L; ly += 18; }
    leg.push({ ...se, x: lx, y: ly });
    lx += w;
  }
  const T = series.length ? ly + 26 : 6;
  const per = Math.max(...groups.map((g) => g.vals.length));
  const gh = lab + per * bh + (per - 1) * gap;
  const H_ = T + groups.length * gh + (groups.length - 1) * ggap + 24;
  const vmax = opt.max || Math.max(...groups.flatMap((g) => g.vals.map((v) => v.v || 0)));
  const nt = narrow ? 2 : 4;
  const { max, ticks } = opt.max ? { max: opt.max, ticks: narrow ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1] } : niceTicks(vmax * 1.02, nt);
  const textRoom = Math.min(narrow ? 120 : 150, W * 0.36);
  const x = (v) => L + (v / max) * (W - L - R - textRoom);
  const s = sv("svg", { class: "chart", viewBox: `0 0 ${W} ${H_}`, width: W, height: H_, role: "img" });
  for (const se of leg) {
    s.append(sv("rect", { x: se.x, y: se.y, width: 11, height: 11, rx: 3, fill: se.color }));
    s.append(sv("text", { x: se.x + 16, y: se.y + 10, class: "tick-label", text: se.name }));
  }
  const yb = H_ - 22;
  for (const v of ticks) {
    s.append(sv("line", { x1: x(v), x2: x(v), y1: T - 4, y2: yb, class: v ? "gridline" : "axisline" }));
    s.append(sv("text", { x: x(v), y: yb + 16, "text-anchor": "middle", class: "tick-label", text: opt.axis ? opt.axis(v) : v }));
  }
  if (opt.unit && !narrow) s.append(sv("text", { x: W - R, y: yb + 16, "text-anchor": "end", class: "tick-label", text: opt.unit }));
  groups.forEach((g, gi) => {
    const y0 = T + gi * (gh + ggap);
    const grp = sv("g", { style: g.link ? "cursor:pointer" : null, onclick: g.link ? () => { location.href = g.link; } : null });
    grp.append(narrow
      ? sv("text", { x: L, y: y0 + 12, class: "cat-label", text: g.label })
      : sv("text", { x: L - 10, y: y0 + gh / 2 + 4, "text-anchor": "end", class: "cat-label", text: g.label }));
    g.vals.forEach((v, i) => {
      const yy = y0 + lab + i * (bh + gap), w = Math.max(0, x(v.v || 0) - x(0)), r = Math.min(4, w / 2);
      const bar = sv("g", {});
      bar.append(sv("path", { class: "mark", fill: v.color, d: w > 0
        ? `M${x(0)},${yy} H${x(0) + w - r} Q${x(0) + w},${yy} ${x(0) + w},${yy + r} V${yy + bh - r} Q${x(0) + w},${yy + bh} ${x(0) + w - r},${yy + bh} H${x(0)} Z`
        : `M${x(0)},${yy} h2 v${bh} h-2 Z` }));
      bar.append(sv("text", { x: x(0) + w + 6, y: yy + bh - 3, class: "value-label", text: narrow ? v.text.split("  ")[0] : v.text }));
      bar.append(sv("rect", { class: "hit", x: 0, y: yy - gap / 2, width: W, height: bh + gap }));
      grp.append(hoverable(bar, () => [tipRow(v.text.split("  ")[0], v.tip, v.color)]));
    });
    s.append(grp);
  });
  return s;
}
