// Compare the simulator's model with the bench: median stall share per policy and client under the fault mix.
//   node presentation/simulator/check.js [bench metrics.csv ...]
// 8 trials per cell with per-trial fault shifts in [0, 4] s, as on the bench.
const fs = require("fs");
const S = require("./model.js");

function median(a) { const b = [...a].sort((x, y) => x - y); const n = b.length; return n ? (n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2) : NaN; }

const bench = {};   // `${condition}|${policy}` -> median M2
for (const f of process.argv.slice(2)) {
  const [head, ...rows] = fs.readFileSync(f, "utf8").trim().split("\n");
  const cols = head.split(",");
  const ci = cols.indexOf("condition"), mi = cols.indexOf("model"), si = cols.indexOf("M2_stall_pct");
  const acc = {};
  for (const r of rows) {
    const v = r.split(","); const pol = v[mi].split("_")[0];
    (acc[`${v[ci]}|${pol}`] = acc[`${v[ci]}|${pol}`] || []).push(parseFloat(v[si]));
  }
  for (const k in acc) bench[k] = median(acc[k].filter(x => !isNaN(x)));
}

const clients = { drtc: "drtc__mix", patient: "upstream_patient__mix", shipped: "upstream_shipped__mix" };
console.log("policy      client    sim stall%  bench stall%  sim runs ended");
for (const pol of Object.keys(S.POLICIES)) {
  for (const [cl, cond] of Object.entries(clients)) {
    const stalls = [], ended = [];
    for (let k = 1; k <= 8; k++) {
      const shift = 4 * ((k * 0.6180339887) % 1);
      const lane = S.runTrial({ policy: pol, client: cl, faults: S.FAULTS.mix, shift, seed: k });
      const st = lane.stats(); stalls.push(st.stallPct); if (st.ended) ended.push(st.ended.reason);
    }
    const b = bench[`${cond}|${pol}`];
    console.log(`${pol.padEnd(11)} ${cl.padEnd(9)} ${median(stalls).toFixed(1).padStart(10)}  ${(b === undefined ? "-" : b.toFixed(1)).padStart(12)}  ${ended.length}/8 ${ended[0] || ""}`);
  }
}
