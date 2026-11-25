// Record a clip of the DRTC Fault Simulator for the slides (headless Chrome over the DevTools protocol, then ffmpeg).
//   node presentation/simulator/record.js [policy] [faults] [out.mp4]
// Defaults: SmolVLA, spike + drops + outage, presentation/out/drtc_simulator_clip.mp4. Needs google-chrome, ffmpeg,
// and a built page (python presentation/simulator/build.py).
const { spawn, execFileSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const policy = process.argv[2] || "smolvla";
const faults = process.argv[3] || "mix";
const ROOT = path.resolve(__dirname, "..");
const out = path.resolve(process.argv[4] || path.join(ROOT, "out", "drtc_simulator_clip.mp4"));
const page = path.join(ROOT, "out", "drtc_simulator.html");
const FPS = 25, SPEED = 0.5, DUR = 14, W = 1280, H = 1180;   // half speed: 28 s of video for the 14 s trial
const PORT = 9300 + Math.floor(Math.random() * 400);

async function main() {
  const frames = fs.mkdtempSync(path.join(os.tmpdir(), "drtc-sim-frames-"));
  const prof = fs.mkdtempSync(path.join(os.tmpdir(), "drtc-sim-chrome-"));
  const chrome = spawn("google-chrome", ["--headless=new", `--remote-debugging-port=${PORT}`, "--disable-gpu",
    "--hide-scrollbars", `--user-data-dir=${prof}`, `--window-size=${W},${H}`, "about:blank"], { stdio: "ignore" });
  try {
    let target;
    for (let i = 0; i < 50 && !target; i++) {
      await new Promise(r => setTimeout(r, 200));
      try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find(t => t.type === "page"); } catch (e) { /* starting */ }
    }
    if (!target) throw new Error("Chrome did not start");
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
    let id = 0; const waiting = new Map();
    ws.onmessage = m => { const d = JSON.parse(m.data); if (waiting.has(d.id)) { waiting.get(d.id)(d); waiting.delete(d.id); } };
    const send = (method, params = {}) => new Promise(res => { const i = ++id; waiting.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
    await send("Page.enable");
    await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
    await send("Page.navigate", { url: `file://${page}?policy=${policy}&faults=${faults}&t=0` });
    await new Promise(r => setTimeout(r, 2500));
    const n = Math.round(DUR / SPEED * FPS);
    for (let k = 0; k <= n; k++) {
      const t = Math.min(DUR, k / FPS * SPEED);
      await send("Runtime.evaluate", { expression: `window.DRTCSimPage.setTime(${t})` });
      const shot = await send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: W, height: H, scale: 1 } });
      fs.writeFileSync(path.join(frames, `f${String(k).padStart(5, "0")}.png`), Buffer.from(shot.result.data, "base64"));
    }
    // hold the last frame for 2 s
    const last = path.join(frames, `f${String(n).padStart(5, "0")}.png`);
    for (let k = 1; k <= 2 * FPS; k++) fs.copyFileSync(last, path.join(frames, `f${String(n + k).padStart(5, "0")}.png`));
    ws.close();
    execFileSync("ffmpeg", ["-loglevel", "error", "-y", "-framerate", String(FPS), "-i", path.join(frames, "f%05d.png"),
      "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out]);
    console.log(`wrote ${out} (${n + 1 + 2 * FPS} frames)`);
    console.log(`frames kept in ${frames} (temporary folder)`);
  } finally {
    chrome.kill();
  }
}
main().catch(e => { console.error(e); process.exit(1); });
