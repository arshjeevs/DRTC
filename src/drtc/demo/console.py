"""The live demo console: `lab demo`. The real client (DRTC and LeRobot's own scheduling) against a policy served on
this machine's GPU, the replay robot as the arm, faults from buttons, everything live in a browser page.

    lab demo                     # serve SmolVLA on this GPU, open http://127.0.0.1:8765
    lab demo --saved-only        # no GPU: replay saved runs only
    lab demo --check             # the self-check only

One run at a time (sequential): LeRobot as shipped, LeRobot made patient, or DRTC, 14 s each on a held-out sorting
episode, with the bench's 250 ms reply latency (lab/experiments/demo/). The page's buttons inject a latency spike,
1 s of dropped replies or a 1.5 s outage into the running client (drtc.control.faults). "Run all three" runs the three
clients back to back with the same fault script. Every finished run is saved (results/demo/) and can be played back
without a GPU. The run is the real rollout (drtc.control.run.rollout_once) in a thread of this process; a sampler
reads the engine 30 times a second and streams ticks, requests and faults to the page (server-sent events).
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = "smolvla_sort_one_v1_full"
CLIENTS = {"shipped": "demo/shipped", "patient": "demo/patient", "drtc": "demo/drtc"}
DURATION = 14.0
SCRIPT = [(3.0, "spike"), (6.0, "drops"), (9.0, "outage")]  # "Run all three": the same faults for every client
HERE = Path(__file__).resolve().parent


class Hub:
    """Fan-out of messages to the open pages; a page that connects late first gets the history."""

    def __init__(self):
        self.lock = threading.Lock()
        self.subs: list[queue.Queue] = []
        self.history: list[str] = []

    def emit(self, msg: dict) -> None:
        line = json.dumps(msg, separators=(",", ":"))
        with self.lock:
            if msg.get("type") != "frame":  # frames are not replayed to late pages
                self.history.append(line)
                del self.history[:-20000]
            for q in self.subs:
                q.put(line)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            for line in self.history:
                q.put(line)
            self.subs.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)


class Console:
    def __init__(self, args):
        self.args = args
        self.hub = Hub()
        self.busy = threading.Lock()
        self.engine = None
        self.shutdown = None
        self.client: str | None = None
        self.stop_all = threading.Event()
        self.server_proc: subprocess.Popen | None = None
        self.saved = Path(args.saved)
        self.saved.mkdir(parents=True, exist_ok=True)
        self.status = {"server": "off", "message": ""}

    # --- status -----------------------------------------------------------------------------------------------
    def say(self, server: str, message: str) -> None:
        self.status = {"server": server, "message": message}
        self.hub.emit({"type": "status", **self.status})
        print(f"[demo] {message}", flush=True)

    # --- the policy server --------------------------------------------------------------------------------------
    def start_server(self) -> bool:
        log = self.saved / "serve.log"
        self.say("starting", f"Starting the SmolVLA server on this GPU (port {self.args.port})...")
        with open(log, "w") as fh:
            self.server_proc = subprocess.Popen(
                [sys.executable, "-m", "drtc.cli", "serve", MODEL, "--listen", f"tcp/127.0.0.1:{self.args.port}"],
                stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 600:
            text = log.read_text(errors="replace")
            if "Policy server ready" in text:
                self.say("ready", f"SmolVLA is ready on this GPU ({time.monotonic() - t0:.0f} s). Choose a client to run.")
                return True
            if self.server_proc.poll() is not None:
                tail = " | ".join(text.strip().splitlines()[-3:])
                self.say("failed", f"The policy server stopped: {tail[-300:]}. Saved runs still play.")
                return False
            time.sleep(1)
        self.stop_server()
        self.say("failed", "The policy server was not ready after 10 minutes. Saved runs still play.")
        return False

    def stop_server(self) -> None:
        p = self.server_proc
        if p is not None and p.poll() is None:  # only the server this console started
            import os
            import signal

            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=20)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass

    # --- one run ------------------------------------------------------------------------------------------------
    def run(self, client: str, script: bool) -> None:
        from drtc.control.engine import ON_ENGINE
        from drtc.control.run import build, rollout_once
        from drtc.evals.run import held_out_episodes
        from drtc.registry import Registry

        reg = Registry()
        episode = held_out_episodes(reg, MODEL)[0]
        stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
        run_dir = self.saved / "runs" / f"{stamp}-{client}"
        cmd, _, _ = build(reg, CLIENTS[client], None, "replay", episode, f"tcp/127.0.0.1:{self.args.port}", DURATION,
                          run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        messages: list[dict] = []
        stop = threading.Event()
        threads: list[threading.Thread] = []

        def emit(msg):
            msg["client"] = client
            if msg["type"] != "frame":
                messages.append(msg)
            self.hub.emit(msg)

        def on_engine(engine):
            self.engine = engine
            latest = {}
            engine.observers.append(lambda obs: latest.__setitem__("obs", obs))
            for target, args in [(self.sample, (engine, emit, stop, latest))] + ([(self.scripted, (engine, stop))] if script else []):
                th = threading.Thread(target=target, args=args, daemon=True)
                th.start()
                threads.append(th)

        emit({"type": "start", "duration": DURATION, "episode": episode, "script": script})
        ON_ENGINE.append(on_engine)
        try:
            outcome, error, _ = rollout_once(cmd, run_dir, on_context=lambda ev: setattr(self, "shutdown", ev))
        finally:
            ON_ENGINE.remove(on_engine)
            stop.set()
            for th in threads:
                th.join(timeout=2)
            engine, self.engine, self.shutdown = self.engine, None, None
        rt = getattr(engine, "runtime", None)
        reason = getattr(rt, "failure", None) or error
        t_end = time.monotonic() - engine._t0 if engine is not None else 0.0
        if outcome == "completed":
            t_end = min(t_end, DURATION)
        emit({"type": "end", "t": round(t_end, 3), "outcome": outcome, "reason": reason})
        (self.saved / f"{stamp}-{client}.json").write_text(json.dumps(messages))

    @staticmethod
    def wait_started(engine, stop) -> bool:
        """The engine's clock and the fault clock start in engine.start(), after on_engine (at construction)."""
        while not stop.is_set():
            if engine.faults is not None and engine.faults.t0 is not None:
                return True
            time.sleep(0.005)
        return False

    def scripted(self, engine, stop) -> None:
        if not self.wait_started(engine, stop):
            return
        for at, kind in SCRIPT:
            while not stop.is_set() and time.monotonic() - engine._t0 < at:
                time.sleep(0.01)
            if stop.is_set():
                return
            self.fault(kind)

    def sample(self, engine, emit, stop, latest) -> None:
        """30 times a second: the queue, the trigger level, whether the arm moved, requests, faults, the command."""
        import cv2

        if not self.wait_started(engine, stop):
            return
        rt = engine.runtime
        n_ticks, n_faults = 0, 0
        pend_id, generation = None, rt.generation
        ending: list[tuple] = []  # requests that left `pending`: (id, t, generation then, sample number)
        cmd = None
        k, start = 0, time.monotonic()
        while not stop.is_set():
            t = time.monotonic() - engine._t0
            if t > DURATION + 0.05:  # the trial is over; teardown (the arm going home) is not part of it
                break
            rows = engine._ticks[n_ticks:]
            n_ticks += len(rows)
            moved = any(r[1] for r in rows)
            for r in rows:
                if r[9] is not None:
                    cmd = r[9]
            with rt.lock:
                pending, gen_now = rt.pending, rt.generation
            pid = getattr(pending, "request_id", None)
            for rid, t_end, gen_then, _ in [e for e in ending if e[3] < k]:  # resolved one sample later
                ending.remove(next(e for e in ending if e[0] == rid))
                chunks = {row[2]: row[3] for row in list(getattr(rt, "chunk_log", []))}
                kinds = {kind for _, kind, r in list(engine.recovery) if r == rid}
                if rid in chunks:
                    fate = "accepted" if chunks[rid] else "rejected"
                elif "cooldown_resend" in kinds:
                    fate = "replaced"
                elif "busy_backoff" in kinds:
                    fate = "busy"
                elif gen_now != gen_then:
                    fate = "cancelled"
                else:
                    fate = "ended"
                emit({"type": "req_end", "id": rid, "t": t_end, "fate": fate})
            if pid != pend_id:
                if pend_id is not None:
                    ending.append((pend_id, round(t, 3), generation, k))
                if pid is not None:
                    emit({"type": "req", "id": pid, "t": round(t, 3)})
                pend_id = pid
            generation = gen_now
            if engine.faults is not None:
                for ft, kind, _ in engine.faults.events[n_faults:]:
                    if kind in ("drop_action", "disconnect_reply", "loss", "disconnect_obs", "drop_obs"):
                        emit({"type": "lost", "t": ft, "id": pend_id, "what": kind})
                n_faults = len(engine.faults.events)
            try:
                with rt.lock:
                    thr = rt.effective_refill / rt.interval
            except Exception:  # noqa: BLE001
                thr = None
            emit({"type": "tick", "t": round(t, 3), "q": rt.queue.qsize(), "moved": moved,
                  "thr": None if thr is None else round(thr, 2), "pending": pid is not None,
                  "held": bool(getattr(rt, "held", False)), "cmd": cmd})
            if k % 5 == 0 and "obs" in latest:  # the camera input, 6 frames a second
                img = next((v for key, v in latest["obs"].items() if hasattr(v, "shape") and len(v.shape) == 3), None)
                if img is not None:
                    h, w = img.shape[:2]
                    small = cv2.resize(img, (320, int(320 * h / w)))
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        emit({"type": "frame", "img": base64.b64encode(jpg.tobytes()).decode()})
            k += 1
            time.sleep(max(0.0, start + k / 30.0 - time.monotonic()))

    # --- actions from the page ------------------------------------------------------------------------------------
    def fault(self, kind: str) -> dict:
        eng = self.engine
        if eng is None or eng.faults is None:
            return {"ok": False, "message": "No run in progress: start one first."}
        f = eng.faults
        if kind == "spike":
            t, dur = f.spike_now(1.0), 1.0
        elif kind == "drops":
            t, dur = f.drop_replies_now(1.0), 1.0
        elif kind == "outage":
            t, dur = f.outage_now(1.5), 1.5
        else:
            return {"ok": False, "message": f"unknown fault {kind}"}
        self.hub.emit({"type": "fault", "client": self.client, "kind": kind, "start": round(t, 3), "dur": dur})
        return {"ok": True}

    def start_runs(self, clients: list[str], script: bool) -> dict:
        if self.status["server"] != "ready":
            return {"ok": False, "message": "The policy server is not ready: " + self.status["message"]}
        if not self.busy.acquire(blocking=False):
            return {"ok": False, "message": "A run is in progress."}

        def go():
            self.stop_all.clear()
            try:
                for c in clients:
                    if self.stop_all.is_set():
                        break
                    self.client = c
                    try:
                        self.run(c, script)
                    except Exception as e:  # noqa: BLE001 - a failed run is shown, the console stays up
                        self.hub.emit({"type": "end", "client": c, "t": 0, "outcome": "error",
                                       "reason": f"{type(e).__name__}: {e}"})
                    time.sleep(1.5)  # the server closes the last session
            finally:
                self.client = None
                self.busy.release()

        threading.Thread(target=go, daemon=True).start()
        return {"ok": True}

    def stop_run(self) -> dict:
        self.stop_all.set()
        if self.shutdown is not None:
            self.shutdown.set()
            return {"ok": True}
        return {"ok": False, "message": "No run in progress."}

    def play_saved(self, name: str) -> dict:
        path = (self.saved / name).resolve()
        if path.parent != self.saved.resolve() or path.suffix != ".json" or not path.is_file():
            return {"ok": False, "message": "No such saved run."}
        try:
            msgs = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            return {"ok": False, "message": f"That saved run cannot be read: {e}"}
        if not self.busy.acquire(blocking=False):
            return {"ok": False, "message": "A run is in progress."}

        def go():
            try:
                t0 = time.monotonic()
                for m in msgs:
                    at = m.get("t", 0) if m["type"] not in ("start",) else 0
                    while time.monotonic() - t0 < (at or 0):
                        time.sleep(0.005)
                    self.hub.emit({**m, "saved": True})
            finally:
                self.busy.release()

        threading.Thread(target=go, daemon=True).start()
        return {"ok": True}

    def saved_runs(self) -> list[str]:
        return sorted(p.name for p in self.saved.glob("*.json"))


def make_handler(con: Console):

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                f = HERE / "console.html"
                page = f.read_bytes() if f.is_file() else b"<p>console.html is missing</p>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q = con.hub.subscribe()
                try:
                    self.wfile.write(f"data: {json.dumps({'type': 'status', **con.status})}\n\n".encode())
                    while True:
                        try:
                            line = q.get(timeout=10)
                            self.wfile.write(f"data: {line}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                except OSError:  # the page went away
                    pass
                finally:
                    con.hub.unsubscribe(q)
            elif self.path == "/saved":
                self._json(con.saved_runs())
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                body = {}
            if self.path == "/run":
                clients = [c for c in body.get("clients", []) if c in CLIENTS]
                self._json(con.start_runs(clients, bool(body.get("script"))) if clients else
                           {"ok": False, "message": "Choose a client."})
            elif self.path == "/fault":
                self._json(con.fault(str(body.get("kind"))))
            elif self.path == "/stop":
                self._json(con.stop_run())
            elif self.path == "/play":
                self._json(con.play_saved(str(body.get("name"))))
            else:
                self.send_error(404)

    return Handler


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def check(args) -> list[tuple[bool, str]]:
    """The self-check, in plain words."""
    out = []
    try:
        import torch

        gpu = torch.cuda.is_available()
        out.append((gpu, f"GPU: {torch.cuda.get_device_name(0)}" if gpu else
                    "GPU: none found. The policy cannot run here; use --saved-only to play saved runs."))
    except Exception as e:  # noqa: BLE001
        out.append((False, f"GPU: PyTorch failed to load ({e})."))
    out.append((port_free(args.http), f"Console port {args.http}: " + ("free." if port_free(args.http) else
                "in use. Pick another with --http.")))
    if not args.saved_only:
        out.append((port_free(args.port), f"Policy server port {args.port}: " + ("free." if port_free(args.port) else
                    "in use (is another lab serve running?). Pick another with --port.")))
    try:
        from drtc.registry import Registry

        hub_id = Registry().resolve(MODEL).hub
        from huggingface_hub.constants import HF_HUB_CACHE

        repo = Path(HF_HUB_CACHE) / ("models--" + hub_id.replace("/", "--"))
        if not any(repo.glob("snapshots/*/**/model.safetensors")):  # the served checkpoint's weights
            raise FileNotFoundError(repo)
        out.append((True, f"Model {MODEL}: downloaded."))
    except Exception:  # noqa: BLE001
        out.append((False, f"Model {MODEL}: not found on this machine. Download it once with internet "
                           f"(lab serve {MODEL} does), or use --saved-only."))
    saved = sorted(Path(args.saved).glob("*.json"))
    out.append((True, f"Saved runs: {len(saved)} in {args.saved}."))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab demo", description=__doc__.split("\n\n")[0])
    ap.add_argument("--http", type=int, default=8765, help="the console's port (never 8080)")
    ap.add_argument("--port", type=int, default=7955, help="the policy server's port")
    ap.add_argument("--saved", default="results/demo", help="where finished runs are saved and played from")
    ap.add_argument("--saved-only", action="store_true", help="no policy server: play saved runs only")
    ap.add_argument("--check", action="store_true", help="run the self-check and exit")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    results = check(args)
    for ok, msg in results:
        print(("  ok   " if ok else "  NO   ") + msg)
    if args.check:
        return 0 if all(ok for ok, _ in results) else 1
    if not port_free(args.http):
        print(f"lab demo: port {args.http} is in use; choose another with --http", file=sys.stderr)
        return 1
    con = Console(args)

    def on_term(*_):  # a stop from outside (kill, a closed terminal) cleans up like Ctrl-C
        raise KeyboardInterrupt

    import signal

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)
    ThreadingHTTPServer.daemon_threads = True  # an open page never holds the console open
    httpd = ThreadingHTTPServer(("127.0.0.1", args.http), make_handler(con))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.http}"
    print(f"[demo] console at {url} (Ctrl-C stops it and the policy server)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        if args.saved_only:
            con.say("saved", "Saved runs only (no policy server). Choose a saved run to play.")
        else:
            threading.Thread(target=con.start_server, daemon=True).start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[demo] stopping", flush=True)
    finally:
        con.stop_run()
        if con.busy.acquire(timeout=20):  # the run tears down before its server goes
            con.busy.release()
        con.stop_server()
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
