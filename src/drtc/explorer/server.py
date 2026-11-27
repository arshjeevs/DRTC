"""`lab explore`: a local web app over the downloaded results (lab/docs/EXPLORER.md).

    lab explore                  # http://127.0.0.1:8770 : Results (the whole picture) and Trials (videos + timeline)
    lab explore --port 8771 --no-open
    lab explore --host 0.0.0.0   # reachable from another device on the same network (a projector laptop, a tablet)
    lab explore --present        # for the panel: the chosen videos only, no "Choose videos" button

Read only: it reads results/evals/<eval>/ (pull first with `lab eval pull <eval>`) and never writes there. Runs from
the repository root; no GPU, no network. Videos are served with byte ranges, so scrubbing works in every browser.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from drtc.explorer import data as D

WEB = Path(__file__).resolve().parent / "web"
REPO = Path(__file__).resolve().parents[3]
FONTS = REPO / "presentation" / "fonts"
CHUNK = 256 * 1024


class Store:
    """The bundle (built once, rebuilt on ?reload=1) and the trial lookup id -> folder."""

    def __init__(self, reg, names, editable: bool = True, videos: Path = D.SELECTION):
        self.reg, self.names, self.editable, self.videos = reg, names, editable, videos
        self.lock = threading.Lock()
        self.bundle: dict = {}
        self.paths: dict[tuple[str, str], Path] = {}
        self.reload()

    def reload(self) -> None:
        with self.lock:
            from drtc.evals import spec as S
            from drtc.evals import store

            self.bundle = D.build(self.reg, self.names)
            paths = {}
            for e in self.bundle["evals"]:
                if e.get("missing"):
                    continue
                root = S.results_root(S.load(self.reg, e["name"]))
                for t in store.trials(e["name"], root):
                    paths[(e["name"], f"{t.session}/{t.condition}/{t.model}/t{t.number:02d}")] = t.path.resolve()
            self.paths = paths

    def body(self) -> bytes:
        """The bundle with the saved video choice (read fresh: it changes without a rebuild)."""
        return json.dumps({**self.bundle, "selection": D.load_selection(self.videos), "editable": self.editable},
                          allow_nan=False, default=str).encode()

    def save_selection(self, sel: dict) -> dict:
        with self.lock:
            return D.save_selection(sel, set(self.paths), self.videos)

    def trial_path(self, name: str, tid: str) -> Path | None:
        return self.paths.get((name, tid))


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """`bytes=a-b`, `bytes=a-` or `bytes=-n` -> (first, last) inclusive; None when absent or not satisfiable."""
    if not header:
        return None
    m = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*", header)
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return None
    a, b = m.group(1), m.group(2)
    if a == "":
        n = int(b)
        first, last = max(0, size - n), size - 1
    else:
        first = int(a)
        last = min(int(b), size - 1) if b else size - 1
    if first > last or first >= size:
        return None
    return first, last


def make_handler(st: Store):
    class H(BaseHTTPRequestHandler):
        server_version = "drtc-explorer"

        def log_message(self, fmt, *args):  # quiet: only errors
            if args and str(args[1]).startswith(("4", "5")):
                sys.stderr.write("explore: " + fmt % args + "\n")

        def send_bytes(self, body: bytes, ctype: str, code: int = 200, cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def send_json(self, obj, code: int = 200):
            self.send_bytes(json.dumps(obj, allow_nan=False, default=str).encode(), "application/json", code)

        def send_file(self, p: Path, cache: bool = False):
            if not p.is_file():
                return self.send_bytes(b"not found", "text/plain", 404)
            size = p.stat().st_size
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            rng = parse_range(self.headers.get("Range"), size)
            if self.headers.get("Range") and rng is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            first, last = rng or (0, size - 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(last - first + 1))
            if rng:
                self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-cache")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with open(p, "rb") as f:
                    f.seek(first)
                    left = last - first + 1
                    while left > 0:
                        buf = f.read(min(CHUNK, left))
                        if not buf:
                            break
                        self.wfile.write(buf)
                        left -= len(buf)
            except (BrokenPipeError, ConnectionResetError):  # the browser seeks and drops the old request
                pass

        def do_POST(self):
            if urlparse(self.path).path != "/api/selection":
                return self.send_bytes(b"not found", "text/plain", 404)
            if not st.editable:
                return self.send_json({"error": "started with --present: the choice cannot be changed"}, 403)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                sel = json.loads(self.rfile.read(min(n, 2_000_000)) or b"{}")
                if not isinstance(sel, dict) or not all(isinstance(v, list) for v in sel.values()):
                    raise ValueError("expected {eval: [trial ids]}")
            except (ValueError, json.JSONDecodeError) as e:
                return self.send_json({"error": str(e)}, 400)
            saved = st.save_selection(sel)
            print(f"lab explore: saved {sum(len(v) for v in saved.values())} chosen videos to {st.videos}", flush=True)
            return self.send_json({"selection": saved})

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            u = urlparse(self.path)
            path, q = unquote(u.path), parse_qs(u.query)
            if path in ("/", "/results", "/index.html"):
                return self.send_file(WEB / "index.html")
            if path in ("/trials", "/trials.html"):
                return self.send_file(WEB / "trials.html")
            if path in ("/videos", "/videos.html"):
                return self.send_file(WEB / "videos.html")
            if path.startswith("/static/"):
                return self.send_file(_inside(WEB, path[len("/static/"):]))
            if path.startswith("/fonts/"):
                return self.send_file(_inside(FONTS, path[len("/fonts/"):]), cache=True)
            if path == "/api/data":
                if q.get("reload"):
                    st.reload()
                return self.send_bytes(st.body(), "application/json")
            if path == "/api/trial":
                p = st.trial_path(q.get("eval", [""])[0], q.get("id", [""])[0])
                if p is None:
                    return self.send_json({"error": "unknown trial"}, 404)
                return self.send_json(D.trial_detail(p))
            m = re.fullmatch(r"/video/([^/]+)/(.+)/([a-z0-9_]+)\.mp4", path)
            if m:  # only a camera file of a known trial, never an arbitrary path
                p = st.trial_path(m.group(1), m.group(2))
                if p is None:
                    return self.send_bytes(b"not found", "text/plain", 404)
                return self.send_file(p / "video" / f"{m.group(3)}.mp4", cache=True)
            return self.send_bytes(b"not found", "text/plain", 404)

    return H


def _inside(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    return p if p.is_relative_to(base.resolve()) else base / "__missing__"


def port_free(host: str, port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab explore", description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8770, help="the explorer's port (default 8770; never 8080)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open it from another device on this network")
    ap.add_argument("--evals", nargs="+", default=list(D.EVALS), help="the evals to show (default: E4 + baselines)")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    ap.add_argument("--present", action="store_true", help="for the panel: no 'Choose videos' button, the saved choice only")
    ap.add_argument("--videos", type=Path, default=D.SELECTION, help=f"the file of chosen videos (default {D.SELECTION})")
    a = ap.parse_args(argv)
    if a.port == 8080:
        print("lab explore: 8080 is the robot session's port; pick another with --port", file=sys.stderr)
        return 2
    cwd = os.getcwd()
    os.chdir(REPO)
    if not port_free(a.host, a.port):
        print(f"lab explore: port {a.port} is in use; choose another with --port", file=sys.stderr)
        return 2
    from drtc.registry import Registry

    print("lab explore: reading the results ...", flush=True)
    videos = REPO / D.SELECTION if a.videos == D.SELECTION else Path(os.path.abspath(Path(cwd) / a.videos))
    st = Store(Registry(), a.evals, editable=not a.present, videos=videos)
    sel = D.load_selection(videos)
    print(f"  chosen videos ({a.videos}): " + ("none saved yet, every trial is shown" if sel is None else
                                                  ", ".join(f"{k} {len(v)}" for k, v in sel.items()) or "none"))
    for e in st.bundle["evals"]:
        print(f"  {e['name']}: " + (e["missing"] if e.get("missing") else
                                     f"{len(e['trials'])} trials, {sum(1 for t in e['trials'] if t['cameras'])} with video"))
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(st))
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}"
    print(f"lab explore: {url}  (Results), {url}/trials  (Trials) and {url}/videos  (the chosen videos, one after "
          "another). Ctrl-C to stop.", flush=True)
    if not a.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
