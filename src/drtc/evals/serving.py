"""Serving an eval's models, and reaching them from the robot computer (lab/EVALUATION.md, "Roles").

Model server side, `lab eval serve <eval>` (laptop GPU, A100, or the H200 container):
- every model of the frozen eval (or `--models` a subset) in its own policy server, started one after the other;
- each serves exactly its frozen Hub commit (only the served checkpoint is downloaded), under a deployment name that
  contains the commit (drtc.control.serve.deployment_name), so a client can only reach the frozen version;
- ports: each eval has its own block of 10 (from its name, so two evals can be served on one machine at once), and
  each model the port at its place in the eval's list; listening on 127.0.0.1, or on all interfaces inside the H200's
  container (only reachable from the H200 host, which the tunnel goes through);
- logs in logs/serve/<eval>/<model>.log (debugging only: nothing for analysis is stored on a model server).

Robot computer side:
- laptop: the servers are local (tcp/127.0.0.1:<port>);
- A100: one SSH connection (key login) forwarding every model's port;
- H200: one SSH connection (the password is typed once) that finds the container's address and forwards every port
  into it. Local tunnel ports are the server ports + 10000, so a tunnel never clashes with servers on the laptop.
- probe(): asks a deployment to describe itself (no session is opened): ready, checkpoint identity, software.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from drtc.evals import spec as S
from drtc.registry import Registry, RegistryError

PORT_BLOCKS_START = 7500  # eval blocks: 7500-9999, 10 ports each
TUNNEL_OFFSET = 10000
MAX_MODELS = 10
SERVERS = ("laptop", "a100", "h200")


def deployment(spec: S.EvalSpec, model: str) -> str:
    from drtc.control.serve import deployment_name

    return deployment_name(model, spec.frozen[model])


def base_port(spec: S.EvalSpec) -> int:
    """The first port of the eval's block: the same on every machine (computed from the eval's description and models,
    which identify it), different for different evals."""
    import zlib

    key = (spec.description + "|" + ",".join(spec.models)).encode()
    return PORT_BLOCKS_START + (zlib.crc32(key) % 250) * 10


def port(spec: S.EvalSpec, model: str) -> int:
    if len(spec.models) > MAX_MODELS:
        raise RegistryError(f"an eval serves at most {MAX_MODELS} models")
    return base_port(spec) + spec.models.index(model)


def port_free(p: int, host: str = "127.0.0.1") -> bool:
    import socket

    with socket.socket() as sk:
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sk.bind((host, p))
            return True
        except OSError:
            return False


def in_container() -> bool:
    return Path("/.dockerenv").exists() or bool(os.environ.get("LAB_IN_CONTAINER"))


# --- model server side --------------------------------------------------------------------------------------------


def serve(reg: Registry, name: str, models: list[str] | None, device: str, ready_timeout_s: float = 900) -> int:
    sp = S.load(reg, name)
    if not S.is_frozen(sp):
        raise RegistryError(f"{name} is not frozen: run `lab eval freeze {name}` on the laptop, commit, push, pull here")
    chosen = models or sp.models
    unknown = [m for m in chosen if m not in sp.models]
    if unknown:
        raise RegistryError(f"not in {name}: {', '.join(unknown)} (its models: {', '.join(sp.models)})")
    listen_host = "0.0.0.0" if in_container() else "127.0.0.1"
    busy = [f"{m} (port {port(sp, m)})" for m in chosen if not port_free(port(sp, m), listen_host)]
    if busy:
        raise RegistryError(f"ports in use on this machine: {', '.join(busy)}. Is `lab eval serve {name}` already "
                            f"running (tmux ls)? Stop it there first (Ctrl-C in its window)")
    logs = Path("logs") / "serve" / name
    logs.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[str, subprocess.Popen, Path]] = []
    print(f"serving {name} ({len(chosen)} models) on {listen_host}; logs in {logs}/", flush=True)
    try:
        for m in chosen:
            log = logs / f"{m}.log"
            cmd = [sys.executable, "-m", "drtc.cli", "serve", m, "--revision", sp.frozen[m], "--deployment",
                   deployment(sp, m), "--listen", f"tcp/{listen_host}:{port(sp, m)}", "--device", device]
            fh = open(log, "w")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            procs.append((m, proc, log))
            print(f"  {m:32s} port {port(sp, m)}  starting (loading the model) ...", end="", flush=True)
            t0 = time.monotonic()
            while True:
                text = log.read_text(errors="replace")
                if "Policy server ready" in text:
                    print(f" ready in {time.monotonic() - t0:.0f} s", flush=True)
                    break
                if proc.poll() is not None:
                    print(" FAILED", flush=True)
                    tail = "\n".join(text.strip().splitlines()[-8:])
                    raise RegistryError(f"{m}: the server exited ({proc.returncode}). Last lines of {log}:\n{tail}")
                if time.monotonic() - t0 > ready_timeout_s:
                    print(" TIMEOUT", flush=True)
                    raise RegistryError(f"{m}: not ready after {ready_timeout_s:.0f} s; see {log}")
                time.sleep(1)
        print(f"all {len(procs)} ready. Leave this running during the session; Ctrl-C stops every server.",
              flush=True)
        while True:
            for m, proc, log in procs:
                if proc.poll() is not None:
                    raise RegistryError(f"{m}: the server stopped ({proc.returncode}); see {log}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nstopping the servers", flush=True)
        return 0
    finally:
        for _, proc, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, _ in procs:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


# --- robot computer side ------------------------------------------------------------------------------------------


@dataclass
class Connection:
    """How this robot computer reaches one model server for an eval."""

    server: str
    endpoints: dict[str, str]  # model -> zenoh endpoint on this machine
    socket: Path | None = None  # the SSH control socket (a100, h200)
    ssh: str | None = None

    def close(self) -> None:
        if self.socket and self.socket.exists():
            subprocess.run(["ssh", "-S", str(self.socket), "-O", "exit", self.ssh], capture_output=True)


def _ssh_target(reg: Registry, server: str) -> tuple[str, str | None]:
    host = reg.hosts.get(server)
    if host is None or not getattr(host, "ssh", None):
        raise RegistryError(f"lab/hosts.yaml has no ssh address for {server!r}")
    return host.ssh, getattr(host, "container", None)


def connect(reg: Registry, spec: S.EvalSpec, server: str, models: list[str] | None = None) -> Connection:
    """Open the way to the eval's servers on `server` (laptop, a100, h200); nothing to open for the laptop."""
    if server not in SERVERS:
        raise RegistryError(f"server must be one of {', '.join(SERVERS)}")
    chosen = models or spec.models
    if server == "laptop":
        return Connection(server, {m: f"tcp/127.0.0.1:{port(spec, m)}" for m in chosen})
    ssh, container = _ssh_target(reg, server)
    sock = Path.home() / ".ssh" / f"lab-{server}.sock"
    if not sock.exists():
        print(f"connecting to {server} ({ssh})" + (": type the server password if asked" if server == "h200" else ""),
              flush=True)
        opts = ["-o", "ServerAliveInterval=15", "-o", "ExitOnForwardFailure=yes"]
        if server != "h200":
            opts += ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        r = subprocess.run(["ssh", "-M", "-S", str(sock), "-o", "ControlPersist=yes", *opts, "-fN", ssh])
        if r.returncode != 0:
            raise RegistryError(f"cannot connect to {server} ({ssh}): is it reachable from here?")
    target = "127.0.0.1"
    if server == "h200" and container:
        fmt = "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"
        r = subprocess.run(["ssh", "-S", str(sock), ssh, f"docker inspect -f {shlex.quote(fmt)} {container}"],
                           capture_output=True, text=True)
        target = r.stdout.strip()
        if r.returncode != 0 or not target:
            raise RegistryError(f"the {container} container is not running on {server}: start it (lab/H200.md)")
    endpoints = {}
    for m in chosen:
        local = port(spec, m) + TUNNEL_OFFSET
        r = subprocess.run(["ssh", "-S", str(sock), "-O", "forward", "-L", f"{local}:{target}:{port(spec, m)}", ssh],
                           capture_output=True, text=True)
        if r.returncode != 0 and "already" not in r.stderr.lower():
            raise RegistryError(f"cannot forward port {local} to {server}: {r.stderr.strip()}")
        endpoints[m] = f"tcp/127.0.0.1:{local}"
    return Connection(server, endpoints, sock, ssh)


def probe(endpoint: str, deployment_name: str, timeout_s: float = 5.0) -> dict:
    """Ask one deployment to describe itself (no session is opened). Returns its descriptor, or {'error': ...}."""
    from uuid import uuid4

    from lerobot.remote_inference.codec import decode_message, encode_message
    from lerobot.remote_inference.protocol import Envelope, MessageType, deployment_prefix
    from lerobot.transport import ZenohConfig, ZenohTransport

    transport = ZenohTransport(ZenohConfig(mode="peer", connect_endpoints=[endpoint], open_timeout_s=timeout_s))
    try:
        transport.open()
    except Exception as e:  # noqa: BLE001
        return {"error": f"no server at {endpoint} ({type(e).__name__})"}
    try:
        request = Envelope(MessageType.DESCRIBE, request_id=uuid4().hex)
        try:
            replies = transport.query(deployment_prefix(deployment_name) + "/describe", encode_message(request),
                                      timeout_s)
        except TimeoutError:
            return {"error": f"{deployment_name} did not answer at {endpoint} (another model, or another version?)"}
        for payload in replies:
            body = decode_message(payload).body
            if body.get("deployment") == deployment_name:
                return body
        return {"error": f"{deployment_name} did not answer at {endpoint}"}
    finally:
        transport.close()
