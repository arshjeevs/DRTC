"""Job files and the per-host worker (lab/PLAN.md, "Lab system").

    lab submit <model> [--host H] [--steps N] [-- <lerobot-train flags>]   queue a training job (local or on host H)
    lab worker [--max-jobs N]                                                run queued jobs on this machine
    lab jobs                                                                 list this machine's jobs

A job file pins what runs: the model, its arguments, the git commit and uv.lock it was submitted from, and the GPU
memory it needs. The worker starts a job only when its commit is checked out here (it never changes the checkout
itself; it reports the mismatch) and enough GPU memory is free (the H200 is shared). Queue layout, in the repo:
    .queue/pending/<id>.yaml   .queue/running/<id>.yaml   .queue/done/<id>.yaml   .queue/failed/<id>.yaml
    logs/<id>.log
Submitting to another host copies the job file into that host's .queue/pending over SSH (hosts.yaml: ssh, repo).
The person submitting runs `lab submit`; Claude only writes job files.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from drtc.registry import Registry, RegistryError

QUEUE = Path(".queue")
LOGS = Path("logs")
STATES = ("pending", "running", "done", "failed")
# Rough peak GPU memory per policy for a full fine-tune at the recipe's batch size, GB. Recipes can set their own
# (train.gpu_mem_gb is not a lerobot flag; it only tells the worker how much to wait for). Estimates, refined from
# measured runs.
GPU_MEM_GB = {"act": 6, "diffusion": 12, "smolvla": 24, "pi0": 70, "pi05": 70, "pi0_fast": 70, "xvla": 45,
              "groot_n15": 45, "groot": 45, "molmoact2": 70}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def lock_sha() -> str:
    return hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()[:12]


def new_job(reg: Registry, model: str, steps: int | None, extra: list[str]) -> dict:
    r = reg.resolve(model)
    if not r.trainable:
        raise RegistryError(f"{model!r} cannot be trained (no recipe/dataset)")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RegistryError("uncommitted changes: commit (and push) first, so the job pins exactly what runs")
    commit = git("rev-parse", "HEAD")
    pushed = subprocess.run(["git", "branch", "-r", "--contains", commit], capture_output=True, text=True).stdout
    if not pushed.strip():
        raise RegistryError(f"commit {commit[:10]} is not on the remote yet: push it, so other hosts can fetch it")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args = ([f"--steps={steps}"] if steps else []) + extra
    return {
        "id": f"{stamp}-{model}",
        "kind": "train",
        "model": model,
        "args": args,
        "commit": commit,
        "uv_lock": lock_sha(),
        "gpu_mem_gb": GPU_MEM_GB.get(r.policy, 24),
        "submitted": datetime.now().isoformat(timespec="seconds"),
        "submitted_from": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
    }


def write_job(job: dict, state: str = "pending") -> Path:
    (QUEUE / state).mkdir(parents=True, exist_ok=True)
    path = QUEUE / state / f"{job['id']}.yaml"
    path.write_text(yaml.safe_dump(job, sort_keys=False))
    return path


def move(job: dict, src: str, dst: str) -> None:
    (QUEUE / dst).mkdir(parents=True, exist_ok=True)
    (QUEUE / src / f"{job['id']}.yaml").rename(QUEUE / dst / f"{job['id']}.yaml")
    (QUEUE / dst / f"{job['id']}.yaml").write_text(yaml.safe_dump(job, sort_keys=False))


def cmd_submit(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lab submit")
    ap.add_argument("model")
    ap.add_argument("--host", help="queue on this host (hosts.yaml) instead of here")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--gpu-mem", type=float, help="GPU memory the job needs, GB (default: estimate per policy at the "
                                                    "recipe's batch size)")
    args, extra = ap.parse_known_args(argv)
    extra = [a for a in extra if a != "--"]
    reg = Registry()
    try:
        job = new_job(reg, args.model, args.steps, extra)
    except (RegistryError, subprocess.CalledProcessError) as e:
        print(f"lab submit: {e}", file=sys.stderr)
        return 1
    if args.gpu_mem:
        job["gpu_mem_gb"] = args.gpu_mem
    path = write_job(job)
    if args.host:
        h = reg.hosts.get(args.host)
        if not h or not h.ssh or not h.repo:
            print(f"lab submit: host {args.host!r} needs `ssh` and `repo` in hosts.yaml", file=sys.stderr)
            return 1
        remote = f"{h.ssh}:{h.repo.rstrip('/')}/.queue/pending/"
        subprocess.run(["ssh", h.ssh, f"mkdir -p {h.repo.rstrip('/')}/.queue/pending"], check=True)
        subprocess.run(["scp", "-q", str(path), remote], check=True)
        path.unlink()  # the job lives on the host now
        print(f"queued {job['id']} on {args.host} (commit {job['commit'][:10]}, needs ~{job['gpu_mem_gb']} GB GPU)")
    else:
        print(f"queued {job['id']} here: {path}")
    return 0


def free_gpu_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split()
    return max(float(x) for x in out) / 1024


def cmd_worker(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lab worker")
    ap.add_argument("--max-jobs", type=int, default=1, help="jobs running at once (default 1: the GPU is shared)")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--once", action="store_true", help="start what can start now, wait for it, then exit")
    a = ap.parse_args(argv)
    for s in STATES:
        (QUEUE / s).mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    running: dict[str, tuple[subprocess.Popen, dict]] = {}
    told: set[tuple[str, str]] = set()
    print(f"lab worker: queue {QUEUE.resolve()}, up to {a.max_jobs} job(s) at once; Ctrl-C stops the worker "
          f"(running jobs get a clean stop)", flush=True)
    try:
        while True:
            for jid, (proc, job) in list(running.items()):
                if proc.poll() is not None:
                    job["finished"] = datetime.now().isoformat(timespec="seconds")
                    job["exit_code"] = proc.returncode
                    move(job, "running", "done" if proc.returncode == 0 else "failed")
                    print(f"{job['finished']} {jid}: {'done' if proc.returncode == 0 else 'FAILED'} "
                          f"(exit {proc.returncode}, log {LOGS / (jid + '.log')})", flush=True)
                    del running[jid]
            head = git("rev-parse", "HEAD")
            for path in sorted((QUEUE / "pending").glob("*.yaml")):
                if len(running) >= a.max_jobs:
                    break
                job = yaml.safe_load(path.read_text())
                jid = job["id"]
                why = None
                if job["commit"] != head:
                    why = f"needs commit {job['commit'][:10]}, this checkout is at {head[:10]} (git pull / checkout)"
                elif job.get("uv_lock") != lock_sha():
                    why = "uv.lock differs from the submitted one (uv sync)"
                else:
                    free = free_gpu_gb()
                    if free < job["gpu_mem_gb"]:
                        why = f"waiting for GPU memory: {free:.0f} GB free, needs ~{job['gpu_mem_gb']} GB"
                if why:
                    if (jid, why) not in told:  # say each reason once, not every poll
                        told.add((jid, why))
                        print(f"{jid}: {why}", flush=True)
                    continue
                job["started"] = datetime.now().isoformat(timespec="seconds")
                move(job, "pending", "running")
                log = open(LOGS / f"{jid}.log", "w")
                proc = subprocess.Popen([sys.executable, "-m", "drtc.cli", "train", job["model"], *job["args"]],
                                        stdout=log, stderr=subprocess.STDOUT)
                running[jid] = (proc, job)
                print(f"{job['started']} {jid}: started (pid {proc.pid}, log {LOGS / (jid + '.log')})", flush=True)
            if a.once and not running:
                return 0
            time.sleep(a.poll)
    except KeyboardInterrupt:
        print("lab worker: stopping; asking running jobs to save and stop", flush=True)
        for jid, (proc, job) in running.items():
            from drtc.train import stop_file

            stop_file(job["model"]).parent.mkdir(parents=True, exist_ok=True)
            stop_file(job["model"]).touch()
        for jid, (proc, job) in running.items():
            proc.wait()
            job["finished"] = datetime.now().isoformat(timespec="seconds")
            job["exit_code"] = proc.returncode
            job["stopped_by_worker"] = True
            move(job, "running", "failed" if proc.returncode else "done")
        return 0


def cmd_jobs(argv: list[str]) -> int:
    for s in STATES:
        for path in sorted((QUEUE / s).glob("*.yaml")) if (QUEUE / s).is_dir() else []:
            job = yaml.safe_load(path.read_text())
            extra = f" exit {job['exit_code']}" if "exit_code" in job else ""
            print(f"  {s:8s} {job['id']:52s} commit {job['commit'][:10]}{extra}")
    return 0

