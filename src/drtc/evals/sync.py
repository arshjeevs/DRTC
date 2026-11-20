"""The results store (lab/docs/EVALUATION.md rule 5): the private Hub dataset in lab/lab.yaml `results_repo`.

Its layout mirrors results/: evals/<eval>/sessions/<session>/... (robot) and bench/<eval>/sessions/... (replay robot).
- pull: downloads the sessions this machine does not have yet. A session already here is never overwritten (it may
  hold an edit, e.g. a void, that is not pushed yet), except one whose pull stopped part-way: it holds `.pulling`
  until its last file is here, and the next pull finishes it (2026-11-20: a pull cut by a network error left
  sessions with 23 of 64 trials, which the next pull skipped). Each file is retried on a network error.
- push: uploads this machine's sessions that changed since their last push; a pushed session holds `.pushed` (the
  store's commit). Re-pushing a session (after a void) replaces its files there.
"""

from __future__ import annotations

import time
from pathlib import Path

from drtc.evals import store
from drtc.registry import Registry, RegistryError


def repo(reg: Registry) -> str:
    r = reg.settings.results_repo
    if not r:
        raise RegistryError("lab/lab.yaml has no results_repo")
    return r


def _prefix(kind: str, name: str) -> str:
    return f"{kind}/{name}/sessions/"


def _download(repo_id: str, filename: str, root: Path, tries: int = 3) -> None:
    """One file from the store into root/, retried: a transient network error must not stop a pull."""
    from huggingface_hub import hf_hub_download

    for k in range(tries):
        try:
            hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=root)
            return
        except Exception:  # noqa: BLE001 - seen 2026-11-20: an intermittent TLS error on one request
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def pull(reg: Registry, name: str, kind: str = "evals", root: Path = Path("results")) -> list[str]:
    """Download the eval's sessions that are not here yet, and finish those whose pull stopped part-way; returns
    their names. Raises on no connection."""
    from huggingface_hub import HfApi

    prefix = _prefix(kind, name)
    files = [f for f in HfApi().list_repo_files(repo(reg), repo_type="dataset") if f.startswith(prefix)]
    local = root / kind / name / "sessions"
    here = {p.name: p for p in local.iterdir()} if local.is_dir() else {}
    stored = {f[len(prefix):].split("/", 1)[0] for f in files}
    todo = sorted(s for s in stored if s not in here or (here[s] / store.PULLING).is_file())
    for s in todo:
        (local / s).mkdir(parents=True, exist_ok=True)
        (local / s / store.PULLING).write_text("pull from the results store in progress\n")
        for f in files:
            if f[len(prefix):].split("/", 1)[0] == s:
                _download(repo(reg), f, root)  # files already here are skipped by huggingface_hub
        (local / s / store.PUSHED).write_text("pulled from the results store\n")
        (local / s / store.PULLING).unlink()
    return todo


def push(reg: Registry, session: Path, kind: str = "evals", name: str | None = None) -> str:
    """Upload one session folder; returns the store's commit. Marks the session as pushed."""
    from huggingface_hub import HfApi

    name = name or session.parent.parent.name
    info = HfApi().upload_folder(folder_path=str(session), path_in_repo=_prefix(kind, name) + session.name,
                                 repo_id=repo(reg), repo_type="dataset", ignore_patterns=[store.PUSHED, store.PULLING],
                                 commit_message=f"{kind}/{name}: session {session.name}")
    oid = getattr(info, "oid", None) or str(info)
    (session / store.PUSHED).write_text(f"{oid}\n")
    return oid


def pull_offline(reg: Registry, datasets: set[str], root: Path = Path("results")) -> int:
    """Download the offline results (M9, M10) of these datasets; returns the number of files."""
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files(repo(reg), repo_type="dataset")
             if f.startswith("offline/") and f.split("/")[1] in datasets]
    for f in files:
        hf_hub_download(repo(reg), f, repo_type="dataset", local_dir=root, force_download=True)
    return len(files)
