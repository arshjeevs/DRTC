"""The results store (lab/EVALUATION.md rule 5): the private Hub dataset in lab/lab.yaml `results_repo`.

Its layout mirrors results/: evals/<eval>/sessions/<session>/... (robot) and bench/<eval>/sessions/... (replay robot).
- pull: downloads the sessions this machine does not have yet. A session already here is never overwritten (it may
  hold an edit, e.g. a void, that is not pushed yet).
- push: uploads this machine's sessions that changed since their last push; a pushed session holds `.pushed` (the
  store's commit). Re-pushing a session (after a void) replaces its files there.
"""

from __future__ import annotations

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


def pull(reg: Registry, name: str, kind: str = "evals", root: Path = Path("results")) -> list[str]:
    """Download the eval's sessions that are not here yet; returns their names. Raises on no connection."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    prefix = _prefix(kind, name)
    files = [f for f in api.list_repo_files(repo(reg), repo_type="dataset") if f.startswith(prefix)]
    local = root / kind / name / "sessions"
    have = {p.name for p in local.iterdir()} if local.is_dir() else set()
    new = sorted({f[len(prefix):].split("/", 1)[0] for f in files} - have)
    for f in files:
        session = f[len(prefix):].split("/", 1)[0]
        if session in new:
            hf_hub_download(repo(reg), f, repo_type="dataset", local_dir=root)
    for s in new:
        (local / s / store.PUSHED).write_text("pulled from the results store\n")
    return new


def push(reg: Registry, session: Path, kind: str = "evals", name: str | None = None) -> str:
    """Upload one session folder; returns the store's commit. Marks the session as pushed."""
    from huggingface_hub import HfApi

    name = name or session.parent.parent.name
    info = HfApi().upload_folder(folder_path=str(session), path_in_repo=_prefix(kind, name) + session.name,
                                 repo_id=repo(reg), repo_type="dataset", ignore_patterns=[store.PUSHED],
                                 commit_message=f"{kind}/{name}: session {session.name}")
    oid = getattr(info, "oid", None) or str(info)
    (session / store.PUSHED).write_text(f"{oid}\n")
    return oid
