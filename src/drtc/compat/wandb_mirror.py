"""Every training run logs to wandb as well as Trackio (user's choice, 2026-11-19).

LeRobot 0.7 logs through `lerobot.common.trackio_utils.TrackioLogger` only. `install()` mirrors that logger's calls to
wandb: the same run name (so a Trackio run and its wandb run are found by one name), the same config, the same
metrics under the same keys (`train/loss`, `eval/eval_loss`, ...), on the same step. wandb also records the GPU and
CPU of the machine. Trackio keeps its local database on each machine (its shared Space needs a Hugging Face PRO
subscription).

wandb needs WANDB_API_KEY (the GPU servers: ~/.drtc.env or .secrets.env; the laptop: `wandb login`). Without it, or
without the package, training runs with Trackio only and says so; a wandb failure never stops a training run.
"""

from __future__ import annotations

import logging
import os
import re

_state: dict = {"run": None}


def _run_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)[:120]


def _start(trackio_logger, cfg, project: str, entity: str | None) -> None:
    try:
        import wandb
    except ImportError:
        logging.warning("wandb is not installed: this run logs to Trackio only")
        return
    if not (os.environ.get("WANDB_API_KEY") or wandb.api.api_key):
        logging.warning("no WANDB_API_KEY: this run logs to Trackio only")
        return
    try:
        name = trackio_logger.cfg.run_name
        _state["run"] = wandb.init(project=project, entity=entity, name=name, id=_run_id(name),
                                   group=getattr(trackio_logger, "_group", None), config=cfg.to_dict(),
                                   resume="allow", job_type="train")
        logging.info(f"wandb: {_state['run'].url}")
    except Exception as e:  # noqa: BLE001 - logging must never stop training
        logging.warning(f"wandb could not start ({type(e).__name__}: {e}); this run logs to Trackio only")
        _state["run"] = None


def _log(d: dict, step: int | None, mode: str) -> None:
    run = _state["run"]
    if run is None:
        return
    data = {f"{mode}/{k}": v for k, v in d.items() if isinstance(v, (int, float))}
    if not data:
        return
    try:
        run.log(data, step=step)
    except Exception as e:  # noqa: BLE001
        logging.warning(f"wandb log failed ({e}); continuing")


def problem() -> str | None:
    """Why wandb cannot log on this machine (package, key), or None. No network call."""
    try:
        import wandb
    except ImportError:
        return "wandb is not installed here (uv sync --extra gpu; on the A100 rerun lab/bin/a100-setup)"
    if not (os.environ.get("WANDB_API_KEY") or wandb.api.api_key):
        return ("no wandb key here: `wandb login` (laptop), or WANDB_API_KEY in this host's secrets file "
                "(lab/hosts.yaml `secrets`)")
    return None


def install(project: str, entity: str | None = None) -> None:
    from lerobot.common import trackio_utils

    cls = trackio_utils.TrackioLogger
    if getattr(cls, "_drtc_wandb", False):
        return
    init, log_dict, finish = cls.__init__, cls.log_dict, cls.finish

    def __init__(self, cfg, *args, **kwargs):
        init(self, cfg, *args, **kwargs)
        _start(self, cfg, project, entity)

    def _log_dict(self, d, step=None, mode="train", custom_step_key=None):
        log_dict(self, d, step=step, mode=mode, custom_step_key=custom_step_key)
        _log(d, step if custom_step_key is None else None, mode)

    def _finish(self):
        finish(self)
        if _state["run"] is not None:
            try:
                _state["run"].finish()
            finally:
                _state["run"] = None

    cls.__init__, cls.log_dict, cls.finish = __init__, _log_dict, _finish
    cls._drtc_wandb = True
