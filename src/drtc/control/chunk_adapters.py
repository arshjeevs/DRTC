"""Chunk-serving adapters for upstream policies that LeRobot 0.7's default chunk runner cannot serve as-is.

`install()` patches them in place, once; `lab serve` and tests/smoke/serve_check.py call it before loading a
deployment. Upstream code is not edited.

Diffusion Policy:
- its config has `horizon` / `n_action_steps` but no `chunk_size`, so the default `chunk_inference_spec` refuses it
  ("requires a positive chunk_size"). `predict_action_chunk` returns `n_action_steps` actions (the slice that starts
  at the current observation), so that is both the prediction and the execution length. Only `n_obs_steps == 1` is
  admitted: the server sends one observation, never a history.
- with empty observation queues `predict_action_chunk` adds the observation-time axis to images only, while
  `generate_actions` expects every observation as (B, n_obs_steps, ...). The server sends state as (B, D), so the
  state (and environment state) get that axis here.
"""

from __future__ import annotations

import functools

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _install_diffusion()
    _installed = True


def _install_diffusion() -> None:
    from lerobot.policies.chunk import ChunkPolicySpec, ExecutionMode
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

    def chunk_inference_spec(self) -> ChunkPolicySpec:
        cfg = self.config
        if cfg.n_obs_steps != 1:
            raise ValueError("Diffusion chunk serving needs n_obs_steps=1 (the server sends one observation).")
        return ChunkPolicySpec(prediction_steps=cfg.n_action_steps, execution_steps=cfg.n_action_steps,
                               modes=(ExecutionMode.CHUNK,))

    original = DiffusionPolicy.predict_action_chunk

    @functools.wraps(original)
    def predict_action_chunk(self, batch, *args, **kwargs):
        if not any(len(q) > 0 for q in self._queues.values()):
            batch = dict(batch)
            for key in (OBS_STATE, OBS_ENV_STATE):
                if key in batch and batch[key].ndim == 2:
                    batch[key] = batch[key].unsqueeze(1)
        return original(self, batch, *args, **kwargs)

    DiffusionPolicy.chunk_inference_spec = chunk_inference_spec
    DiffusionPolicy.predict_action_chunk = predict_action_chunk
