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

π0.5 and MolmoAct 2 (fine-tuned from a base checkpoint, whose nominal input_features the fine-tune keeps):
- the default `validate_chunk_input_features` demands the checkpoint's exact names and shapes. Both policies resize
  camera images themselves (π0.5: resize_with_pad to image_resolution; MolmoAct 2: its image processor), so any RGB
  resolution is admitted, as upstream does for VLA-JEPA.
- π0.5 also has three camera slots of which our rigs fill two (it gives missing slots an empty, masked image, as in
  training) and pads the state to max_state_dim: supplied names must be a subset (state and at least one camera
  included) and the state no longer than the checkpoint's. MolmoAct 2 keeps exact names and state shape.
"""

from __future__ import annotations

import functools

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _install_diffusion()
    _install_resizing_vlas()
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


def _rgb_ok(feature, policy_feature) -> bool:
    shape = policy_feature.shape
    return len(shape) == 3 and shape[0] == 3 and feature.kind == "rgb" and len(feature.shape) == 3 \
        and feature.shape[2] == 3


def validate_pi05_features(self, features) -> None:
    from lerobot.configs.types import FeatureType

    expected = self.config.input_features or {}
    supplied = {feature.name: feature for feature in features}
    if not supplied.keys() <= expected.keys():
        raise ValueError(f"Features unknown to the checkpoint: {sorted(supplied.keys() - expected.keys())}.")
    visual = [n for n, f in expected.items() if f.type == FeatureType.VISUAL]
    if not any(n in supplied for n in visual):
        raise ValueError("π0.5 needs at least one camera.")
    for name, policy_feature in expected.items():
        if name not in supplied:
            if policy_feature.type != FeatureType.VISUAL:
                raise ValueError(f"Missing non-camera input: {name}.")
            continue  # an empty camera slot, as in training
        feature = supplied[name]
        if policy_feature.type == FeatureType.VISUAL:
            if not _rgb_ok(feature, policy_feature):
                raise ValueError(f"π0.5 requires RGB camera inputs: {name}.")
        elif feature.kind != "tensor" or len(feature.shape) != 1 or feature.shape[0] > policy_feature.shape[0]:
            raise ValueError(f"Tensor feature {name} {feature.shape} does not fit the checkpoint's "
                             f"{tuple(policy_feature.shape)} (padded, never truncated).")


def validate_molmoact2_features(self, features) -> None:
    from lerobot.configs.types import FeatureType

    expected = self.config.input_features or {}
    supplied = {feature.name: feature for feature in features}
    if supplied.keys() != expected.keys():
        raise ValueError("Canonical feature names must exactly match policy input_features.")
    for name, policy_feature in expected.items():
        feature = supplied[name]
        if policy_feature.type == FeatureType.VISUAL:
            if not _rgb_ok(feature, policy_feature):
                raise ValueError(f"MolmoAct 2 requires RGB camera inputs: {name}.")
        elif feature.kind != "tensor" or feature.shape != tuple(policy_feature.shape):
            raise ValueError(f"Tensor feature shape differs from checkpoint: {name}.")


def _install_resizing_vlas() -> None:
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    PI05Policy.validate_chunk_input_features = validate_pi05_features
    MolmoAct2Policy.validate_chunk_input_features = validate_molmoact2_features
