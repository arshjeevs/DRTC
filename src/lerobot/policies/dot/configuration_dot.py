#!/usr/bin/env python

# Copyright 2025 Ilia Larchenko. All rights reserved.
# Ported to this LeRobot version (processor pipeline, (loss, dict) forward, train-script frame resize).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decoder-Only Transformer (DOT) policy: https://github.com/IliaLarchenko/dot_policy

Defaults are the author's real-task (ALOHA insertion) settings converted from 50 fps to our 30 fps
SO101 data: 3 s training horizon, 2 s inference horizon, a 0.6 s look-back (+/- 0.1 s augmentation),
action weights per step = author's ** (50 / 30), LR 3e-5 -> 1e-5 cosine, gradient clip 50.
Frames are resized to `rescale_shape` (the author used 96x96 for PushT, full 480x640 for insertion;
240x320 keeps our 600x800 frames' 4:3 shape at a quarter of the pixels).
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("dot")
@dataclass
class DOTConfig(PreTrainedConfig):
    """Configuration for the DOT policy.

    FPS / horizon dependent (adjust for other data):
    - train_horizon: future steps predicted during training
    - inference_horizon: future steps used at inference (<= train_horizon)
    - lookback_obs_steps / lookback_aug: the extra past observation, and its random jitter in training
    - alpha / train_alpha: per-step decay of action weights at inference / in the loss

    Inference speed:
    - predict_every_n: run the model every n control steps (reuse the shifted last prediction between)
    - return_every_n: return the n-th future action instead of the next one (faster, less precise)
    """

    # Input / output structure.
    n_obs_steps: int = 3  # (n_obs_steps - 1) most recent observations + 1 from lookback_obs_steps ago
    train_horizon: int = 90
    inference_horizon: int = 60
    lookback_obs_steps: int = 18
    lookback_aug: int = 3

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Architecture.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    pre_norm: bool = True
    lora_rank: int = 20  # LoRA on every backbone conv: keeps the pretrained ResNet from overfitting (0: none)
    merge_lora: bool = False  # merge the backbone LoRA into the conv weights when loading

    dim_model: int = 128
    n_heads: int = 8
    dim_feedforward: int = 512
    n_decoder_layers: int = 8
    rescale_shape: tuple[int, int] = (240, 320)  # (height, width) every camera frame is resized to

    # Augmentation (decays every training step by noise_decay).
    crop_scale: float = 0.8
    state_noise: float = 0.01
    noise_decay: float = 0.999995

    # Training and loss computation.
    dropout: float = 0.1

    # Weighting and inference.
    alpha: float = 0.967
    train_alpha: float = 0.983
    predict_every_n: int = 1
    return_every_n: int = 1

    # Training preset.
    optimizer_lr: float = 3.0e-5
    optimizer_min_lr: float = 1.0e-5
    optimizer_lr_cycle_steps: int = 100_000  # shortened automatically to the run's length
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 50.0

    def __post_init__(self):
        super().__post_init__()
        if self.predict_every_n > self.inference_horizon:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be <= inference_horizon ({self.inference_horizon})."
            )
        if self.return_every_n > self.inference_horizon:
            raise ValueError(
                f"return_every_n ({self.return_every_n}) must be <= inference_horizon ({self.inference_horizon})."
            )
        if self.predict_every_n > self.inference_horizon // self.return_every_n:
            raise ValueError(
                f"predict_every_n ({self.predict_every_n}) must be <= inference_horizon // return_every_n "
                f"({self.inference_horizon // self.return_every_n})."
            )
        if self.train_horizon < self.inference_horizon:
            raise ValueError(
                f"train_horizon ({self.train_horizon}) must be >= inference_horizon ({self.inference_horizon})."
            )
        if self.n_obs_steps < 2:
            raise ValueError(f"n_obs_steps must be >= 2 (recent + look-back), got {self.n_obs_steps}.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_min_lr,
            num_warmup_steps=0,
            num_decay_steps=self.optimizer_lr_cycle_steps,
        )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def dataloader_frame_size(self) -> tuple[int, int]:
        """The training script resizes camera frames to this in the data-loading workers (less shared
        memory per batch; the policy would resize them anyway)."""
        return tuple(self.rescale_shape)

    @property
    def observation_delta_indices(self) -> list[int]:
        far_past = list(range(-self.lookback_aug - self.lookback_obs_steps, self.lookback_aug + 1 - self.lookback_obs_steps))
        recent = list(range(2 - self.n_obs_steps, 1))
        return far_past + recent

    @property
    def action_delta_indices(self) -> list[int]:
        far_past = list(range(-self.lookback_aug - self.lookback_obs_steps, self.lookback_aug + 1 - self.lookback_obs_steps))
        recent_and_future = list(range(2 - self.n_obs_steps, self.train_horizon))
        return far_past + recent_and_future

    @property
    def reward_delta_indices(self) -> None:
        return None
