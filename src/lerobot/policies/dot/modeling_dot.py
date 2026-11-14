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

Differences from the original code:
- Normalization happens in the processor pipeline (processor_dot.py), not inside the policy; actions
  are ensembled in normalized units (the same result: the weights sum to 1 and min-max is affine).
- `forward` returns (loss, dict) and, in eval mode (held-out validation), skips augmentation and uses
  the middle look-back frame, so validation is deterministic.
- `predict_action_chunk` exists (DRTC server, validation): from a training-format batch it uses the
  stored history; from a single observation it keeps the same rolling buffers as `select_action`,
  so each call counts as one control step.
- Frames are resized with bilinear + antialias (the original used nearest, chosen for PushT's 96x96).
- Random-crop sizes are rounded to 16 px steps (a few fixed shapes, so cudnn.benchmark stays fast).
"""

import math

import torch
import torchvision
from torch import Tensor, nn
from torchvision import transforms
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF  # noqa: N812

from lerobot.policies.dot.configuration_dot import DOTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


def resize_frames(frames: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize (..., C, H, W) frames; a no-op when they already have that size."""
    if tuple(frames.shape[-2:]) == tuple(size):
        return frames
    return TF.resize(frames, list(size), interpolation=InterpolationMode.BILINEAR, antialias=True)


class DOT(nn.Module):
    def __init__(self, config: DOTConfig):
        super().__init__()
        self.config = config

        self.projections = nn.ModuleDict()
        self.n_features = 0

        self.image_names = sorted(config.image_features.keys())

        # One backbone for all cameras; its output is projected to the model dimension.
        if len(self.image_names) > 0:
            backbone = getattr(torchvision.models, self.config.vision_backbone)(
                weights=self.config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            backbone.fc = nn.Linear(backbone.fc.in_features, self.config.dim_model)

            # lora_rank 0: a plain backbone (all trainable; LoRA can then come from finetune: lora).
            self.projections["images"] = (
                add_lora_to_backbone(backbone, rank=config.lora_rank) if config.lora_rank > 0 else backbone
            )
            self.n_features += len(self.image_names) * self.config.n_obs_steps

        if self.config.robot_state_feature:
            self.projections["state"] = nn.Linear(self.config.robot_state_feature.shape[0], self.config.dim_model)
            self.n_features += self.config.n_obs_steps

        if self.config.env_state_feature:
            self.projections["environment_state"] = nn.Linear(
                self.config.env_state_feature.shape[0], self.config.dim_model
            )
            self.n_features += self.config.n_obs_steps

        self.projections_names = sorted(self.projections.keys())
        obs_mapping = {"images": OBS_IMAGES, "state": OBS_STATE, "environment_state": OBS_ENV_STATE}
        self.obs_mapping = {k: v for k, v in obs_mapping.items() if k in self.projections_names}

        # Extra trainable vector prepended to the input features.
        self.prefix_input = nn.Parameter(torch.randn(1, 1, config.dim_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.config.dim_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=self.config.pre_norm,
        )
        decoder_norm = nn.LayerNorm(self.config.dim_model)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.config.n_decoder_layers, norm=decoder_norm)

        # Decoder queries: fixed sinusoidal encodings of the predicted time steps
        # [look-back, recent past..., 0, 1, ..., train_horizon - 1].
        decoder_pos = create_sinusoidal_pos_embedding(
            config.train_horizon + config.lookback_obs_steps, config.dim_model
        )
        decoder_pos = torch.cat(
            [decoder_pos[:1], decoder_pos[-config.train_horizon - config.n_obs_steps + 2 :]], dim=0
        )
        self.register_buffer("decoder_pos", decoder_pos)

        decoder_pos_inf = self.decoder_pos[
            : self.decoder_pos.shape[0] + self.config.inference_horizon - self.config.train_horizon
        ]
        self.register_buffer("decoder_pos_inf", decoder_pos_inf)

        # The first inference_horizon steps must not see the extra training-only steps, so training
        # behaves like inference for them.
        mask = torch.zeros(len(decoder_pos), len(decoder_pos), dtype=torch.bool)
        mask[
            : len(decoder_pos) + config.inference_horizon - config.train_horizon,
            len(decoder_pos) + config.inference_horizon - config.train_horizon :,
        ] = True
        self.register_buffer("mask", mask)

        # Trainable positional embeddings of the input features.
        self.inputs_pos_emb = nn.Parameter(torch.empty(1, self.n_features, self.config.dim_model))
        nn.init.uniform_(
            self.inputs_pos_emb, -((1 / self.config.dim_model) ** 0.5), (1 / self.config.dim_model) ** 0.5
        )

        self.action_head = nn.Linear(self.config.dim_model, self.config.action_feature.shape[0])

    def _process_inputs(self, batch: dict[str, Tensor]) -> Tensor:
        inputs_projections_list = []
        for state in self.projections_names:
            batch_state = self.obs_mapping[state]
            if batch_state in batch:
                bs, n_obs, *obs_shape = batch[batch_state].shape
                enc = self.projections[state](batch[batch_state].reshape(bs * n_obs, *obs_shape)).view(bs, n_obs, -1)
                inputs_projections_list.append(enc)
        return torch.cat(inputs_projections_list, dim=1)

    def forward(self, batch: dict[str, Tensor], full_horizon: bool) -> Tensor:
        """Actions for every decoder step. full_horizon: all training steps with the training mask
        (loss); otherwise only the inference steps (acting)."""
        inputs_projections = self._process_inputs(batch)
        bs = inputs_projections.shape[0]

        inputs_projections = inputs_projections + self.inputs_pos_emb.expand(bs, -1, -1)
        inputs_projections = torch.cat([self.prefix_input.expand(bs, -1, -1), inputs_projections], dim=1)

        if full_horizon:
            decoder_out = self.decoder(self.decoder_pos.expand(bs, -1, -1), inputs_projections, self.mask)
        else:
            decoder_out = self.decoder(self.decoder_pos_inf.expand(bs, -1, -1), inputs_projections)
        return self.action_head(decoder_out)


class DOTPolicy(PreTrainedPolicy):
    config_class = DOTConfig
    name = "dot"

    def __init__(self, config: DOTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.image_names = sorted(config.image_features.keys())
        self.model = DOT(self.config)

        self.state_noise = self.config.state_noise
        self.crop_scale = self.config.crop_scale

        # Weights of the overlapping predictions averaged at inference (newest first).
        action_weights = self.config.alpha ** torch.arange(self.config.inference_horizon).float()
        action_weights /= action_weights.sum()
        self.register_buffer("action_weights", action_weights.view(1, -1, 1), persistent=False)

        # Loss weights: past steps 1, future steps decay with train_alpha; mean 1.
        loss_weights = torch.ones(self.config.train_horizon + self.config.n_obs_steps - 1)
        loss_weights[-self.config.train_horizon :] = (
            self.config.train_alpha ** torch.arange(self.config.train_horizon).float()
        )
        loss_weights /= loss_weights.mean()
        self.register_buffer("loss_weights", loss_weights.view(1, -1, 1), persistent=False)

        self.reset()

    def reset(self):
        self._old_predictions = None
        self._input_buffers = {}
        self.last_action = None
        self.step = 0

    def get_optim_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    # --- inference ---

    def _update_observation_buffers(self, buffer_name: str, observation: Tensor) -> Tensor:
        # Keep the last lookback_obs_steps + 1 observations; return [oldest, most recent n_obs_steps - 1].
        if buffer_name not in self._input_buffers:
            self._input_buffers[buffer_name] = observation.unsqueeze(1).repeat(
                1, self.config.lookback_obs_steps + 1, *([1] * (observation.dim() - 1))
            )
        else:
            self._input_buffers[buffer_name] = self._input_buffers[buffer_name].roll(shifts=-1, dims=1)
            self._input_buffers[buffer_name][:, -1] = observation

        return torch.cat(
            [
                self._input_buffers[buffer_name][:, :1],
                self._input_buffers[buffer_name][:, -(self.config.n_obs_steps - 1) :],
            ],
            dim=1,
        )

    def _has_history(self, batch: dict[str, Tensor]) -> bool:
        """True for a training-format batch (observations carry the look-back window)."""
        key = self.image_names[0] if self.image_names else next(iter(self.model.obs_mapping.values()))
        return batch[key].dim() == (5 if self.image_names else 3)

    def _select_history(self, batch: dict[str, Tensor], lookback_ind: int) -> dict[str, Tensor]:
        """Training-format batch -> the one look-back step + the recent steps (obs and actions)."""
        batch = dict(batch)
        keys = [v for v in self.model.obs_mapping.values() if v != OBS_IMAGES] + self.image_names
        if ACTION in batch:
            keys += [ACTION, "action_is_pad"]
        for k in keys:
            batch[k] = torch.cat(
                [batch[k][:, lookback_ind : lookback_ind + 1], batch[k][:, 2 * self.config.lookback_aug + 1 :]], 1
            )
        return batch

    def _stack_images(self, batch: dict[str, Tensor], crop: transforms.RandomCrop | None = None) -> None:
        # (bs, n_obs, c, h, w) per camera -> batch[OBS_IMAGES]: (bs, n_obs * n_cam, c, h, w), obs-major
        frames = []
        for k in self.image_names:
            bs, n_obs, c, h, w = batch[k].shape
            x = resize_frames(batch[k].reshape(bs * n_obs, c, h, w), self.config.rescale_shape)
            if crop is not None:
                x = crop(x)
            frames.append(x.reshape(bs, n_obs, c, *x.shape[-2:]))
        batch[OBS_IMAGES] = torch.stack(frames, dim=2).flatten(1, 2)

    def _prepare_single_observation(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        if self.image_names:
            batch[OBS_IMAGES] = torch.stack(
                [resize_frames(batch[k], self.config.rescale_shape) for k in self.image_names], dim=1
            )  # bs, n_cam, c, h, w
        for name, batch_name in self.model.obs_mapping.items():
            batch[batch_name] = self._update_observation_buffers(name, batch[batch_name])
        if OBS_IMAGES in batch:
            batch[OBS_IMAGES] = batch[OBS_IMAGES].flatten(1, 2)  # bs, n_obs * n_cam, c, h, w
        return batch

    def _chunk_actions(self, actions: Tensor) -> Tensor:
        # Weighted average of the predictions made for this step by the last inference_horizon calls.
        if self._old_predictions is not None:
            self._old_predictions[:, 0] = actions
        else:
            self._old_predictions = actions.unsqueeze(1).repeat(1, self.config.inference_horizon, 1, 1)

        action = (self._old_predictions[:, :, 0] * self.action_weights).sum(dim=1)
        self._old_predictions = self._old_predictions.roll(shifts=(1, -1), dims=(1, 2))
        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """(bs, inference_horizon, action_dim) normalized actions for the current and next steps."""
        self.eval()
        if self._has_history(batch):
            batch = self._select_history(batch, self.config.lookback_aug)
            if self.image_names:
                self._stack_images(batch)
        else:
            batch = self._prepare_single_observation(batch)
        return self.model(batch, full_horizon=False)[:, -self.config.inference_horizon :]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """One action per control step, ensembled over the overlapping predictions."""
        self.eval()
        batch = self._prepare_single_observation(batch)

        if self.step % self.config.predict_every_n == 0:
            self.last_action = self.model(batch, full_horizon=False)[:, -self.config.inference_horizon :]
        else:  # shift the previous prediction, repeating its last action
            self.last_action = self.last_action.roll(-1, dims=1)
            self.last_action[:, -1] = self.last_action[:, -2]
        self.step += 1

        action = self._chunk_actions(self.last_action)
        for _ in range(self.config.return_every_n - 1):
            self.last_action = self.last_action.roll(-1, dims=1)
            self.last_action[:, -1] = self.last_action[:, -2]
            action = self._chunk_actions(self.last_action)
        return action

    def chunk_targets(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Ground truth aligned with predict_action_chunk on a training-format batch (actions from the
        current step on, and their padding flags)."""
        start = 2 * self.config.lookback_aug + 1 + self.config.n_obs_steps - 2
        return batch[ACTION][:, start:], batch["action_is_pad"][:, start:]

    # --- training ---

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        augment = self.training
        if augment:
            lookback_ind = torch.randint(0, 2 * self.config.lookback_aug + 1, (1,)).item()
        else:
            lookback_ind = self.config.lookback_aug  # validation: the nominal look-back
        batch = self._select_history(batch, lookback_ind)

        if self.image_names:
            crop = None
            if augment:
                # Crop sizes are rounded to 16 px steps: lerobot_train enables cudnn.benchmark, which
                # re-tunes every conv for each new input size (a different size every step: ~5x slower).
                height, width = self.config.rescale_shape
                scale = 1 - torch.rand(1).item() * (1 - self.crop_scale)
                crop_h = max(16, int(height * scale) // 16 * 16)
                crop = transforms.RandomCrop((crop_h, round(crop_h * width / height)))
            self._stack_images(batch, crop)

        if augment and self.state_noise is not None:
            for k in self.model.obs_mapping.values():
                if k != OBS_IMAGES:
                    batch[k] = batch[k] + (torch.rand_like(batch[k]) * 2 - 1) * self.state_noise

        actions_hat = self.model(batch, full_horizon=True)

        loss = nn.functional.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid = (~batch["action_is_pad"]).unsqueeze(-1)
        loss = (loss * valid * self.loss_weights).mean()

        if augment:  # augmentation gets gentler as training goes on
            self.state_noise *= self.config.noise_decay
            self.crop_scale = 1 - (1 - self.crop_scale) * self.config.noise_decay

        return loss, {"l1_loss": loss.item()}

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        """Load, then merge the backbone LoRA when the config asks for it."""
        policy = super().from_pretrained(pretrained_name_or_path, *args, **kwargs)
        if getattr(policy.config, "merge_lora", False):
            policy.model = merge_lora_weights(policy.model)
        return policy


class LoRAConv2d(nn.Module):
    def __init__(self, base_conv: nn.Conv2d, rank: int = 4):
        super().__init__()
        self.base_conv = base_conv

        out_channels, in_channels, kh, kw = base_conv.weight.shape
        self.weight_shape = (out_channels, in_channels, kh, kw)
        fan_in = in_channels * kh * kw

        self.lora_A = nn.Parameter(torch.normal(0, 0.02, (out_channels, rank)))
        self.lora_B = nn.Parameter(torch.normal(0, 0.02, (rank, fan_in)))

    def forward(self, x: Tensor) -> Tensor:
        lora_update = torch.matmul(self.lora_A, self.lora_B).view(self.weight_shape)
        return nn.functional.conv2d(
            x,
            self.base_conv.weight + lora_update,
            self.base_conv.bias,
            stride=self.base_conv.stride,
            padding=self.base_conv.padding,
            dilation=self.base_conv.dilation,
            groups=self.base_conv.groups,
        )

    @torch.no_grad()
    def merge_lora(self) -> nn.Conv2d:
        """Fold the LoRA update into the base convolution and return it as a plain Conv2d."""
        lora_update = torch.matmul(self.lora_A, self.lora_B).view(self.weight_shape)
        self.base_conv.weight.add_(lora_update)
        return self.base_conv


def replace_conv2d_with_lora(module: nn.Module, rank: int = 4) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank))
        else:
            replace_conv2d_with_lora(child, rank)
    return module


def merge_lora_weights(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, LoRAConv2d):
            setattr(module, name, child.merge_lora())
        else:
            merge_lora_weights(child)
    return module


def add_lora_to_backbone(backbone: nn.Module, rank: int = 4) -> nn.Module:
    """LoRA on every conv; only the LoRA matrices and the new fc layer train."""
    replace_conv2d_with_lora(backbone, rank)
    for name, param in backbone.named_parameters():
        param.requires_grad = "lora_" in name or name.startswith("fc")
    return backbone


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    position = torch.arange(num_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float) * (-math.log(10000.0) / dimension))
    pe = torch.zeros(num_positions, dimension)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
