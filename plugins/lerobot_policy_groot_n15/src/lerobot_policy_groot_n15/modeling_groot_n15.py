#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""
Groot Policy Wrapper for LeRobot Integration

Minimal integration that delegates to Isaac-GR00T components where possible
without porting their code. The intent is to:

- Download and load the pretrained GR00T model via GR00TN15.from_pretrained
- Optionally align action horizon similar to gr00t_finetune.py
- Expose predict_action via GR00T model.get_action
- Provide a training forward that can call the GR00T model forward if batch
  structure matches.

Notes:
- Dataset loading and full training orchestration is handled by Isaac-GR00T
  TrainRunner in their codebase. If you want to invoke that flow end-to-end
  from LeRobot, see `GrootN15Policy.finetune_with_groot_runner` below.
"""

import os
from collections import deque

import torch
from torch import Tensor

from lerobot_policy_groot_n15.configuration_groot_n15 import GrootN15Config
from lerobot_policy_groot_n15.groot_n1 import GR00TN15
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


class GrootN15Policy(PreTrainedPolicy):
    """Wrapper around external Groot model for LeRobot integration."""

    name = "groot_n15"
    config_class = GrootN15Config

    def __init__(self, config: GrootN15Config, **kwargs):
        """Initialize Groot policy wrapper."""
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize GR00T model using ported components
        self._groot_model = self._create_groot_model()
        self.init_rtc_processor()

        self.reset()

    def supports_rtc(self) -> bool:
        """Guided RTC through the flow-matching action head (see _velocity_guidance)."""
        return True

    def init_rtc_processor(self) -> None:
        """Build the RTC processor from `config.rtc_config` (set by the policy server for RTC modes)."""
        rtc_config = getattr(self.config, "rtc_config", None)
        self.rtc_processor = RTCProcessor(rtc_config) if rtc_config is not None else None

    def _create_groot_model(self):
        """Create and initialize the GR00T model using Isaac-GR00T API.

        This is only called when creating a NEW policy (not when loading from checkpoint).

        Steps (delegating to Isaac-GR00T):
        1) Download and load pretrained model via GR00TN15.from_pretrained
        2) Align action horizon with data_config if provided
        """
        # Handle Flash Attention compatibility issues
        self._handle_flash_attention_compatibility()

        model = GR00TN15.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
        )

        model.compute_dtype = "bfloat16" if self.config.use_bf16 else model.compute_dtype
        model.config.compute_dtype = model.compute_dtype

        return model

    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Delegates to Isaac-GR00T model.forward when inputs are compatible.
        """
        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Run GR00T forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original GR00T finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.forward(groot_inputs)

        # Isaac-GR00T returns a BatchFeature; loss key is typically 'loss'
        loss = outputs.get("loss")

        loss_dict = {"loss": loss.item()}

        return loss, loss_dict

    def _velocity_guidance(self, rtc_kwargs: dict):
        """Guided RTC prefix inpainting for GR00T's flow-matching head (lab addition, LeRobot 0.7 RTCProcessor).

        RTCProcessor is written for the PI0 / SmolVLA convention (time 1 = noise -> 0 = clean,
        x1 = x - time * v). GR00T integrates the other way (t 0 = noise -> 1 = clean, x += dt * v), so its
        velocity is passed in negated with time = 1 - t and the result negated back, as upstream's
        `euler_integrate` does for NOISE_AT_ZERO policies.

        The prefix (`prev_chunk_left_over`, model coordinates) has the robot's action width, while GR00T
        denoises `max_action_dim` padded dims. Only the real dims are guided: the padded ones are held at
        their current values inside the guided call and keep their unguided velocity (otherwise RTC would
        pull them to the prefix's zero padding). Returns None when there is nothing to apply.
        """
        rtc = getattr(self, "rtc_processor", None)
        prefix = rtc_kwargs.get("prev_chunk_left_over")
        if rtc is None or prefix is None or not rtc.rtc_config.enabled:
            return None
        width = prefix.shape[-1]
        delay = rtc_kwargs.get("inference_delay") or 0
        horizon = rtc_kwargs.get("execution_horizon")

        def guidance(actions, t, num_steps, velocity):
            padding = actions[..., width:].detach()
            full = {}

            def real_dims_velocity(x_real):
                v = velocity(torch.cat([x_real, padding.to(x_real.dtype)], dim=-1))
                full["v"] = v
                return -v[..., :width]

            v_real = -rtc.denoise_step(
                x_t=actions[..., :width],
                prev_chunk_left_over=prefix.to(device=actions.device, dtype=actions.dtype),
                inference_delay=delay,
                time=1.0 - t,
                original_denoise_step_partial=real_dims_velocity,
                execution_horizon=horizon,
            )
            v = full["v"].detach().clone()
            v[..., :width] = v_real.to(v.dtype)
            return v

        return guidance

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **rtc_kwargs) -> Tensor:
        """Predict a chunk of actions for inference by delegating to Isaac-GR00T.

        Returns a tensor of shape (B, n_action_steps, action_dim). `rtc_kwargs` (inference_delay,
        prev_chunk_left_over, overlap_end) come from the DRTC policy server; see _velocity_guidance.
        """
        self.eval()

        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        # Preprocessing is handled by the processor pipeline, so we just filter the batch
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        allowed_base = {"state", "state_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(groot_inputs, self._velocity_guidance(rtc_kwargs))

        actions = outputs.get("action_pred")

        original_action_dim = self.config.output_features["action"].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues by setting environment variables.

        This addresses the common 'undefined symbol' error that occurs when Flash Attention
        is compiled against a different PyTorch version than what's currently installed.
        """

        # Set environment variables to handle Flash Attention compatibility
        # These help with symbol resolution issues
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")

        # Try to import flash_attn and handle failures gracefully
        try:
            import flash_attn

            print(f"[GROOT] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[GROOT] Flash Attention not available: {e}")
            print("[GROOT] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[GROOT] Flash Attention compatibility issue detected: {e}")
                print("[GROOT] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[GROOT] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[GROOT] Continuing with fallback attention mechanism")
            else:
                print(f"[GROOT] Flash Attention error: {e}")
                print("[GROOT] Continuing with fallback attention mechanism")
