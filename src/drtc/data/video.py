"""One video setup on every machine: how recordings are written and how training reads them.

Recording writes video (camera frames -> file); training reads it back (file -> frames). LeRobot 0.7 writes only
with PyAV, and reads with torchcodec or PyAV. Both sit on FFmpeg, and the file format is what keeps everything
consistent:
- ENCODER: every recording is AV1 (libsvtav1), yuv420p, a keyframe every 2 frames, CRF 30. These are LeRobot 0.7's
  defaults and how so101_cubebox_v1 was recorded; `lab record` passes them explicitly, so a LeRobot upgrade cannot
  change them unnoticed.
- DECODER: training, frame caches and checks read with torchcodec (3.5 ms per frame on the laptop, PyAV 8.7 ms). On
  our files the two give identical frames (checked 2026-11-19). LeRobot silently falls back to PyAV where
  torchcodec cannot load FFmpeg (the H200 container had none), so `lab train` checks first and stops with
  instructions instead; `--dataset.video_backend=pyav` trains with PyAV deliberately.
"""

from __future__ import annotations

from drtc.registry import RegistryError

DECODER = "torchcodec"
ENCODER = {"vcodec": "libsvtav1", "pix_fmt": "yuv420p", "g": 2, "crf": 30}


def encoder_flags() -> dict:
    """lerobot-record flags for ENCODER."""
    return {f"dataset.rgb_encoder.{k}": v for k, v in ENCODER.items()}


def check_decoder() -> str:
    """DECODER, after checking it can load on this machine (no silent fallback to PyAV)."""
    from lerobot.utils.import_utils import get_safe_default_video_backend

    got = get_safe_default_video_backend()
    if got != DECODER:
        raise RegistryError(
            f"video decoding: {DECODER} cannot load FFmpeg on this machine, so LeRobot would fall back to {got} "
            f"(the same frames, about 2.5x slower). Install FFmpeg 4-9, e.g. `apt-get update && apt-get install -y "
            f"ffmpeg` (in the H200 container), or pass --dataset.video_backend={got} to use it deliberately.")
    return got
