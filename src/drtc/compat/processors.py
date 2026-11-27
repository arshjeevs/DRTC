"""Load the saved pre/post-processors of a checkpoint carried over from the minor.

GR00T N1.5 checkpoints name their processor steps groot_pack_inputs_v3, ...; LeRobot 0.7 registers rejecting stubs
under those names (N1.5 was removed upstream), and our plugin registers the same steps as groot_n15_*. The
processor files are copied to a local folder with the step names mapped, then loaded the normal way. Checkpoints
without renamed steps load directly from the Hub.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

STEP_MAP = {
    "groot_pack_inputs_v3": "groot_n15_pack_inputs_v3",
    "groot_eagle_encode_v3": "groot_n15_eagle_encode_v3",
    "groot_eagle_collate_v3": "groot_n15_eagle_collate_v3",
    "groot_action_unpack_unnormalize_v1": "groot_n15_action_unpack_unnormalize_v1",
}
CACHE = Path(os.environ.get("DRTC_CACHE", Path.home() / ".cache" / "drtc")) / "compat_processors"


def processor_path(hub: str, revision: str | None = None) -> str:
    """Path to load the processors from: the Hub repo, or a local copy with renamed steps."""
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(hub, revision=revision, allow_patterns=["policy_*"]))
    files = sorted(src.glob("policy_*processor.json"))
    needs = any(step.get("registry_name") in STEP_MAP
                for f in files for step in json.loads(f.read_text()).get("steps", []))
    if not needs:
        return hub
    dst = CACHE / hub.replace("/", "--") / src.name  # src.name: the snapshot's commit
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".json":
            data = json.loads(f.read_text())
            for step in data.get("steps", []):
                if step.get("registry_name") in STEP_MAP:
                    step["registry_name"] = STEP_MAP[step["registry_name"]]
            (dst / f.name).write_text(json.dumps(data, indent=2))
        else:
            shutil.copyfile(f, dst / f.name)
    return str(dst)


def load_processors(policy_config, hub: str, device: str, revision: str | None = None):
    """-> (preprocessor, postprocessor) for a kept checkpoint."""
    from lerobot.policies.factory import make_pre_post_processors

    return make_pre_post_processors(policy_config, pretrained_path=processor_path(hub, revision),
                                    preprocessor_overrides={"device_processor": {"device": device}})
