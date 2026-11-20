"""Serve a registry model with LeRobot 0.7's policy server: `lab serve <model>`.

The deployment (what the server declares and every client must match) comes from the registry:
- features: the model's dataset (joint names, camera shapes as HWC uint8 RGB);
- semantics: joint units from the dataset (use_degrees) and camera roles from its rig, so a model trained in one
  unit system can never be admitted for a robot configured in the other;
- action rate: the dataset's fps.
The server process is upstream's (lerobot-policy-server); this only writes its config and starts it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from drtc.registry import Registry, RegistryError

# LeRobot ships the degrees convention; the -100..100 one (LeRobot 0.4.3 datasets) is ours.
JOINT_SEMANTICS = {True: "so101-degrees-gripper-percent-v1", False: "so101-range100-gripper-percent-v1"}


def deployment_name(model_id: str, revision: str | None = None) -> str:
    """The name a client asks for. With a revision (eval serving), the Hub commit is part of the name, so a client
    can only ever reach the model version its frozen eval names."""
    name = model_id.replace("_", "-")
    return f"{name}-{revision[:8]}" if revision else name


# Policies that declare guided RTC in LeRobot 0.7 (SmolVLA upstream; GR00T N1.5 through our plugin). ACT and Diffusion
# have none: an RTC condition on them is refused by `lab eval check`.
RTC_POLICIES = {"smolvla", "groot_n15", "pi0", "pi05"}
RTC_EXECUTION_HORIZON = 10  # upstream's default (RTCConfig.execution_horizon)


# What a policy server needs from a model repo: the root (the served checkpoint). checkpoints/<step>/ holds the
# training state for resuming (optimizer, several GB for the VLAs) and is never downloaded for serving.
SERVE_IGNORE = ["checkpoints/*", "checkpoints/**"]


def fetch_checkpoint(repo: str, revision: str | None = None) -> Path:
    """Download (or find in the Hub cache) only the served checkpoint of a model repo."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, ignore_patterns=SERVE_IGNORE))


def semantics_for(reg: Registry, model_id: str) -> dict:
    """Joint semantics and one semantics label per camera, for the model's dataset."""
    r = reg.resolve(model_id)
    if r.dataset is None:
        raise RegistryError(f"{model_id!r} has no dataset in the registry, so its units and cameras are unknown")
    if r.dataset.use_degrees is None:
        raise RegistryError(f"dataset {r.dataset.id!r} does not state its joint units (use_degrees)")
    rig = reg.rigs[r.dataset.rig]
    cams = {k: f"rgb-{c.role}-v1" for k, c in rig.cameras.items()}
    return {"joints": JOINT_SEMANTICS[r.dataset.use_degrees], "cameras": cams}


def camera_renames(reg: Registry, model_id: str) -> dict[str, str]:
    """Dataset camera key -> the checkpoint's input name, from the recipe's `cameras` (the training rename_map).

    The server declares the checkpoint's names (LeRobot 0.7 requires them to match the policy's input_features);
    the robot client maps its camera keys onto them with the same map (`lab run` passes it as --rename_map).
    """
    r = reg.resolve(model_id)
    return dict((r.flags or {}).get("rename_map") or {})


def server_config(reg: Registry, model_id: str, path: str | None = None, device: str = "cuda",
                  listen: str = "tcp/127.0.0.1:7447", modes: list[str] | None = None) -> dict:
    from lerobot.datasets import LeRobotDatasetMetadata

    r = reg.resolve(model_id)
    sem = semantics_for(reg, model_id)
    renames = camera_renames(reg, model_id)
    meta = LeRobotDatasetMetadata(r.dataset.repo, revision=r.dataset.revision)
    state = meta.features["observation.state"]
    action = meta.features["action"]
    features = [{"name": "observation.state", "shape": list(state["shape"]), "dtype": "float32",
                 "names": list(state["names"]), "semantics": sem["joints"]}]
    for key, f in meta.features.items():
        if f["dtype"] != "video":
            continue
        cam = key.removeprefix("observation.images.")
        h, w, c = f["shape"]
        features.append({"name": renames.get(key, key), "shape": [h, w, c], "dtype": "uint8", "kind": "rgb",
                         "semantics": sem["cameras"].get(cam, f"rgb-{cam}-v1")})
    execution = {"supported_modes": modes or ["chunk"], "action_fps": float(meta.fps)}
    if any(m != "chunk" for m in execution["supported_modes"]):
        execution["rtc"] = {"enabled": True, "mode": "guided", "execution_horizon": RTC_EXECUTION_HORIZON}
    return {
        "deployment": deployment_name(model_id),
        "model": {"repo_or_path": path or r.hub, "revision": None, "device": device},
        "execution": execution,
        "zenoh": {"mode": "peer", "listen_endpoints": [listen]},
        "semantics": sem["joints"],
        "robot_type": reg.rigs[r.dataset.rig].robot.type if reg.rigs[r.dataset.rig].robot else "so101_follower",
        "features": features,
        "action_feature": {"name": "action", "shape": list(action["shape"]), "dtype": "float32",
                           "names": list(action["names"]), "semantics": sem["joints"]},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab serve", description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--path", help="serve this local checkpoint folder instead of the model's Hub repo")
    ap.add_argument("--revision", help="serve this Hub commit of the model repo (lab eval serve passes the frozen one)")
    ap.add_argument("--deployment", help="the deployment name clients ask for (default: from the model's name)")
    ap.add_argument("--rtc", action="store_true", help="also serve guided RTC (SmolVLA, GR00T N1.5); clients choose it "
                                                       "per session with an rtc_enabled experiment")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--listen", default="tcp/127.0.0.1:7447", help="Zenoh endpoint to listen on")
    ap.add_argument("--robot-type", help="declared robot type (default: the dataset rig's), e.g. replay for tests")
    ap.add_argument("--dry-run", action="store_true", help="print the server config and exit")
    a, extra = ap.parse_known_args(argv)
    reg = Registry()
    try:
        if a.rtc and reg.resolve(a.model).policy not in RTC_POLICIES:
            raise RegistryError(f"{a.model}: its policy ({reg.resolve(a.model).policy}) has no RTC")
        cfg = server_config(reg, a.model, a.path, a.device, a.listen, modes=["chunk", "rtc_guided"] if a.rtc else None)
    except RegistryError as e:
        print(f"lab serve: {e}", file=sys.stderr)
        return 1
    if a.robot_type:
        cfg["robot_type"] = a.robot_type
    if a.deployment:
        cfg["deployment"] = a.deployment
    out = Path("outputs/serve") / f"{a.deployment or a.model}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"server config: {out}\n" + yaml.safe_dump(cfg, sort_keys=False) if a.dry_run else f"server config: {out}")
    if a.dry_run:
        return 0
    from lerobot.scripts import lerobot_policy_server
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    from drtc.control.chunk_adapters import install as install_chunk_adapters

    install_chunk_adapters()  # Diffusion Policy chunk serving
    if not a.path:
        from drtc.compat.serve import install_loader, needs_compat, prepare_checkpoint

        hub = cfg["model"]["repo_or_path"]
        if needs_compat(hub, a.revision):  # a model carried over from the minor (old checkpoint format)
            local = prepare_checkpoint(hub, a.revision)
            print(f"old-format checkpoint {hub}: serving the converted copy {local}", flush=True)
            install_loader()
        else:
            local = fetch_checkpoint(hub, a.revision)
            print(f"serving {hub} at {a.revision or 'its latest commit'} from {local}", flush=True)
        cfg["model"]["repo_or_path"] = str(local)
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    sys.argv = ["lerobot-policy-server", f"--config_path={out}", *extra]
    return lerobot_policy_server.main() or 0


if __name__ == "__main__":
    sys.exit(main())
