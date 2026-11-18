# Old-vs-new parity of the kept models

Checks that the models carried over from the minor (`lab/models/kept.yaml`) compute the same actions in the new
setup (LeRobot 0.7 + `lerobot_policy_groot_n15` + `drtc.compat`) as in the old one (tag `minor-final`).

| Step | Venv | Command |
|---|---|---|
| Reference: old setup | `~/Projects/drtc/.venv` (SmolVLA) or `.venv-groot` (GR00T) | `python tests/parity/make_reference.py <model> --device cuda` |
| Check: new setup | `.venv` | `python tests/parity/check.py <model> --device cuda` |
| Noise level: old vs old on two machines | any | `python tests/parity/compare_refs.py <ref A> <ref B>` |
| Old-setup inference speed | old venv | `python tests/parity/bench_old.py <model>` |

- **The reference:** the exact inputs (dataset frames as uint8, state, task), every `torch.randn`/`torch.normal`
  draw and the outputs.
- **The check:** replays the same draws in the same order, so the noise is identical. It also compares the
  preprocessed batch stage by stage.
- **Tolerances:** the old setup's own GPU-to-GPU variation, measured on 2026-11-18.
- **Paths:** `HF_LEROBOT_HOME` (datasets), `DRTC_PARITY_DIR` (results, default `~/Projects/drtc-snapshots/parity`),
  `DRTC_CACHE`.
- Run one GPU job at a time; GR00T N1.5 needs ~7.3 GB.
