# Major project plan

*Working plan for the major phase, from 2026-11-18. It replaces §9–§10 of `lab/PROPOSAL.md` for the major
(the proposal stays as the research framing). Each phase starts only when the previous one is done, so
nothing has to be redone. Items marked **[open]** are decisions still to make.*

| Phase | What | Done when | Needs |
|---|---|---|---|
| 1 | Move to the latest LeRobot, with our code as a separate layer on top | The robot runs through DRTC on the new version; every `lab/bin` tool works; a short test training passes for each kept policy | — |
| 2 | Shortlist the models, each with the recipe others used to make it work | Short list in `models.yaml`, each entry with recipe and source | 1 (research can start during 1) |
| 3 | Faster training for the shortlisted models | Measured time per run, before and after, per model | 1, 2 |
| 4 | Train the shortlist once on trimmed data, then robot trials | We know which models work | 3 |
| 5 | Smoothing test bench, then smoothing methods | Any method can be compared on any model with one command | 4 (bench can be built during 3–4) |
| 6 | DRTC experiments | — | 4, 5 |
| Later | Edge fallback | Only if ACT gives good results | 4 |

**[open]** Major presentation date. It decides how much of Phase 6 fits in.

---

## Status at the end of the 2026-11-18 session (handover)
**Phase 1 is done except for what needs other machines. These items wait until the user says the hardware is
available:**
- **D7:** Zenoh across machines (an H200 server with the laptop client), the GR00T N1.5 kept-model serving test
  on the H200, and the Pi as the client. Needs H200 access, then the robot.
- **Pi:** the new setup's Pi install (`uv sync`), then Zenoh to the laptop or H200 server.
- **Robot-side checks:**
  - `lab check --rig` with everything plugged in;
  - `lab calibrate` against the explicit `so101_*` calibration files;
  - a short `lab record` test (record → trim → push);
  - real `lab run` trials, including the replication conditions with option (b).

Small open items: robot-run logging to the Trackio Space plus a results mirror; DOT and SFP plugins (only if
shortlisted); best-checkpoint selection after training.

**Next session:** Phase 2, the shortlist. The findings so far are in the Phase 2 section below. Then start the
trainings on the H200 (`lab submit` / `lab worker`, or `lab train`; all 8 recipes train there).

## Phase 1: LeRobot upgrade (option c: LeRobot as a dependency, our code on top)

### Where we are
- The repo is a fork of `jackvial/drtc`, built on LeRobot from 2025-12-31 (version 0.4.3).
- Upstream is at **v0.7.0** (tagged 2026-11-05). `main` reads 0.7.1 and is 788 commits ahead of our base.
- Our changes since that base:
  - 267 commits
  - about 100 files outside `lab/`
  - about 19k lines, mostly the DRTC client/server and our policy ports

### Target layout
LeRobot is no longer copied into the repo; it is installed as a pinned package. A future upgrade means
bumping the pin and fixing what broke in our layer.

```
pyproject.toml              lerobot[...] == 0.7.0 (pinned), plus our packages
src/drtc/                   our code (one Python package; the `lab` command is drtc.cli)
  control/                  remote control on upstream remote_inference with Jack Vial's DRTC ideas
                            (freshest-wins merging, latency estimation, trigger cooldown, RTC), fault
                            injection, smoothing A/B/C
  registry/                 loads and checks lab/ configs, expands recipes and sweeps, lineage
  jobs/                     job files, the per-host worker, submission over SSH
  logging/                  wandb runs and artifacts, Hub mirror of results
  compat/                   loader for old LoRA adapters, importers for the old models.yaml / rig.yaml / index.csv
src/lerobot_policy_dot/     plugins, found automatically by LeRobot (lerobot_policy_* prefix)
src/lerobot_policy_sfp/
src/lerobot_policy_groot_n15/   GR00T N1.5 kept as a plugin (decision 1b)
lab/                        configs and docs (the registry, see "Lab system" below); bin/ keeps the old
                            command names as aliases for `lab ...`
tests/
```

Git and folders:
- The end-of-minor state is tag `minor-final`: the `drtc` branch plus the uncommitted rig edits. It is the code
  the minor results and checkpoints came from.
- Untracked files and the exact venv package lists are in `~/Projects/drtc-snapshots/minor-final-2026-11-18/`.
- New work is on branch `major`, in a separate worktree `~/Projects/drtc-major` with its own venv.
  `~/Projects/drtc` (branch `drtc`) and its venvs stay as they are and keep working for the robot until Phase 1
  passes its checks. Then `major` is merged into `drtc`.
- `lab/` history is kept.

### Lab system (accepted 2026-11-18)
Replaces the old `lab/bin` + `lab/tools` + single `models.yaml`/`rig.yaml`. It is built in Phase 1 in place of
porting the old tools.

```
  laptop (control + robot side)                      compute hosts (H200 now; A100 / more later)
  ┌───────────────────────────────┐                 ┌──────────────────────────────────────┐
  │ lab CLI: record, submit,      │   ssh           │ lab worker  (user starts it, tmux)   │
  │          run, status          │ ──────────────▶ │  ├─ job queue (local file)           │
  │ registry (git): rigs, hosts,  │                 │  ├─ checks out the job's git commit  │
  │   datasets, recipes, models,  │                 │  ├─ runs train / eval / bench / serve│
  │   protocols                   │                 │  └─ waits for free GPU memory        │
  │ Zenoh router  ◀───────────────┼── policy server ┤                                      │
  └──────────────┬────────────────┘                 └──────────────────────────────────────┘
                 │ Zenoh                 all hosts: datasets + checkpoints via HF Hub,
        Pi / laptop robot client         logs + lineage to wandb, results mirrored to a Hub repo
```

| Part | Design |
|---|---|
| Registry | YAML files per type in `lab/`: `rigs/`, `hosts.yaml`, `datasets/`, `recipes/`, `models/`, `protocols/`, `experiments/`. A checker validates references and names. Lineage (rig → dataset → derived dataset → model → evaluation) is recorded in wandb artifacts by every run |
| Recipes | One file per policy recipe, with its source (who made it work, on what robot). A model = dataset × recipe × base checkpoint + overrides. A sweep entry expands to N models. Our own expander on draccus, not Hydra |
| Rig | Versioned rig profiles. Every dataset records its rig version and calibration fingerprint; a run refuses a model trained for other cameras |
| Hosts and compute | `hosts.yaml`: roles (record, client, serve, train), GPUs, SSH/container, cache paths. Adding compute = one entry + `lab worker` there |
| Jobs | A job file holds the kind (train, eval, bench, serve), model, GPU memory and count needed, git commit and env lock. The worker checks out that commit, runs jobs when enough GPU memory is free (several side by side), multi-GPU via upstream `--parallelism.*`. After training: best checkpoint by validation, push, offline eval, benchmark. Other backends (SLURM, cloud) can replace the worker later with the same job files |
| H200 rule | The user starts workers and submits jobs (`lab submit <job> --host h200`). Claude writes job files and never submits them |
| Serving | Policy servers on any host connect to a Zenoh router on the laptop; the client asks for a deployment by name. Replaces `lab/bin/tunnel`; several servers at once |
| Collection | Leader-arm recording; every robot trial also saved as a dataset. After recording, automatically: checks (dropped frames, camera rate, joint limits), autotrim, stats, per-episode quality report, Hub push with a dataset card. DAgger after Phase 4 |
| Testing | Tiers: code tests → offline evaluation → replay bench → robot protocol (fixed start positions in random order, alternating conditions, set trial count, success and failure reason per trial) |
| Logging | wandb for everything (training, offline eval, robot trials, per-tick logs), mirrored to a private Hub dataset repo |
| CLI | One `lab` command, installed as a package, the same on laptop, Pi, H200 and Windows. `lab status`: jobs, latest models, what each still needs |
| Compatibility | Old `lab/bin/*` names stay as aliases. An import tool converts `models.yaml`, `rig.yaml`, `results/index.csv` |

### What happens to each of our changes

| Our change | Upstream 0.7 | Plan |
|---|---|---|
| DRTC client/server, LWW registers, latency estimation, RTC guidance, fault injection, metrics, simulation (`async_inference/`, ~8k lines) | `async_inference` removed; new Zenoh-based `remote_inference` with chunk alignment, blending, RTC, refill timing, position hold | Rebuild on upstream `remote_inference` with DRTC's ideas (decision 1a). The original DRTC stays runnable from `minor-final` for comparison |
| gRPC proto changes (`transport/`) | `transport/services.proto` still exists | Not needed: upstream's transport is used (Zenoh) |
| Training: held-out validation, LoRA, clean stop, progress line, uint8 frames in shared memory | Has held-out eval split, PEFT, accelerate (bf16, FSDP2/DDP), GPU-compiled image transforms | Use upstream for validation, LoRA, precision and parallelism. Keep only clean stop, progress line and `val_action_mse`, if still needed, in a thin wrapper around `lerobot-train`, not a copy of it |
| Our own LoRA (`policies/lora.py`) | PEFT built in | New training uses upstream PEFT (`--peft.*`). Keep our loader in `src/drtc/` so old adapters (e.g. the working GR00T LoRA) still load |
| π0-FAST port | In-tree | Drop ours |
| π0/π0.5 `embed_tokens` tie fix | Fixed (copies `lm_head` into `embed_tokens` when loading) | Drop |
| DOT, SFP | Not in LeRobot | Plugins `lerobot_policy_dot`, `lerobot_policy_sfp` |
| GR00T N1.5 (with our RTC for method B) | Only N1.7 (Qwen3-VL, 40-action chunks, RTC) | Plugin `lerobot_policy_groot_n15`, ported from our code (decision 1b) |
| ACT, Diffusion small edits (validation targets, DRTC server queue, frame resize) | Diffusion is not supported by upstream remote inference (needs observation history) | Drop; redo only if ACT/Diffusion are shortlisted |
| Jack's camera, record and teleop speed patches | Camera has a background reading thread with `async_read` and `read_latest` | Drop; measure the camera rate in 1.8, and patch only if it is too slow |
| Dataset batched video encoding (Jack) | New `dataset_writer` with batched encoding; dataset format still v3.0 | Drop; check that v1/v1_trim load and that recording works in 1.8 |

### Steps
- [x] 1.1 Snapshot: tag `minor-final`, archive and venv package lists in `~/Projects/drtc-snapshots/` (2026-11-18). To do: push the tag (`git push origin drtc minor-final`)
- [x] 1.2 Branch `major` in worktree `~/Projects/drtc-major` (created 2026-11-18); `pyproject.toml` pinned to lerobot 0.7.0. Check how many venvs we need now that every policy uses transformers 5.22 (today: separate venvs per model family)
  - Done 2026-11-18. The vendored fork was removed from `major`. `pyproject.toml` (package `drtc-lab`) pins
    `lerobot==0.7.0`; `uv.lock` is committed.
  - Base install is the robot side (Pi); extra `gpu` is training and serving.
  - One venv covers SmolVLA, π0/π0.5/π0-FAST, X-VLA and GR00T N1.7 (transformers 5.22.2, torch 2.14.1+cu130). It
    works on the 4060. Still open: whether GR00T N1.5's Eagle code runs on transformers 5 (step 1.5).
  - **The H200 cannot be changed** (no driver or system changes, not even cleaning caches). Its driver 580
    supports CUDA up to 13.0, so torch is pinned to **2.14.1+cu130** (the newest CUDA 13.0 build) for Linux x86_64.
    It is tested on the laptop (GPU, bf16, video decoding, all tests). The new setup uses its own dataset cache
    folder on the servers, so old caches there are never read.
- [x] 1.3 Finish the change table above (2026-11-18)
- [ ] 1.4 Build `src/drtc/` on upstream `remote_inference` (DRTC's scheduling ideas, our fault injection and logging) and run it against a test policy on the laptop
  - [x] D1: replay robot plugin `lerobot_robot_replay` (`--robot.type=replay`). It plays a recorded episode
    (real frames, 18 ms per observation); its state follows commands, it logs them, and it declares position hold.
  - [x] D2: `lab serve <model>`: the deployment comes from the registry. Joint semantics follow the dataset's units
    (`so101-range100-…` for v0/v1, `so101-degrees-…` for new data); camera semantics follow the rig's roles.
    Upstream's server and client ran end to end on the laptop: 29.8 Hz, 0 starved ticks.
  - [x] D3: `lab run <experiment> [--model] [--robot rig|replay]` plus `lab results`.
    - Per run: `results/<exp>/<model>/<run_id>/` with config.yaml, ticks.csv (per control tick), faults.csv,
      run.json (outcome and fault reason), rollout.log and the robot's commands; plus `results/index.csv`.
    - Failed runs are recorded too.
  - [x] D4 (option a): `--inference.type=drtc`. It extends upstream's remote engine and changes only the request
    threshold (DRTC: queue ≤ H − s_min) and the latency estimate (jk / max_last_10 / fixed / upstream).
    Replay test: the DRTC scheduler requested 32 chunks (queue ≥ 79), upstream 8 (queue down to 14); 0 stalls each.
  - [x] D5: fault injection at the transport (observations published, replies received), for all 10 replication
    conditions. **Replay results with the test ACT** (3.3 s chunks, ~45 ms turnaround, DRTC trigger):

    | Condition | Outcome on upstream's runtime |
    |---|---|
    | baseline, dup_action, spikes_jk, spikes_max_last_10 | 25 s, 0 stalls (duplicate replies rejected; 0.5 s and 1 s spikes absorbed) |
    | reorder_action | 25 s, 5 short stalls |
    | drop_obs, drop_action, disconnect, reorder_obs, mixture | **terminal fault after 6–10 s**: "action starvation grace expired" |
    | dup_obs | **terminal fault after 3 s**: the server rejects the second copy as "Duplicate or completed request", and the client treats it as a protocol error |

    **Why the lost-message runs end:** with one request in flight, a lost message blocks requests until its 5 s
    deadline. The queue runs out, and upstream's starvation recovery sends one fresh request with 1 s of grace,
    which falls in the next fault window. **This is the measured case for option (b)** (DRTC's cooldown
    re-request).
  - [x] D4b (decided 2026-11-18, built): the cooldown re-request. A request unanswered for its cooldown (latency
    estimate in steps, capped at H/2, + ε; s_min + ε before the first measurement) is given up, and a fresh
    observation is sent. Upstream's server rules are kept:
    - a `BUSY` reply backs off one action interval;
    - a `STALE` "duplicate" reply for the request in flight is ignored;
    - the aligned-mode anchor is cleared after a give-up.

    Every recovery is logged (`recovery.csv`). **Validated on the quiet laptop (2026-11-18): all 11
    replication conditions complete their full 25 s** (replay robot, test ACT; at most one 0.03 s stall). Without
    (b), 6 of them ended in a terminal fault.

    | Condition | Upstream runtime alone | With (b) |
    |---|---|---|
    | drop_obs / drop_action / disconnect | terminal after ~6 s | 25 s, 1 stall of 0.03 s, 64–70 re-sends |
    | reorder_obs | terminal after ~10 s | 25 s, 0 stalls (12 re-sends, 1 BUSY back-off) |
    | dup_obs | terminal after 3 s (protocol error) | 25 s, 0 stalls (9 duplicates ignored) |
    | mixture | terminal after ~10 s | 25 s, 0 stalls (37 re-sends, 2 duplicates ignored) |
    | baseline, dup_action, reorder_action, spikes ×2 | 25 s | 25 s, 0 stalls |
  - [x] D6 (decided 2026-11-18: only this for now): the Butterworth action filter, ported unchanged (order 2,
    3 Hz, gain 1.4; identical to the old filter to 1e-12, `tests/test_action_filter.py`). It is applied to every
    action before the robot and reset when motion is invalidated. Smoothing A/B/C: decided later.
- [x] Serving the kept models (`drtc.compat.serve`, used by `lab serve`, 2026-11-18). An old-format checkpoint is
  converted to a local copy (upgraded config, mapped processor step names). Upstream's server builds the policy with
  our old-LoRA loader. `smolvla_cubebox_v0_lora_raw` was served and run on the replay robot: 15 s, 0 stalls,
  ~280 ms turnaround on the 4060. GR00T N1.5 uses the same path; its serving test is on the H200 (D7).
- [ ] 1.5 Policy plugins (DOT, SFP, GR00T N1.5)
  - [x] GR00T N1.5 plugin `lerobot_policy_groot_n15` (type `groot_n15`, steps `groot_n15_*`), ported to
    transformers 5. Changes:
    - kw-only config dataclass
    - `post_init`
    - lazy Beta distribution (meta-device init)
    - image-processor imports moved
    - its own Eagle cache folder, so the old setup's stays untouched
  - [x] Loading the kept models (`drtc.compat`): our old LoRA adapter format, config upgrade, N1.5 processor step
    names mapped
  - [x] **Old-vs-new parity of both kept models (2026-11-18)** (`tests/parity`; results in
    `~/Projects/drtc-snapshots/parity/`):
    - **Method:** same inputs and the same recorded random draws through the old setup (`minor-final` venvs)
      and the new one.
    - **Preprocessing:** bit-identical for both, including GR00T's Eagle pixel values and tokens.
    - **Actions:** they differ only within the old setup's own GPU-to-GPU noise. That noise was measured as the
      same old code on the laptop vs the H200 (`compare_refs.py`): GR00T 0.31°, SmolVLA 2.84° (largest at the
      rest pose; 50-step chunks amplify rounding). Old vs new:
      - GR00T: 0.29° on the H200
      - SmolVLA: 1.86° on the H200, 1.06° on the laptop
    - **Laptop GR00T N1.5 inference:** 613 ms median (636 mean, 765 p90), 7.27 GB peak. Too slow to serve the
      robot from the laptop (a 16-action chunk is 533 ms), so it is served from the H200. This is the baseline for
      the efficiency work.
  - [ ] DOT, SFP plugins (only if shortlisted in Phase 2)
- [ ] 1.6 Training wrapper (clean stop, progress line), recipes → 0.7 `lerobot-train` flags
  - [x] `lab train <model>` (`drtc.train`, 2026-11-18). It builds the 0.7 `lerobot-train` command from the
    registry:
    - the dataset pinned to its revision; a stale local copy is refused;
    - our seeded held-out split, mapped onto 0.7's eval split (training episodes first, held-out last);
    - epochs → steps, and ~20 held-out evals per run;
    - recipe flags checked against the policy config;
    - camera roles → `rename_map`.

    It adds a clean stop (Ctrl-C / SIGTERM / `lab stop`: checkpoint at the next step, pushed with training
    state, `--resume` anywhere) and pushes the final or a chosen checkpoint (`--push <step>`). 0.7's own
    progress bar shows the time left.
  - [x] Tested on the laptop with ACT: 200 steps with held-out eval; stop at 55, resume to 200 (the same losses
    as an uninterrupted run).
  - [x] `lab submit` / `lab worker` / `lab jobs` (`drtc.jobs`). A job pins the pushed commit, `uv.lock` and its
    GPU memory need; the worker starts it only then, one at a time by default. Tested locally.
  - [ ] Dry runs: the commands build for all 9 trainable planned models. DOT and SFP need plugins (only if
    shortlisted).
  - [ ] **[open]** Logging: 0.7 has no wandb, only Trackio. Choose a Trackio Space, wandb through our wrapper,
    or both. Until then runs log to local Trackio, project `drtc-lab`.
  - [x] **All 8 planned recipes train on LeRobot 0.7 on the H200** (2026-11-18, `tests/smoke/train.sh`: 30
    steps + one held-out eval each, no checkpoints):

    | Model | Held-out loss | Peak GPU | Time |
    |---|---|---|---|
    | ACT | 0.81 | 44.6 GB | 195 s |
    | Diffusion | 1.01 | 15.8 GB | 175 s |
    | SmolVLA | 0.46 | 14.7 GB | 162 s |
    | X-VLA | 0.94 | 32.0 GB | 170 s |
    | π0-FAST | 7.38 (token loss) | 37.6 GB | 390 s |
    | π0 | 0.24 | 43.8 GB | 466 s |
    | π0.5 | 0.26 | 45.5 GB | 344 s |
    | GR00T N1.5 (plugin) | 0.22 | 20.5 GB | 250 s |

    - The old base-revision pins of π0/π0.5/π0-FAST work on 0.7.
    - The worker's GPU estimates are these peaks plus ~10%.
    - The GR00T plugin had to become its own package (`plugins/lerobot_policy_groot_n15`), because LeRobot finds
      plugins by distribution name.
    - **For Phase 3:** in the H200 container, torchcodec cannot decode the videos and LeRobot falls back to PyAV,
      so data loading is slower.
  - [ ] After training (later): pick the best checkpoint by held-out loss, then offline eval and benchmark.
- [ ] 1.7 Lab system (see above), in this order:
  - [x] registry + checker + import of the old `models.yaml` / `rig.yaml` (2026-11-18)
    - `src/drtc/registry/`, files in `lab/rigs`, `lab/hosts.yaml`, `lab/datasets`, `lab/recipes`, `lab/models`
    - The import produced 2 rigs, 4 hosts, 8 datasets, 34 recipes and 46 models
    - Parity tests (`tests/test_registry_parity.py`, 82 tests):
      - every old `models.yaml` entry comes out identical
      - every experiment's composed run config comes out identical (Linux and Windows rigs)
      - frozen old files are in `tests/fixtures/legacy/`
    - Recipes take camera roles (wrist/scene), so `rename_map` follows the dataset's rig
    - `lab list` and `lab check [--rig] [--data]` work: registry, devices, calibration (warns about LeRobot's
      default folder), episode counts
  - [x] clean start: renames, pinned datasets, registry reset, `lab check --data --online` (2026-11-18)
  - [x] `lab trim` (`drtc.data.trim`, 2026-11-18): the minor's autotrim detection, unchanged, with the crops
    applied on LeRobot 0.7 into a new folder.
    - **Method:** no re-encoding (each episode's video window moves), mp4 files hardlinked, numeric stats
      recomputed with LeRobot's functions.
    - **Parity:** on `so101_cubebox_v1_raw` (`tests/test_trim_parity.py`) it gives the same noise floor, the same
      110 crops, the same data rows and episode windows as the old setup. The per-episode stats are exact; the old
      editor's std was float32-imprecise, up to ~10% on the wrist roll.
    - **Generic:** on `so101_cubebox_v0_raw` it measures its own noise (0.0058) and cuts 19%.
    - **Dataset quantiles (decided 2026-11-18): LeRobot's current convention** (the installed LeRobot's
      `aggregate_stats`, an envelope: q01 −50.6 on joint 1 of v1). LeRobot 0.4.3 used the frame-weighted mean of
      the episodes' quantiles (−22.9, which the current v1 keeps); the true value is −46.5. `--quantiles
      weighted_mean` only reproduces old datasets.
  - [x] `lab record` (`drtc.data.record`): records `<name>_raw` with 0.7's `lerobot-record` (its terminal key
    listener works on Wayland/SSH), checks it, trims into `<name>`, pushes both and writes pinned registry entries.
    0.7's RecordConfig parses the generated command, with the right calibration folders and joint units. Not yet
    run with the robot.
  - **Joint units (decided 2026-11-18): LeRobot's current convention, degrees, for new recordings.** LeRobot
    0.7's SO101 default is `use_degrees: true`.
    - Each dataset entry states its own units: v1, v1_raw and v0_raw are −100..100.
    - A model runs the robot in its dataset's units (control system, D).
    - The user accepts retraining for new data; v2+ in degrees is not mixable with v1.
  - [x] `lab` CLI with the old names as aliases (2026-11-18).
    - **Aliases:** list, train, stop, run, record, preflight → `lab check --rig`, analyze → `lab results`,
      calibrate → `lab calibrate` (explicit calibration file, backup), server → `lab serve <model>`.
    - **Kept as is:** phonecam and session (they read the rig through `lab rig <key>`).
    - **Not ported (they say so):** bench, eval-offline, logstats, ablate-cameras, tunnel, edit-dataset, venv,
      h200-setup.
    - **Removed from `major`:** the old `lab/tools`, `models.yaml` and `rig.yaml`. They stay in `minor-final` and
      the test fixtures.
    - Hosts can declare their `hostname`, so `lab` knows where it runs (the laptop is `arshjeevs`).
  - [x] `lab import-results`: the minor's 9 robot runs plus its offline/benchmark folders were copied to
    `results/minor/` under the new model names.
  - [ ] jobs + worker + `lab submit` (train, eval, bench, serve)
  - [ ] logging to the Trackio Space for robot runs and a Hub mirror of results (training already logs to Trackio); the results import is done
  - [ ] Zenoh serving across hosts (replaces `tunnel`), Pi and H200 setup
    - 2026-11-18, for the parity check, done by Claude with the user's permission (no deletions):
      - `~/drtc_major` was cloned on the H200, with `.server.env` and `.venv` from `uv.lock` (6.8 GB, `uv` 0.10.4
        in the container).
      - torch 2.14.1+cu130 runs on the H200 NVL with driver 580.
      - The Hugging Face model cache is shared with `drtc_lab` (files only added).
      - `/home` is 97% full: 94 GB free afterwards.
      - Once the migration is done, the user will delete the whole DRTC space there and set up fresh.
    - H200 (cannot be changed; see `lab/H200.md` for the existing setup): the new setup lives in `~/drtc_major`
      (branch `major`), next to the old `~/drtc_lab`, inside the same `drtc_lab` container (the guide's image, which
      already mounts all of `/home/jayaraj`). It has its own `.venv` from `uv.lock`.
      - Shared with the old setup: the Hugging Face model cache (`drtc_lab/.cache/huggingface`, so no second copy
        of large checkpoints; `/home` is ~88% full) and `drtc_lab/.secrets.env`.
      - Its own dataset folder (`HF_LEROBOT_HOME=drtc_major/.cache/lerobot`), so the stale untrimmed v1 cached by
        the old setup is never read.
      - Nothing in `~/drtc_lab` is modified.
- [ ] 1.8 Checks:
  - `so101_cubebox_v1_trim` loads
  - a short training per kept policy on the laptop and the H200
  - a recorded test episode
  - the robot runs through DRTC (laptop, then Pi)
  - everything in the compatibility list below

### Clean start (2026-11-18)
The new setup starts clean instead of carrying the whole old one. Hub and local renames were run by Claude with
the user's approval; the log is in `~/Projects/drtc-snapshots/minor-final-2026-11-18/rename_hub.log`, and the
name table is in `lab/renames.yaml`.

**Datasets.** The trimmed copy is now v1. Untrimmed recordings keep a `_raw` name, so a name without `_raw` is
always trimmed.

| Old | New | |
|---|---|---|
| `so101_cubebox_v1_trim` | `so101_cubebox_v1` | 110 episodes, 25,067 frames |
| `so101_cubebox_v1` | `so101_cubebox_v1_raw` | untrimmed, 29,525 frames |
| `train_new_1` | `so101_cubebox_v0_raw` | earlier two-scene-camera setup, 50 episodes |
| `so101_cubebox_v1_scene` | deleted | regenerate from v1 when needed |

**Models.** Only the two that worked are kept: `groot_cubebox_v1_lora_raw` (was `groot_cubebox_v1_lora`) and
`smolvla_cubebox_v0_lora_raw` (was `smolvla_train_new_1_lora`). Jack's `jack_smolvla_cube` stays as an external
model for the replication experiments.
- Old models on raw v1 were renamed `<name>_raw` on the Hub, which frees the v1 names. They are not in the
  registry.
- The three `*_cubebox_v1_full_trim` models are untouched on the Hub and not in the registry.
- New models are `<policy>_cubebox_v1_full` (planned).

**Pins.** Every dataset entry pins its Hub revision (the commit of the v3.0 tag), episode count and frame count.
`lab check --data [--online]` and every job refuse a copy that doesn't match. That catches stale caches under
reused names, such as the untrimmed v1 still cached on the servers.

**wandb project.** `drtc-lab` (`lab/lab.yaml`). `drtc` stays as the minor's project.

**Recording from now on.** `lab record` pipeline:
1. Record into `<name>_raw`.
2. Run the checks.
3. Autotrim, saved as `<name>` (the trim report and parameters go in the dataset card and registry).
4. Push both, public.

Autotrim is rewritten on LeRobot 0.7's dataset tools, without the dataset editor. A test checks it makes the same
cuts as the old version on `so101_cubebox_v1_raw`. It must be done before the next recording.

### Compatibility (required)
Everything carried over must work as in `minor-final`:
- **Kept models:** the two kept models load and give identical outputs.
  - Their entries equal the old ones apart from the renamed repos (tests pass).
  - The old-vs-new output check runs in 1.8.
- **Experiments:** every experiment composes the same run config (tests pass, Linux and Windows).
  - The 10 fault conditions exist on the new system.
- **Calibration:** the files work without recalibrating.
- **Rig:** the same rig (cameras, ports, Pi and laptop).
- **Commands:** the old `lab/bin/*` names stay as aliases.
- **Results:** `results/index.csv` imports with the renames applied.

Findings so far (2026-11-18):
- **Calibration:** LeRobot 0.7 looks for calibration in `calibration/robots/so_follower/` and
  `teleoperators/so_leader/`, not `so101_*`. Both folders already exist on the laptop, but hold **old calibrations
  from 2026-11-05**; the ones in use are `so101_*` from 2026-11-10. Relying on the default would silently use the
  wrong calibration. The rig profile therefore names the calibration files explicitly, the lab layer always
  passes `calibration_dir`, and a fingerprint check catches mismatches.
- **Datasets:** `so101_cubebox_v1` (29,525 frames) and `so101_cubebox_v1_trim` (25,067 frames, trim saved) load in
  0.7, video decoding included.

### Decisions (2026-11-18)
- **1a: our own system on upstream `remote_inference`**, using Jack Vial's DRTC ideas (ii). We are building our own
  rather than replicating DRTC exactly. The original DRTC (`minor-final`) stays runnable, so the two can be
  compared and the better one kept.
- **1b: GR00T N1.5 kept** as a plugin (i).
- **1c: full fine-tunes on trimmed data wait** until Phase 1 is done (i).

## Phase 2: Shortlist
**Started 2026-11-18, paused to finish Phase 1. Findings so far:**
- **ArmnetBench v0.1** (arXiv 2607.24481), the strongest evidence.
  - **Setup:** SO-101 arms, 3 cameras, 20 fps, **50 teleop demos per task**, trained per task, 12 tasks, 2,518
    real rollouts.
  - **Overall strict success:** π0.5 **47.6%**, π0 35.1%, GR00T N1.7 29.4%, Diffusion 26.7%, ACT 19.2%,
    MolmoAct 2 18.9%, SmolVLA 15.0%.
  - **Closest tasks to ours:** transfer_cube (π0.5 86%, π0 47%, GR00T N1.7 47%, ACT 0%) and eye_drops_to_basket
    (π0 70%, π0.5 67%, GR00T N1.7 67%, ACT 63%).
  - **Recipes:** read on 2026-11-18, see "ArmnetBench recipes" below.
- **Project-IRA π0.5 on SO-101** (Hub model cards): `lerobot/pi05_base`, batch 32, 30k steps, checkpoints every 2k.
  The best checkpoints were at **8k–10k steps**; after ~20k they were unusable on the robot (overfitting, like
  our minor's validation curves).
- **SmolVLA:** the LeRobot docs recommend ~50 episodes and 20k steps. Community results are weak: 0–30% with
  ~20–60 episodes; 89% *grasping* with 340 episodes.
- **GR00T:** N1.5 and N1.7 SO-101 community cards (e.g. raymondlo84-nvidia, 10k steps, batch 8, lr 1e-4) have no
  success rates. NVIDIA's N1.7 path: 16-step absolute action chunks, ≥40 GB to fine-tune.
- **MolmoAct 2:** 56.7% on SO-100/101 in its paper (pretrained on 38k community SO-100/101 episodes), but 18.9% in
  ArmnetBench.
- **X-VLA, EVO1, LaWAM:** no SO-101 success rates found.
- **SO-101 failure and recovery benchmark** (arXiv 2606.08881): 100 demos per task, 4 tasks, 20 trials each.
  - **Average success:** π0.5 **56.25%**, Wall-X 51.25%, ACT 33.75%, SmolVLA 32.5%.
  - **Multi-object packing into a box:** π0.5 55%, ACT 10%, SmolVLA 10%.
  - **Recovery after a failure:** π0.5 30.8%, ACT 6.5%.
  - **Failures:** execution failures (grasp instability, repetition loops) dominate for all models; π0.5 has the
    fewest state mismatches.
  - No recipes in the paper.
- **"Does validation loss predict real-robot success?"** (arXiv 2611.00283): SO-101, a green ball among
  distractors into a basket, 225 demos, budgets 12 to 200, LeRobot 0.6.2 default recipes. Recipes:
  - ACT: 100k steps;
  - Diffusion: 100k steps;
  - SmolVLA: 20k steps, from pretrained;
  - π0.5: action expert only, 6k steps.

  Findings:
  - **Success at 200 episodes:** π0.5 **73%**, SmolVLA 38% (71% of the runs where it started moving); ACT 37/40
    at 100 episodes.
  - **"Never started" (the arm stays at rest) is a major failure mode:** untrimmed Diffusion 96% of rollouts, **trimmed
    Diffusion 9%**; SmolVLA 47%. This is direct support for trimming every dataset.
  - **Validation loss is a weak guide:**
    - held-out *action error* tracks success within a policy, but no better than the number of episodes;
    - held-out *loss* rises during training for diffusion/flow-matching models while action error falls, so it is
      not a stopping rule for them;
    - final checkpoints were within 5.3% of the best;
    - **use held-out action error to decide when to stop, not to choose models; judge only on the robot**, and
      report "never started" separately from success.
  - **Plateaus:** ACT at 20k–40k steps, SmolVLA at 2k–10k, Diffusion at 10k–30k (with ≤50 episodes).
- **Tentative shortlist (for the user to decide):** π0.5, π0, GR00T N1.7 (and N1.5 for continuity), Diffusion as
  the strongest small model; ACT for the edge fallback. SmolVLA and MolmoAct 2 rank low in independent tests.

### ArmnetBench recipes (read 2026-11-18)
**Where they are.** The HF collection holds only the three datasets. The checkpoint links are in the dataset
README (`armnet/armnetbench_v01_lerobot_so101`, `..._bimanual_so101`), one checkpoint per (task, policy).
Each rollout's `policy_repo_id` and `success` are in `meta/episodes`. The formats differ by family:
- ACT and Diffusion: LeRobot `train_config.json`;
- GR00T N1.7: Isaac-GR00T `experiment_cfg/conf.yaml`;
- π0 and π0.5: openpi JAX checkpoints with no config; the configs are in GitHub `pravsels/openpi`, branch
  `task/v50-retrain`, `src/openpi/training/config.py` (`_make_v50`, `LeRobotSO101DataConfig`).

**Their data:** 50 demos per task, 20 fps, front/top 576×1024 and wrist 720×1280 cameras, ~9.1k frames per task.
Ours: 110 episodes, 30 fps, 25,067 frames.

**Success by embodiment,** recomputed from `meta/episodes`. Single-arm is our setting; transfer_cube is bimanual.

| | π0.5 | π0 | GR00T N1.7 | Diffusion | ACT | MolmoAct 2 | SmolVLA |
|---|---|---|---|---|---|---|---|
| Single-arm, 8 tasks | **45.4%** | 36.4% | 34.6% | 22.1% | 20.3% | 14.8% | 13.6% |
| Bimanual, 4 tasks | **52.1%** | 32.5% | 19.2% | 35.8% | 2.0% | 8.8% | 12.0% |
| eye_drops_to_basket (single) | 67% | 70% | 67% | 43% | 63% | 33% | 23% |

**Recipes:**

| | π0.5 / π0 (identical except base) | GR00T N1.7 | ACT | Diffusion |
|---|---|---|---|---|
| Trainer | openpi (JAX), fork with per-timestep action norm | Isaac-GR00T, 4× RTX 5090 DDP | LeRobot | LeRobot |
| Base | `pi05_base` / `pi0_base` | `nvidia/GR00T-N1.7-3B` | ResNet18 ImageNet | ResNet18 ImageNet |
| Trained | everything (openpi default, no LoRA in params) | projector + DiT + VLLN; LLM and vision frozen | everything | everything |
| Steps × batch | **25k × 32** (2 GPU × 16) | **20k × 32** | **20k × 16** (10k + 10k warm restart) | 20k × 32 on 5 single-arm tasks; 80k × 8 on 3; several resumed from an earlier checkpoint, so these are not totals |
| ≈ epochs of their data | ~88 | ~70 | ~35 | ~70 |
| LR | 2.5e-5 peak, 1k warmup, cosine to 2.5e-6 over 100k (so nearly flat over 25k); EMA 0.999 | 1e-4 cosine, 5% warmup, wd 1e-5, grad clip 1 | 1e-5 (backbone too), wd 1e-4, constant | 1e-4, diffusers schedule, 500 warmup, wd 1e-6 |
| Chunk | 30 predicted (1.5 s), executed per their client | 16 used (0.8 s) of the 40-step head | 30 / 30 executed | horizon 64, 32 executed, 2 obs steps |
| Actions | **delta joints, absolute gripper**; per-timestep action normalization (per-step mean/std/q01/q99) | **relative arm, absolute gripper**; percentile stats | absolute, mean/std | absolute |
| Cameras | front + wrist (top dropped) | front, wrist, top | top, wrist, front at 480×640 | all three at full resolution, no crop, one encoder per camera |
| Other | default prompt per task | colour jitter, crop 244→224, state dropout 0.2 | image augmentation on (jitter, sharpness, small affine), KL 10 | DDPM 50 steps (DDIM 20 on 2 tasks), down_dims 512/1024/2048, no augmentation |

**Caveats:**
- The Diffusion recipe is not uniform. `cable_clip_remote_v2_diffusion`'s config names the block_stack dataset,
  which looks like an upload mistake.
- MolmoAct 2: the `quantile_norm_fix_25k` checkpoints are what the README links. The dataset has 60 rollouts per
  task because the earlier `_20k` (before the fix) was also evaluated, so the 14.8% mixes both. The fixed
  checkpoints alone: 19.6% single-arm, 63% on eye_drops_to_basket.

**The user's view (2026-11-18):** these are not the recipes to follow; they are reference only. The user is
looking for the right source. The training configs are confirmed after that research.

### Shortlist (decided 2026-11-18)
**Models:**
- SmolVLA
- GR00T N1.5
- GR00T N1.7
- π0.5
- MolmoAct 2
- ACT, also the edge fallback
- Diffusion Policy

**Retired:** π0, π0-FAST, X-VLA, DOT, SFP (`status: retired` in `lab/models/cubebox_v1.yaml`).

**DRTC compatibility.** DRTC's own parts work for any policy that LeRobot 0.7's policy server can serve as action
chunks:
- the trigger (request when the queue is ≤ H − s_min);
- the latency estimators;
- the cooldown re-request.

RTC (inpainting the next chunk onto the actions still queued) is extra. It needs a flow-matching policy that
declares `supports_rtc()`.

| Model | Chunk H (current recipe) | Served by the 0.7 server | RTC in 0.7 | To do |
|---|---|---|---|---|
| SmolVLA | 50 | yes (kept model run on the replay robot, ~280 ms on the 4060) | yes | — |
| GR00T N1.5 (our plugin) | 16 | yes (same path as the kept model) | **yes since 2026-11-18** (see below) | 613 ms on the laptop vs a 533 ms chunk, so serve from the H200. H − s_min = 8 steps is tight |
| GR00T N1.7 (upstream `groot`) | 40 | not tested | yes | smoke train + serve on the H200. Recipe fixed: `policy: groot` + `policy.base_model_path` |
| π0.5 | 50 | yes (smoke-trained) | yes | — |
| MolmoAct 2 | 30 | not tested | only with `inference_action_mode: continuous` (its default) | smoke train + serve on the H200 |
| ACT | 100 | yes (serve check on the laptop: 27 ms per chunk) | no (not flow matching): chunk switching only | temporal ensembling must stay off (the server refuses it) |
| Diffusion | 16 executed | yes once trained with `n_obs_steps: 1` (now in `diffusion_full`) | no | — |

**Done on 2026-11-18 for the test runs:**
- **GR00T N1.5 plugin, guided RTC on 0.7.**
  - `rtc_config` config field; the server sets it for `rtc_guided`.
  - An `RTCProcessor` built at init, `supports_rtc()`, and the 0.7 `denoise_step` call.
  - The time and sign mapping is upstream's NOISE_AT_ZERO one.
  - Only the robot's 6 action dims are guided; GR00T's 26 padded dims keep their unguided velocity.
  - Tests in `tests/test_groot_n15_rtc.py`: the prefix is pulled in, nothing changes past the execution horizon,
    the padding is untouched, the server's RTC call interface is accepted, and the config round-trips.
  - The old setup's smoothness guidance (method C) is not ported; it belongs to Phase 5.
- **Recipes and registry:**
  - `groot_n17_nvidia` now uses `policy: groot` and `policy.base_model_path`.
  - `diffusion_full` now uses `n_obs_steps: 1` and `drop_n_last_frames: 16`.
  - GR00T N1.7 and MolmoAct 2 are `planned`, so the test run can reach them.
  - MolmoAct 2 is in the registry's RTC policies.
  - The naming rule accepts `grootn17_*` for `groot`. The kept N1.5 model still carries the old type `groot`.
- **`tests/smoke/train.sh`:** the 7 shortlisted models.
  - Each one trains 30 steps and saves the step-30 checkpoint.
  - Then `tests/smoke/serve_check.py` runs on that checkpoint, using upstream's `load_deployment`. That covers
    admission and warmup in `chunk` and, for RTC policies, `rtc_guided`.
  - It times each request and compares it with H − s_min at 30 fps.
  - Results: `logs/smoke/summary.txt` and `logs/smoke/serve_<model>.json`.
- **Serve check on the laptop (RTX 4060, s_min 20, 30 fps):**

  | Model | Checkpoint | Modes admitted | H | Time per chunk (median) | DRTC margin |
  |---|---|---|---|---|---|
  | ACT | the Phase 1 300-step test | chunk | 100 | chunk 27 ms (1 step) | 80 steps |
  | SmolVLA | 5 steps, batch 2 (removed after the check) | chunk + rtc_guided | 50 | chunk 157 ms (5 steps); RTC 229 ms (7 steps) | 30 steps |

  Peak GPU: ACT 0.3 GB, SmolVLA 1.5 GB. Guided RTC costs about +45% per chunk for SmolVLA (the autograd
  correction at each denoising step).

ACT and Diffusion get DRTC scheduling with the Butterworth filter and the Phase 5 smoothing (method A), but no
RTC. The other five can be compared with and without RTC.

**Next:**
- [ ] Recipes after the user's research. The choices to make:
  - the chunk length at 30 fps;
  - steps vs epochs, and the stopping rule;
  - relative vs absolute actions.
- [ ] Test runs of all 7 on the H200: `bash tests/smoke/train.sh` (the user runs it).
  - **First run (2026-11-18, commit c4d24663).** Serve times are the H200's; 30 fps, s_min 20:

    | Model | Train (30 steps) | Peak train GPU | Serve check |
    |---|---|---|---|
    | ACT | ok, 170 s, eval 0.81 | 44.6 GB | admitted chunk; 13.7 ms (1 step) vs margin 80 |
    | SmolVLA | ok, 170 s, eval 0.46 | 14.7 GB | admitted chunk + rtc_guided; 175 ms (6 steps) / RTC 301 ms (10 steps) vs margin 30 |
    | Diffusion | ok, 167 s, eval 1.00 | 9.6 GB | refused: no `chunk_size` (fix 1) |
    | π0.5 | ok, 585 s, eval 0.27 | 45.5 GB | refused: feature names ≠ checkpoint inputs (fix 2) |
    | GR00T N1.5 | ok, 261 s, eval 0.22 | 20.5 GB | refused: postprocessor returned one action, not the chunk (fix 3) |
    | GR00T N1.7 | failed: gated Hub repo (`nvidia/Cosmos-Reason2-2B`, the backbone) | — | — |
    | MolmoAct 2 | failed: the base config has 4 fields 0.7 dropped (fix 4) | — | — |

  - **Fixes:**
    1. `drtc.control.chunk_adapters` (installed by `lab serve` and the serve check) gives Diffusion a chunk spec
       (prediction = execution = `n_action_steps`; `n_obs_steps` must be 1). It also adds the observation-time
       axis to state: without it the server's (B, D) state fails in `generate_actions`, as a test showed.
       Tests: `tests/test_chunk_adapters.py`.
    2. Camera renames. `lab serve` declares the checkpoint's camera names (the recipe's `cameras`, the training
       `rename_map`), and `lab run` passes the same map as `--rename_map`. Camera semantics are kept.
    3. The GR00T N1.5 `action_unpack_unnormalize` step keeps every timestep of a (B, T, D) chunk. Upstream's N1.5
       port kept only the last one; single actions are unchanged. Test in `tests/test_groot_n15_rtc.py`.
    4. `lab train` passes a base with an outdated config as an upgraded local copy (`drtc.train.upgraded_base`:
       files linked, unknown fields dropped and printed). `molmoact2_full` now also sets
       `policy.train_mode_vlm: fft`; 0.7's default is LoRA on the VLM.
    5. GR00T N1.7: the user accepts the gate of `nvidia/Cosmos-Reason2-2B` (auto-approved) with the Hub account
       whose token is on the H200.
  - **Second run (2026-11-18, commit d30fef52), the 5 that failed:**

    | Model | Train | Peak train GPU | Serve check |
    |---|---|---|---|
    | Diffusion | ok, 177 s, eval 1.00 | 9.6 GB | **admitted** chunk; 54.9 ms (2 steps) vs margin 8 |
    | GR00T N1.5 | ok, 245 s, eval 0.22 | 20.5 GB | **admitted** chunk + rtc_guided; 77.7 ms (3 steps) / RTC 99.2 ms (p90 218) vs margin 8. The plugin's RTC works on the real model |
    | π0.5 | ok, 416 s, eval 0.27 | 45.5 GB | refused: feature names ≠ checkpoint inputs (fix 6) |
    | MolmoAct 2 | ok, 747 s, eval 0.23 (the base upgrade works) | 55.8 GB | refused: RGB shape ≠ checkpoint (fix 6) |
    | GR00T N1.7 | failed: gated repo again (fix 5 not done yet) | — | — |

  - **Fix 6.** A fine-tune keeps its base checkpoint's nominal input_features:
    - π0.5: 3 camera slots at 224×224 and a 32-d state;
    - MolmoAct 2: cam0/cam1 at 224×224.

    The 0.7 server demands an exact match. `drtc.control.chunk_adapters` overrides `validate_chunk_input_features`
    for both, as upstream does for VLA-JEPA:
    - any RGB resolution, since both policies resize images themselves;
    - π0.5: a subset of the camera slots (the missing one gets the same empty, masked image as in training) and a
      state no longer than max_state_dim (padded);
    - MolmoAct 2: exact names and state.

    Tests in `tests/test_chunk_adapters.py`.
  - [ ] Third run: π0.5, MolmoAct 2 and GR00T N1.7.
- [x] GR00T N1.5 plugin: guided RTC on 0.7, with `supports_rtc()` (2026-11-18).

**What LeRobot 0.7 covers:**
- π0.5, π0 and GR00T: `use_relative_actions` with `relative_exclude_joints: [gripper]`.
- EMA for any policy: `--ema.type`.
- Not covered: openpi's per-timestep action normalization for π. It would need a processor step or plugin.

**How our current recipes differ:**
- `pi05_full_openpi` / `pi0_full_openpi`: absolute actions, chunk 50, 10 epochs (~7.8k steps), no EMA.
- `groot_n17_nvidia`: chunk 40, absolute actions.
- `act_100ep`: batch 64, ~39k steps, no augmentation, chunk 100 (default).
- `diffusion_full`: horizon 32 / 16 executed, 240×320 with crop, DDIM 10.

- [ ] For each candidate, who made it work on an SO100/SO101-class arm: ArmnetBench read (reference only). The
      recipe source is being researched by the user.
  - [ ] Inference speed against DRTC's timing: the H200 test runs above.
- [x] Selection criteria: the user's choice (2026-11-18).
- [x] Candidates reviewed; shortlist of 7 (2026-11-18).
- [x] `lab/models/cubebox_v1.yaml`: the shortlist `planned`, the rest `retired`.
  - [ ] Recipes confirmed after the research.

## Phase 3: Faster training
- [ ] **[open]** Hardware available (H200 only, or the A100 too; one GPU or several)
- [ ] Measure the time per step for each shortlisted model on 0.7 as-is: bf16, compiled image transforms,
      dataloader workers, video decoding
- [ ] Options to compare:
  - upstream FSDP2/DDP (`--parallelism.*`), if there is more than one GPU
  - `torch.compile` of the policy
  - LoongForge (π0.5, X-VLA, GR00T N1.6/N1.7; its speedups were measured on 8 GPUs; DCP checkpoints need
    conversion)
- [ ] Settle on one setup per model, recorded in `models.yaml`

## Phase 4: Train and choose
- [ ] Train each shortlisted model once (`*_full_trim`) with the Phase 3 setup
- [ ] Rank offline (`eval-offline`), then robot trials with a fixed protocol (trial count, cube positions)

## Phase 5: Smoothing
- [ ] Test bench:
  - replay recorded observations and chunks offline, with simulated latency
  - the same metrics for every run: stalls, jerk, jumps between chunks, action error
  - a fixed robot procedure, so a method can be compared on any model with one command
- [ ] Baselines: no smoothing; upstream alignment, blending and RTC
- [ ] Methods: A (adaptive playback), B (RTC), C (smoothness guidance); new ones later

## Phase 6: DRTC experiments
- [ ] Fault conditions (spikes, drops, duplicates, reordering, disconnects), on the chosen model(s)
- [ ] Latency estimators, offline on the logged runs
- [ ] Comparison with upstream remote inference
- [ ] **[open]** Which further items from the proposal (predictive triggering, adaptive chunk length, adaptive
      steps, caching, shared server)
