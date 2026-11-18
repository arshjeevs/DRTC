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
- [ ] 1.5 Policy plugins (DOT, SFP, GR00T N1.5)
- [ ] 1.6 Training wrapper (clean stop, progress line), recipes → 0.7 `lerobot-train` flags
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
  - [ ] `lab record` pipeline: raw → checks → autotrim (on LeRobot 0.7) → trimmed + raw pushed
  - [ ] `lab` CLI with the old names as aliases (record, calibrate, preflight, session, run, bench, eval-offline, analyze, edit-dataset, list, logstats)
  - [ ] jobs + worker + `lab submit` (train, eval, bench, serve)
  - [ ] logging (wandb trials, lineage, Hub mirror), import of `results/index.csv`
  - [ ] Zenoh serving across hosts (replaces `tunnel`), Pi and H200 setup
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
- [ ] For each candidate: who made it work on an SO100/SO101-class arm, with which recipe (steps, batch, what was
      frozen, chunk length, data size), and how fast it runs at inference (does it fit DRTC's timing?)
- [ ] **[open]** Selection criteria (results others got, architecture variety, inference speed, or a mix)
- [ ] Candidates: GR00T N1.5/N1.7, π0, π0.5, π0-FAST, SmolVLA, X-VLA, MolmoAct2, ACT, DOT, Diffusion, SFP, and
      models new in 0.7 (EVO1, LaWAM, VLA-JEPA, ...)
- [ ] Update `models.yaml`: the shortlist as `*_full_trim` entries with their recipes; the rest marked as not kept

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
