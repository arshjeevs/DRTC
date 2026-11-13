# Reliable and Efficient Remote VLA Control on Low-Cost Robots

*Project proposal and working plan. Status: draft for discussion. Items marked **[open]** are not decided yet.*

## 1. Summary

Vision-Language-Action (VLA) models can perform manipulation tasks from camera images and a language instruction, but they are too large to run on the computers that low-cost robots carry. The usual answer is to run the model on a remote GPU server, which makes robot control depend on the network: delays, dropped messages and outages directly disrupt the robot.

DRTC (asynchronous action chunking with real-time-chunking inpainting and latency estimation) addresses part of this problem. This project:

1. **replicates** DRTC on a low-cost setup (SO101 arm, Raspberry Pi 5, laptop GPU) with our own data and fine-tuned models,
2. **characterises** how model choice, fine-tuning method, amount of data, cameras and compute affect performance on this hardware, and
3. **extends** DRTC with methods that make remote control robust to outages and adapt inference to network and hardware conditions.

## 2. Problem and motivation

- **VLAs are heavy.** SmolVLA has ~450M parameters, π0.5 ~3.4B. A Raspberry Pi cannot run them at control rate; an 8 GB laptop GPU can run only the smallest.
- **Remote inference is fragile.** With the model on a server, every action depends on a network round trip. Latency spikes cause the robot to stall; outages stop it entirely.
- **Choices are unclear.** Building a VLA robot involves many decisions (which model, full or LoRA fine-tuning, how many demonstrations, which cameras, which GPU) with little guidance for low-cost setups.

## 3. Gap

- DRTC keeps control smooth under latency spikes, drops, duplicates and reordering, **but a disconnect longer than the buffered action queue stops the robot** (to be confirmed in Phase 1).
- DRTC uses a fixed amount of computation per inference (fixed denoising steps, fixed chunk length, no caching), regardless of network or hardware conditions.
- Published VLA comparisons use different robots, tasks and compute; few compare models, fine-tuning methods, data amounts and camera setups on one low-cost robot.

## 4. Objective and research questions

**Objective:** make remote VLA control on low-cost robots reliable under realistic network conditions and efficient on modest hardware, and identify which design choices matter most.

| | Research question |
|---|---|
| **RQ1 Robustness** | How well does DRTC hold up on a low-cost robot, and how can control survive network outages? |
| **RQ2 Efficiency** | How much can inference be accelerated or adapted to network and hardware conditions without lowering task success? |
| **RQ3 Design choices** | Which model, fine-tuning method, amount of data and camera configuration give the best results on low-cost hardware? |

## 5. Existing system (done)

| Component | Status |
|---|---|
| Hardware | SO101 follower + leader arms, wrist camera (Zebronics) + scene camera (phone), laptop RTX 4060 8 GB, Raspberry Pi 5 |
| Data collection | `lab/bin/record`: teleoperation with keyboard controls, Rerun viewer, Hub upload |
| Training | `lab/bin/train`: model registry (`lab/models.yaml`), full or LoRA fine-tuning, resume, Hub push, wandb logging; SmolVLA, π0, π0.5, X-VLA, ACT |
| Serving | DRTC policy server (GPU) + robot client, full and LoRA checkpoints, automatic venv selection for π0 / π0.5 |
| Experiments | `lab/bin/run`: 10 network-fault conditions (latency spikes, drops, duplicates, reordering, disconnects, mixture), per-trial metrics and success labels |
| Platforms | Linux laptop, Raspberry Pi, Windows (`lab/WINDOWS.md`), Docker on GPU servers |
| Pre-validation | Models run on the hardware and complete the task; the full pipeline works end to end |

## 6. Methods

Items are numbered as in the planning discussion.

### 6.1 Replication (Phase 1)

| # | Method | Description | Metrics |
|---|---|---|---|
| 0 | **DRTC replication** | Run the trained model under every fault condition, including the Jacobson-Karels vs. max-of-last-10 latency estimators | Success rate, stalls, latency estimate vs. measured, chunk disagreement |

### 6.2 RQ1: Robustness

| # | Method | Description | Hypothesis |
|---|---|---|---|
| 1 | **Edge fallback** | A small policy (ACT) runs on the Pi and takes over when the server is unreachable, handing back control when it returns | Success under disconnects rises substantially; handover is smooth if actions are blended |
| 2 | **Graceful slowdown** | Play buffered actions more slowly as the queue drains | Fewer and shorter stalls at the cost of slower task completion |
| 4 | **Predictive triggering** | Request the next chunk earlier when the latency trend rises | Fewer stalls under gradual latency increases |
| 5 | **Latency estimators** | Compare Jacobson-Karels with percentile, Kalman and learned predictors, first offline on recorded latency traces, then the best on the robot | A better estimator reduces stalls without excess requests |
| 9 | **Adaptive chunk length** | Request longer chunks when the network is unreliable | Longer chunks reduce stalls under bad networks; shorter chunks are more reactive |

### 6.3 RQ2: Efficiency

| # | Method | Description | Hypothesis |
|---|---|---|---|
| 6 | **Adaptive denoising steps** | Fewer flow-matching steps when latency is high | Lower latency with small quality loss |
| 7 | **Prefix / KV caching** | Reuse the vision-language computation when images barely change | Significant latency reduction at high control rates |
| 8 | **Inference optimisation** | fp32 vs. bf16 vs. `torch.compile` vs. 8-bit, per model | bf16 and compile give large speed-ups on the 4060 with no success loss |
| 18 | **Hardware profile** | Inference time and memory per model on laptop 4060, A100, H200, Pi 5 CPU | Defines which models each device can serve at control rate |
| 19 | **Shared server** | Several clients (real or simulated) on one policy server | Identifies the load at which DRTC's latency handling breaks down |

### 6.4 RQ3: Design choices

| # | Study | Models (`lab/models.yaml`) |
|---|---|---|
| 14 | **Full vs. LoRA fine-tuning** | `smolvla_cubebox_v1_full`, `smolvla_cubebox_v1_lora` |
| 15 | **Data efficiency** (10 / 25 / 45 episodes) | `smolvla_cubebox_v1_lora_ep10`, `_ep25`, `smolvla_cubebox_v1_lora` |
| 16 | **Camera ablation** (wrist / scene / both) | `smolvla_cubebox_v1_lora_wrist`, `_scene`, `smolvla_cubebox_v1_lora` |
| 13 | **Model comparison** | `smolvla_cubebox_v1_lora`, `act_cubebox_v1_full`, `xvla_cubebox_v1_lora`, `pi05_cubebox_v1_lora` |
| 12 | **Action filtering** | Existing Butterworth filter settings, 2–3 configurations on one model |

All RQ3 models use the same recipe within a study (same steps, batch size and LoRA settings) so only the studied factor changes. The data-efficiency runs keep the number of training steps fixed.

## 7. Experimental design

### 7.1 Dataset

- **New dataset** `orangemoon599/so101_cubebox_v1` (being recorded; see the naming convention below): one task, "Pick up the green cube and put it inside the box.", current camera layout (wrist + scene), 30 fps.
- **Split:** episodes `0-44` for training, the remaining episodes held out for offline evaluation. Set once in `lab/models.yaml` (`_study.train_episodes`).
- **Camera-ablated copies:** `lab/bin/ablate-cameras orangemoon599/so101_cubebox_v1` creates `_wrist` and `_scene` on the Hub.

### 7.2 Naming convention

All new datasets and models live under the `orangemoon599` Hub account. Older datasets and models keep their names (`_legacy` in `lab/models.yaml`).

| What | Pattern | Example |
|---|---|---|
| Dataset | `so101_<task>_v<N>[_<variant>]`; bump `vN` when the setup changes | `so101_cubebox_v1`, `so101_cubebox_v1_wrist` |
| Model (registry entry) | `<policy>_<task>_v<N>_<full\|lora>[_<variant>]` | `smolvla_cubebox_v1_lora_ep10` |
| Model (Hub repo) | `orangemoon599/<registry entry>`; one entry, one repo | `orangemoon599/smolvla_cubebox_v1_lora` |
| Results | `results/<experiment>/<registry entry>/` | `results/replication/baseline/smolvla_cubebox_v1_lora/` |

Model variants: `ep10` (episode subset), `wrist` / `scene` (camera), `r16` (LoRA rank), `s40k` (steps). `lab/bin/list` and `lab/bin/train` check new entries against these rules.

### 7.3 Evaluation

**Offline (no robot):** action-prediction error of each model on the held-out episodes. Cheap, runs on the GPU servers, ranks all models before robot time is spent. **[to build]**

**On the robot:**
- **Success definition** (fixed before trials): the cube ends inside the box and the gripper is open, within the trial duration.
- **Fixed start positions:** marked cube positions on the table, cycled in the same order for every condition.
- **Interleaving:** alternate conditions rather than running each condition in one block, to avoid drift (lighting, motor temperature) biasing one condition.
- **Trial counts:** 10 per condition for headline results; 5 for exploratory comparisons. **[open]**
- **Statistics:** success rates with Wilson 95% confidence intervals; latency and stall metrics as median and interquartile range over trials.

### 7.4 Metrics

| Metric | Source |
|---|---|
| Task success | Label after each trial (`lab/bin/run`) |
| Stalls (queue empty) | Per-step log |
| Measured vs. estimated latency | Per-step log |
| Chunk disagreement (`chunk_mean_l2`) | Per-step log |
| Inference time, GPU memory | Benchmark script **[to build]** |
| Training time, checkpoint size | Training logs, Hub |

## 8. Compute plan

| Machine | Role |
|---|---|
| **H200** (141 GB) | π0.5 LoRA, X-VLA LoRA, camera-ablation runs; can run several jobs at once |
| **A100** | SmolVLA full / LoRA / data-efficiency runs, ACT |
| **Laptop RTX 4060** (8 GB) | Policy server for SmolVLA and ACT on the robot; inference benchmarks |
| **Raspberry Pi 5** | Robot client; ACT inference for the edge fallback |

π0.5 does not fit in 8 GB, so it must be served from a server GPU for robot trials. That requires network access from the robot to the server (e.g. an SSH tunnel to port 8080) **[open: check access]**. It also gives a real long-distance network for RQ1.

### Training matrix

| Model entry | Study | Server | Notes |
|---|---|---|---|
| `smolvla_cubebox_v1_full` | 13, 14 | A100 | |
| `smolvla_cubebox_v1_lora` | 13, 14, 15, 16 | A100 | Shared reference model |
| `smolvla_cubebox_v1_lora_ep10` | 15 | A100 | |
| `smolvla_cubebox_v1_lora_ep25` | 15 | A100 | |
| `smolvla_cubebox_v1_lora_wrist` | 16 | H200 | Needs `_wrist` dataset first |
| `smolvla_cubebox_v1_lora_scene` | 16 | H200 | Needs `_scene` dataset first |
| `act_cubebox_v1_full` | 13, fallback (1) | A100 | |
| `xvla_cubebox_v1_lora` | 13 | H200 | Server cannot serve X-VLA yet **[to build]** |
| `pi05_cubebox_v1_lora` | 13 | H200 | Separate venv, created automatically |

## 9. Minor and major presentation split

### Minor presentation: "Replicating and characterising remote VLA control on a low-cost robot"

| # | Item |
|---|---|
| 0 | DRTC replication under all fault conditions |
| 14 | Full vs. LoRA fine-tuning |
| 18 | Hardware profile (inference time and memory per device) |
| 12 | Action filtering |
| (+) | Optional, depending on robot time: first results of 13, 15, 16 **[open]** |

Ends with the observation motivating the major: outages stop the robot, and inference cost limits which models can be used.

### Major presentation: "Making remote VLA control reliable and efficient on low-cost robots"

| Group | Items |
|---|---|
| Robustness | 1 (edge fallback, main contribution), 2, 4, 5, 9 |
| Efficiency | 6, 7, 8, 19 |
| Design choices | 13, 15, 16 (full trial counts) |
| Final evaluation | Ablation: DRTC alone vs. with each extension; Pi as the real client over Wi-Fi |

Nothing shown in the minor is repeated; its results become the baselines of the major.

## 10. Work breakdown

### Done
- [x] Data collection, training, serving and experiment pipeline
- [x] Pre-validation on the hardware
- [x] Training entries for all study models (`lab/models.yaml`)

### Next (minor)
- [ ] Record `orangemoon599/so101_cubebox_v1`
- [ ] `lab/bin/ablate-cameras orangemoon599/so101_cubebox_v1`
- [ ] Launch training runs on H200 / A100
- [ ] Offline evaluation script (held-out action error) **[to build]**
- [ ] Inference benchmark script (items 8, 18) **[to build]**
- [ ] Robot trials: replication + full vs. LoRA (+ filtering)
- [ ] Results analysis script (success table with confidence intervals, latency / stall figures) **[to build]**
- [ ] Minor presentation

### Then (major)
- [ ] X-VLA support in the policy server
- [ ] Edge fallback on the Pi (ACT inference, trigger, handover)
- [ ] Client scheduling extensions (graceful slowdown, predictive triggering, adaptive chunk length)
- [ ] Latency estimators (offline on recorded traces)
- [ ] Adaptive denoising steps, caching, inference optimisation
- [ ] Shared-server experiment
- [ ] Full design-choice evaluation (13, 15, 16)
- [ ] Final ablation and write-up

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Robot trial time is the bottleneck | Rank models offline first; reserve full trial counts for headline comparisons |
| Small dataset limits absolute success | Report relative differences between conditions; keep the scene fixed |
| π0.5 cannot run on the laptop | Serve from A100 / H200 over the network |
| π0 / π0.5 dependency conflicts | Separate venv, selected automatically |
| X-VLA not yet served | Add server support before major evaluation; offline evaluation works without it |
| Scene drift between trials | Marked start positions, interleaved conditions, fixed lighting |
| Network to GPU servers blocked | SSH tunnel; otherwise evaluate π0.5 offline only |

## 12. Open decisions

- **[open]** Trial counts per condition (5 vs. 10) and which studies get robot time for the minor.
- **[open]** Whether the minor includes early results of 13, 15 and 16, or keeps them for the major.
- **[open]** Robot client for the minor: laptop or Pi.
- **[open]** Whether the robot can reach the A100 / H200 (needed for π0.5 robot trials).
- **[open]** Number of held-out episodes (default: 5 of 50).
