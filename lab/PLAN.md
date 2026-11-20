# Major project plan

*The plan from 2026-11-20 on: the scope locked with the user, one item at a time, and the order of work. Everything before it (the phases of 2026-11-18 to 20, decisions, measurements) is in [docs/history/PLAN-2026-11-18-to-20.md](docs/history/PLAN-2026-11-18-to-20.md). The research framing: [docs/PROPOSAL.md](docs/PROPOSAL.md).*

## LOCKED SCOPE (2026-11-20, with the user): this is the project from here on

Decided one item at a time; it replaces the evaluation plans further down wherever they differ. Today's earlier robot
evals and benches (sort_one_baseline_v2, sort_one_faults, the bench_* evals) are side results: valid, shown in the
report, not the main experiment.

**Archive (2026-11-20, the user: "start fresh with respect to the DRTC experiments"):** every result made before
this scope (robot evals and benches) moved unchanged to `results/archive/2026-11-20-before-scope/` (laptop and H200;
the results store keeps its copy), their eval files to `lab/evals/archive/`. Side results, usable in the report.

**Claim.** In client-server robot inference, network faults lower the task scores; **DRTC recovers them**; **smoothing**
then improves the motion.

**Datasets** (as they are, no new recording or edits): `so101_cubebox_v0` (50 = 45 + 5 held out),
`so101_cubebox_v1` (110 = 100 + 10), `so101_sort_one_v1` (88 = 80 + 8).

**Models and what each dataset gets** (the 7 shortlisted, each with its fixed original recipe: SmolVLA, GR00T N1.5,
GR00T N1.7, π0.5, MolmoAct 2, ACT, Diffusion):

| Dataset | Trained | Robot evaluation |
|---|---|---|
| `so101_cubebox_v0` (initial, temporary research) | SmolVLA, GR00T N1.5 | baseline (condition 1) only |
| `so101_cubebox_v1` | all 7 | baseline (condition 1) for all 7; the DRTC experiment only if time remains |
| `so101_sort_one_v1` | all 7 | baseline (condition 1) for all 7; **the DRTC experiment (conditions 1-4) for SmolVLA and GR00T N1.5** |

**Trainings** (✓ trained and on the Hub, ▶ running, · to do):

| Model | cubebox v0 | cubebox v1 | sort_one |
|---|---|---|---|
| SmolVLA | ✓ | ▶ H200 | ✓ |
| GR00T N1.5 | ▶ H200 | ✓ | ✓ |
| ACT | (✓, not needed) | ▶ H200 | ✓ |
| Diffusion | – | ▶ H200 | ✓ |
| GR00T N1.7 | – | · | · |
| π0.5 | – | · | · |
| MolmoAct 2 (LoRA, its original recipe) | – | · | · |

The 4 running ones started 2026-11-20 on the H200 (tmux `train` in the drtc_lab container; logs
`logs/train_<model>.log`). Still to do: 6 (GR00T N1.7, π0.5, MolmoAct 2 on cubebox v1 and sort_one); π0.5 takes ~27 h
each.

**Conditions** (per model, per task; the same order and rules everywhere):

| # | Condition | Network | Robot side |
|---|---|---|---|
| 1 | baseline | normal | upstream LeRobot remote inference, as it ships: no DRTC, no filter, and it ends the run after 1 s without actions (starvation grace 1.0 s) |
| 2 | baseline + faults | the fault mix | as 1 |
| 3 | DRTC + faults | the fault mix | DRTC, our version (the 2026-11-20 fixes: cooldown never shorter than the measured latency incl. the slowest of the last 10 replies, cold-start wait, request delay capped at the chunk); **no** Butterworth filter; holds the arm through stalls and carries on |
| 4 | DRTC + smoothing + faults | the fault mix | as 3 + the Butterworth filter + adaptive playback (method A) |

Claim 1 = 2 against 1 (faults hurt) and 3 against 2 (DRTC recovers); claim 2 = 4 against 3 (smoothing).

**Fault mix** (conditions 2-4, every task, times from the trial start): latency spike +1 s at 2 s; every reply dropped
4-5 s; full outage (both directions) 6-7.5 s. Each fault type is measured separately on the bench only.

**Rules.** 10 trials per condition, per model, per task (rounds, as before). A terminal fault after the arm has started
moving counts as a failure (M1 = 0, label `fault`); a fault before the first action is a setup problem and is
repeated. The report shows the fraction of trials ended by a fault. Metrics v2 ([METRICS.md](METRICS.md)).

**Robot computer:** the laptop for every experiment; the Raspberry Pi is set up only after everything is complete. The
presentation and report say the laptop was the robot computer.

**Order of work** (approved; the user decides when each step starts):

| # | What | Who | Status |
|---|---|---|---|
| 1 | This scope into PLAN.md, the rule changes into METRICS.md (v2); earlier results archived | Claude | ✓ done |
| 2 | Build the 4 conditions, the fault mix, the new rules (fault = failure, M1f) and the eval files; test on the replay robot | Claude | ✓ done (A100 check 2026-11-20: upstream gave up under the faults, DRTC got through, smoothing cut jerk 3-4x) |
| 3 | Trainings (table above) | Claude, A100 + H200 | ▶ 4 running; 6 to do |
| 4 | **DRTC experiment on sort_one:** SmolVLA + GR00T N1.5 x 4 conditions x 10 trials = 80 trials (~1 h): eval `sort_one_drtc_main`, served on the A100 | the user | ▶ ready |
| 5 | **Baselines:** condition 1 x 10 trials: cubebox v0 (2 models), cubebox v1 (7), sort_one (7), as the models finish | the user | · after 2, and 3 per model |
| 6 | Bench (each fault type separately, sort_one, SmolVLA + GR00T) and offline M9/M10 for every model | Claude, H200 | · |
| 7 | One report with all results | Claude | · |
| 8 | If time: the DRTC experiment on cubebox v1 | the user | · optional |
| 9 | The Raspberry Pi setup | later | · |
