# Major project plan

*The plan from 2026-11-20 on: the scope locked with the user, one item at a time, and the order of work. Everything before it (the phases of 2026-11-18 to 20, decisions, measurements) is in [docs/history/PLAN-2026-11-18-to-20.md](docs/history/PLAN-2026-11-18-to-20.md). The research framing: [docs/PROPOSAL.md](docs/PROPOSAL.md).*

## DRTC + SMOOTHING EXPERIMENT SETUP v2 (LOCKED 2026-11-22, with the user)

Replaces, for DRTC and smoothing, the four conditions, the fault mix and the DRTC parts of the order of work in the
locked scope below (the datasets, models, trainings and baselines there still hold). Why it changed: in the first
run (`sort_one_drtc_main`, 2026-11-20) all 20 "baseline + faults" trials were ended by upstream's own 1 s starvation
grace at the first fault (the spike), so DRTC was compared with a system that gave up by design; DRTC also differed
from it in two things at once (its scheduling and a 25 s grace), and smoothing added two methods at once. That run
and the old benches are archived as side results.

**Claim** (unchanged): network faults lower the task score; DRTC recovers it, more than just waiting; smoothing then
improves the motion without lowering success.

**Systems**
- **upstream (patient)**: LeRobot's own scheduler with the starvation grace raised to the whole trial (25 s). The
  baseline everywhere: raising the grace is one flag, so DRTC must beat upstream at its best.
- **upstream (as shipped)**: the same with the 1 s grace. Bench only, to show the default breaks.
- **DRTC**: its trigger (`H - s_min`), the Jacobson-Karels estimator, the cooldown re-request with our 2026-11-20
  fixes (no H/2 cap), the 25 s grace.
- **Smoothing methods**: adaptive playback (have), Butterworth filter (have), RTC (in the code; flow-matching models
  only: SmolVLA, GR00T N1.5/N1.7, later π0.5 and MolmoAct 2), temporal ensembling (build), zero-lag queue smoothing
  (build), soft stop + resume ramp (build). Dropped: smoothness guidance (same job as RTC), crossfade (weaker
  ensembling), minimum-jerk transition (inside the resume ramp), One Euro filter, step limiter (a safety cap).

**Faults**: none; spike (one reply +1 s); drops (every reply lost for 1 s); outage (1.5 s, both directions); mix
(spike, then drops 2 s later, then outage 4 s later, as before); rise (+400 ms on every reply for 5 s, new); loss
(10% of replies lost for the whole run, new). Fault times are shifted by a seeded random 0-4 s per trial number,
the same shift for trial k in every condition.

**The bench** (replay robot on the held-out episodes, simulated network 250 +- 20 ms, 8 runs per cell, 10 s each):
datasets `so101_sort_one_v1` and `so101_cubebox_v1`; models SmolVLA, GR00T N1.5, GR00T N1.7, ACT, Diffusion (10
model-dataset pairs); π0.5 and MolmoAct 2 join with the same cells when trained (4 more pairs). Per pair:

| Exp. | What | Conditions x faults | Runs |
|---|---|---|---|
| E1 | DRTC against upstream | upstream patient (all 7 faults), upstream as shipped (6, no "none"), DRTC (all 7) | 160 |
| E2 | DRTC ablations | A1 no cooldown, A2 upstream's estimator, A3 upstream's trigger, A4 DRTC's original H/2 cooldown cap: each under all 7; A5 `s_min` at 0.5x and 1.5x (SmolVLA 10/30, GR00T 4/12, ...): mix and none | 256 |
| E3a | each smoothing method alone, on DRTC + mix | none, Butterworth, adaptive, RTC (flow-matching only), ensembling, zero-lag, resume ramp, Butterworth + adaptive (the old condition 4) | 64 (56 without RTC) |
| E3b | the final stack and its ablations, on DRTC + mix | the stack, the stack minus each of its 4 parts, the stack with no faults | 48 |

The final stack = adaptive playback + one seam method (RTC or ensembling) + one jitter method (zero-lag or
Butterworth) + the resume ramp, chosen from E3a by a rule fixed in the eval setup **before** E3a runs; models without
RTC use ensembling. Size: ~520 runs per pair, ~5,200 runs for the 10 pairs, ~20 h of GPU time on one server at the
13.5 s per run measured on the last bench (~10 h with a server per dataset); π0.5 + MolmoAct 2 add ~8 h. Where and
when: the user decides.

**The robot (E4)**: sort_one, SmolVLA + GR00T N1.5, the laptop, 10 trials per condition: 1 upstream patient, no
faults; 2 upstream patient + mix; 3 DRTC + mix; 4 DRTC + the final stack + mix. 80 trials. Claims: 2 against 1
(faults hurt even when waiting), 3 against 2 (DRTC's scheduling beats waiting), 4 against 3 (smoothing). The DRTC
robot run on cubebox v1 stays optional.

**Build (2026-11-22, commit 779d5b95 and after)**: conditions generated in `lab/experiments/v2/` and `v2_robot/`
(`_make.py`); bench evals `bench_v2_drtc_<dataset>` (E1 + E2 in one eval, 52 conditions), `bench_v2_smooth_<dataset>`
(E3a), `bench_v2_smooth_rtc_<dataset>` (E3a's RTC cell: SmolVLA + GR00T N1.5); E3b and E4 are written once the stack
is chosen. Bench runs are 14 s (the latest fault ends at 11.5 s with the shift). Local check on the laptop's replay
robot (ACT, SmolVLA): every fault, the shift, the ablations and the smoothing methods work. Found: GR00T N1.7 gets no
RTC cell (relative actions: LeRobot turns its RTC prefix off); upstream (patient) still ends a run when a reply is
lost, after its 5 s request deadline, because it never re-sends (what DRTC's cooldown adds); the first RTC request
after a server start compiles for > 5 s, so RTC evals get one test round first (WORKFLOW).

**Robot evals: core and how they grow** (locked 2026-11-22). Core, 10 trials per condition (~1 min per trial with
the reset, ~3 h): E4 above (80 trials; waits for the final stack); the baselines `sort_one_baseline` and
`cubebox_v1_baseline` for SmolVLA, GR00T N1.5, GR00T N1.7, ACT, Diffusion (50 + 50 trials; condition: upstream
patient, no faults, the same reference as E4). Trials run in rounds (every model x condition once per round, in a
seeded order); always stop at the end of a round. They grow append-only, without redoing anything:
- **more trials**: raise `trials` (10 -> 20) and run the same command; it continues with round 11;
- **more models** (π0.5, MolmoAct 2 when trained): add them to `models` and `lab eval freeze`: only the new models
  are frozen, the frozen ones never change; the next rounds run the new models until they catch up (no check model);
- **a new condition or task**: a new eval file, which runs its own reference condition again.

When there is time, in this order: 1 E4 10 -> 20 trials (+80); 2 the DRTC robot run on cubebox v1 (2 models, the 4
conditions, 80); 3 E4 on a third model (GR00T N1.7 if its baseline is good, +40); 4 the baselines 10 -> 20 (+100 to
140); 5 a robot run of a bench fault that surprised us (40 each); 6 optional: `cubebox_v0_baseline` (2 models, 20,
frozen).

**Order**: 1 build (the systems, the new faults and the shift, the three smoothing methods, RTC with DRTC and
GR00T N1.7 served with RTC (to check), the eval setup, bench eval files per dataset, each tested locally first);
2 E1 + E2 on all pairs (need no new smoothing); 3 E3a, then the stack is fixed, then E3b; 4 E4 on the robot.
Later: π0.5 and MolmoAct 2 on the bench when trained.

## LOCKED SCOPE (2026-11-20, with the user): this is the project from here on

*For DRTC and smoothing, the conditions, the fault mix and steps 4, 6 (bench) and 8 below are replaced by the setup
v2 above (2026-11-22).*

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
| SmolVLA | ✓ | ✓ | ✓ |
| GR00T N1.5 | ✓ | ✓ | ✓ |
| ACT | (✓, not needed) | ✓ | ✓ |
| Diffusion | – | ✓ | ✓ |
| GR00T N1.7 | – | ✓ | ✓ |
| π0.5 | – | · | · |
| MolmoAct 2 (LoRA, its original recipe) | – | · | ▶ A100 |

Running:
- MolmoAct 2 on sort_one, on the A100 (tmux `drtc`, window `molmo`; resumed from step 1000 after it was stopped).
  ~10 s/step; at ~1,760 of 50,000 steps late on 2026-11-20, ~5.5 days left.

Done 2026-11-21 (the user): ACT and Diffusion on cubebox v1 (H200); GR00T N1.7 on cubebox v1 and sort_one (H200, 20,000 steps each, on the Hub).

Still to do: 3 (π0.5 on cubebox v1 and sort_one, MolmoAct 2 on cubebox v1).
π0.5 takes ~27 h and ~77 GB each; the A100 has ~35 GB free next to MolmoAct, so π0.5 needs the H200. Where and when:
the user decides. After each training: the offline evaluation (WORKFLOW.md, Part 3), then its baseline.

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
| 3 | Trainings (table above) | Claude, A100 + H200 | ▶ 1 running (MolmoAct 2 sort_one); 3 to do (where and when: the user decides) |
| 4 | **DRTC experiment on sort_one:** SmolVLA + GR00T N1.5 x 4 conditions x 10 trials = 80 trials (~1 h): eval `sort_one_drtc_main`, served on the A100 | the user | ✓ done (80/80 trials, 2026-11-20); **replaced by setup v2 (E4), this run becomes a side result** |
| 5 | **Baselines:** condition 1 x 10 trials: cubebox v0 (2 models), cubebox v1 (7), sort_one (7), as the models finish | the user | ▶ cubebox v0: eval `cubebox_v0_baseline` frozen 2026-11-20, ready for its 20 trials; cubebox v1 and sort_one: evals `cubebox_v1_baseline` and `sort_one_baseline` written 2026-11-21, frozen when all 7 are trained. Commands: WORKFLOW.md, "The plan, step by step" |
| 6 | Bench (each fault type separately, sort_one, SmolVLA + GR00T; **replaced by setup v2, E1-E3**) and offline M9/M10 for every model | bench: Claude; offline: the user (WORKFLOW.md, "The plan, step by step", step 6) | ▶ bench ✓ (4 benches x 64 runs, A100, 2026-11-20); offline M9/M10 ✓ for the 8 trained models (H200, 2026-11-20); the others as their trainings finish |
| 7 | One report with all results | Claude | ▶ started 2026-11-20: the LaTeX report `report/` (built from the results, `make`) and the slides (REPORT.md section 12); extended as results come in |
| 8 | If time: the DRTC experiment on cubebox v1 (on the robot; its bench is in setup v2) | the user | · optional |
| 10 | **Setup v2** (top of this file): build, then E1 + E2, E3a, E3b on the bench (sort_one + cubebox v1, 5 models), then E4 on the robot | build: Claude; bench: where and when the user decides; E4: the user | ▶ locked 2026-11-22; build next |
| 9 | The Raspberry Pi setup | Claude | ✓ set up and tested 2026-11-20 (moved earlier, the user's call): `lab/bin/pi-setup`, calibration from the laptop, the Pi's key on the A100; a test trial ran (29.4-29.9 Hz, round trip ~2x the laptop's). Official trials stay on the laptop unless decided otherwise |
