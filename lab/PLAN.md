# Major project plan

*What the project is doing now and where each part stands (2026-11-22). The scope was locked with the user on
2026-11-20 and the DRTC and smoothing experiments were redesigned as setup v2 on 2026-11-22. Earlier plans and every
decision on the way: [docs/history/](docs/history/) (PLAN-2026-11-18-to-20.md, PLAN-2026-11-20-to-22.md,
PROPOSAL.md). The commands: [WORKFLOW.md](WORKFLOW.md). The metrics: [METRICS.md](METRICS.md) (v3). The results so
far: [REPORT.md](REPORT.md).*

## The claim

In client-server robot inference, network faults lower the task score; **DRTC recovers it**, more than just waiting
does; **smoothing** then improves the motion without lowering success.

## Datasets and models

**Datasets** (as they are, no new recording): `so101_cubebox_v0` (50 = 45 + 5 held out; the first dataset, baseline
only, optional), `so101_cubebox_v1` (110 = 100 + 10), `so101_sort_one_v1` (88 = 80 + 8).

**Models:** 7 policies, each with its authors' fixed recipe (`lab/recipes/`): SmolVLA, GR00T N1.5, GR00T N1.7, π0.5,
MolmoAct 2 (LoRA, its recipe), ACT, Diffusion.

**Trainings** (✓ on the Hub, ▶ running, · to do; the user runs and reports them):

| Model | cubebox v0 | cubebox v1 | sort_one |
|---|---|---|---|
| SmolVLA | ✓ | ✓ | ✓ |
| GR00T N1.5 | ✓ | ✓ | ✓ |
| ACT | (✓, not needed) | ✓ | ✓ |
| Diffusion | – | ✓ | ✓ |
| GR00T N1.7 | – | ✓ | ✓ |
| π0.5 | – | · | · |
| MolmoAct 2 | – | · | ▶ A100 (last known) |

Still to do: π0.5 on cubebox v1 and sort_one (~27 h and ~77 GB each: the H200), MolmoAct 2 on cubebox v1. After each
training: its offline evaluation (WORKFLOW, Step 6), then it joins the baselines and the bench (they grow).

## Setup v2: the DRTC and smoothing experiments (locked 2026-11-22)

**Why it was redesigned.** In the first run (`sort_one_drtc_main`, 2026-11-20) every "baseline + faults" trial was
ended by upstream's own 1 s starvation grace at the first fault, so DRTC was compared with a system that gave up by
design; DRTC also differed from it in two things at once, and smoothing added two methods at once. That run and its
benches are side results.

**Systems**
- **upstream (patient)**: LeRobot's own scheduler with its two give-up limits, the starvation grace (1 s) and the
  request deadline (`action_timeout`, 5 s), raised to the whole trial. The baseline everywhere: each is one setting,
  so DRTC must beat upstream at its best, and no default ends a run.
- **upstream (as shipped)**: LeRobot's defaults. Bench only, to show the default breaks.
- **DRTC**: its trigger (`H - s_min`), the Jacobson-Karels latency estimate, the cooldown re-request with our
  2026-11-20 fixes (no H/2 cap), the same whole-trial limits as upstream (patient).
- **Smoothing methods**: adaptive playback, the Butterworth filter, RTC (flow-matching models with absolute actions:
  SmolVLA, GR00T N1.5; GR00T N1.7 is trained with relative actions, for which LeRobot turns RTC off), temporal
  ensembling, zero-lag queue smoothing, the resume ramp (its soft stop was dropped after the local check: it let a
  reply from a 1.3 s old observation be accepted and the next chunk jumped).

**Faults**: none; spike (one reply +1 s); drops (every reply lost for 1 s); outage (1.5 s, both directions); mix
(spike, drops 2 s later, outage 4 s later); rise (+400 ms on every reply for 5 s); loss (10% of replies lost, the
whole run). Timed faults move by a seeded random 0-4 s per trial number, the same for trial k in every condition.

**The bench** (replay robot on the held-out episodes, simulated network 250 ± 20 ms, 8 runs per cell, 14 s each),
on `so101_sort_one_v1` and `so101_cubebox_v1`, for SmolVLA, GR00T N1.5, GR00T N1.7, ACT, Diffusion (π0.5 and
MolmoAct 2 join when trained):

| Exp. | Eval(s) | What | Runs per dataset |
|---|---|---|---|
| E1 + E2 | `bench_v2_drtc_<dataset>` | upstream patient (7 faults), as shipped (6), DRTC (7); the ablations A1 no cooldown, A2 upstream's estimator, A3 upstream's trigger, A4 DRTC's original H/2 cooldown cap (7 faults each), A5 `s_min` x0.5 and x1.5 (none, mix) | 2,080 |
| E3a | `bench_v2_smooth_<dataset>`, `bench_v2_smooth_rtc_<dataset>` | each smoothing method alone on DRTC + mix (RTC: SmolVLA, GR00T N1.5) | 280 + 32 |
| E3b | `bench_v2_stack_<dataset>` (+ `_nortc_` when the stack has RTC) | the final stack, without faults, and without each of its parts | ~240 |

**The final stack** = adaptive playback + the resume ramp + one seam method (RTC or ensembling) + one jitter method
(zero-lag or Butterworth), chosen from E3a by the rule fixed before E3a ran (`python -m drtc.evals.stack`;
METRICS.md rule 9). Models without RTC use ensembling.

**Chosen 2026-11-25 (V3): adaptive + RTC + Butterworth + ramp** (ACT, Diffusion, GR00T N1.7, π0.5, MolmoAct 2:
ensembling instead of RTC). Seam: RTC won 1 pair (GR00T N1.5 sort_one), ensembling none; the other 3 RTC-capable pairs
ruled both out. Jitter: Butterworth won 2 pairs (ACT x2), zero-lag 1 (Diffusion cubebox); in the other 11 pairs both
were ruled out, Butterworth mostly by M12 (+1.0 to 1.6 deg from the plan), zero-lag by M11 recovery. Output of
`python -m drtc.evals.stack`; E3b: `bench_v2_stack{,_nortc}_<dataset>`, E4: `sort_one_drtc_v2`.

**The robot (E4, `sort_one_drtc_v2`)**: sort_one, SmolVLA + GR00T N1.5, the laptop, 10 trials per condition:
upstream patient without faults and with the mix, DRTC + mix, DRTC + the stack + mix (80 trials). Primary claims:
faults hurt even when waiting, DRTC's scheduling beats waiting, smoothing improves the motion.

**Robot evals grow** (locked 2026-11-22; one laptop, one session at a time). Core, 10 trials per condition: E4 (80)
and the baselines `sort_one_baseline` and `cubebox_v1_baseline` (upstream patient, no faults; the 5 trained models,
50 + 50). They only grow: more `trials`, or more `models` (freezing adds only the new ones); anything else is a new
eval. When there is time, in this order: E4 to 20 trials; the DRTC robot run on cubebox v1; E4 on a third model
(GR00T N1.7 if its baseline is good); the baselines to 20 trials; a robot run of a bench fault that surprised us;
optional, `cubebox_v0_baseline`.

**Rules** ([METRICS.md](METRICS.md) v3): rounds; a run the system ends after the first action is a failure (label
`fault`, M1f), whatever key is pressed; jerk only while the arm moves; M11 recovery per fault; M12 deviation from the
model's plan; each eval names its comparisons, and only the primary ones are tested (Holm). The robot computer is
the laptop for every official experiment (the Pi is set up and tested).

## Order of work (the user decides when each step starts)

| # | What | Who | Status |
|---|---|---|---|
| 1 | Trainings (table above) | the user | ▶ π0.5 x2 and MolmoAct 2 cubebox v1 left |
| 2 | Offline M9/M10 for every model | the user (WORKFLOW Step 6) | ✓ 10 models (SmolVLA, GR00T N1.5 x3 datasets; ACT, Diffusion x2); GR00T N1.7 x2 ready to run; the rest as they finish |
| 3 | Setup v2 build and eval setup (metrics v3), tested on the laptop's replay robot | Claude | ✓ 2026-11-22 |
| 4 | V0: freeze the bench evals | the user | · next |
| 5 | V1 + V2: E1 + E2 and E3a on the bench | the user, on a GPU server (WORKFLOW Setup v2) | · |
| 6 | V3: choose the stack, write E3b and E4 | Claude (from the stack command's output) | ✓ 2026-11-25 (written; to freeze) |
| 7 | V4: E3b on the bench | the user | · |
| 8 | Baselines on the robot (5 models now, sort_one + cubebox v1) | the user (WORKFLOW Step 5) | · ready to freeze any time |
| 9 | V5: E4 on the robot | the user | · after the stack |
| 10 | Growth when there is time (list above) | the user | · |
| 11 | The report (`report/`, LaTeX, built from the results) and the slides | Claude | ▶ extended as results come in |

Side results already in [REPORT.md](REPORT.md): the first DRTC run (`sort_one_drtc_main`, 80 trials) and its four
benches (`bench_main_*`), recomputed with metrics v3; offline M9/M10 of 10 models; the Raspberry Pi test.
