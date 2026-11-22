# Metrics v3

**Fixed on 2026-11-20 with the user (v1), revised the same day for the locked scope (v2), and on 2026-11-22 for setup
v2 (v3, see Changes).** Every official result and the final presentation use these definitions. A change makes a new
version: the definitions here stay as they are, the change and its date are
added below, and `lab eval report` recomputes every trial from its raw files, so old and new results stay comparable.

Where the data comes from: [EVALUATION.md](docs/EVALUATION.md) (one folder per trial; raw files never edited).

## Terms

- **Trial:** one attempt at the task on the robot (or on the replay robot, Phase 7-8 bench), from pressing Enter to
  the operator's key or the time limit.
- **Condition:** what is varied in a study: baseline, a smoothing method, a network fault, an estimator.
- **Tick:** one control step of the robot computer (30 per second). The robot computer is the laptop or the Pi.
- **Executed action / commanded joints:** the joint targets actually sent to the arm, after DRTC's queue and any
  filter. **Arm joints** are the five joints in degrees (shoulder pan, shoulder lift, elbow flex, wrist flex, wrist
  roll); the gripper (0-100 %) is reported separately where it matters, never mixed into degree averages.
- **First action:** the first tick that sends an action from a model chunk. Start-up (loading, the first request)
  is before it and is not counted.
- **Time** is measured with each machine's own monotonic clock, never across machines, so clock offsets (the H200's
  clock is off; the Pi's may be) cannot distort a metric.

## Task

| # | Metric | Definition | Per trial | Across trials |
|---|---|---|---|---|
| M1 | **Success rate** | The operator presses the success key before the time limit, and the result meets the task's written success rule (in the eval file, fixed before the first trial). A failure key, the time limit, or a **terminal fault after the first action** (the system stopped the arm; label `fault`) is a failure. A fault before the first action is a setup problem: the trial is not counted and is run again. | 1 or 0 | successes / valid trials, with a Wilson 95% interval |
| M1f | **Ended by a fault** | Trials that a terminal fault ended after the first action (they count as failures in M1). Read from the run's own record (its outcome and first action), whatever key the operator pressed (v3). | 1 or 0 | fraction of valid trials |
| M1t | **Time to success** | Seconds from the first action to the success key press. | seconds (successful trials only) | median and interquartile range |

## Network and timing (headline for Phase 8)

| # | Metric | Definition | Per trial | Across trials |
|---|---|---|---|---|
| M2 | **Stall fraction** | Ticks after the first action at which no new action from a chunk was available (the arm holds its position), divided by all ticks after the first action. | % | median, IQR |
| M3 | **Stall events, longest stall** | A stall event is a run of consecutive stall ticks. The longest stall is the longest run, in seconds (ticks / 30). | count; seconds | median, IQR |
| M4 | **Turnaround** | For every accepted chunk: time on the robot computer from sending the observation to receiving its chunk. | median and 95th percentile, ms | median, IQR of the per-trial values |
| M5 | **Latency estimate error** | For every accepted chunk: the scheduler's latency estimate at the moment the request was sent, minus that request's measured turnaround (positive = overestimate). | median signed error and median absolute error, ms | median, IQR |
| M6 | **Server inference time** | For every accepted chunk: the time the server reports for producing it (LeRobot's `server_durations`, total). M4 - M6 is the network and queueing share. | median, ms | median, IQR |

## Smoothness (headline for Phase 7)

| # | Metric | Definition | Per trial | Across trials |
|---|---|---|---|---|
| M7 | **Jerk** | The commanded arm-joint positions on the 30 Hz tick grid (held through stalls), third finite difference x 30^3, root mean square over the five arm joints and **the samples while the arm moves** (v3: a sample spans 4 ticks and counts when at least one of them executed a new action, so stops and restarts count and a long hold does not). Lower is smoother. | 10^3 deg/s^3 | median, IQR |
| M8a | **Chunk switch disagreement** | At every switch to a new chunk: over the steps both the old and the new chunk cover (the overlap), the mean Euclidean distance between their arm-joint actions (DRTC's `chunk_mean_l2`). | median over switches, deg | median, IQR |
| M8b | **Largest step** | The largest change of any arm joint between two consecutive commanded positions. | deg | median, IQR |
| M12 | **Deviation from the plan** (v3) | On every executed tick: the mean over the five arm joints of \|commanded position - the model's own action for that step\| (the newest accepted chunk's raw action, before ensembling, zero-lag smoothing, the ramp or a filter; `plan.*` in `ticks.csv`). How far smoothing moves the arm off what the model asked for; 0 without smoothing. | deg | median, IQR |

## Recovery (v3)

| # | Metric | Definition | Per trial | Reported |
|---|---|---|---|---|
| M11 | **Recovery time** | For each timed fault (spike: until its delayed reply is released; drops, outage, rise: until the window ends; times include the trial's shift): seconds from the fault's end until the arm moves again, i.e. the end of the first stall still going at that moment or starting up to 1 s after it (0 when there is none). **Not recovered** when the arm never moves again, or the system ended the run before the fault ended. Not counted for that fault when the operator's key or success ended the run first. `loss` has no end and no M11. | seconds, or not recovered, per fault | recovered / counted; median [IQR] of the recovered |

## Without the robot

| # | Metric | Definition | Reported |
|---|---|---|---|
| M9 | **Held-out action error** | On the dataset's held-out episodes (seeded split, never trained on): from every 10th frame, the model predicts a chunk from the recorded observation; the first 15 predicted actions (0.5 s, which every model's chunk covers) are compared with the 15 recorded actions that follow. Mean absolute error over those actions and frames: arm joints in degrees (mean of the five, and each), gripper in %. Flow and diffusion models: one sample, fixed seed 0. | per model, per kept checkpoint; the root (served) checkpoint marked |
| M10 | **Inference time, GPU memory** | Server inference time (as M6) over 200 requests made from held-out observations after 10 warm-up requests, and peak GPU memory, per model per GPU. | median and 95th percentile, ms; GB |

## Rules for reporting

1. **Trials:** 10 valid trials per condition, per model, per task (`trials: 10` in the eval file; v1: 20). The operator chooses the
   layouts; all trials of one round use the same layout.
2. **Order:** in rounds. Each round runs every model x condition once, in a seeded random order, so drift (light,
   motor temperature, battery) affects all of them alike.
3. **Valid trials:** a trial voided for a setup mistake (`lab eval void`, with a reason) is kept and listed in the
   report, but not counted. Voiding is for setup mistakes only (cube not on its mark, wrong layout, cable caught),
   never for a model's failure.
4. **One value per trial first:** every metric is reduced to one value per trial (column "Per trial"), then
   summarised over trials, so a long trial does not weigh more than a short one.
5. **Comparisons:** an improvement is always the same model on the same task, inside the same eval (sessions
   interleaved). Each eval names its comparisons (`comparisons:`, "<condition> vs <reference>"; older evals: every
   condition against the first and each against the one before). Differences use Fisher's exact test (M1) and the
   Mann-Whitney U test (the others); p-values are reported, and the intervals shown.
6. **Grouping:** a report never averages over different robot computers (laptop, Pi) or servers (laptop GPU, A100,
   H200): when an eval has several, the timing metrics (M2-M6) are shown per robot computer and server.
7. **Primary comparisons (v3):** each eval names the few comparisons it tests (`primary:`), before its runs. Their
   metrics: M1 on the robot; on the bench M2 and the M11 of the condition's own faults (a run that never moved again
   ranks last). Their p-values are Holm-corrected over all primary tests of the eval, and the report counts in how
   many models the condition is better. Every other comparison is exploratory: effect and uncorrected p-value, marked
   as such. Setup v2: the bench's primary comparisons are DRTC against upstream (patient) per fault; E4's are its
   three claims (faults hurt, DRTC recovers, smoothing).
8. **Replay robot (bench):** M2-M8, M11, M12. M1 needs the real robot.
9. **The smoothing stack (setup v2):** chosen from the E3a bench by `python -m drtc.evals.stack` (rule in its
   docstring and lab/PLAN.md): per pair, candidates are ruled out if they add > 1 deg of M12 or raise M2 or any M11
   by > 10% (floors 0.5 percentage points, 0.05 s) against DRTC alone in their own eval; of the rest, the lower jerk
   relative to that reference wins; the candidate winning more pairs is chosen.

## Changes

**v3 (2026-11-22, setup v2, lab/PLAN.md; each item decided with the user):**
- M1f and the `fault` label: from the run's own record, whatever key was pressed. Before, an `f` pressed while the
  stopped arm returned home replaced `fault` (sort_one_drtc_main: 7 of 20 such trials; M1f was 4/10 and 9/10, is
  10/10 and 10/10). New trials store the key separately (`key`); no trial file was rewritten.
- M7: only while the arm moves. A long hold has zero jerk, so a system that stalled more looked smoother.
- New M11 (recovery time per fault) and M12 (deviation from the model's plan; needs `plan.*` in `ticks.csv`, logged
  from 2026-11-22, so older trials have no M12).
- Rules 5, 7, 9: declared comparisons, primary comparisons with Holm, the stack rule.
- Every report is rebuilt with v3 from the raw files (`lab eval report`), including the evals made under v2.

**v2 (2026-11-20, the locked scope, lab/PLAN.md):**
- M1: a terminal fault after the first action is a failure (v1: such trials were not counted and run again). Under
  network faults the fault *is* the outcome: a system that gives up must score lower, not be retried until it gets
  lucky. New M1f, the fraction of trials ended this way.
- Rule 1: 10 trials per condition, per model, per task (v1: 20).
- Every report is rebuilt with v2 from the raw files (`lab eval report`), including the evals made under v1.

**v1** (2026-11-20, morning): as above, except: a trial ended by a terminal fault was not counted and was run again
in a later round; 20 trials per model per condition.

Clarification (2026-11-20, still v1, before any official trial): while the arm holds through a stall, the control
loop does not ask for actions, so those ticks have no row in `ticks.csv`. M2, M3, M7 and M8b therefore use the tick
grid from the first action to the end of the run (marked by the last row): every tick without a new action is a
stall tick, and the commanded position is held through it.
