# Metrics v1

**Fixed on 2026-11-20 with the user.** Every official result in Phases 6-8 and in the final presentation uses these
definitions. A change makes a new version (v2): the definitions here stay as they are, the change and its date are
added below, and `lab eval report` recomputes every trial from its raw files, so old and new results stay comparable.

Where the data comes from: [EVALUATION.md](EVALUATION.md) (one folder per trial; raw files never edited).

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
| M1 | **Success rate** | The operator presses the success key before the time limit, and the result meets the task's written success rule (in the eval file, fixed before the first trial). A failure key, or the time limit, is a failure. | 1 or 0 | successes / valid trials, with a Wilson 95% interval |
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
| M7 | **Jerk** | The commanded arm-joint positions on the 30 Hz tick grid (held through stalls), third finite difference x 30^3, root mean square over the five arm joints and all ticks after the first action. Lower is smoother. | 10^3 deg/s^3 | median, IQR |
| M8a | **Chunk switch disagreement** | At every switch to a new chunk: over the steps both the old and the new chunk cover (the overlap), the mean Euclidean distance between their arm-joint actions (DRTC's `chunk_mean_l2`). | median over switches, deg | median, IQR |
| M8b | **Largest step** | The largest change of any arm joint between two consecutive commanded positions. | deg | median, IQR |

## Without the robot

| # | Metric | Definition | Reported |
|---|---|---|---|
| M9 | **Held-out action error** | On the dataset's held-out episodes (seeded split, never trained on): from every 10th frame, the model predicts a chunk from the recorded observation; the first 15 predicted actions (0.5 s, which every model's chunk covers) are compared with the 15 recorded actions that follow. Mean absolute error over those actions and frames: arm joints in degrees (mean of the five, and each), gripper in %. Flow and diffusion models: one sample, fixed seed 0. | per model, per kept checkpoint; the root (served) checkpoint marked |
| M10 | **Inference time, GPU memory** | Server inference time (as M6) over 200 requests made from held-out observations after 10 warm-up requests, and peak GPU memory, per model per GPU. | median and 95th percentile, ms; GB |

## Rules for reporting

1. **Trials:** 20 valid trials per model per condition (`trials: 20` in the eval file). The operator chooses the
   layouts; all trials of one round use the same layout.
2. **Order:** in rounds. Each round runs every model x condition once, in a seeded random order, so drift (light,
   motor temperature, battery) affects all of them alike.
3. **Valid trials:** a trial voided for a setup mistake (`lab eval void`, with a reason) is kept and listed in the
   report, but not counted. Voiding is for setup mistakes only (cube not on its mark, wrong layout, cable caught),
   never for a model's failure.
4. **One value per trial first:** every metric is reduced to one value per trial (column "Per trial"), then
   summarised over trials, so a long trial does not weigh more than a short one.
5. **Comparisons:** an improvement is always the same model on the same task, inside the same eval (sessions
   interleaved), against that eval's baseline condition. Differences are tested with Fisher's exact test (M1) and the
   Mann-Whitney U test (the others); p-values are reported, and the intervals shown.
6. **Grouping:** a report never averages over different robot computers (laptop, Pi) or servers (laptop GPU, A100,
   H200): when an eval has several, the timing metrics (M2-M6) are shown per robot computer and server.
7. **Replay robot (bench):** M2-M8 only. M1 needs the real robot.

## Changes

None yet (v1).
