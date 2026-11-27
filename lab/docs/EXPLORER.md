# Results explorer (`lab explore`)

A local web app for showing the robot results to the panel: one window with the whole picture of each evaluation,
one window with every trial's videos and what happened inside the run. It reads the results already on this computer
and changes nothing; no GPU, no internet.

## Start it

```bash
cd ~/Projects/drtc-major
.venv/bin/lab eval pull cubebox_v1_drtc_v2      # only if the results are not on this computer yet (same for the two baselines)
.venv/bin/lab explore                            # opens http://127.0.0.1:8770 in the browser
```

| Option | What it does |
|---|---|
| `--port 8771` | another port (8080 is refused: that is the robot session's) |
| `--no-open` | do not open the browser |
| `--host 0.0.0.0` | open it from another device on the same network (a projector laptop, a tablet): `http://<this laptop's IP>:8770` |
| `--present` | for the panel: the chosen videos only, no "Choose videos" button (nothing can be changed) |
| `--videos FILE` | another file of chosen videos (default `lab/explorer/videos.yaml`), e.g. one per talk |
| `--evals A B` | show other evals (default: the three below) |

Ctrl-C stops it. It reads the results once at start; after a new `lab eval pull`, restart it (or open `/api/data?reload=1`).

## What it shows

Three evaluations: **DRTC on the robot** (E4, `cubebox_v1_drtc_v2`), **Baseline: cube in box** (`cubebox_v1_baseline`)
and **Baseline: sorting** (`sort_one_baseline`).

**Results** (`/`) opens on **Key findings**: the main results, the ones that make the work new, from the E4 trials
(numbers as the report's abstract and claims):

- The headline: success 68% → 13% under the fault mix → 65% with DRTC → 75% with the smoothing stack, 3.2× smoother
  (pooled over the four policies), and what is new in the work.
- Finding 1: the waiting client itself ended 32 of 40 runs under faults (DRTC 3, the stack 2).
- Finding 2: after the link went down the arm moved again in 31 of 33 runs with DRTC, 2 of 34 with the waiting client.
- Finding 3: every policy wins success back; GR00T N1.7 0 → 8 of 10, significant after Holm over all 12 tests.
- Finding 4: jerk 52 → 16 with the stack, 3.0-5.6× per policy, success 26/40 → 30/40.
- Finding 5: the limit: GR00T's chunk (0.53 s) is shorter than its round trip (0.86-0.97 s), so it stands still 65% of
  the time even without faults (the report's Figure 11.1).

Each finding has "▶" links to watch the trials behind it. The cards at the top switch to one evaluation in full:

- The setup: task, success rule, plan, Pi → server, dates, counted trials; for E4 the four conditions and the fault
  mix drawn on a time axis (with the random 0-4 s shift).
- Headline tiles: successes of all policies together per condition (E4: 27/40 → 5/40 → 26/40 → 30/40).
- Success per policy with 95% Wilson intervals.
- The planned tests (E4): Fisher's exact test per policy, Holm-corrected over the 12 tests, exactly as
  `lab eval report` (only GR00T N1.7, DRTC against patient under faults, is significant after the correction); the
  pooled rows are exploratory.
- How the runs ended: success, failed (operator), time limit, or ended by the client during a fault.
- Timing and smoothness: pick a measure (stalled time, longest stall, jerk, time to success, turnaround, ...), with a
  one-line explanation of each.
- Recovery after each fault (E4): in how many runs the arm moved again after the spike, the lost replies and the link
  down, and how fast.
- Every number (median [IQR]) and the sessions / trials not counted, folded at the bottom.

Every tile, bar, dot and table row opens the matching trials in the Trials window.

**Trials & videos** (`/trials`):

- Left: filters (condition, policy, ending, with video only) and the trial grid: one square per trial (✓ success,
  ✗ failed, ⏱ time limit, ⚡ ended by the fault, – not counted; a dot = no video). `←` / `→` go to the previous / next
  trial, space plays.
- The two cameras (scene and wrist) play in sync; speed 0.25× to 2×.
- "What happened, on one clock": the fault windows (orange), every request from sending to its reply, lost requests
  and replies (✕), DRTC's re-sends (▲), the action queue, when the arm stood still (red), and one joint, commanded and
  measured. The playhead follows the video; click or drag on the timeline to jump; hover for the values at that moment.
- This trial against its group (the median of its 10 trials), the run's facts and its events (click one to jump).

Good trials to show: the same policy and trial number under the four E4 conditions (the fault pattern is the same
for a given trial number), e.g. GR00T N1.7: patient under faults ends at the link down, DRTC re-sends and finishes.

## Choosing which videos to show

1. Trials & videos → **☆ Choose videos** (top right). Every trial is shown while you choose.
2. Open a trial (click its square) and press **☆ Show this video** (or the `C` key); it turns into **★ Chosen**, and
   its square gets a blue ring. Only trials with a video can be chosen. Switch evaluations in the same mode.
3. **Save**. From then on the Trials window shows **only the chosen trials** (the grid, the filters, ← / → and the
   evaluation list); an evaluation with none chosen is left out. **Cancel** drops the changes.

The choice is saved in `lab/explorer/videos.yaml` (one list of trial ids per evaluation, in grid order). Commit it to
take it to another laptop. To change it, press **Choose videos** again; to show every trial again, delete the file (or
an evaluation's list). For the panel, start with `lab explore --present`: the same chosen videos, without the button.
The Results window is not affected (it shows no videos); its links open the chosen trials of that group, if any.

## How it works

`src/drtc/explorer/`: `data.py` builds the numbers from the raw trial files with the same code as `lab eval report`
(`drtc.evals.metrics`), so the explorer and each eval's report.md agree; `server.py` serves the two pages
(`web/`), the data (`/api/data`, `/api/trial`) and the videos (byte ranges, only camera files of known trials).
Fonts come from `presentation/fonts`. The choice: `POST /api/selection` (refused with `--present`) writes the file.
