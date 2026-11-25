# Contributing DRTC to LeRobot

The changes are ready on a local branch. **Nothing has been pushed or opened**: a pull request is public, so you
submit it.

- Clone: `~/Projects/lerobot-drtc-pr` (huggingface/lerobot, `main` at `b07adcc5`, 2026-11-25)
- Branch: `remote-inference-unreliable-networks`, two commits:
  1. `fix(remote_inference): cap the request delay at the policy's prediction steps`, a bug fix that stands alone
  2. `feat(remote_inference): opt-in scheduling for slow or unreliable links (DRTC)`, the feature
- Checked with LeRobot's pinned tools: `ruff 0.14.1` (format + check), `prettier 3.6.2` (docs), `typos`; LeRobot's
  `tests/inference` and `tests/remote_inference`: **277 passed, 2 skipped** (11 new tests).

## What is in it

| Setting (`--inference.`) | Default | What it does |
|---|---|---|
| `request_trigger=chunk_margin`, `trigger_margin_steps` | `playback`, 20 | DRTC's trigger: request at H - s_min |
| `latency_estimate=smoothed` | `recent_max` | Jacobson-Karels instead of the max of the last 100 |
| `request_cooldown`, `cooldown_margin_steps` | off, 1 | DRTC's cooldown re-request, with our fixes 1-3 built in |

Fix 4 of our report (the "Invalid RTC delay" crash) is commit 1. Our fixes 1-3 are the cooldown's rules: never
shorter than the measured latency (no H/2 cap), covering the slowest of the last ten replies, and the startup budget
before the first reply. The smoothing methods and the fault injector are **not** in the PR: they are a separate,
larger discussion; start with the scheduling.

Not checked yet: an end-to-end run with the PR's code (the lab's own engine, `src/drtc/control/engine.py`, is what
the bench measured; the PR is a fresh port of it into LeRobot's runtime). Before opening the PR, run one replay
session with it, e.g. on the laptop GPU:

```bash
cd ~/Projects/drtc-major
PYTHONPATH=~/Projects/lerobot-drtc-pr/src .venv/bin/lerobot-rollout ... \
  --inference.type=remote --inference.request_trigger=chunk_margin --inference.trigger_margin_steps=20 \
  --inference.latency_estimate=smoothed --inference.request_cooldown=true
```

## How to submit

1. Fork huggingface/lerobot on GitHub (your account).
2. `cd ~/Projects/lerobot-drtc-pr && git remote add fork git@github.com:<you>/lerobot.git`
3. `git fetch origin && git rebase origin/main` (main moves daily), rerun the tests:
   `PYTHONPATH=src ~/Projects/drtc-major/.venv/bin/python -m pytest -q tests/inference tests/remote_inference`
4. Either one PR with both commits, or (what LeRobot reviewers usually prefer) two: push commit 1 alone first
   (`git push fork 4d7c86ea:refs/heads/fix-rtc-delay-cap`), then the branch.
5. `git push fork remote-inference-unreliable-networks` and open the PR with the text below. LeRobot asks for an
   issue or discussion first for features; the bug fix can go straight in.

## PR text (feature)

**Title:** feat(remote_inference): opt-in scheduling for slow or unreliable links (DRTC)

**What this does**

Remote inference over Wi-Fi or a shared network sees late and lost replies. Today the client keeps only the refill
threshold of actions in reserve (about 0.5 s) and waits for each reply until its deadline, so one lost reply empties
the queue and the robot holds until the starvation grace ends the run. This adds three opt-in settings after
Distributed Real-Time Chunking (DRTC, J. Vial): an earlier request trigger (`request_trigger=chunk_margin`), TCP's
Jacobson-Karels latency estimate (`latency_estimate=smoothed`), and a cooldown that replaces an unanswered request
with a fresh observation (`request_cooldown=true`). Defaults are unchanged.

**Why the cooldown rules look like this.** On an SO-101 with GR00T N1.5 (16-action chunks, 0.53 s at 30 Hz) served
over Wi-Fi (0.45 s turnaround), DRTC's original cooldown, capped at half a chunk, gave up every request before its
reply arrived. The cooldown here is never shorter than the measured latency, also covers the slowest of the last ten
replies (otherwise replies near the estimate are given up and the server, still computing them, slows further), and
uses the startup budget before the first reply.

**Evidence** (our project on LeRobot 0.7, SO-101, sorting task, 10 trials per cell): under a fault mix of a 1 s
latency spike, 1 s of lost replies and a 1.5 s outage, the client as shipped ended every faulted trial (success 0%
for SmolVLA and GR00T N1.5); with these scheduling changes success was 60% and 80%, and the arm moved again within
about 1.5 s of every fault. A replay bench with 7 policies compares it with the client given whole-trial limits
(TODO: add E1/E2 numbers from report Chapter 9 when the bench finishes).

**How it was tested**: `tests/inference/test_unreliable_network_scheduling.py` (trigger, estimator, cooldown before
and after the first reply, replacement and stale late reply, BUSY backoff, validation); the existing inference and
remote-inference suites pass.

## PR text (bug fix)

**Title:** fix(remote_inference): cap the request delay at the policy's prediction steps

`estimate_delay` turns the turnaround into steps without a bound. When the turnaround is longer than one predicted
chunk (for example GR00T's 16 steps at 30 Hz over a 0.6 s link gives 18), the server rejects the request with
`Invalid RTC delay` and the run ends. The runtime now gets the policy's prediction steps from the server's
capabilities and caps the delay there. Tests in `tests/inference/test_aligned_execution.py`.
