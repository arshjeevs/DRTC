# Plan: original vs extras (2026-11-26)

Two lists. **Original** = the course project; it goes in the report and the deck. **Extras** = built on the side,
kept out of the report and deck. An extra moves to Original only when the user says so (then it is written into the
report/deck). Extras are worked by the Gemini swarm (branch `swarm-experiment`) in parallel; Claude only checks the
results and merges what the user approves.

## Original (Claude)

| # | Task | State |
|---|------|-------|
| O1 | Robot runs, smoothing study, edge chapter | done, in report |
| O2 | Bench E3a/E3b (smoothing stacks) | done, in report |
| O3 | Bench E1/E2 sorting (sort_one) | done, in report; regenerated figures not committed yet |
| O4 | Bench E1/E2 cube in box | bench COMPLETE 2912/2912 on the H200 -> metrics on the H200 -> ch. 9 text + viva todo |
| O5 | E4 robot stack (RTC or no RTC) | **user decision** -> unblocks tab_e4, tab_base_sort_one, tab_base_cubebox_v1 |
| O6 | Robot baselines (sort_one_baseline, cubebox_v1_baseline), all 7 models x 10 = 70 each, plain LeRobot, **Pi as the robot computer**, H200 serving | sort_one DONE 70/70 (pushed). cubebox run 1 DISCARDED (labels entered by another person; archived in the store at evals/_archive/cubebox_v1_baseline/ with ARCHIVED.md); re-run from scratch. **The report must say a first cube-in-box run was discarded and why.** |
| O6b | E4 robot runs (DRTC) | after O6, if time; RTC choice still open (O5) |
| O7 | Report final pass + PDF | after O4-O6 |
| O8 | Deck final pass | after O4-O6 |
| O9 | Viva prep (viva/viva.tex) + the 3 Gemini questions + "patient re-sends once" fix | after O4 |

## Extras (swarm, parallel)

| # | Extra | State | Next |
|---|-------|-------|------|
| X1 | DRTC Fault Simulator (presentation/simulator/) | built | user decides: promote? |
| X2 | Live demo console (`lab demo`) | built, tested | user decides |
| X3 | LIBERO simulation (sim/) | eval done 240/240 (results/sim/libero_smolvla.jsonl) | tabulate into SIM-PLAN.md only |
| X4 | Paper (paper/) | 7-page draft | authors later (user) |
| X5 | Website (website/) | built locally, not published | user: what it is / publish? |
| X6 | Project video | 2:36, has simulator + E1 scenes | keep scenes as is for now |
| X7 | LeRobot PR (~/Projects/lerobot-drtc-pr) | branch ready, not pushed | never opened without the user |
| X8 | Publishable repo (README, CI) | done | repo URL for X5 |

## Decisions (2026-11-26)

- Robot runs: baselines (no DRTC) first, official from the Pi; E4 (DRTC) after if time. RTC choice for E4 still open.
- Website: user asked what it is; open.
- Paper authors: later.
- Video scenes: not for now (leave the video as it is).
