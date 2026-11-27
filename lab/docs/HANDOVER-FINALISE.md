# Handover: finalising the project (2026-11-26)

For a fresh session that picks up from here. Everything below is on branch `major`, head `51e9c836`, pushed.

## The split: core vs extras

- **Core project** (goes in the report `report/` and the deck `presentation/`): the robot runs, the setup-v2 bench
  (E1-E3), the smoothing study, the edge chapter. The report is auto-filled from `report/generated/numbers.tex`
  (`make -C report`); the deck is `python presentation/build.py`.
- **Extras** (built, on `major`, but **kept out of the report and deck** until the user finalises them, see memory
  [[extras-not-in-report]]): the DRTC Fault Simulator (`presentation/simulator/`), the live demo console
  (`src/drtc/demo/`, `lab demo`), the LIBERO simulation (`sim/`, `lab/docs/SIM-PLAN.md`), the paper (`paper/`), the
  website (`website/`, built locally, **not published**). The project video (`presentation/out/drtc_video.mp4`) does
  include two extra scenes (simulator, E1); the user has not yet said to remove them.

## Two jobs were running when this session ended — check them first

1. **Cube-in-box bench (core)** on the H200 via the `cm-h200` socket. When it finishes (all 2912 counted + pushed):
   compute metrics ON THE H200 at low priority (the local pull is very slow):
   `docker exec drtc_lab bash -c "cd /workspace/drtc_major && nice -n 19 ionice -c3 .venv/bin/lab eval report bench_v2_drtc_cubebox_v1"`, then `cat` back `results/bench/bench_v2_drtc_cubebox_v1/{metrics,trials}.csv` + `report.md`,
   `make -C report data && make -C report`. Then write the cube-in-box E1/E2 text in
   `report/chapters/09_results.tex` (the sorting text is the model; keys are `v2/cubebox_v1/e1/...` and
   `v2/cubebox_v1/e2/...`, exported by `report/scripts/v2_results.py`), fill the viva `\todo` for cube in box
   (`viva/viva.tex`, the E1 answer ends "Cube in box is \todo{...}"), and rebuild. The watcher script is
   `/tmp/claude-1000/drtc-demo/watch.sh` (re-arm after any H200 reconnect; the socket drops — reconnect command is in
   memory [[h200-user-runs-commands]], the ServerAlive variant is safer).
2. **LIBERO eval (extra)** on the laptop: `sim/eval.py`, 240 episodes to `results/sim/libero_smolvla.jsonl`
   (log `/tmp/claude-1000/drtc-demo/libero_eval.log`). When done, tabulate success per client x fault and write it
   ONLY into `lab/docs/SIM-PLAN.md`, not the report. It restarts + skips finished episodes if re-run.

## Decisions still needed from the user (these unblock the pending report tables)

1. **E4 robot stack: RTC or no-RTC.** The pre-registered rule chose RTC; the E3b ablation says no-RTC is better. This
   is the user's call and must not be changed without them. It unblocks `tab_e4` and the robot baselines
   (`tab_base_sort_one`, `tab_base_cubebox_v1`) — 3 tables still "(pending)" in the report.
2. **Website:** publish or not, where, and the real repo URL (currently placeholder `OWNER/REPO`; pass
   `--repo <url>` to `website/build.py`). Not published yet.
3. **Paper author names** (`paper/paper.tex`, "Author names to be added").
4. Whether to **remove the simulator/E1 scenes from the video** (user asked to keep extras out of report + deck; did
   not mention the video).

## What was finished this session (all pushed)

Stages 0-5 of the extras plan ([[extras-order]], [[autonomous-extras]]): swarm tool (branch `swarm-experiment`,
worktree `~/Projects/drtc-major-swarm`, Hydrogen only — [[swarm-experiment]]); simulator; demo console (laptop GPU,
tested; `lab/docs/DEMO.md`); E1/E2 sorting in report+viva+deck+PR notes; LIBERO smoke test passed; paper draft;
website; proofreading; video 2:36; two new setup slides (hardware; VLA-in-one-loop) before "The problem"; slide-6
animation + factual fixes; an animated DRTC explainer slide. LeRobot PR is prepared only (`lab/docs/LEROBOT-PR.md`),
never pushed/opened.

## Standing rules (from memory)

Weekly Claude usage must stay **under 40%** ([[claude-budget]]; check with `claude -p "/usage"`, was 38%). Offload
reading/drafting to the Gemini swarm. **Never delete** without a fresh explicit yes ([[no-delete-without-permission]]).
No E4 stack change, no public PR, no website publish without the user. H200: work only in `drtc_major`, kill only own
PIDs, never touch other processes. Do **not** attack or probe third-party sites.
