# Report plan (draft for approval, 2026-11-26)

The final project report: `report/` (LaTeX, NIT Calicut format, builds to `report/report.pdf`).
Nothing below is built until the user approves this plan.

## 1. Ground rules (decided)

| Rule | Meaning |
|---|---|
| Format | NIT Calicut final-year template (done, e001ca6b): Times 12 pt, 1.15 spacing, CHAPTER N headings, "Fig 3.1" captions, table captions below, NITC front matter. Every chapter ends with a Summary. |
| Scope | Every piece of **core** work done for the project. |
| Extras | Simulator, demo console, LIBERO, paper, website: **out**. Decided later, one by one. |
| Older DRTC results | First robot run (2026-11-20), first benches, bench E1-E3: **out for now**. Their sections stay, with a marked placeholder ("results to be added"). |
| E4 (robot DRTC) | The DRTC result. Sections, tables and charts are laid out now; they fill in from the trials when the runs finish. |
| Results that stay in | Robot baselines (7 policies, 2 tasks), offline evaluation, dataset statistics, training logs, Pi latency. |
| Figures | About 45-50: TikZ diagrams (drawn from the code, so they are accurate) and charts generated from the results files (they update themselves). |
| Length | About 100-120 pages. |

## 2. Chapters, sections, figures and tables

D = diagram (TikZ), C = chart (generated from data), P = photo (user provides; labelled placeholder until then),
T = table. "new" = not in the report yet; "have" = exists already.

### Chapter 1. Introduction
Sections: 1.1 General overview · 1.2 Motivation · 1.3 Problem statement · 1.4 Objectives · 1.5 Contributions ·
1.6 Organisation of the report · 1.7 Summary
- D1.1 new: **the project in one flow**: hardware -> data collection -> training -> serving -> evaluation -> results.
- D1.2 new: **the problem**: the arm's action queue over time; a network fault arrives, the queue runs dry, the arm stops.

### Chapter 2. Literature Review
Sections: 2.1 Introduction · 2.2 Learning manipulation from demonstrations · 2.3 Vision-language-action models ·
2.4 Action chunking · 2.5 The policies used in this work · 2.6 Asynchronous and remote inference ·
2.7 Real-time chunking (RTC) · 2.8 Distributed real-time chunking (DRTC) · 2.9 Smoothing robot commands ·
2.10 Limitations of existing systems · 2.11 Summary
- D2.1 new: **how a VLA works**: camera images + instruction + arm state -> vision encoder -> language-model backbone
  -> action expert (flow matching / diffusion) -> a chunk of future actions -> the arm.
- D2.2 new: **imitation learning loop**: demonstrations -> dataset -> training -> policy -> robot.
- T2.1 new: **the 7 policies side by side**: architecture, parameters, chunk length, action type, inputs.
- D2.3 new: **action chunking on a timeline**: chunks requested, received and executed.
- D2.4 new: **RTC**: a new chunk blended into the end of the old one.
- D2.5 new: **DRTC**: when it sends a request, the latency estimate, the cooldown re-request.

### Chapter 3. System Design
Sections: 3.1 Introduction · 3.2 Overall system architecture · 3.3 Hardware (arms, cameras, phone camera, laptop,
Raspberry Pi 5, GPU servers) · 3.4 Software: a layer on LeRobot 0.7 · 3.5 The registry · 3.6 Machines and how work
moves between them · 3.7 Summary
- D3.1 new: **system architecture**: arms, cameras, phone, laptop, Pi (cable/WiFi), H200, A100, Hugging Face Hub, wandb.
- D3.2 new: **software layers**: LeRobot 0.7 -> our `lab` layer (registry, plugins, DRTC, smoothing, faults, evals) -> commands.
- D3.3 new: **how work moves**: git, Hub (datasets, models, results store), SSH tunnel.
- P3.1-P3.3: the arm with leader/follower, the cameras, the full setup with Pi and laptop.
- have: the existing platform figures and tables (kept, restyled).

### Chapter 4. Data Collection
Sections: 4.1 Introduction · 4.2 Tasks (sorting, cube in box) · 4.3 Teleoperation and recording · 4.4 Trimming and
editing · 4.5 Versions and splits · 4.6 Summary
- D4.1 new: **data pipeline**: leader-follower teleop -> record -> trim idle start/end -> version -> Hub.
- D4.2 new: **task layouts** (top view of the taped marks) for sorting and cube in box.
- C4.1 new: **dataset statistics**: episodes per dataset, episode lengths before/after trimming.
- P4.1-P4.2: the two task layouts on the table.
- have: dataset table.

### Chapter 5. Policies and Training
Sections: 5.1 Introduction · 5.2 Choosing the policies · 5.3 Original recipes · 5.4 Training infrastructure ·
5.5 What was trained · 5.6 Offline evaluation (moved here from Results) · 5.7 Summary
- D5.1 new: **training pipeline**: recipe -> automatic dataset preparation -> GPU server -> wandb/Trackio -> Hub.
- C5.1 new: **training loss curves** of the 7 policies (exported from wandb).
- C5.2 new: **model size against inference time**.
- have: the recipe/model tables; the offline-evaluation figure and table.

### Chapter 6. Remote Inference and DRTC
Sections: 6.1 Introduction · 6.2 The client-server loop · 6.3 LeRobot's client (as shipped, patient) · 6.4 DRTC in
LeRobot's runtime, and our four fixes · 6.5 Smoothing methods · 6.6 Network fault injection · 6.7 Summary
- D6.1 new: **request/reply sequence diagram** between robot computer and model server.
- D6.2 new: **flowchart, LeRobot's client**: what it does when a reply is late or lost.
- D6.3 new: **flowchart, DRTC**: trigger, latency estimator, cooldown, hold.
- D6.4 new: **smoothing pipeline** (block diagram of the six methods and where each acts).
- D6.5 new: **the seven fault types on a timeline**.
- have: the existing control figures and tables.

### Chapter 7. Evaluation Framework
Sections: 7.1 Introduction · 7.2 Three kinds of evaluation (offline, robot, bench) · 7.3 Evaluation plans (freeze,
rounds, voids) · 7.4 Records and the results store · 7.5 Metrics · 7.6 Statistics · 7.7 Choosing the smoothing stack ·
7.8 Summary
- D7.1 new: **evaluation workflow**: plan -> freeze -> serve -> run -> push -> report.
- D7.2 new: **metric definitions on a timeline**: stall, jerk, recovery time, plan deviation.
- D7.3 new: **stack-selection rule** as a flowchart.

### Chapter 8. Experiment Design
Sections: 8.1 Introduction · 8.2 Setup v2: the bench experiments E1-E3 · 8.3 The robot experiment E4 (4 policies x
2 tasks x 4 conditions, Pi over cable) · 8.4 Robot baselines · 8.5 Claims and primary comparisons · 8.6 Protocol ·
8.7 Summary
- have: the setup v2 figure (redrawn in the new style).
- D8.1 new: **the E4 condition ladder**: no faults -> faults -> DRTC -> DRTC + stack, and what each step shows.
- The first run (2026-11-20) is **held out** with its results (see 3).

### Chapter 9. Results
Sections: 9.1 Introduction · 9.2 Robot baselines (have, in) · 9.3 Bench: DRTC against LeRobot's client (E1) and the
ablations (E2) **placeholder** · 9.4 Bench: smoothing (E3a, E3b) **placeholder** · 9.5 Robot: DRTC under network faults
(E4) **placeholder, fills in from the trials** · 9.6 Summary
- C9.1 new: **baseline chart**: success of 7 policies x 2 tasks with 95% intervals.
- C9.2-C9.5: E4 success, stall, jerk and recovery charts per task (laid out now, fill in from the trials).
- E1-E3 charts and tables: slots kept, shown as "results to be added".

### Chapter 10. Edge Deployment on the Raspberry Pi
Sections: 10.1 Introduction · 10.2 Setup · 10.3 The Pi as the robot computer for the baselines · 10.4 WiFi against
cable · 10.5 Summary
- C10.1 new: **round trip per policy, WiFi against cable** (from the baseline and E4 trial logs).
- D10.1 new: **where the round-trip time goes**: image upload, inference, reply.

### Chapter 11. Discussion
Sections: 11.1 Introduction · 11.2 What the baselines show · 11.3 DRTC on the robot **placeholder (after E4)** ·
11.4 Short-chunk policies and the round trip · 11.5 Lessons from the implementation · 11.6 Limitations · 11.7 Summary
- D11.1 new: **when DRTC can help**: chunk length against round trip.

### Chapter 12. Conclusion and Future Work
Written last, after E4.

### Front and back matter
Title page, certificate, acknowledgement (done) · Abstract (rewritten without the older DRTC numbers) · Contents,
figures, tables, abbreviations · References · Appendices A recipes, B metric formulas, C conditions, D reproducibility.

## 3. What happens to the existing text

| Existing content | Plan |
|---|---|
| Chapters 2-7 (background, platform, data, policies, control, evaluation) | Kept; sections renamed and reordered as above; diagrams added. |
| Ch. 8 "The first run", "Why it was redesigned" | Held out (moved to `report/held/`, not built, nothing deleted). |
| Ch. 9 first run, first benches | Held out. |
| Ch. 9 bench E1-E3 text, tables, figures | Held out; placeholders in their sections. |
| Ch. 9 offline evaluation | Moved to chapter 5. |
| Ch. 9 baselines, E4 | Kept (E4 with placeholders). |
| Ch. 11 discussion of the first run and the bench | Held out; kept: lessons, limitations. |
| Ch. 12, abstract, introduction (they quote first-run numbers) | Rewritten without them. |

## 4. Build order and review stops

1. **Skeleton**: new structure, the held-out moves, every figure as a labelled slot saying what it will show.
   -> PDF to the user for review.
2. **Diagrams**: the TikZ diagrams, chapter by chapter (D1.1 ... D11.1). -> review.
3. **Charts**: generated from the results files (C4.1, C5.1, C5.2, C9.1, C10.1; E4 charts wired to fill in). -> review.
4. **Text**: introduction, abstract, chapter summaries, consistency pass. -> review.
5. **E4 results**: when the runs finish, the tables and charts fill in; then the discussion (11.3) and the conclusion.
6. Then the presentation, from the finished report.

## 5. Open points for the user

1. Photos: about 5 (arm, cameras, full setup with Pi and laptop, the two task layouts). Placeholders until then.
2. Bench E1-E3: they are DRTC results from setup v2 (not the old first run). Bring them back after E4, or keep them out?
3. Training loss curves come from wandb (project drtc-lab); if a run is not there, its curve is left out.
