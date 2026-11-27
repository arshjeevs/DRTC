# The live demo console

`lab demo` runs the real robot client (LeRobot's scheduling and DRTC) against SmolVLA on this laptop's GPU, with
the replay robot as the arm, and shows it live in a browser. Faults come from buttons. Nothing at the venue needs a
network, a Pi or the arm.

## Before the day (with internet)

```bash
.venv/bin/lab demo --check          # GPU, ports, model downloaded, saved runs
```

If the model is missing, run `lab serve smolvla_sort_one_v1_full` once with internet and stop it when it says
"Policy server ready". Then run the console once and do "Run all three with the same faults": the three runs are
saved in `results/demo/` and are your fallback.

## At the venue

```bash
.venv/bin/lab demo                  # opens http://127.0.0.1:8765; SmolVLA takes 1-4 min to compile on first start
```

- **Run all three with the same faults**: LeRobot as shipped, LeRobot made patient and DRTC, 14 s each, with a
  latency spike at 3 s, 1 s of dropped replies at 6 s and a 1.5 s outage at 9 s. As shipped ends its run; DRTC
  keeps moving.
- **Run one client** and press the orange buttons yourself while it runs.
- **Play saved run**: a recorded console session, without the GPU. If the GPU or the model fails, start
  `lab demo --saved-only` and play the saved runs.

Ctrl-C in the terminal stops the console and the policy server it started. The console uses ports 8765 (page) and
7955 (policy server); change them with `--http` and `--port` if they are taken. It never uses 8080.

## What is real

The client code, its scheduling, its deadlines and SmolVLA are the real ones. The robot is a held-out sorting episode
replayed as the cameras and joints (the camera keeps playing the recording; the arm drawing moves only with the
commands the client sends). The network is localhost with the bench's 250 ms reply delay added, and the faults are
injected in the client's transport (`drtc.control.faults`), as on the bench.
