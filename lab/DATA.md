# Datasets: recording and editing

Everything on the laptop, in the `major` setup:

```bash
cd ~/Projects/drtc-major && source .venv/bin/activate      # then `lab ...` works
```

## The two copies of a dataset

| Copy | Hub name | What it is |
|---|---|---|
| raw | `orangemoon599/<name>_raw` | the recording as made; **the only copy you edit** |
| trimmed | `orangemoon599/<name>` | built from the raw copy: idle start/end cut (autotrim) plus your manual trims; **the one models train on** |

You never touch the trimmed copy. Every `lab data` command edits the raw copy and rebuilds the trimmed one by
itself. Manual trims are stored inside the raw copy (`meta/drtc.json`), so they travel with it to the Hub.

## Episodes: training + held out

`--episodes N --held-out K` records **N + K** episodes: N to train on, K held out (standard since 2026-11-19).
Held-out episodes are never trained on. They give the held-out loss during training, pick the best checkpoint,
and are used in the offline evaluation. They are picked by a fixed seed (0), so every model on the dataset gets
the same ones.

Example: `--episodes 80 --held-out 8` records 88 episodes. Balance the rotation over all 88 (4 combinations x 22).

## Recording

```bash
lab record so101_sort_two_v1 --task "Put each cube in the square of the same color." --episodes 100 --held-out 10
```

| Option | Default | Meaning |
|---|---|---|
| `--task "..."` | required | the instruction, identical for every episode; keep it exactly the same on `--resume` |
| `--episodes N` | 50 | training episodes |
| `--held-out K` | 10 | held-out episodes, recorded on top |
| `--episode-time S` | 30 | the longest an episode can last (press → to end it early) |
| `--reset-time S` | 10 | time to reset the scene between episodes |
| `--resume` | | continue an unfinished recording up to N + K; give the same `--episodes`/`--held-out` |
| `--no-push` | | keep it local; upload later with `lab data push <name>` |
| `--no-sounds` | sounds on | mute the spoken prompts ("Recording episode 3", "Reset the environment") |
| `--no-display` | on with a screen | no Rerun viewer |
| `--dry-run` | | print what would run (and how many episodes are left), record nothing |

Keys during recording (LeRobot's): **→** end the episode early and go on, **←** discard this episode and record it
again, **Esc** stop the session (what is recorded so far is kept and pushed).

After the session, `lab record` checks the recording, builds the trimmed copy, pushes both and writes their entries
in `lab/datasets/recorded.yaml`. **Commit that file** (or ask Claude), so the GPU servers see the new revision.

Stopped early? Continue with the same command plus `--resume`:

```bash
lab record so101_sort_two_v1 --task "Put each cube in the square of the same color." --episodes 100 --held-out 10 --resume
```

A new name refuses to start if `<name>` already exists locally (use `--resume`, or a new name).

## Looking at a dataset

```bash
lab data show so101_sort_one_v1           # every episode: length, kept range, trim status, warnings; unpushed edits
lab data view so101_sort_one_v1 12        # open episode 12 (raw) in the Rerun viewer
lab data view so101_sort_one_v1 12 --trimmed   # the same episode after trimming
```

In `show`, `!` marks an episode with a problem (e.g. rows and video out of step). Very short episodes are worth a
look in `view`.

## Editing

Edits stay **local until you push**. Each edit can be undone (one level).

```bash
lab data delete so101_sort_one_v1 7           # delete one episode (asks for 'yes'; --yes skips the question)
lab data delete so101_sort_one_v1 3 5-9       # several; the later episodes are renumbered (LeRobot needs 0..N-1)
lab data trim so101_sort_one_v1 12 --start 1.5 --end 9.0   # keep 1.5-9.0 s of episode 12 instead of autotrim
lab data trim so101_sort_one_v1 12 --end 9.0               # cut only the end
lab data trim so101_sort_one_v1 12 --auto                  # back to autotrim
lab data undo so101_sort_one_v1               # undo the last edit (until the next push or recording)
lab data push so101_sort_one_v1               # upload both copies, update lab/datasets/recorded.yaml
```

- Trim times are seconds from the start of the **raw** episode (what `view` shows without `--trimmed`).
- `push` keeps the registered held-out count; `--held-out K` changes it.
- Deleting episodes changes the total, so the seeded held-out pick changes too. Delete before training, not
  between models you want to compare.
- After a push, commit `lab/datasets/recorded.yaml`, then `git pull` on the GPU server.

## Starting over

```bash
lab data remove so101_sort_one_v1             # delete it locally (type the name to confirm); the Hub copies stay
lab data remove so101_sort_one_v1 --hub       # also delete both Hub repos; this cannot be undone
```

Both remove the entries from `lab/datasets/recorded.yaml`. Commit that file afterwards.

## Good to know

- Where the files are: `~/.cache/huggingface/lerobot/orangemoon599/<name>_raw` and `.../<name>`.
- The laptop does not prepare datasets for training. Each GPU server does that itself, at its first `lab train`
  or with `lab prepare <name>` (see [TRAIN.md](TRAIN.md)).
- Joints are recorded in degrees (rig `so101_v2`). Old v0/v1 datasets use -100..100 and are never mixed with new
  ones.
- Registry checks: `lab check` (registry), `lab check --rig` (cameras, arms, calibration), `lab check --data --online`
  (dataset pins against the Hub).
