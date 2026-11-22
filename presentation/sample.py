"""Three sample slides to check the look: the title, a result with click builds, and two videos side by side.

    .venv/bin/python presentation/sample.py      # -> presentation/out/sample.pptx
"""

from __future__ import annotations

from pathlib import Path

from deck import (ACCENT, FIGS, GREY, H, INK, MARGIN, REPO, SANS, SERIF, W, click_builds, fade_transition, footer,
                  header, movie, new_deck, picture, rule, text, _blank)
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parent / "out"
TRIALS = REPO / "results" / "evals" / "sort_one_drtc_main" / "sessions" / "20261120-1751-laptop" / "trials"


def title_slide(prs):
    s = _blank(prs)
    text(s, MARGIN, Inches(1.55), Inches(9), Inches(0.3), "PROJECT PRESENTATION", size=12, color=ACCENT, bold=True,
         spacing=2)
    text(s, MARGIN, Inches(1.95), Inches(10.5), Inches(1.9),
         ["Recovering remote robot inference", "from network faults"], size=44, font=SERIF, line=1.0)
    text(s, MARGIN, Inches(3.75), Inches(10), Inches(0.5),
         "Distributed Real-Time Chunking and motion smoothing on a low-cost SO-101 arm", size=20, color=GREY)
    rule(s, MARGIN, Inches(4.75), Inches(0.9))
    text(s, MARGIN, Inches(5.0), Inches(9), Inches(1.0),
         [[("Name One", {"bold": True}), ("   ·   ", {"color": GREY}), ("Name Two", {"bold": True})],
          [("Guide: Prof. Name   ·   Department, Institution   ·   2026", {"color": GREY, "size": 14})]],
         size=16, line=1.4)
    fade_transition(s)
    return s


def result_slide(prs, n):
    s = _blank(prs)
    header(s, "First run · task success", "Network faults break the shipped client")
    fig = picture(s, FIGS / "fig_main_success.png", MARGIN, Inches(2.0), w=Inches(7.4))
    x, w = Inches(8.75), Inches(3.85)
    items = [("90% → 0%", "SmolVLA's success when the network faults; GR00T N1.5 fell from 70% to 0%."),
             ("20 of 20", "faulted trials were ended by the client itself, at the very first fault."),
             ("60% · 80%", "with DRTC under the same faults: SmolVLA and GR00T N1.5 recover.")]
    groups = []
    for i, (big, small) in enumerate(items):
        y = Inches(2.15 + 1.55 * i)
        a = text(s, x, y, w, Inches(0.6), big, size=30, font=SERIF, color=ACCENT)
        b = text(s, x, y + Inches(0.62), w, Inches(0.8), small, size=15, color=GREY, line=1.15)
        groups.append([a, b])
    footer(s, n)
    fade_transition(s)
    click_builds(s, groups)
    return s


def video_slide(prs, n):
    s = _blank(prs)
    header(s, "First run · the same faults, two clients", "Giving up, or holding and recovering")
    vw, vh = Inches(5.55), Inches(4.16)  # the scene camera's 4:3
    xs = [MARGIN, W - MARGIN - vw]
    clips = [(TRIALS / "baseline_faults" / "smolvla_sort_one_v1_full" / "t03" / "video" / "scene.mp4",
              "LeRobot as shipped", "The spike empties the queue; one second later the run is ended."),
             (TRIALS / "drtc_faults" / "smolvla_sort_one_v1_full" / "t03" / "video" / "scene.mp4",
              "DRTC", "The arm holds through the spike, the lost replies and the outage, and finishes.")]
    for x, (clip, label, line) in zip(xs, clips):
        movie(s, clip, x, Inches(1.95), vw, vh)
        text(s, x, Inches(6.22), vw, Inches(0.7),
             [[(label, {"bold": True, "color": INK}), ("   " + line, {"color": GREY})]], size=14, line=1.15)
    footer(s, n)
    fade_transition(s)
    return s


def main() -> None:
    OUT.mkdir(exist_ok=True)
    prs = new_deck()
    title_slide(prs)
    result_slide(prs, 2)
    video_slide(prs, 3)
    out = OUT / "sample.pptx"
    prs.save(out)
    print(out)


if __name__ == "__main__":
    main()
