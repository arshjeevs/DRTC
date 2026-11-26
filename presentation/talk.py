"""The final talk (20-25 min, for a panel new to the project): order A of the plan of 2026-11-26.

    .venv/bin/python presentation/talk.py               # -> presentation/out/talk.pptx (+ .pdf and PNG previews)
    .venv/bin/python presentation/talk.py --no-preview

Story: a hook -> how a VLA drives the arm -> the problem (the client-server loop, the seven faults) -> our solution
(DRTC, smoothing) -> what we built (hardware, data, the first attempt, training, testing) -> results -> close.

Media slots: every video and animation is a named slot (MEDIA below). Drop presentation/media/<key>.mp4 (or .png/.jpg)
and the next build embeds it in place; until then the slot is a dashed box saying what goes there. So a video can be
redone without touching the rest of the deck. Numbers come from the report (results.py), photos from report/assets,
charts from report/generated/slides. Style: style.py (the deck's look). Each slide has speaker notes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import json
import subprocess

import numpy as np

from build import ASSETS, CW, M, TOP, card, notes, para, photo, stat, table
from deck import FIGS, W, bar_panel, box, builds, line, movie, new_deck, picture, text
from drtc.evals.metrics import wilson
from results import R, title_page
from style import Style
from talk_draw import (JOINT_COLOURS, JOINTS, chunk_plot, curve, dot, e4_trial, fault_labels, fault_row, fault_windows,
                       first_chunk_curves, first_state, mapper, marker, mini_panel, moving_segments, plot_curve,
                       problem_curves, resends, time_axis, timeline_row)

S = Style()
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
MEDIA = HERE / "media"
BOTTOM = S.bottom

# key -> (kind, title, length, what it shows)
MEDIA_SLOTS = {
    "hook": ("video", "Hook", "~15 s", "A real E4 trial: the arm starts the cube-in-box task, a network fault hits, "
             "the arm freezes. Caption: the robot's brain is on another computer."),
    "vla": ("video", "V1  From camera to motion", "~40 s",
            "Real frames + real servo angles from one trial -> into the model with the instruction -> a chunk of 50 "
            "future joint positions comes out -> the arm plays them at 30 Hz -> it asks again."),
    "why_server": ("animation", "Why the model is not on the robot", "click-in",
                   "Model sizes (0.05B to 5.6B parameters) next to what the Raspberry Pi can run."),
    "loop": ("animation", "The loop over the network", "click-in",
             "Robot computer -> 2.9 MB of images -> GPU server -> a chunk back; the queue of moves filling and draining."),
    "fault": ("video", "V2  What a fault does", "~30 s",
              "A real trial beside its queue: the reply is late, the queue runs dry, the arm stops; LeRobot as shipped "
              "ends the run after 1 s without actions."),
    "drtc": ("video or animation", "V3  DRTC at work", "~45 s",
             "LeRobot and DRTC side by side under the same fault: one queue empties and the arm stops; the other asks "
             "again and keeps going. (Video or animated slides: decided when we get here.)"),
    "jerk": ("video", "V4  Why smoothing", "~30 s",
             "The arm at a chunk switch and after a fault: the jump in the joint curve, then the same with the stack."),
    "hardware": ("animation", "The setup, labelled", "click-in",
                 "The setup photo with each part labelled in turn: follower arm, wrist camera, phone camera, leader arm, "
                 "Raspberry Pi, laptop; the GPU servers off to the side."),
    "record": ("video", "V5  Recording a demonstration", "~30 s",
               "YOUR FOOTAGE: the leader arm moved by hand, the follower copying it; then the Rerun screen recording of "
               "what is saved (both cameras, the joint traces)."),
    "robot_eval": ("video", "Robot evaluation, live", "~30 s",
                   "YOUR FOOTAGE: a real-time robot evaluation: the layout set, a trial running from the Pi, a fault "
                   "hitting, the operator labelling it."),
    "e4_trials": ("video", "V6  Same fault, three clients", "~30 s",
                  "Three real E4 trials side by side under the fault mix: LeRobot stops, DRTC recovers, DRTC + the "
                  "stack recovers smoothly."),
}


def media_file(key):
    for ext in (".mp4", ".mov", ".png", ".jpg"):
        if (MEDIA / f"{key}{ext}").is_file():
            return MEDIA / f"{key}{ext}"
    return None


def fit_picture(s, path, x, y, w, h):
    """A picture scaled to fit inside the box, centred in it."""
    pic = picture(s, path, x, y, w=w)
    if pic.height > h:
        pic.width = int(pic.width * h / pic.height)
        pic.height = int(h)
    pic.left = int(x + (w - pic.width) / 2)
    pic.top = int(y + (h - pic.height) / 2)
    return pic


def slot(s, key, x, y, w, h):
    """A media slot: presentation/media/<key>.* when present; else a stand-in from media/fallback/ (a real frame or
    figure); else a dashed box describing what goes there."""
    f = media_file(key)
    if f is not None and f.suffix in (".mp4", ".mov"):  # fitted to the box at the video's own aspect ratio
        vw, vh = (int(v) for v in subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of",
             "csv=p=0", str(f)], capture_output=True, text=True, check=True).stdout.strip().split(","))
        fw, fh = (w, int(w * vh / vw)) if w * vh / vw <= h else (int(h * vw / vh), h)
        return movie(s, f, int(x + (w - fw) / 2), int(y + (h - fh) / 2), fw, fh, poster_at=2.0)
    if f is not None:
        return fit_picture(s, f, x, y, w, h)
    for fb in (MEDIA / "fallback" / f"{key}.jpg", MEDIA / "fallback" / f"{key}.png", MEDIA / "diagrams" / f"{key}.png"):
        if fb.is_file():
            return fit_picture(s, fb, x, y, w, h)
    kind, title, length, what = MEDIA_SLOTS[key]
    return box(s, x, y, w, h, fill=S.panel, line=S.alert, dash=True, weight=Pt(1.5), radius=S.card_radius,
               runs=[[(f"{kind.upper()}  ·  {length}", {"bold": True, "color": S.alert, "size": 12})],
                     [(title, {"bold": True, "color": S.ink, "size": 20})],
                     [(what, {"color": S.muted, "size": 13})],
                     [(f"slot: presentation/media/{key}.mp4", {"color": S.muted, "size": 10})]],
               size=16, font=S.body, margin=Inches(0.35), line_spacing=1.2)


def fig(s, name, x, y, *, w=None, h=None, centre=True, maxw=None):
    """A chart from the report, centred on the slide unless placed beside text; never wider than maxw."""
    pic = picture(s, FIGS / f"{name}.png", x, y, w=w, h=h)
    if maxw and pic.width > maxw:
        pic.height = int(pic.height * maxw / pic.width)
        pic.width = int(maxw)
    if centre:
        pic.left = int((W - pic.width) / 2)
    return pic


def bullets(s, x, y, w, items, *, size=18, gap=Inches(0.62), marker=None):
    """Lead + grey rest lines, one per click."""
    return [para(s, x, y + gap * i, w, lead, rest, size=size, marker=marker, h=gap) for i, (lead, rest) in
            enumerate(items)]


def n_of(key):
    v = R(key)
    return int(v.rstrip("%")) // 10 if v.endswith("%") else None


# --- the slides ------------------------------------------------------------------------------------------------------

def t_title(prs, n):
    meta = title_page()
    meta["kind"] = "Final-year project"
    s = S.title_slide(prs, meta)
    notes(s, "Introduce the team and the guide. One sentence: we made a robot arm keep working when the network that "
             "connects it to its AI model fails.")


def t_hook(prs, n):
    s = S.content(prs, n, "Why this matters", "The robot's brain is on another computer")
    v = slot(s, "hook", M, TOP - Inches(0.1), Inches(7.2), Inches(4.4))
    x = M + Inches(7.6)
    q = text(s, x, TOP + Inches(0.6), Inches(3.9), Inches(2.6),
             [[("What happens to the robot", {"bold": True})], [("when the network fails?", {"bold": True})],
              [("")], [("That question is this project.", {"color": S.muted, "size": 18})]],
             size=26, font=S.heading, line=1.15)
    builds(s, [[q]])
    notes(s, "Play the clip. The arm is doing its task; the model that drives it runs on a GPU server. A network fault "
             "hits and the arm stops. Everything today is about that moment.")


def t_outline(prs, n):
    s = S.content(prs, n, "Outline", "Six steps")
    parts = [("1", "How a robot is driven by an AI model", "cameras and joints in, motion out"),
             ("2", "The problem", "control over a network, and the seven faults"),
             ("3", "Our solution", "DRTC and motion smoothing"),
             ("4", "What we built", "hardware, our lab platform, data, training"),
             ("5", "How we evaluate", "offline, the bench, live robot trials"),
             ("6", "Results", "four questions, answered")]
    rows = [para(s, M, TOP - Inches(0.1) + Inches(0.75) * i, CW, f"{a}   {b}", c, size=21, h=Inches(0.7))
            for i, (a, b, c) in enumerate(parts)]
    builds(s, rows)
    notes(s, "The order of the talk.")


# Part 1 -----------------------------------------------------------------------------------------------------------

def t_vla_what(prs, n):
    s = S.content(prs, n, "1 · Robot control with a VLA", "From camera to motion: inside a VLA")
    ex = json.loads((MEDIA / "data" / "vla_example.json").read_text())
    st = MEDIA / "stills"
    y0, h = TOP - Inches(0.05), Inches(4.15)
    iw, aw, mw = Inches(3.45), Inches(0.45), Inches(3.75)
    ix, mx = M, M + iw + aw
    ox = mx + mw + aw
    ow = W - M - ox
    pad = Inches(0.16)
    eb = dict(size=11, bold=True, color=S.muted, font=S.body, spacing=1.2)
    # 1. the inputs of one moment
    inp = [text(s, ix, y0 - Inches(0.32), iw, Inches(0.3), "INPUTS  ·  ONE MOMENT", **eb),
           box(s, ix, y0, iw, h, fill=S.panel, line=S.panel_line, radius=0.04)]
    pw = (iw - 2 * pad - Inches(0.1)) / 2
    for k, (f, cap) in enumerate((("vla_ep0_wrist.jpg", "wrist camera"), ("vla_ep0_scene.jpg", "scene camera"))):
        x = int(ix + pad + (pw + Inches(0.1)) * k)
        inp.append(picture(s, st / f, x, int(y0 + pad), w=int(pw)))
        inp.append(text(s, x, int(y0 + pad + pw * 0.75 + Inches(0.02)), int(pw), Inches(0.25), cap, size=10,
                        color=S.muted, font=S.body))
    yi = y0 + Inches(1.6)
    inp.append(text(s, ix + pad, yi, iw - 2 * pad, Inches(0.25), "INSTRUCTION", size=10, bold=True, color=S.muted,
                    font=S.body, spacing=1.2))
    inp.append(text(s, ix + pad, yi + Inches(0.22), iw - 2 * pad, Inches(0.5),
                    "“Pick up the green cube and put it in the box”", size=13, italic=True, color=S.ink,
                    font=S.body, line=1.1))
    yj = y0 + Inches(2.45)
    inp.append(text(s, ix + pad, yj, iw - 2 * pad, Inches(0.25), "JOINT POSITIONS NOW", size=10, bold=True,
                    color=S.muted, font=S.body, spacing=1.2))
    labels = ["shoulder pan", "shoulder lift", "elbow", "wrist flex", "wrist roll", "gripper"]
    cw_ = (iw - 2 * pad) / 3
    for k, (lab, v) in enumerate(zip(labels, ex["state"])):
        x = int(ix + pad + cw_ * (k % 3))
        y = int(yj + Inches(0.27) + Inches(0.6) * (k // 3))
        inp.append(text(s, x, y, int(cw_), Inches(0.6), [[(lab, {"size": 9, "color": S.muted})],
                                                          [(f"{v:+.1f}", {"size": 16, "bold": True,
                                                                          "color": S.ink})]],
                        size=12, font=S.body, line=1.05))
    # 2. inside the policy, step by step
    ymid = y0 + h / 2
    a1 = line(s, ix + iw + Inches(0.06), ymid, mx - Inches(0.06), ymid, color=S.ink, weight=Pt(1.75), arrow=True)
    frame = [text(s, mx, y0 - Inches(0.32), mw, Inches(0.3), "THE VLA POLICY  ·  e.g. SmolVLA, 0.45 B", **eb),
             box(s, mx, y0, mw, h, fill=S.dark, radius=0.04)]
    bx, bw = mx + pad, mw - 2 * pad
    sh = Inches(0.82)
    gap = (h - 2 * pad - 4 * sh) / 3
    steps_txt = [("1  Into tokens", "each image is cut into patches; words and joint angles become tokens too"),
                 ("2  Vision-language backbone", "reads all the tokens together: what is where, and what was asked"),
                 ("3  Action expert", "starts from random noise and refines it into actions in a few steps"),
                 ("4  An action chunk", "50 future positions for each of the 6 joints")]
    inner = []
    for k, (t1, t2) in enumerate(steps_txt):
        yy = y0 + pad + (sh + gap) * k
        g = [box(s, bx, yy, bw, sh, fill="#232A34", radius=0.08,
                 runs=[[(t1, {"bold": True})], [(t2, {"color": S.dark_muted, "size": 11})]], size=13,
                 font=S.body, color=S.dark_text, align=PP_ALIGN.LEFT, margin=Inches(0.14))]
        if k == 0:  # a strip of tokens: image (blue), words (green), joints (orange)
            for j, col in enumerate([S.accent] * 6 + [S.series["drtc"]] * 3 + [S.alert]):
                g.append(box(s, bx + bw - Inches(0.15) - Inches(0.13) * (10 - j), yy + Inches(0.1), Inches(0.1),
                             Inches(0.1), fill=col))
        if k:
            g.insert(0, line(s, bx + bw / 2, yy - gap + Inches(0.02), bx + bw / 2, yy - Inches(0.02),
                             color=S.dark_muted, weight=Pt(1.5), arrow=True))
        inner.append(g)
    # 3. the output, played at 30 Hz
    a2 = line(s, mx + mw + Inches(0.06), ymid, ox - Inches(0.06), ymid, color=S.ink, weight=Pt(1.75), arrow=True)
    out = [text(s, ox, y0 - Inches(0.32), ow, Inches(0.3), "OUTPUT  ·  THE CHUNK", **eb),
           box(s, ox, y0, ow, h, fill=S.panel, line=S.panel_line, radius=0.04)]
    chunk = np.array(ex["chunk"])
    lw_ = Inches(0.85)
    gx, gw = ox + pad + lw_, ow - 2 * pad - lw_
    rh = (h - 2 * pad - Inches(0.45)) / 6
    curves = []
    for k, (lab, col) in enumerate(zip(labels, JOINT_COLOURS)):
        ry = y0 + pad + rh * k
        out.append(text(s, ox + pad, ry, lw_, rh, lab, size=10, color=S.ink, font=S.body, anchor=MSO_ANCHOR.MIDDLE))
        out.append(line(s, gx, ry + rh - Inches(0.04), gx + gw, ry + rh - Inches(0.04), color=S.grid,
                        weight=Pt(0.5)))
        v = chunk[:, k]
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 3:
            lo, hi = lo - 3, hi + 3
        f = mapper(gx, ry + Inches(0.07), gw, rh - Inches(0.16), 0, len(v) - 1, lo, hi)
        curves.append(curve(s, [f(i, a) for i, a in enumerate(v)], col, weight=Pt(2)))
    ay = y0 + h - pad - Inches(0.36)
    out += [text(s, gx, ay, Inches(0.8), Inches(0.25), "now", size=9, color=S.muted, font=S.body),
            text(s, gx, ay + Inches(0.2), gw, Inches(0.25), "played at 30 per second", size=9, color=S.muted,
                 font=S.body, align=PP_ALIGN.CENTER),
            text(s, gx + gw - Inches(0.8), ay, Inches(0.8), Inches(0.25), "+1.7 s", size=9, color=S.muted,
                 font=S.body, align=PP_ALIGN.RIGHT)]
    loop = [line(s, ox + ow / 2, y0 + h + Inches(0.08), ox + ow / 2, y0 + h + Inches(0.3), color=S.accent,
                 weight=Pt(1.5)),
            line(s, ox + ow / 2, y0 + h + Inches(0.3), ix + iw / 2, y0 + h + Inches(0.3), color=S.accent,
                 weight=Pt(1.5)),
            line(s, ix + iw / 2, y0 + h + Inches(0.3), ix + iw / 2, y0 + h + Inches(0.08), color=S.accent,
                 weight=Pt(1.5), arrow=True),
            text(s, ix + iw / 2 + Inches(0.2), y0 + h + Inches(0.31), ox + ow / 2 - ix - iw / 2 - Inches(0.4),
                 Inches(0.25), "before the chunk runs out, the next moment is sent: the loop repeats", size=11,
                 color=S.accent, font=S.body, align=PP_ALIGN.CENTER)]
    builds(s, [inp, [a1] + frame, inner[0], inner[1], inner[2], inner[3],
               [a2] + out + [(c, "wipe_right") for c in curves], [(x_, "fade") for x_ in loop]])
    notes(s, "Real data from one moment of a recorded demonstration (cube in box). In: the wrist and scene frames, "
             "the instruction and the six joint positions. Inside the VLA: everything becomes tokens; a vision-"
             "language backbone, pretrained on web images and text, reads them together; a smaller action expert "
             "starts from noise and refines it into a chunk of 50 future positions per joint. Out: the next 1.7 s of "
             "motion (shown here: what the person actually did next, which the policy learns to predict). The arm "
             "plays it at 30 per second, and before it runs out the next moment is sent: the loop repeats.")


def t_vla_video(prs, n):
    s = S.content(prs, n, "1 · Robot control with a VLA", "From camera to motion")
    slot(s, "vla", M, TOP - Inches(0.1), CW, Inches(4.7))
    notes(s, "Walk through the video: capture, model, chunk, playback, ask again. Thirty times a second.")


def t_chunks(prs, n):
    s = S.content(prs, n, "1 · Robot control with a VLA", "Chunk after chunk: the arm moves while the server computes")
    st = MEDIA / "stills"
    t0, t1 = 1.2, 7.45
    fw = Inches(2.17)
    times = [1.8, 2.9, 3.7, 4.9, 6.6]
    frames = [picture(s, st / f"chunk_frame_{t}.jpg", M + (fw + Inches(0.175)) * i, TOP - Inches(0.25), w=fw)
              for i, t in enumerate(times)]
    fh = frames[0].height
    jy = TOP - Inches(0.1) + fh + Inches(0.15)
    jbox = (M + Inches(1.2), jy, CW - Inches(1.2), Inches(1.15))
    qbox = (M + Inches(1.2), jy + Inches(1.35), CW - Inches(1.2), Inches(0.8))
    stat_ = [text(s, M, jy + Inches(0.3), Inches(1.15), Inches(0.6), [[("shoulder", {"bold": True})],
                                                                       [("command", {"color": S.muted})]],
                  size=12, font=S.body, line=1.1),
             text(s, M, qbox[1] + Inches(0.1), Inches(1.15), Inches(0.6), [[("queue", {"bold": True})],
                                                                           [("moves left", {"color": S.muted})]],
                  size=12, font=S.body, line=1.1),
             line(s, qbox[0], qbox[1] + qbox[3], qbox[0] + qbox[2], qbox[1] + qbox[3], color=S.grid, weight=Pt(1))]
    segs, fj, fq = chunk_plot(s, jbox, qbox, t0, t1)
    cap = text(s, M, qbox[1] + qbox[3] + Inches(0.1), CW, Inches(0.35),
               "A real trial (SmolVLA, cube in box, from the Raspberry Pi). Each colour is one chunk of 50 moves; the "
               "queue jumps up when a chunk arrives and drains at 30 moves a second.", size=11, color=S.muted,
               font=S.body)
    steps = [stat_ + [cap]]
    for k, (jc, qc, a, nxt) in enumerate(segs):
        step = [(jc, "wipe_right"), (qc, "wipe_right")]
        if k < len(frames):
            step.append((frames[k], "fade"))
        if nxt is not None:
            x, y = fq(nxt, 0)
            tri = text(s, int(x - Inches(0.3)), int(qbox[1] - Inches(0.28)), Inches(0.6), Inches(0.25), "ask",
                       size=10, bold=True, color=S.accent, align=PP_ALIGN.CENTER, font=S.body)
            step.append((tri, "fade", 300))
        steps.append(step)
    builds(s, steps)
    notes(s, "Click by click, one chunk at a time. The arm plays a chunk from the queue (bottom) while the next one is "
             "computed; 'ask' marks when the next request was sent. LeRobot asks late, so at the end the queue runs "
             "dry and the arm waits: this is the moment a network fault makes much worse.")


def t_why_server(prs, n):
    s = S.content(prs, n, "1 · Robot control with a VLA", "Too big for the robot: the model runs on a GPU server")
    a = stat(s, M, TOP, Inches(3.5), "0.05-5.6 B", "parameters in the policies we trained")
    b = stat(s, M, TOP + Inches(1.5), Inches(3.5), "Raspberry Pi 5", "the robot's computer: no GPU")
    c = stat(s, M, TOP + Inches(3.0), Inches(3.5), "A100 · H200", "the GPU servers that run the models")
    x0 = M + Inches(4.2)
    if media_file("why_server"):
        slot(s, "why_server", x0, TOP - Inches(0.1), CW - Inches(4.2), Inches(4.6))
        builds(s, [a, b, c])
        return
    w = W - M - x0
    nw = Inches(1.6)
    half = (w - nw - Inches(0.3)) / 2
    hdr = [text(s, x0 + nw, TOP - Inches(0.25), half, Inches(0.3), "SIZE (BILLIONS OF PARAMETERS)", size=10,
                bold=True, color=S.muted, font=S.body, spacing=1.1),
           text(s, x0 + nw + half + Inches(0.3), TOP - Inches(0.25), half, Inches(0.3), "TIME PER CHUNK ON THE GPU",
                size=10, bold=True, color=S.muted, font=S.body, spacing=1.1)]
    rh = Inches(0.58)
    rows = []
    for i, (name, size, ms) in enumerate(SIZE_SPEED):
        y = TOP + Inches(0.15) + rh * i
        bw1 = int((half - Inches(0.75)) * size / 5.6)
        bw2 = int((half - Inches(0.75)) * ms / 450)
        x2 = x0 + nw + half + Inches(0.3)
        rows.append([text(s, x0, y, nw, rh, name, size=13, color=S.ink, font=S.body, anchor=MSO_ANCHOR.MIDDLE),
                     box(s, x0 + nw, y + Inches(0.15), max(bw1, Pt(2)), rh - Inches(0.3), fill=S.accent),
                     text(s, x0 + nw + max(bw1, Pt(2)) + Inches(0.06), y, Inches(0.7), rh, f"{size:g} B", size=11,
                          color=S.ink, font=S.body, anchor=MSO_ANCHOR.MIDDLE),
                     box(s, x2, y + Inches(0.15), bw2, rh - Inches(0.3), fill=S.series["smooth"]),
                     text(s, x2 + bw2 + Inches(0.06), y, Inches(0.8), rh, f"{ms} ms", size=11, color=S.ink,
                          font=S.body, anchor=MSO_ANCHOR.MIDDLE)])
    cap = text(s, x0, TOP + Inches(0.2) + rh * 7, w, Inches(0.4),
               "Even on an H200 the larger models need up to 0.43 s per chunk; a Raspberry Pi has no GPU at all.",
               size=12, color=S.muted, font=S.body)
    builds(s, [a, b, c, hdr + sum(rows, []) + [cap]])
    notes(s, "So the model runs on a GPU server, and the robot sends it images over the network. Even there, the "
             "big models take a few hundred milliseconds per chunk (median over our robot trials). This is remote "
             "inference: next, how the arm keeps moving while it waits.")


def t_loop(prs, n):
    s = S.content(prs, n, "2 · The problem", "Every move now depends on the network")
    x = M + Inches(7.8)
    rows = bullets(s, x, TOP, Inches(3.8), [
        ("Up:", "2 images + 6 joints, 2.9 MB"),
        ("Down:", "a chunk of moves"),
        ("Round trip:", "0.6-1 s from the Pi"),
        ("Queue:", "the moves still waiting to be played"),
    ], size=17, gap=Inches(0.8))
    if media_file("loop"):
        slot(s, "loop", M, TOP - Inches(0.1), Inches(7.4), Inches(4.6))
        builds(s, rows)
        return
    bw, bh = Inches(2.3), Inches(1.45)
    ry, sx = TOP + Inches(0.25), M + Inches(5.0)
    robot = box(s, M, ry, bw, bh, fill=S.dark, radius=S.card_radius,
                runs=[[("Robot computer", {"bold": True})], [("Raspberry Pi 5, 30 Hz", {"color": S.dark_muted,
                                                                                         "size": 12})],
                      [("cameras + arm", {"color": S.dark_muted, "size": 12})]], size=16, font=S.body,
                color=S.dark_text)
    server = box(s, sx, ry, bw, bh, fill=S.dark, radius=S.card_radius,
                 runs=[[("GPU server", {"bold": True})], [("the policy (H200)", {"color": S.dark_muted, "size": 12})]],
                 size=16, font=S.body, color=S.dark_text)
    net = box(s, M + bw + Inches(0.75), ry + bh / 2 - Inches(0.2), Inches(1.2), Inches(0.4), fill=S.panel,
              line=S.panel_line, radius=0.5, runs="network", size=12, font=S.body, color=S.muted)
    up = [line(s, M + bw + Inches(0.05), ry + Inches(0.35), sx - Inches(0.05), ry + Inches(0.35), color=S.accent,
               weight=Pt(2.25), arrow=True),
          text(s, M + bw, ry - Inches(0.2), sx - M - bw, Inches(0.3), "observation: 2 images + 6 joints", size=12,
               color=S.accent, font=S.body, align=PP_ALIGN.CENTER)]
    down = [line(s, sx - Inches(0.05), ry + bh - Inches(0.35), M + bw + Inches(0.05), ry + bh - Inches(0.35),
                 color=S.series["drtc"], weight=Pt(2.25), arrow=True),
            text(s, M + bw, ry + bh - Inches(0.3), sx - M - bw, Inches(0.3), "a chunk of 50 actions", size=12,
                 color=S.series["drtc"], font=S.body, align=PP_ALIGN.CENTER)]
    qy = ry + bh + Inches(0.55)
    cells = []
    for k in range(20):
        filled = k < 13
        cells.append(box(s, M + Inches(0.13) * 0 + Inches(0.29) * k, qy, Inches(0.25), Inches(0.42),
                         fill=S.series["drtc"] if filled else "#ECEAE3", radius=0.15))
    qlab = text(s, M, qy + Inches(0.5), Inches(6.2), Inches(0.6),
                [[("The queue", {"bold": True, "color": S.ink}),
                  ("  actions waiting to be played, one every 33 ms; the next chunk must arrive before it runs dry",
                   {"color": S.muted})]], size=13, font=S.body, line=1.15)
    arm = [line(s, M + Inches(0.29) * 20 + Inches(0.05), qy + Inches(0.21), M + Inches(6.6), qy + Inches(0.21),
                color=S.ink, weight=Pt(1.5), arrow=True),
           text(s, M + Inches(6.65), qy, Inches(0.8), Inches(0.42), "arm", size=14, bold=True, color=S.ink,
                font=S.body, anchor=MSO_ANCHOR.MIDDLE)]
    builds(s, [[robot, server, net], up + rows[0], down + rows[1], cells + [qlab] + arm + rows[3], rows[2]])
    notes(s, "The robot computer sends the images and joint angles; the server answers with a chunk; the robot keeps "
             "a queue of moves and plays one every 33 ms. As long as the next chunk arrives before the queue empties, "
             "the arm moves smoothly.")


def t_fault(prs, n):
    s = S.content(prs, n, "2 · The problem", "When a reply is lost, the arm stops for good")
    x = M + Inches(7.8)
    rows = bullets(s, x, TOP, Inches(3.8), [
        ("LeRobot as shipped", "ends the run after 1 s without moves"),
        ("Even when patient", "set to wait the whole trial, it never asks again after a lost reply"),
    ], size=17, gap=Inches(1.0))
    if media_file("fault"):
        slot(s, "fault", M, TOP - Inches(0.1), Inches(7.4), Inches(4.6))
        builds(s, rows)
        return
    p, d = e4_trial("upstream_patient__mix"), e4_trial("drtc__mix")
    faults = fault_windows(d)
    tmax = 18
    w = Inches(7.4)
    segs = moving_segments(p)
    last = max(b for a, b in segs)
    ry = TOP + Inches(0.6)
    base, mv, fl, ak, en = timeline_row(s, M, ry, w, tmax, segs, faults, label="Patient LeRobot",
                                        sub="GR00T N1.7, trial 1", colour=S.series["baseline"],
                                        end=(16.5, "run ended"), end_ok=False, row_h=Inches(0.75))
    axis = time_axis(s, M, ry + Inches(0.95), w, tmax)
    fls = fault_labels(s, M, ry - Inches(0.3), w, tmax, faults)
    leg = text(s, M, TOP - Inches(0.25), w, Inches(0.3),
               [[("■ ", {"color": S.series["drtc"]}), ("arm moving   ", {"color": S.muted}),
                 ("■ ", {"color": S.alert}), ("network fault", {"color": S.muted})]], size=11, font=S.body)
    expl = text(s, M, ry + Inches(1.45), w, Inches(1.8),
                [[("A real robot trial. ", {"bold": True, "color": S.ink}),
                  (f"The arm moved until {last:.1f} s. A reply lost in the faults was never asked for again, so no "
                   "new actions came, even after the network was back. After 10 s without actions LeRobot ended the "
                   "run.", {"color": S.muted})]], size=14, font=S.body, line=1.25)
    builds(s, [base + axis + [leg] + mv, fl + fls, en + [expl]] + rows)
    notes(s, "A real trial from our robot experiment: GR00T N1.7 with LeRobot's client set to be patient. The green "
             "bars are when the arm moved. The orange windows are the injected faults: a late reply, a second of lost "
             "replies, and a 1.5 s outage. One request was lost; the client never asked again, so the arm stood "
             "still until LeRobot's 10 s deadline ended the run. LeRobot as shipped would have ended it after 1 s.")


FAULTS = [("none", "a good network", "the cost of being careful"),
          ("spike", "one reply 1 s late", "a single slow moment"),
          ("drops", "every reply lost for 1 s", "messages that never arrive"),
          ("outage", "the link dead both ways, 1.5 s", "WiFi dropping out"),
          ("rise", "every reply 0.4 s slower, for 5 s", "a network getting busy"),
          ("loss", "10 % of replies lost at random", "a flaky connection"),
          ("mix", "spike + drops + outage", "a bad few seconds: our robot test")]


def t_faults(prs, n):
    s = S.content(prs, n, "2 · The problem", "Seven kinds of network fault")
    nx, tx, dx = M, M + Inches(1.35), M + Inches(7.0)
    tw = Inches(5.3)
    y0 = TOP - Inches(0.3)
    hdr = [text(s, nx, y0, Inches(1.2), Inches(0.3), "FAULT", size=11, bold=True, color=S.muted, font=S.body),
           text(s, tx, y0, tw, Inches(0.3), "REPLIES FROM THE SERVER OVER 10 s  (one mark = one reply)", size=11,
                bold=True, color=S.muted, font=S.body),
           text(s, dx, y0, Inches(4.5), Inches(0.3), "WHAT HAPPENS  ·  LIKE", size=11, bold=True,
                color=S.muted, font=S.body)]
    for t in (0, 5, 10):
        hdr.append(text(s, int(tx + tw * t / 10 - Inches(0.3)), y0 + Inches(0.25), Inches(0.6), Inches(0.25),
                        f"{t} s", size=10, color=S.muted, align=PP_ALIGN.CENTER, font=S.body))
    rh = Inches(0.6)
    groups = []
    for i, (name, what, like) in enumerate(FAULTS):
        y = y0 + Inches(0.5) + rh * i
        g = [text(s, nx, y, Inches(1.2), rh, name, size=17, bold=True, color=S.alert if name != "none" else S.ink,
                  font=S.body, anchor=MSO_ANCHOR.MIDDLE)]
        g += fault_row(s, tx, y, tw, rh, name)
        g.append(text(s, dx, y, Inches(4.6), rh, [[(what, {"color": S.ink})], [(like, {"color": S.muted,
                                                                                         "size": 12})]],
                      size=14, font=S.body, anchor=MSO_ANCHOR.MIDDLE, line=1.05))
        g.append(line(s, nx, y + rh, W - M, y + rh, color=S.grid, weight=Pt(0.5)))
        groups.append([(x, "fade") for x in g])
    builds(s, [[(x, "fade") for x in hdr]] + groups)
    notes(s, "Each mark is a reply from the server. Spike: one reply arrives a second late. Drops: replies vanish "
             "for a second. Outage: the link is dead both ways for 1.5 s. Rise: every reply is 0.4 s slower for five "
             "seconds. Loss: one reply in ten disappears at random. Mix: spike, drops and outage in one run; that is "
             "the robot test. Every fault is shifted by a random 0-4 s per trial, the same for every client.")


def t_question(prs, n):
    s = S.statement_slide(prs, n, "The question")
    q = text(s, M, Inches(2.0), CW, Inches(2.6),
             [[("Can a smarter client keep the arm working through these faults,")],
              [("and keep it moving smoothly?")]], size=36, bold=True, font=S.heading, color=S.on_accent, line=1.15)
    sub = text(s, M, Inches(4.9), CW, Inches(1.0),
               "Without changing the model, the server, or the network: only how the robot asks.", size=20,
               color=S.on_accent_muted, font=S.body)
    builds(s, [[sub]])
    notes(s, "Our research question.")


# Part 3 -----------------------------------------------------------------------------------------------------------

def t_drtc_idea(prs, n):
    s = S.content(prs, n, "3 · Our solution", "DRTC: ask early, learn the delay, ask again")
    cw, gap = Inches(3.6), Inches(0.37)
    ideas = [("1", "Ask early", "request the next chunk with about 1 s of moves left",
              "when the queue falls to H − s_min moves (SmolVLA: 30 of 50)"),
             ("2", "Learn the delay", "keep a running estimate of how long replies take",
              "smoothed mean + 1.5 × deviation of reply times, as TCP does"),
             ("3", "Ask again", "no reply in that time: assume it is lost and send a fresh request",
              "wait at least as long as the slowest of the last 10 replies; the newest reply wins")]
    groups = []
    for i, (a, b, c, d) in enumerate(ideas):
        x = M + (cw + gap) * i
        g = card(s, x, TOP, cw, Inches(2.75), a, b, c)
        g.append(text(s, x + Inches(0.25), TOP + Inches(2.12), cw - Inches(0.5), Inches(0.6), d, size=11,
                      color=S.accent, font=S.body, line=1.15))
        groups.append(g)
    foot = text(s, M, TOP + Inches(3.1), CW, Inches(1.2),
                [[("Distributed Real-Time Chunking (Vial, 2026)", {"bold": True, "color": S.ink}),
                  ("   We ported its scheduling into LeRobot's own runtime; the model and server are unchanged.",
                   {"color": S.muted})]], size=16, font=S.body)
    builds(s, groups + [[foot]])
    notes(s, "DRTC was designed for unreliable networks. Three ideas. The third, asking again after the cooldown, "
             "turns out to carry almost all of the benefit.")


def _seq_column(s, x, w, title, colour, lost_again):
    """A message diagram for one client: robot and server lifelines, time flowing down."""
    rx, sx = x + Inches(0.75), x + w - Inches(0.75)
    top, bot = TOP + Inches(0.25), TOP + Inches(4.15)
    yy = lambda f: int(top + (bot - top) * f)  # noqa: E731
    head = [text(s, x, TOP - Inches(0.45), w, Inches(0.35), title, size=18, bold=True, color=colour, font=S.body,
                 align=PP_ALIGN.CENTER),
            box(s, rx - Inches(0.6), top - Inches(0.12), Inches(1.2), Inches(0.3), fill=S.dark, radius=0.3,
                runs="robot", size=11, bold=True, color=S.dark_text, font=S.body),
            box(s, sx - Inches(0.6), top - Inches(0.12), Inches(1.2), Inches(0.3), fill=S.dark, radius=0.3,
                runs="GPU server", size=11, bold=True, color=S.dark_text, font=S.body),
            line(s, rx, yy(0.06), rx, bot, color=S.grid, weight=Pt(1.5)),
            line(s, sx, yy(0.06), sx, bot, color=S.grid, weight=Pt(1.5))]

    def msg(f0, f1, words, up=True, ok=True, cut=False, col=S.ink):
        a, b = (rx, sx) if up else (sx, rx)
        y_a, y_b = yy(f0), yy(f1)
        if cut and not up:  # a lost reply travels down from the server, like every reply
            y_a, y_b = min(y_a, y_b), max(y_a, y_b)
        if cut:
            mx_ = (a + b) / 2
            my_ = (y_a + y_b) / 2
            return [line(s, a, y_a, mx_, my_, color=S.alert, weight=Pt(1.75)),
                    text(s, int(mx_ - Inches(0.2)), int(my_ - Inches(0.2)), Inches(0.4), Inches(0.4), "×",
                         size=22, bold=True, color=S.alert, font=S.body, align=PP_ALIGN.CENTER,
                         anchor=MSO_ANCHOR.MIDDLE),
                    text(s, int(mx_ - Inches(0.95)), int(my_ + Inches(0.12)),
                         Inches(1.9), Inches(0.3), words, size=11, bold=True, color=S.alert, font=S.body,
                         align=PP_ALIGN.CENTER)]
        return [line(s, a, y_a, b, y_b, color=col, weight=Pt(1.75), arrow=True),
                text(s, rx + Inches(0.1), int((y_a + y_b) / 2 - Inches(0.27)), sx - rx - Inches(0.2), Inches(0.26),
                     words, size=11, color=col, font=S.body, align=PP_ALIGN.CENTER)]

    def note(f0, f1, words, col):
        return [line(s, rx - Inches(0.12), yy(f0), rx - Inches(0.12), yy(f1), color=col, weight=Pt(3)),
                text(s, x - Inches(0.3), yy(f0) - Inches(0.05), Inches(0.9), yy(f1) - yy(f0) + Inches(0.1), words,
                     size=9, bold=True,
                     color=col, font=S.body, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.RIGHT, line=1.0)]

    normal = msg(0.08, 0.16, "request: images + joints") + msg(0.2, 0.28, "a chunk: the arm moves", up=False,
                                                                      col=S.series["drtc"])
    req2 = msg(0.3, 0.36, "request")
    lost = msg(0.4, 0.46, "reply lost", up=False, cut=True)
    return head, normal, req2 + lost, yy, msg, note, rx, sx, bot


def t_drtc_media(prs, n):
    s = S.content(prs, n, "3 · Our solution", "DRTC at work: what happens when a reply is lost")
    if media_file("drtc"):
        slot(s, "drtc", M, TOP - Inches(0.1), CW, Inches(4.7))
        return
    cw = (CW - Inches(0.5)) / 2
    lx, dx = M, M + cw + Inches(0.5)
    L = _seq_column(s, lx, cw, "Patient LeRobot", S.series["baseline"], False)
    D = _seq_column(s, dx, cw, "DRTC", S.series["drtc"], True)
    yy = L[3]
    band = [box(s, x_ + Inches(0.75), yy(0.37), cw - Inches(1.5), yy(0.6) - yy(0.37), fill=S.alert, alpha=0.12)
            for x_ in (lx, dx)]
    band += [text(s, x_ + Inches(0.85), yy(0.375), Inches(1.2), Inches(0.25), "outage", size=9, bold=True,
                  color=S.alert, font=S.body) for x_ in (lx, dx)]
    # LeRobot: waits forever
    _, _, _, _, msgL, noteL, rxL, sxL, botL = L
    onceL = msgL(0.53, 0.57, "one more try: lost too", cut=True)
    waitL = noteL(0.6, 0.86, "waiting; arm still", S.alert)
    endL = [box(s, lx + Inches(0.3), yy(0.88), cw - Inches(0.6), Inches(0.42), fill="#FFFFFF", line=S.alert,
                radius=0.3, runs="nothing more is sent: 10 s later the run is ended ✗", size=12, bold=True,
                color=S.alert, font=S.body)]
    # DRTC: asks again after its cooldown
    _, _, _, _, msgD, noteD, rxD, sxD, botD = D
    cool1 = noteD(0.47, 0.53, "cooldown", S.accent)
    again1 = msgD(0.53, 0.57, "asks again: lost too", cut=True)
    cool2 = noteD(0.58, 0.71, "cooldown", S.accent)
    again2 = msgD(0.72, 0.77, "asks again: fresh images", col=S.accent)
    reply = msgD(0.8, 0.86, "a chunk: the arm moves again", up=False, col=S.series["drtc"])
    endD = [box(s, dx + Inches(0.3), yy(0.88), cw - Inches(0.6), Inches(0.42), fill="#FFFFFF",
                line=S.series["drtc"], radius=0.3, runs="task done ✓", size=12, bold=True,
                color=S.series["drtc"], font=S.body)]
    foot = text(s, M, TOP + Inches(4.3), CW, Inches(0.35),
                "A real trial (GR00T N1.7, trial 1, the same faults for both): LeRobot's arm stopped at 6.4 s and the run "
                "was ended at 16.5 s; DRTC asked again at 9.9 and 11.0 s and finished the task at 13.1 s.", size=11,
                color=S.muted, font=S.body, align=PP_ALIGN.CENTER)
    builds(s, [L[0] + D[0], L[1] + D[1], L[2] + D[2] + band, onceL + cool1 + again1, waitL + endL, cool2 + again2,
               reply + endD, [foot]])
    notes(s, "Both clients, side by side, as messages between the robot and the server; time flows down. At first "
             "both work: a request goes up, a chunk comes back. Then the outage: a reply is lost. LeRobot, even when "
             "patient, sends one fresh request when its queue runs dry; when that is lost too it never asks again: it "
             "waits, the arm stands still, and after 10 s LeRobot ends the run. DRTC "
             "waits only for its cooldown, about one round trip; then it sends a fresh request with new images. The "
             "first one falls into the outage too; the next gets through, a chunk comes back, and the task is done. "
             "This is exactly what happened in the real trial on the previous slides' timeline.")


SMOOTH = [("seam", "Seams", "a new chunk starts somewhere else: the arm jumps",
           "Temporal ensembling", "blend the overlapping chunks at the seam", "weights e^(−0.01 i) over chunks"),
          ("stall", "Stalls", "the queue runs dry: the arm halts mid-motion",
           "Adaptive playback", "play slower when the queue runs low", "rate = q / (n + 4), between 0.5 and 1"),
          ("jitter", "Jitter", "small shakes inside a chunk",
           "Butterworth filter", "remove the fast shakes from every command", "2nd order, 3 Hz cutoff at 30 Hz"),
          ("restart", "Restarts", "after a fault the arm lurches to the new chunk",
           "Resume ramp", "ease from the held pose into the new chunk", "minimum-jerk blend over 10 commands")]


def _panels(s, title_of, sub_of, detail_of, draw):
    cw, gap = Inches(2.67), Inches(0.29)
    steps = []
    curves = problem_curves()
    for i, row in enumerate(SMOOTH):
        x = M + (cw + gap) * i
        static, plot = mini_panel(s, x, TOP - Inches(0.1), cw, Inches(4.0), title_of(row), sub_of(row),
                                  detail_of(row))
        steps.append([(sh, "fade") for sh in static] + draw(s, plot, curves[row[0]]))
    return steps


def t_jerk(prs, n):
    s = S.content(prs, n, "3 · Our solution", "Recovering is not enough: the motion jerks")

    def draw(s, plot, c):
        before, _, mk = c
        out = [(plot_curve(s, plot, before, S.alert), "wipe_right", 200)]
        m = marker(s, plot, mk, before)
        if m is not None:
            out.append((m, "fade", 700))
        return out
    steps = _panels(s, lambda r: r[1], lambda r: r[2], lambda r: None, draw)
    builds(s, steps)
    notes(s, "Four ways the motion goes wrong, drawn as one joint's position over time. Seams: each chunk is planned "
             "from a slightly different moment, so the arm jumps. Stalls: the queue runs dry and the arm halts. "
             "Jitter: small shakes. Restarts: after a fault the arm lurches to the new chunk.")


def t_stack(prs, n):
    s = S.content(prs, n, "3 · Our solution", "Smoothing methods: one for each problem")

    def draw(s, plot, c):
        before, after, _ = c
        return [(plot_curve(s, plot, before, S.muted, dash=True, weight=Pt(1.5)), "fade", 100),
                (plot_curve(s, plot, after, S.accent), "wipe_right", 400)]
    steps = _panels(s, lambda r: r[3], lambda r: f"{r[4]}  (fixes {r[1].lower()})", lambda r: r[5], draw)
    foot = text(s, M, S.bottom - Inches(0.6), CW, Inches(0.55),
                [[("Two more we built: ", {"bold": True, "color": S.ink}),
                  ("RTC (the model is asked to continue the moves still queued, so the seam is smooth at the source) "
                   "and zero-lag queue smoothing. Which to keep was decided on the bench.", {"color": S.muted})]],
                size=13, font=S.body)
    builds(s, steps + [[(foot, "fade")]])
    notes(s, "Six methods in total. Each of these four answers one problem from the previous slide; dashed is before, "
             "blue is with the method. We also built RTC, real-time chunking, which guides the model itself to "
             "continue the queued actions, and a zero-lag smoother. The bench decided which ones earn their place.")


def t_hardware(prs, n):
    s = S.content(prs, n, "4 · What we built", "The hardware")
    st = MEDIA / "stills"
    ph = Inches(4.55)
    big = picture(s, st / "hw_Hardware_setup_full.jpg", M, TOP - Inches(0.25), h=ph)
    bw = big.width
    labels = [("leader arm", 0.07, 0.50), ("laptop", 0.28, 0.27), ("follower arm", 0.62, 0.30),
              ("task board", 0.62, 0.52), ("phone camera", 0.86, 0.26)]
    lab_steps = []
    for name, fx, fy in labels:
        cx, cy = big.left + bw * fx, big.top + ph * fy
        d = dot(s, cx, cy, Inches(0.07), S.alert)
        lx = min(max(int(cx - Inches(0.7)), big.left + Inches(0.08)), big.left + bw - Inches(1.48))
        t = box(s, lx, int(cy - Inches(0.42)), Inches(1.4), Inches(0.3), fill=S.dark,
                radius=0.3, runs=name, size=11, font=S.body, color=S.dark_text, bold=True)
        lab_steps.append([(d, "fade"), (t, "fade")])
    x = big.left + bw + Inches(0.3)
    sw = (W - M - x - Inches(0.2)) / 3
    smalls, caps = [], []
    for i, (f, c) in enumerate([("hw_follower_arm_with_wrist_cam.jpg", "follower arm + wrist camera"),
                                ("hw_leader_arm.jpg", "leader arm, moved by hand"),
                                ("hw_phone_camera_and_arm.jpg", "phone as the scene camera")]):
        p = picture(s, st / f, int(x + (sw + Inches(0.1)) * i), TOP - Inches(0.25), w=int(sw))
        if p.height > Inches(2.0):
            p.crop_top = p.crop_bottom = (1 - Inches(2.0) / p.height) / 2
            p.height = Inches(2.0)
        smalls.append(p)
        caps.append(text(s, int(x + (sw + Inches(0.1)) * i), TOP + Inches(1.8), int(sw), Inches(0.5), c, size=11,
                         color=S.muted, font=S.body, line=1.1))
    rows = bullets(s, x, TOP + Inches(2.45), W - M - x, [
        ("Raspberry Pi 5", "the robot's computer (8 GB, no GPU)"),
        ("Laptop", "RTX 4060; recording, the Pi's network"),
        ("A100 + H200", "shared GPU servers for the models"),
    ], size=14, gap=Inches(0.5))
    builds(s, lab_steps + [[(p, "fade") for p in smalls] + [(c, "fade") for c in caps], sum(rows, [])])
    notes(s, "The SO-101 arm and its leader arm, a webcam on the gripper, a phone on a tripod as the scene camera. "
             "The Raspberry Pi drives the arm; the models run on the university's shared GPU servers.")


def _steps_row(s, y, items, *, size=13):
    """Numbered steps in one row under a wide video."""
    n_ = len(items)
    cw = (CW - Inches(0.25) * (n_ - 1)) / n_
    return [text(s, M + (cw + Inches(0.25)) * i, y, cw, Inches(0.9),
                 [[(a, {"bold": True, "color": S.ink})], [(b, {"color": S.muted, "size": size - 1})]], size=size,
                 font=S.body, line=1.1) for i, (a, b) in enumerate(items)]


def t_record(prs, n):
    s = S.content(prs, n, "4 · What we built", "Teaching by demonstration")
    slot(s, "record", M, TOP - Inches(0.25), CW, Inches(3.75))
    steps = _steps_row(s, TOP + Inches(3.65), [
        ("1  Move the leader arm", "by hand; the follower arm copies it"),
        ("2  Record", "both cameras and the joints, 30 per second"),
        ("3  Review", "in Rerun; idle starts and ends trimmed automatically"),
        ("4  Hold back", "some episodes for testing, never trained on")])
    builds(s, steps)
    notes(s, "A real demonstration from our cube-in-box dataset on the Hugging Face Hub, as recorded: the scene camera "
             "and the wrist camera. A person moves the leader arm and the follower copies it; every frame, with the "
             "joint positions, becomes training data.")


def t_first_attempt(prs, n):
    s = S.content(prs, n, "4 · What we built", "Our first attempt failed: the cameras could not see the grasp")
    st = MEDIA / "stills"
    pw, ph = Inches(2.65), Inches(1.99)
    lab = {"size": 13, "color": S.muted, "font": S.body}
    a = [text(s, M, TOP - Inches(0.15), Inches(5.6), Inches(0.35), "FIRST ATTEMPT: two cameras in front", size=12,
              bold=True, color=S.alert, font=S.body)]
    a += [picture(s, st / f"first_attempt_t6_camera{i}.jpg", M + (pw + Inches(0.15)) * (i - 1), TOP + Inches(0.25),
                  w=pw) for i in (1, 2)]
    x2 = M + Inches(6.0)
    b = [text(s, x2, TOP - Inches(0.15), Inches(5.6), Inches(0.35), "NOW: a camera on the wrist", size=12, bold=True,
              color=S.accent, font=S.body)]
    b += [picture(s, st / f"first_attempt_v1_camera{i}.jpg", x2 + (pw + Inches(0.15)) * (i - 1), TOP + Inches(0.25),
                  w=pw) for i in (1, 2)]
    sa = stat(s, M, TOP + Inches(2.55), Inches(5.4), "0 / 10",
              "for every model trained on it (50 demonstrations): the gripper could not tell where the cube was",
              color=S.alert)
    sb = stat(s, x2, TOP + Inches(2.55), Inches(5.4), f"up to {R('base_cubebox_v1/diffusion/baseline/M1')}",
              "on cube in box after adding the wrist camera and recording again (Diffusion Policy)")
    builds(s, [a, [*sa], b, [*sb]])
    notes(s, "Our first dataset (train_6) had two cameras in front of the arm and none on the gripper. We trained two "
             "models on it and both scored 0 out of 10: the arm reached the cube but could not grasp it. Moving a "
             "camera onto the wrist fixed this; every dataset after that has one.")


def t_data(prs, n):
    s = S.content(prs, n, "4 · What we built", "Tasks and datasets: how they evolved")
    pw = Inches(2.35)
    ph_ = Inches(1.6)
    tasks = []
    for k, (f, name, what) in enumerate((("task_cubebox.jpg", "Cube in box", "pick up the green cube, put it in the box"),
                                         ("task_sort_one.jpg", "Sorting", "put the cube on the square of its colour"))):
        y = TOP - Inches(0.2) + (ph_ + Inches(0.75)) * k
        tasks.append(photo(s, M, y, pw, ph_, f, name))
        tasks.append(text(s, M, y + ph_ + Inches(0.04), pw, Inches(0.6),
                          [[(name, {"bold": True})], [(what, {"color": S.muted, "size": 11})]], size=14,
                          font=S.body, line=1.05))
    x0 = M + pw + Inches(0.45)
    nw, ag = Inches(2.55), Inches(0.42)
    r1 = TOP - Inches(0.2) + Inches(0.25)
    r2 = TOP - Inches(0.2) + ph_ + Inches(0.75) + Inches(0.25)
    nh = Inches(1.15)
    first = node(s, x0, r1, nw, nh, "First attempt", "50 demos, two cameras in front, no wrist camera: 0 / 10",
                 edge=S.alert, size=14)
    v0 = node(s, x0 + nw + ag, r1, nw, nh, "Cube in box v0",
              f"{R('data/so101_cubebox_v0/episodes')} demos; first setup with a wrist camera", edge=S.accent,
              size=14)
    v1 = node(s, x0 + 2 * (nw + ag), r1, nw, nh, "Cube in box v1",
              f"new recording, final camera setup: "
              f"{int(R('data/so101_cubebox_v1/episodes')) - int(R('data/so101_cubebox_v1/held'))} demos + "
              f"{R('data/so101_cubebox_v1/held')} held out", edge=S.series["drtc"], size=14)
    a1 = line(s, x0 + nw + Inches(0.05), r1 + nh / 2, x0 + nw + ag - Inches(0.05), r1 + nh / 2, color=S.ink,
              weight=Pt(1.5), arrow=True)
    a2 = line(s, x0 + 2 * nw + ag + Inches(0.05), r1 + nh / 2, x0 + 2 * (nw + ag) - Inches(0.05), r1 + nh / 2,
              color=S.ink, weight=Pt(1.5), arrow=True)
    lab1 = text(s, x0 + nw - Inches(0.3), r1 + nh + Inches(0.05), Inches(1.0) + ag, Inches(0.3), "wrist camera",
                size=10, color=S.accent, font=S.body, align=PP_ALIGN.CENTER)
    lab2 = text(s, x0 + 2 * nw + ag - Inches(0.75), r1 + nh + Inches(0.05), Inches(1.9) + ag, Inches(0.3),
                "re-recorded from scratch", size=10, color=S.accent, font=S.body, align=PP_ALIGN.CENTER)
    srt = node(s, x0 + 2 * (nw + ag), r2, nw, nh, "Sorting v1",
               f"{int(R('data/so101_sort_one_v1/episodes')) - int(R('data/so101_sort_one_v1/held'))} demos + "
               f"{R('data/so101_sort_one_v1/held')} held out", edge=S.series["drtc"], size=14)
    a3 = line(s, x0 + nw + ag + nw / 2, r1 + nh, x0 + nw + ag + nw / 2, r2 + nh / 2, color=S.ink, weight=Pt(1.5))
    a4 = line(s, x0 + nw + ag + nw / 2, r2 + nh / 2, x0 + 2 * (nw + ag) - Inches(0.05), r2 + nh / 2, color=S.ink,
              weight=Pt(1.5), arrow=True)
    lab3 = text(s, x0 + nw + ag + nw / 2 + Inches(0.12), r2 + nh / 2 - Inches(0.3), nw / 2 + ag + Inches(0.6), Inches(0.3),
                "second task", size=10, color=S.accent, font=S.body)
    used = text(s, x0 + 2 * (nw + ag), r1 - Inches(0.32), nw, Inches(0.3), "USED IN THE EXPERIMENTS", size=10,
                bold=True, color=S.series["drtc"], font=S.body, spacing=1.2)
    foot = text(s, x0, r2 + nh + Inches(0.25), W - M - x0, Inches(0.7),
                [[("Every recording: ", {"bold": True, "color": S.ink}),
                  ("fixed, taped layouts with balanced positions; held-out episodes recorded on top of the training "
                   "ones; trimmed automatically (raw copy kept).", {"color": S.muted})]], size=13, font=S.body)
    builds(s, [tasks, [first], [a1, lab1, v0], [a2, lab2, v1, used], [a3, a4, lab3, srt], [foot]])
    notes(s, "How our data grew. The first attempt had no wrist camera and every model scored 0 out of 10. Adding a "
             "wrist camera gave cube in box v0. v1 is not v0 with more episodes: it was recorded again from scratch "
             "with the final camera arrangement, with more demonstrations and a held-out set. Sorting was recorded on "
             "the same final setup. v1 and sorting are the datasets the experiments use.")


def t_models(prs, n):
    s = S.content(prs, n, "4 · What we built", "Seven policies, trained with their authors' recipes")
    rows = [["ACT", "0.05 B", "no", "100 (3.3 s)"], ["Diffusion Policy", "0.27 B", "no", "24 (0.8 s)"],
            ["SmolVLA", "0.45 B", "yes", "50 (1.7 s)"], ["GR00T N1.5", "2.7 B", "yes", "16 (0.5 s)"],
            ["GR00T N1.7", "3.1 B", "yes", "16 (0.5 s)"], ["π0.5", "4.1 B", "yes", "50 (1.7 s)"],
            ["MolmoAct 2", "5.6 B", "yes", "30 (1.0 s)"]]
    g = table(s, M, TOP - Inches(0.15), [Inches(2.6), Inches(1.4), Inches(1.6), Inches(1.9)],
              ["POLICY", "SIZE", "LANGUAGE", "CHUNK"], rows, size=15, row_h=Inches(0.5))
    x = M + Inches(7.9)
    st = stat(s, x, TOP, Inches(3.6), "A100 · H200", "trained on the shared GPU servers; every run logged")
    st2 = stat(s, x, TOP + Inches(1.6), Inches(3.6), "2 tasks", "a model per policy and task")
    why = text(s, x, TOP + Inches(3.1), Inches(3.6), Inches(1.4),
               [[("Why these seven: ", {"bold": True, "color": S.ink}),
                 ("chosen from published SO-101 results before any training, from the smallest and fastest (ACT) to "
                  "the strongest VLA (\u03c00.5).", {"color": S.muted})]], size=14, font=S.body, line=1.2)
    builds(s, [sum(g, []), [*st, *st2], [why]])
    notes(s, "Two small classic policies without language (ACT, Diffusion Policy) and five VLAs. Short chunks will "
             "matter later: GR00T's 16 moves last only half a second.")


MONO = "Consolas"


def node(s, x, y, w, h, title, sub, *, edge=None, fill=None, ink=None, sub_ink=None, size=15):
    return box(s, x, y, w, h, fill=fill or S.panel, line=edge or S.panel_line, weight=Pt(1.5), radius=S.card_radius,
               runs=[[(title, {"bold": True, "color": ink or S.ink})],
                     [(sub, {"color": sub_ink or S.muted, "size": 12})]], size=size, font=S.body, line_spacing=1.1)


def edge_label(s, x, y, w, words):
    return text(s, x, y, w, Inches(0.3), words, size=11, color=S.accent, font=S.body, align=PP_ALIGN.CENTER)


def t_lab(prs, n):
    s = S.content(prs, n, "4 · What we built", "Our lab platform: one layer on top of LeRobot")
    lw = Inches(7.6)
    base = box(s, M, TOP + Inches(2.85), lw, Inches(0.95), fill=S.grid, radius=S.card_radius,
               runs=[[("LeRobot 0.7  ·  unchanged", {"bold": True})],
                     [("training · policy server · robot drivers · safety checks",
                       {"color": S.muted, "size": 12})]], size=15, font=S.body)
    bw, gap = Inches(1.78), Inches(0.163)
    blocks = [("Registry", "small files: rig, dataset, recipe, model, host, condition, evaluation"),
              ("Plugins", "our GR00T N1.5 port; the replay robot for the bench"),
              ("Control layer", "DRTC, smoothing, fault injector"),
              ("Evaluation", "frozen plans, trial records, metrics, reports")]
    mids = [node(s, M + (bw + gap) * i, TOP + Inches(0.75), bw, Inches(1.95), a, b, edge=S.accent)
            for i, (a, b) in enumerate(blocks)]
    top = box(s, M, TOP - Inches(0.25), lw, Inches(0.8), fill=S.dark, radius=S.card_radius,
              runs=[[("lab", {"bold": True, "color": S.dark_accent, "font": MONO, "size": 18}),
                     ("   record · trim · train · serve · eval",
                      {"color": S.dark_text, "font": MONO, "size": 15})],
                    [("one command for every step; it builds the exact LeRobot command",
                      {"color": S.dark_muted, "size": 11})]], size=15, font=S.body)
    x = M + lw + Inches(0.45)
    sw = W - M - x
    st = [stat(s, x, TOP - Inches(0.2), sw, R("software/tests"), "automated tests, run without a GPU or a robot",
               size=30),
          stat(s, x, TOP + Inches(1.3), sw, "7 kinds of file", "everything an experiment depends on, named and "
               "versioned in git", size=26),
          stat(s, x, TOP + Inches(2.8), sw, "Pinned", "LeRobot is never edited: every change of ours is in our "
               "own, tested code", size=26)]
    builds(s, [[base], mids, [top], st[0], st[1], st[2]])
    notes(s, "We did not fork LeRobot. Everything of ours is a layer on top: the registry says what an experiment "
             "uses, the lab command runs each step, two plugins add GR00T N1.5 and the replay robot, and the control "
             "layer holds DRTC, smoothing and the fault injector.")


def node2(s, x, y, w, h, title, sub, accent, *, size=15):
    """A card with a coloured stripe on its left: the machine diagrams' node."""
    out = [box(s, x, y, w, h, fill="#FFFFFF", line=S.panel_line, weight=Pt(1), radius=0.08),
           box(s, x, y + Inches(0.12), Inches(0.07), h - Inches(0.24), fill=accent)]
    out.append(text(s, x + Inches(0.25), y, w - Inches(0.35), h,
                    [[(title, {"bold": True, "color": S.ink})], [(sub, {"color": S.muted, "size": 12})]],
                    size=size, font=S.body, anchor=MSO_ANCHOR.MIDDLE, line=1.1))
    return out


def zone(s, x, y, w, h, label):
    return [box(s, x, y, w, h, fill=S.panel, line=S.panel_line, weight=Pt(1), radius=0.04),
            text(s, x + Inches(0.2), y + Inches(0.12), w - Inches(0.4), Inches(0.3), label.upper(), size=10,
                 bold=True, color=S.muted, font=S.body, spacing=1.2)]


def t_machines(prs, n):
    s = S.content(prs, n, "4 · What we built", "Four computers, and how work moves between them")
    zw, zh = Inches(3.5), Inches(3.0)
    y0 = TOP - Inches(0.3)
    lx, rx = M, W - M - zw
    nw, nh = zw - Inches(0.4), Inches(1.0)
    robot = zone(s, lx, y0, zw, zh, "At the robot")
    pi = node2(s, lx + Inches(0.2), y0 + Inches(0.5), nw, nh, "Raspberry Pi 5", "drives the arm and cameras; "
               "the client at 30 Hz", S.series["baseline"])
    lap = node2(s, lx + Inches(0.2), y0 + Inches(1.85), nw, nh, "Laptop  ·  RTX 4060", "recording, development; "
                "routes the Pi's traffic", S.series["baseline"])
    cx = lx + zw / 2
    link1 = [line(s, cx, y0 + Inches(0.5) + nh, cx, y0 + Inches(1.85), color=S.ink, weight=Pt(1.5)),
             text(s, cx + Inches(0.1), y0 + Inches(1.5), Inches(1.5), Inches(0.3), "WiFi or cable", size=10,
                  color=S.accent, font=S.body)]
    gpu = zone(s, rx, y0, zw, zh, "University GPU servers")
    a100 = node2(s, rx + Inches(0.2), y0 + Inches(0.5), nw, nh, "A100  ·  80 GB", "training and serving",
                 S.series["faults"])
    h200 = node2(s, rx + Inches(0.2), y0 + Inches(1.85), nw, nh, "H200  ·  141 GB", "training, serving, "
                 "the bench", S.series["faults"])
    # the network between the two zones: one bus, right angles only
    ya, yh = y0 + Inches(0.5) + nh / 2, y0 + Inches(1.85) + nh / 2
    xl, xr = lx + Inches(0.2) + nw, rx + Inches(0.2)
    xb = rx - Inches(0.45)
    net = [line(s, xl, yh, xb, yh, color=S.ink, weight=Pt(1.5)),
           line(s, xb, ya, xb, yh, color=S.ink, weight=Pt(1.5)),
           line(s, xb, ya, xr, ya, color=S.ink, weight=Pt(1.5), arrow=True),
           line(s, xb, yh, xr, yh, color=S.ink, weight=Pt(1.5), arrow=True)]
    mx0, mw = lx + zw + Inches(0.25), rx - (lx + zw) - Inches(0.5)
    net.append(text(s, mx0, y0 + Inches(0.45), mw - Inches(0.5), Inches(1.4),
                    [[("Campus network", {"bold": True, "color": S.ink, "size": 16})],
                     [("the Pi's own SSH tunnel, routed through the laptop", {"color": S.muted})],
                     [("↑  2 images + 6 joints, 2.9 MB", {"color": S.accent})],
                     [("↓  a chunk of actions", {"color": S.accent})]], size=13, font=S.body, line=1.3,
                    align=PP_ALIGN.CENTER))
    # shared online services
    by, bh = y0 + zh + Inches(0.45), Inches(1.35)
    band = zone(s, M, by, CW, bh, "Shared online")
    sw = (CW - Inches(0.4) - Inches(0.5)) / 3
    svc = []
    for k, (t1, t2, col) in enumerate((("GitHub", "code and the registry", S.muted),
                                       ("Hugging Face Hub", "datasets, models, the results store", S.series["drtc"]),
                                       ("wandb + Trackio", "every training run logged", S.series["smooth"]))):
        svc += node2(s, M + Inches(0.2) + (sw + Inches(0.25)) * k, by + Inches(0.42), sw, Inches(0.78), t1, t2, col,
                     size=14)
    down = [line(s, cx, y0 + zh, cx, by, color=S.ink, weight=Pt(1.5)),
            line(s, rx + zw / 2, y0 + zh, rx + zw / 2, by, color=S.ink, weight=Pt(1.5))]
    builds(s, [robot + pi + lap + link1, gpu + a100 + h200 + net, band + svc + down])
    notes(s, "Left, at the robot: the Raspberry Pi drives the arm; the laptop records and is the Pi's network "
             "gateway. Right: the university's two shared GPU servers train and serve the models and run the bench. "
             "Between them the campus network. The Pi opens its own SSH tunnel to the server, with its own key; the laptop "
             "only routes its traffic (hotspot, or the cable for E4). Every request carries two images and the joints "
             "up, and a chunk of actions comes back. Below: GitHub keeps the code and the registry; the Hugging Face Hub keeps datasets, "
             "models and every trial's record; wandb and Trackio log every training run.")


RERUN = {"bg": "#17191C", "grid": "#2B2F35", "text": "#C9CDD3", "muted": "#7E848D",
         "lines": ["#E05FB5", "#E0A33A", "#C9D24A", "#F07F50", "#4C8FE6", "#5DC9C2"]}


def t_trimming(prs, n):
    s = S.content(prs, n, "4 · What we built", "Trimming: automatic, not by hand")
    d = json.loads((MEDIA / "data" / "trim_episode0.json").read_text())
    a = np.array(d["action"])
    fps, st_, en = d["fps"], d["start"], d["end"]
    t = np.arange(len(a)) / fps
    names = [nm.replace(".pos", "") for nm in d["names"]]
    why = text(s, M, TOP - Inches(0.35), CW, Inches(0.35),
               [[("Why: ", {"bold": True, "color": S.ink}),
                 ("trained on still starts, a model learns that not moving is a valid first move (“never "
                  "starts”).", {"color": S.muted})]], size=15, font=S.body)
    # a Rerun-style panel
    bx, by, bw_, bh = M, TOP + Inches(0.1), CW, Inches(2.6)
    panel = [box(s, bx, by, bw_, bh, fill=RERUN["bg"], radius=0.03),
             text(s, bx + Inches(0.15), by + Inches(0.07), Inches(3), Inches(0.25), "action  ·  episode 0 "
                  "(raw recording)", size=11, bold=True, color=RERUN["text"], font=S.body)]
    px, py, pw, ph = bx + Inches(0.55), by + Inches(0.4), bw_ - Inches(2.2), bh - Inches(0.75)
    f = mapper(px, py, pw, ph, 0, t[-1], -110, 110)
    for v in (-100, 0, 100):
        _, yy = f(0, v)
        panel.append(line(s, px, yy, px + pw, yy, color=RERUN["grid"], weight=Pt(0.75)))
        panel.append(text(s, bx + Inches(0.05), int(yy - Inches(0.11)), Inches(0.45), Inches(0.22), str(v), size=9,
                          color=RERUN["muted"], font=S.body, align=PP_ALIGN.RIGHT))
    for sec in range(0, int(t[-1]) + 1):
        xx, _ = f(sec, 0)
        panel.append(line(s, xx, py, xx, py + ph, color=RERUN["grid"], weight=Pt(0.5)))
        panel.append(text(s, int(xx - Inches(0.3)), py + ph + Inches(0.04), Inches(0.6), Inches(0.22), f"{sec} s",
                          size=9, color=RERUN["muted"], font=S.body, align=PP_ALIGN.CENTER))
    curves = [curve(s, [f(tt, vv) for tt, vv in zip(t, a[:, j])], RERUN["lines"][j], weight=Pt(1.5))
              for j in range(a.shape[1])]
    lx = px + pw + Inches(0.2)
    leg = [box(s, lx, py, Inches(1.35), Inches(1.45), fill="#23262B", radius=0.08)]
    for j, nm in enumerate(names):
        yy = py + Inches(0.08) + Inches(0.22) * j
        leg.append(dot(s, lx + Inches(0.14), yy + Inches(0.1), Inches(0.045), RERUN["lines"][j]))
        leg.append(text(s, lx + Inches(0.25), yy, Inches(1.1), Inches(0.21), nm, size=9, color=RERUN["text"],
                        font=S.body))
    fx = lambda tt: int(f(tt, 0)[0])  # noqa: E731
    cut = []
    for x0, x1, secs in ((0, st_ / fps, st_ / fps), (en / fps, t[-1], (len(a) - en) / fps)):
        cut.append(box(s, fx(x0), py, fx(x1) - fx(x0), ph, fill=S.alert, alpha=0.22))
        xc = x1 if x0 == 0 else x0
        cut.append(line(s, fx(xc), py, fx(xc), py + ph, color=S.alert, weight=Pt(1.75), dash=True))
        cut.append(text(s, fx(x0), py + ph - Inches(0.32), fx(x1) - fx(x0), Inches(0.25), f"cut {secs:.1f} s",
                        size=10, bold=True, color="#FFFFFF", font=S.body, align=PP_ALIGN.CENTER))
    cw, gap = Inches(3.6), Inches(0.37)
    cy = TOP + Inches(2.9)
    cards = [("Automatic trim", "finds the still start and end from the joints' own noise; no manual editing",
              S.accent),
             ("Raw copy kept", "the recording as made; the trim never alters it", S.panel_line),
             ("Trimmed copy", "rebuilt automatically from the raw one; the copy models train on", S.panel_line)]
    cs = [node(s, M + (cw + gap) * i, cy, cw, Inches(1.1), a_, b_, edge=e_) for i, (a_, b_, e_) in enumerate(cards)]
    num = text(s, M, cy + Inches(1.2), CW, Inches(0.35),
               [[("All three datasets trimmed automatically: ", {"bold": True, "color": S.ink}),
                 ("69,864 → 58,104 frames (−17 %). Manual tools exist only for the rare bad episode.",
                  {"color": S.muted})]], size=13, font=S.body)
    builds(s, [[why] + panel + leg + [(c, "wipe_right") for c in curves], cut, cs + [num]])
    notes(s, "This is a raw demonstration as Rerun shows it: the six joints over 9.6 s. At the start and the end the "
             "arm is still while the operator gets ready and lets go. Our trim finds those parts automatically, from "
             "how much the joints jitter when nobody moves them, and cuts them; nobody edits episodes by hand. The raw "
             "recording is kept, and the trimmed copy is rebuilt from it.")


def t_offline(prs, n):
    s = S.content(prs, n, "5 · How we evaluate", "Offline: checking each model before robot time")
    sub = text(s, M, TOP - Inches(0.35), CW, Inches(0.4),
               "On held-out demonstrations, each trained model predicts every step; we compare with what the person "
               "did, and time how long it takes to answer. Four policies, sorting task.", size=14, color=S.muted,
               font=S.body)
    err = [("SmolVLA", 2.13), ("Diffusion Policy", 2.43), ("GR00T N1.5", 3.03), ("ACT", 3.20)]
    ms = [("ACT", 42), ("SmolVLA", 97), ("GR00T N1.5", 167), ("Diffusion Policy", 179)]
    pw = (CW - Inches(0.5)) / 2

    def panel(x, title, rows, vmax, fmt, col):
        out = [text(s, x, TOP + Inches(0.3), pw, Inches(0.35), title, size=15, bold=True, color=S.ink,
                    font=S.body)]
        rh = Inches(0.55)
        for i, (name, v) in enumerate(rows):
            y = TOP + Inches(0.8) + rh * i
            out.append(text(s, x, y, Inches(1.8), rh, name, size=14, color=S.ink, font=S.body,
                            anchor=MSO_ANCHOR.MIDDLE))
            bw = int((pw - Inches(2.8)) * v / vmax)
            out.append(box(s, x + Inches(1.85), y + Inches(0.12), bw, rh - Inches(0.24), fill=col))
            out.append(text(s, x + Inches(1.95) + bw, y, Inches(1.0), rh, fmt(v), size=14, bold=True,
                            color=S.ink, font=S.body, anchor=MSO_ANCHOR.MIDDLE))
        return out
    a = panel(M, "How close to the person  (arm error)", err, 3.5, lambda v: f"{v:.1f}°", S.accent)
    b = panel(M + pw + Inches(0.5), "Time to answer one chunk  (H200)", ms, 200, lambda v: f"{v} ms",
              S.series["smooth"])
    take = text(s, M, TOP + Inches(3.25), CW, Inches(0.9),
                [[("Every model learned the task to within 2-3°; their answer times differ fourfold. ",
                   {"bold": True, "color": S.ink}),
                  ("On the robot the larger VLAs are slower still (π0.5 about 0.43 s per chunk), and that time "
                   "adds to every network round trip.", {"color": S.muted})]], size=15, font=S.body)
    builds(s, [[sub], a, b, [take]])
    notes(s, "The first of the three ways to evaluate: no robot at all. Held-out demonstrations never seen in "
             "training; the model predicts each step and we measure how far its arm command is from the person's, "
             "and how long one chunk takes on the server. All are within a few degrees; the slower models will "
             "matter once the network is in the loop.")


def t_bench(prs, n):
    s = S.content(prs, n, "5 · How we evaluate", "The bench: the arm replayed, the model and server real")
    bw, gap = Inches(2.0), Inches(0.39)
    y = TOP + Inches(0.1)
    items = [("Recorded episode", "camera frames + joints from a held-out demonstration", S.panel_line),
             ("Replay robot", "plays them in place of the arm (our plugin)", S.accent),
             ("Client", "LeRobot or DRTC, with or without smoothing", S.accent),
             ("Network + faults", "simulated: 250 ms as measured on the robot, plus this run's fault", S.alert),
             ("Policy server", "the real trained model on the GPU", S.series["drtc"])]
    nodes, arrows = [], []
    for i, (a_, b_, col) in enumerate(items):
        x = M + (bw + gap) * i
        nodes.append(node(s, x, y, bw, Inches(1.45), a_, b_, edge=col, size=14))
        if i:
            arrows.append(line(s, x - gap + Inches(0.04), y + Inches(0.5), x - Inches(0.04), y + Inches(0.5),
                               color=S.ink, weight=Pt(1.25), arrow=True))
    xr = M + (bw + gap) * 4 + bw / 2
    xc = M + (bw + gap) * 2 + bw / 2
    back = [line(s, xr, y + Inches(1.45), xr, y + Inches(1.8), color=S.ink, weight=Pt(1.25)),
            line(s, xr, y + Inches(1.8), xc, y + Inches(1.8), color=S.ink, weight=Pt(1.25)),
            line(s, xc, y + Inches(1.8), xc, y + Inches(1.48), color=S.ink, weight=Pt(1.25), arrow=True),
            edge_label(s, xc, y + Inches(1.85), xr - xc, "the chunk comes back through the same network")]
    sy = TOP + Inches(2.45)
    st = [stat(s, M, sy, Inches(3.6), "2,912", "runs per task for E1 + E2: 7 policies × 52 conditions × 8 "
               "runs", size=30),
          stat(s, M + Inches(3.97), sy, Inches(3.6), "14 s", "per run; timing and motion measured. Nothing moves, "
               "so no task success", size=30),
          stat(s, M + Inches(7.94), sy, Inches(3.6), "Same seed", "every client meets exactly the same faults",
               size=30)]
    builds(s, [nodes[:2] + arrows[:1], nodes[2:3] + arrows[1:2], nodes[3:] + arrows[2:], back, st[0] + st[1] + st[2]])
    notes(s, "On the bench the arm is replaced by a held-out demonstration: its camera frames and joint states are "
             "played back by our replay robot. The client and the trained model on the GPU are real; the network to "
             "the robot is simulated, 250 ms with up to 20 ms of jitter as measured on the robot, plus the run's "
             "fault. So every fault can be run for every policy thousands of times, identically for each client. "
             "Because the images do not react to the actions, the bench has no task success: that is what the robot "
             "experiment is for.")


def t_records(prs, n):
    s = S.content(prs, n, "5 · How we evaluate", "Every trial is recorded")
    tree = [("trials/<condition>/<model>/t01/", S.ink, ""),
            ("  video/scene.mp4, wrist.mp4", S.ink, "both cameras"),
            ("  ticks.csv", S.ink, "every 33 ms: queue, command, measured joints"),
            ("  chunks.csv", S.ink, "every request: sent, received, turnaround"),
            ("  trial.json", S.ink, "outcome, timing, machine, server")]
    paras = []
    for path, col, what in tree:
        paras.append([(path, {"font": MONO, "color": col, "size": 14})] +
                     ([(f"   {what}", {"color": S.muted, "size": 12})] if what else []))
    card_ = box(s, M, TOP - Inches(0.15), Inches(7.0), Inches(2.4), fill=S.panel, line=S.panel_line,
                radius=S.card_radius)
    tr = text(s, M + Inches(0.3), TOP + Inches(0.05), Inches(6.5), Inches(2.1), paras, size=14, font=S.body,
              line=1.45)
    rows = bullets(s, M, TOP + Inches(2.5), Inches(7.0), [
        ("Plans frozen", "models, conditions and trials fixed before the first run"),
        ("Rounds interleaved", "conditions alternate within each session"),
        ("Results store", "every record pushed to the Hugging Face Hub"),
    ], size=15, gap=Inches(0.55))
    x = M + Inches(7.5)
    st = [stat(s, x, TOP - Inches(0.1), Inches(4.0), "300", "robot trials: 140 baselines + 160 in E4", size=32),
          stat(s, x, TOP + Inches(1.35), Inches(4.0), "5,824", "bench runs for E1 + E2 on the two tasks", size=32),
          stat(s, x, TOP + Inches(2.8), Inches(4.0), "1 run discarded", "its labels were entered by another person; "
               "archived with the reason, run again", size=24, color=S.alert)]
    builds(s, [[card_, tr], sum(rows, []), st[0] + st[1], st[2]])
    notes(s, "Every number in the results comes from these raw files, so any table can be rebuilt. One cube-in-box "
             "run was discarded because someone else entered its labels; it is kept in an archive with the reason, "
             "and the run was repeated from scratch.")


def t_map(prs, n):
    s = S.content(prs, n, "5 · How we evaluate", "Five studies, five questions (the experiment design)")
    cw, gap = Inches(2.15), Inches(0.2)
    rows = [("Baselines", "robot", "How well does each policy do the task?", "7 policies × 2 tasks × 10 = 140"),
            ("E1", "bench", "Does DRTC recover where LeRobot's client does not?", "7 policies, 2 tasks, every fault"),
            ("E2", "bench", "Which part of DRTC matters?", "remove one part at a time"),
            ("E3", "bench", "Which smoothing, and does the stack help?", "each method alone, then the stack"),
            ("E4", "robot", "Does it hold on the real arm?", "4 policies × 4 conditions × 10 = 160")]
    groups = []
    for i, (a_, where, q, size) in enumerate(rows):
        x = M + (cw + gap) * i
        col = S.series["drtc"] if where == "robot" else S.accent
        g = [box(s, x, TOP - Inches(0.1), cw, Inches(3.6), fill=S.panel, line=S.panel_line, radius=S.card_radius),
             text(s, x + Inches(0.2), TOP + Inches(0.1), cw - Inches(0.4), Inches(0.5), a_, size=26, bold=True,
                  font=S.heading, color=col),
             text(s, x + Inches(0.2), TOP + Inches(0.7), cw - Inches(0.4), Inches(0.3), where.upper(), size=11,
                  bold=True, color=S.muted, font=S.body, spacing=1.2),
             text(s, x + Inches(0.2), TOP + Inches(1.05), cw - Inches(0.4), Inches(1.4), q, size=15, bold=True,
                  font=S.body, color=S.ink, line=1.1),
             text(s, x + Inches(0.2), TOP + Inches(2.6), cw - Inches(0.4), Inches(0.8), size, size=12,
                  color=S.muted, font=S.body, line=1.15)]
        groups.append(g)
    foot = text(s, M, TOP + Inches(3.75), CW, Inches(0.5),
                [[("Written down before the first run: ", {"bold": True, "color": S.ink}),
                  ("the questions, the comparisons that are tested, and the rule that picks the smoothing stack.",
                   {"color": S.muted})]], size=14, font=S.body)
    builds(s, groups + [[foot]])
    notes(s, "The map of the results. Baselines on the robot first, then three bench studies, then the robot "
             "experiment E4 with DRTC under faults.")


def t_contrib(prs, n):
    s = S.content(prs, n, "Close", "What we contributed")
    items = [("A platform on LeRobot 0.7", "registry, one command, two plugins, four machines"),
             ("Data and policies", "three datasets with automatic trimming; seven policies, original recipes"),
             ("DRTC inside LeRobot's runtime", "with fixes found on the real robot"),
             ("Six smoothing methods", "and a rule fixed in advance for combining them"),
             ("Network fault injection", "seven fault types, seeded per trial"),
             ("An evaluation framework", "offline, bench and robot; frozen plans; raw records"),
             ("The experiments", "140 baseline trials, the bench studies, 160 E4 trials, the Pi as robot computer")]
    rows = [para(s, M, TOP - Inches(0.2) + Inches(0.62) * i, CW, f"{i + 1}   {a_}", b_, size=17, h=Inches(0.6))
            for i, (a_, b_) in enumerate(items)]
    builds(s, rows)
    notes(s, "Seven contributions, in the order of the talk.")


def t_eval_kinds(prs, n):
    s = S.content(prs, n, "5 · How we evaluate", "Three ways to evaluate")
    cw, gap = Inches(3.6), Inches(0.37)
    cards = [("Offline", "No robot", "held-out demonstrations: how close the model's actions are to the person's, and "
              "how long it takes to answer"),
             ("Bench", "Robot replayed", "the real model and server, the arm replayed from recordings: every fault, "
              "thousands of runs, timing and motion"),
             ("Robot", "Live, real time", "the real arm doing the task from the Raspberry Pi, faults injected live, "
              "success judged by the operator")]
    groups = [card(s, M + (cw + gap) * i, TOP, cw, Inches(2.6), a, b, c,
                   num_color=S.series["drtc"] if a == "Robot" else None) for i, (a, b, c) in enumerate(cards)]
    foot = text(s, M, TOP + Inches(2.95), CW, Inches(1.4),
                [[("Measured per trial: ", {"bold": True, "color": S.ink}),
                  ("task success (with a 95 % interval), stalls, jerk, recovery time after each fault, deviation from "
                   "the policy's plan.", {"color": S.muted})],
                 [("Compared with Fisher's exact and Mann-Whitney tests, Holm-corrected; the tested comparisons named "
                   "before the first run.", {"color": S.accent, "size": 13})]], size=15, font=S.body, line=1.35)
    builds(s, groups + [[foot]])
    notes(s, "Three kinds of evaluation, from cheapest to most real. Offline needs no robot; the bench runs the real "
             "software against a recorded arm; the robot is the real thing, live.")


def t_robot_eval(prs, n):
    s = S.content(prs, n, "5 · How we evaluate", "Robot evaluation, live and in real time")
    slot(s, "robot_eval", M, TOP - Inches(0.25), CW, Inches(3.75))
    steps = _steps_row(s, TOP + Inches(3.65), [
        ("1  Set the layout", "taped marks, rotated every round"),
        ("2  Start the trial", "the Pi runs the client; the H200 runs the model"),
        ("3  Faults, live", "the injector applies the trial's seeded faults"),
        ("4  Judge it", "s for success, f for fail; 30 s limit"),
        ("5  Save it", "videos, every tick, every chunk, the outcome")], size=12)
    builds(s, steps)
    notes(s, "A real robot trial as it ran, live, from the Raspberry Pi: the DRTC run of GR00T N1.7, trial 1, under "
             "the fault mix. Trials run in rounds: each round runs every model under every condition once, in a "
             "seeded random order, so slow drifts affect all conditions alike. A trial with a setup mistake is voided "
             "and run again, never deleted.")


def hbars(s, x, y, w, rows, vmax, *, label_w=Inches(3.3), row_h=Inches(0.62), colours=None, unit=""):
    """Horizontal bars: rows = [(label, [values...], note)]; one bar per value (series), stacked closely."""
    colours = colours or [S.accent, S.series["drtc"]]
    gx, gw = x + label_w, w - label_w - Inches(2.0)
    groups = []
    for i, (lab, vals, note) in enumerate(rows):
        yy = y + row_h * i
        g = [text(s, x, yy, label_w - Inches(0.15), row_h, lab, size=14, color=S.ink, font=S.body,
                  anchor=MSO_ANCHOR.MIDDLE, line=1.05)]
        bh = (row_h - Inches(0.16)) / len(vals)
        for k, v in enumerate(vals):
            by = yy + Inches(0.08) + bh * k
            bw = int(gw * max(v, 0) / vmax)
            g.append(box(s, gx, int(by), max(bw, Pt(2)), int(bh - Pt(2)), fill=colours[k]))
            g.append(text(s, gx + max(bw, Pt(2)) + Inches(0.06), int(by - Pt(2)), Inches(0.9), int(bh), f"{v:+.1f}{unit}",
                          size=11, color=S.ink, font=S.body, anchor=MSO_ANCHOR.MIDDLE))
        g.append(text(s, gx + gw + Inches(0.3), yy, Inches(1.75), row_h, note, size=11, color=S.muted, font=S.body,
                      anchor=MSO_ANCHOR.MIDDLE))
        groups.append(g)
    axis = line(s, gx, y - Inches(0.05), gx, y + row_h * len(rows), color=S.ink, weight=Pt(1))
    return axis, groups


def t_e2(prs, n):
    s = S.content(prs, n, "Q3  ·  bench, DRTC taken apart (E2)", "What makes DRTC work: asking again after a lost reply")
    sub = text(s, M, TOP - Inches(0.4), CW, Inches(0.6),
               [[("DRTC has four parts. On the bench we ", {"color": S.muted}),
                 ("switched off one part at a time", {"bold": True, "color": S.ink}),
                 (", ran the same network faults, and measured how much more of the time the arm stood still.",
                  {"color": S.muted})]], size=15, font=S.body, line=1.15)

    def vals(key):
        return [float(R(f"v2/{t}/e2/{key}/mix/delta")) for t in ("sort_one", "cubebox_v1")]
    rows = [("a1_no_cooldown", "Ask again when a reply is lost", "no answer in time → send a fresh request",
             "Needed", S.alert),
            ("a4_half_chunk_cap", "Wait long enough before asking again", "our fix: never give up before the "
             "measured delay", "Needed", S.alert),
            ("a3_upstream_trigger", "Ask early", "request with about 1 s of moves left", "Barely", S.muted),
            ("a2_upstream_estimator", "A careful delay estimate", "a TCP-style estimate instead of a simple one",
             "Not needed", S.muted)]
    nx, nw_ = M, Inches(4.4)
    bx, bw_ = M + Inches(4.6), Inches(5.0)
    vx = W - M - Inches(1.45)
    y0 = TOP + Inches(0.7)
    hdr = [text(s, nx, y0 - Inches(0.35), nw_, Inches(0.3), "PART OF DRTC", size=10, bold=True, color=S.muted,
                font=S.body, spacing=1.2),
           text(s, bx, y0 - Inches(0.35), bw_, Inches(0.3), "SWITCHED OFF: EXTRA SHARE OF THE RUN STANDING STILL", size=10,
                bold=True, color=S.muted, font=S.body, spacing=1.2),
           text(s, vx, y0 - Inches(0.35), Inches(1.45), Inches(0.3), "VERDICT", size=10, bold=True, color=S.muted,
                font=S.body, spacing=1.2)]
    rh = Inches(0.72)
    vmax = 30.0
    groups = []
    for i, (key, name, what, verdict, col) in enumerate(rows):
        y = y0 + rh * i
        v = vals(key)
        mean = max(0.0, sum(v) / len(v))
        lo, hi = max(0, round(min(v))), round(max(v))
        lab = "≈ 0 %" if hi < 1 else (f"+{round(mean)} %" if hi - lo < 2 else f"+{lo} to {hi} %")
        bl = int(bw_ * 0.8 * mean / vmax)
        g = [text(s, nx, y, nw_, rh, [[(name, {"bold": True, "color": S.ink})], [(what, {"color": S.muted,
                                                                                           "size": 12})]],
                  size=16, font=S.body, anchor=MSO_ANCHOR.MIDDLE, line=1.05),
             box(s, bx, y + Inches(0.2), max(bl, Pt(3)), rh - Inches(0.4), fill=col if verdict == "Needed" else S.grid),
             text(s, bx + max(bl, Pt(3)) + Inches(0.12), y, Inches(1.6), rh, lab, size=17, bold=True,
                  color=col if verdict == "Needed" else S.ink, font=S.heading, anchor=MSO_ANCHOR.MIDDLE),
             box(s, vx, y + Inches(0.2), Inches(1.3), rh - Inches(0.4), fill=col if verdict == "Needed" else "#FFFFFF",
                 line=None if verdict == "Needed" else S.panel_line, radius=0.5, runs=verdict, size=12, bold=True,
                 color="#FFFFFF" if verdict == "Needed" else S.muted, font=S.body),
             line(s, M, y + rh, W - M, y + rh, color=S.grid, weight=Pt(0.5))]
        groups.append(g)
    take = box(s, M, y0 + rh * 4 + Inches(0.2), CW, Inches(0.62), fill=S.dark, radius=0.15,
               runs=[[("Switch off \u201cask again\u201d and the arm stands still a quarter more of the run, ",
                       {"color": S.dark_text}), ("for all 7 policies.", {"bold": True, "color": S.dark_accent})]], size=16, font=S.body)
    foot = text(s, M, y0 + rh * 4 + Inches(0.88), CW, Inches(0.3),
                "Bench, all 7 policies, both tasks, the fault mix; median increase in the share of the run the arm "
                "stood still.", size=10, color=S.muted, font=S.body)
    builds(s, [[sub] + hdr] + groups + [[take, foot]])
    notes(s, "DRTC is four ideas. We switched each off in turn, on the bench, with the same faults. Switch off 'ask "
             "again' and the arm stands still for about a quarter more of the run, for every policy. Switch off our "
             "fix, so the client gives up on a reply before it could possibly arrive, and it is almost as bad. Asking "
             "early helps a little; a more careful delay estimate makes no difference. So the one idea that matters "
             "is: when a reply is lost, ask again, but not too soon.")


def t_e3a(prs, n):
    s = S.content(prs, n, "Q4  ·  bench, each method alone (E3a)", "Each smoothing method fixes one thing; RTC made it worse")
    rows = [("Butterworth filter", "jerk down to 0.12-0.28 of DRTC alone, every policy; costs 0.9-1.6° of "
             "deviation from the plan", "helps", S.series["drtc"]),
            ("Adaptive playback", "stalls of the slow policies cut: GR00T N1.5 64→40 %, Diffusion ~50→30 %, "
             "MolmoAct 2 56→32 %", "helps", S.series["drtc"]),
            ("Resume ramp", "helps where the arm restarts often: GR00T N1.5's jerk down to 0.18", "helps",
             S.series["drtc"]),
            ("Zero-lag smoothing", "halved the jerk of the fast policies, raised it for GR00T N1.5 and MolmoAct 2",
             "mixed", S.series["smooth"]),
            ("Temporal ensembling", "changed little, except for ACT", "little", S.muted),
            ("RTC", "more than doubled SmolVLA's stalls (14 → 36-38 %): each request is slower", "hurts",
             S.alert)]
    groups = []
    rh = Inches(0.6)
    y0 = TOP + Inches(0.2)
    for i, (name, what, verdict, col) in enumerate(rows):
        y = y0 + rh * i
        g = [text(s, M, y, Inches(2.6), rh, name, size=16, bold=True, color=S.ink, font=S.body,
                  anchor=MSO_ANCHOR.MIDDLE),
             box(s, M + Inches(2.7), y + Inches(0.14), Inches(0.95), rh - Inches(0.28), fill=col, radius=0.5,
                 runs=verdict, size=11, bold=True, color="#FFFFFF", font=S.body),
             text(s, M + Inches(3.85), y, CW - Inches(3.85), rh, what, size=14, color=S.muted, font=S.body,
                  anchor=MSO_ANCHOR.MIDDLE, line=1.05),
             line(s, M, y + rh, W - M, y + rh, color=S.grid, weight=Pt(0.5))]
        groups.append(g)
    top = text(s, M, TOP - Inches(0.45), CW, Inches(0.35),
               "On DRTC under the fault mix, bench, 7 policies × 2 tasks. Two groups: fast policies stall little "
               "(ACT 3 %, SmolVLA 12-15 %); slow ones a third to two thirds of the run.", size=12, color=S.muted,
               font=S.body)
    builds(s, [[top]] + groups)
    notes(s, "Each method did one job, the same on both tasks. The filter removes jerk; adaptive playback removes "
             "stalls; the ramp smooths restarts. RTC did not help here: asking the model to continue the queue makes "
             "every request slower, and on our network that costs more stalls than the smoother seams save.")


def t_stack_choice(prs, n):
    s = S.content(prs, n, "Q4  ·  bench, the stack (E3b)", "The stack: chosen by a rule fixed in advance, then RTC dropped")
    nw, ag = Inches(2.62), Inches(0.35)
    y = TOP - Inches(0.1)
    steps = [("1  A rule, fixed first", "drop a method that moves the arm >1° off the plan or slows "
              "recovery; then the lowest jerk wins", S.panel_line),
             ("2  The rule picked", "adaptive playback + RTC + Butterworth filter + resume ramp", S.accent),
             ("3  E3b: remove one part", "without RTC the stalls fell in all 4 RTC pairs (SmolVLA −16 to −20 "
              "points) at the same jerk", S.alert),
             ("4  Used on the robot", "adaptive playback + temporal ensembling + Butterworth filter + resume ramp",
              S.series["drtc"])]
    groups = []
    for i, (a_, b_, col) in enumerate(steps):
        x = M + (nw + ag) * i
        g = [node(s, x, y, nw, Inches(1.75), a_, b_, edge=col, size=14)]
        if i:
            g.insert(0, line(s, x - ag + Inches(0.04), y + Inches(0.87), x - Inches(0.04), y + Inches(0.87),
                             color=S.ink, weight=Pt(1.5), arrow=True))
        groups.append(g)
    sy = TOP + Inches(2.0)
    st = [stat(s, M, sy, Inches(3.6), "3-20×", "smoother than DRTC alone, every policy, both tasks (E3b)",
               size=30),
          stat(s, M + Inches(3.97), sy, Inches(3.6), "50 → 27 %", "stalls of Diffusion Policy on sorting; the "
               "slow policies all stall less", size=30),
          stat(s, M + Inches(7.94), sy, Inches(3.6), "0.7-2.3°", "the cost: how far the arm drifts from the "
               "policy's plan", size=30, color=S.alert)]
    foot = text(s, M, TOP + Inches(3.6), CW, Inches(0.6),
                [[("The rule was kept as written; ", {"bold": True, "color": S.ink}),
                  ("dropping RTC is reported as its correction. With a round trip dominated by the network, a slower "
                   "request costs more than a smoother seam saves.", {"color": S.muted})]], size=13, font=S.body)
    builds(s, groups + [st[0] + st[1] + st[2], [foot]])
    notes(s, "How the final stack was decided. The rule was written before the bench ran and applied by a script. It "
             "chose RTC as the seam method. E3b then removed one part at a time: removing RTC lowered the stalls on "
             "every RTC policy-task pair at the same jerk. So the robot experiment uses temporal ensembling for the "
             "seams instead; we report this openly as a correction to the rule.")


SIZE_SPEED = [("ACT", 0.05, 76), ("SmolVLA", 0.45, 65), ("Diffusion Policy", 0.27, 193), ("GR00T N1.5", 2.7, 246),
              ("GR00T N1.7", 3.1, 334), ("MolmoAct 2", 5.6, 353), ("π0.5", 4.1, 434)]


def _split(segs, t):
    return [x for x in segs if x[0] < t], [x for x in segs if x[0] >= t]


def t_fair(prs, n):
    s = S.content(prs, n, "5 · How we evaluate", "A fair test: what we changed after the pilot")
    rows = [("The baseline gave up after 1 s", "beating it would only show LeRobot's default is fragile",
             "Compare with a patient LeRobot client, same time limits as DRTC"),
            ("DRTC changed two things at once", "its scheduling and its patience",
             "Same limits for both: only the scheduling differs"),
            ("Two smoothing methods added at once", "no way to tell which one helped",
             "Each method alone first; then a stack by a rule fixed in advance")]
    hx, lw_ = M, Inches(5.2)
    rx = M + lw_ + Inches(0.9)
    hdr = [text(s, hx, TOP - Inches(0.3), lw_, Inches(0.3), "THE PILOT (20 NOV)", size=11, bold=True,
                color=S.alert, font=S.body, spacing=1.2),
           text(s, rx, TOP - Inches(0.3), W - M - rx, Inches(0.3), "SETUP V2 (22 NOV): WHAT ALL RESULTS USE", size=11,
                bold=True, color=S.series["drtc"], font=S.body, spacing=1.2)]
    rh = Inches(1.05)
    groups = []
    for i, (a_, b_, c_) in enumerate(rows):
        y = TOP + Inches(0.1) + rh * i
        groups.append([
            box(s, hx, y, lw_, rh - Inches(0.2), fill=S.panel, line=S.panel_line, radius=S.card_radius,
                runs=[[(a_, {"bold": True, "color": S.ink})], [(b_, {"color": S.muted, "size": 12})]], size=15,
                font=S.body, align=PP_ALIGN.LEFT, margin=Inches(0.2)),
            line(s, hx + lw_ + Inches(0.1), y + (rh - Inches(0.2)) / 2, rx - Inches(0.1), y + (rh - Inches(0.2)) / 2,
                 color=S.ink, weight=Pt(1.5), arrow=True),
            box(s, rx, y, W - M - rx, rh - Inches(0.2), fill="#FFFFFF", line=S.series["drtc"], radius=S.card_radius,
                runs=c_, size=15, bold=True, font=S.body, color=S.ink, align=PP_ALIGN.LEFT, margin=Inches(0.2))])
    foot = text(s, M, TOP + Inches(3.35), CW, Inches(0.5),
                "The pilot's results are not used anywhere in this work.", size=13, color=S.muted, font=S.body)
    builds(s, [hdr] + groups + [[foot]])
    notes(s, "Our first robot design had three flaws, and we redesigned before collecting the results. First, its "
             "baseline was LeRobot as shipped, which gives up after one second: beating that proves little. Second, "
             "the DRTC condition changed both the scheduling and the waiting time. Third, two smoothing methods were "
             "added at once. Setup v2 fixes each, and every result in this talk comes from it.")


E4_POL = [("grootn17", "GR00T N1.7", "16 (0.5 s)"), ("diffusion", "Diffusion Policy", "24 (0.8 s)"),
          ("smolvla", "SmolVLA", "50 (1.7 s)"), ("pi05", "π0.5", "50 (1.7 s)")]


E4_COND = ["upstream_patient__none", "upstream_patient__mix", "drtc__mix", "drtc_stack_nortc__mix"]


def t_e4_policy(prs, n):
    s = S.content(prs, n, "Q2  ·  robot, per policy (E4)", "DRTC helps most where the chunk is short")
    rows = []
    for key, name, chunk in E4_POL:
        vals = [n_of(f"e4_cubebox_v1/{key}/{c}/M1") for c in E4_COND]
        rows.append([name, chunk] + [f"{v}/10" for v in vals])
    g = table(s, M, TOP - Inches(0.15), [Inches(2.3), Inches(1.5), Inches(1.85), Inches(1.85), Inches(1.85),
                                          Inches(2.2)],
              ["POLICY", "CHUNK", "PATIENT, NO FAULTS", "PATIENT + FAULTS", "DRTC + FAULTS", "DRTC + STACK + FAULTS"],
              rows, size=17, row_h=Inches(0.58))
    take = box(s, M, TOP + Inches(3.2), CW, Inches(1.2), fill=S.dark, radius=0.08,
               runs=[[("Short chunks lose almost everything to the faults, and DRTC wins it back: ",
                       {"color": S.dark_text}),
                      ("GR00T N1.7 0 → 8, Diffusion Policy 1 → 8.", {"bold": True, "color": S.dark_accent})],
                     [("π0.5's long chunk rides out short faults on its own queue: it lost least and gained least.",
                       {"color": S.dark_muted, "size": 14})]], size=16, font=S.body, line_spacing=1.25)
    builds(s, [sum(g, []), [take]])
    notes(s, "The same 160 trials, per policy. GR00T N1.7 and Diffusion Policy have short chunks, half a second to "
             "0.8 s: the faults took them to 0 and 1 success out of 10, and DRTC brought both back to 8. Pi 0.5 has a "
             "1.7 s chunk, enough to ride out short faults, so it lost least and gained least. DRTC matters most where "
             "a policy cannot ride out a fault on its own queue.")


QUESTIONS = [("Q1", "Do network faults hurt?", "the robot: 160 trials, cube in box"),
             ("Q2", "Does DRTC win the success back?", "the robot, then the bench: 7 policies, every fault"),
             ("Q3", "Which part of DRTC matters?", "the bench: DRTC taken apart"),
             ("Q4", "Can the motion be smooth without losing success?", "the bench, then the robot")]


def _qrows(s, y0, rows, *, rh=Inches(0.92), answer=False):
    groups = []
    for i, (q, a_, b_) in enumerate(rows):
        y = y0 + rh * i
        g = [box(s, M, y + Inches(0.12), Inches(0.75), rh - Inches(0.3), fill=S.accent, radius=0.2, runs=q, size=18,
                 bold=True, color="#FFFFFF", font=S.heading),
             text(s, M + Inches(1.0), y, CW - Inches(1.0), rh - Inches(0.05),
                  [[(a_, {"bold": not answer, "color": S.muted if answer else S.ink,
                          "size": 13 if answer else 20})],
                   [(b_, {"bold": answer, "color": S.ink if answer else S.muted, "size": 20 if answer else 13})]],
                  size=16, font=S.body, anchor=MSO_ANCHOR.MIDDLE, line=1.15),
             line(s, M, y + rh, W - M, y + rh, color=S.grid, weight=Pt(0.5))]
        groups.append(g)
    return groups


def t_questions(prs, n):
    s = S.content(prs, n, "6 · Results", "Four questions")
    groups = _qrows(s, TOP - Inches(0.2), [(q, a_, f"answered on {b_}") for q, a_, b_ in QUESTIONS])
    foot = text(s, M, TOP + Inches(3.6), CW, Inches(0.5),
                [[("First, for context: ", {"bold": True, "color": S.ink}),
                  ("how well each policy does the task with no faults at all.", {"color": S.muted})]], size=15,
                font=S.body)
    builds(s, groups + [[foot]])
    notes(s, "The results answer the four questions of the problem statement. The robot answers the first two with "
             "task success; the bench, with every policy and every fault, explains why. First, the baselines.")


def t_baselines(prs, n):
    s = S.content(prs, n, "Context  ·  robot, no faults", "Before any faults, the best policies succeed 9 times in 10")
    fig(s, "fig_baselines", M, TOP - Inches(0.2), w=Inches(7.2), centre=False)
    x = M + Inches(7.55)
    rows = bullets(s, x, TOP - Inches(0.15), W - M - x, [
        ("No single winner", "GR00T N1.7 and SmolVLA 90 % on sorting; Diffusion Policy 90 % on cube in box"),
        ("Size does not decide", "SmolVLA (0.45 B) is among the best; GR00T N1.7 far ahead of N1.5"),
        ("Every failure was the policy's own", "no trial was ended by the client"),
        ("MolmoAct 2: 0 / 20", "its fine-tune is treated as broken"),
    ], size=14, gap=Inches(0.98))
    cap = text(s, M, TOP + Inches(3.95), Inches(7.2), Inches(0.5),
               "Robot, normal network, patient LeRobot, 10 trials per policy and task, from the Raspberry Pi.",
               size=11, color=S.muted, font=S.body)
    builds(s, [[cap]] + rows)
    notes(s, "Before any faults: each policy's success on the real robot, with LeRobot's client set to be patient and "
             "a normal network, ten trials each, from the Pi. These are the reference the faults are measured "
             "against. With ten trials the intervals are wide, about plus or minus 25 points.")


def t_project(prs, n):
    s = S.content(prs, n, "2 · The problem", "What exists, and what this project does")
    lw = Inches(5.3)
    gap_h = text(s, M, TOP - Inches(0.3), lw, Inches(0.3), "THE GAP", size=11, bold=True, color=S.alert,
                 font=S.body, spacing=1.2)
    gaps = bullets(s, M, TOP + Inches(0.05), lw, [
        ("LeRobot's client", "stops the task on a short fault, and never replaces a lost reply"),
        ("DRTC (Vial, 2026)", "designed for unreliable networks, but tested only by its author, on their own setup"),
        ("No study", "on a physical robot against a fair, patient baseline, with task success, motion, and the "
                     "method taken apart"),
    ], size=15, gap=Inches(1.0))
    x = M + lw + Inches(0.5)
    did_h = text(s, x, TOP - Inches(0.3), W - M - x, Inches(0.3), "WHAT WE DID", size=11, bold=True,
                 color=S.accent, font=S.body, spacing=1.2)
    did = bullets(s, x, TOP + Inches(0.05), W - M - x, [
        ("Ported DRTC", "into LeRobot 0.7's own runtime, and fixed it on the real robot"),
        ("Built six smoothing methods", "and chose a stack by a rule fixed in advance"),
        ("Injected seven network faults", "seeded, the same for every client"),
        ("Tested it", "on a replay bench (7 policies, 2 tasks) and on the real arm from a Raspberry Pi"),
    ], size=15, gap=Inches(0.8))
    builds(s, [[gap_h] + sum(gaps, []), [did_h] + sum(did, [])])
    notes(s, "Where this project sits. LeRobot's default client fails on short faults. DRTC was designed for this, but "
             "nobody had tested it apart from its author. Our project ports it into LeRobot, fixes what broke on our "
             "real setup, adds smoothing, and tests it properly: on a bench and on a real low-cost arm, against a fair "
             "baseline.")


E1_POLICIES = ["act", "diffusion", "groot", "grootn17", "molmoact2", "pi05", "smolvla"]


def _frac(key):
    v = R(key)
    if "/" not in v:
        return None
    a, b = v.split("/")
    return int(a), int(b)


def t_e1(prs, n):
    s = S.content(prs, n, "Q2  ·  bench, 7 policies, both tasks (E1)", "On the bench, DRTC gets the arm moving again after 99 % of faults")
    sub = text(s, M, TOP - Inches(0.4), CW, Inches(0.4),
               [[("After each fault, ", {"color": S.muted}), ("did the arm start moving again?", {"bold": True,
                                                                                               "color": S.ink}),
                 ("  Bench, 7 policies, both tasks, 8 runs each.", {"color": S.muted})]], size=15, font=S.body)
    faults = [("spike", "one reply 1 s late"), ("drops", "replies lost for 1 s"),
              ("outage", "link dead for 1.5 s"), ("rise", "replies 0.4 s slower"),
              ("mix", "spike + drops + outage")]
    cols = [("ended", "LeRobot as shipped", "ended the run", S.alert),
            ("patientrec", "Patient LeRobot", "recovered", S.series["baseline"]),
            ("drtcrec", "DRTC", "recovered", S.series["drtc"])]
    nx, nw_ = M, Inches(2.7)
    cw_ = (CW - nw_) / 3
    y0 = TOP + Inches(0.5)
    hdr = [text(s, nx + nw_ + cw_ * j, y0 - Inches(0.42), cw_ - Inches(0.2), Inches(0.4),
                [[(a_, {"bold": True, "color": col})], [(b_, {"color": S.muted, "size": 11})]], size=13,
                font=S.body, line=1.0) for j, (_, a_, b_, col) in enumerate(cols)]
    rh = Inches(0.52)
    totals = {k: [0, 0] for k, *_ in cols}
    groups = []
    for i, (f, what) in enumerate(faults):
        y = y0 + rh * i
        g = [text(s, nx, y, nw_, rh, [[(f, {"bold": True, "color": S.ink})], [(what, {"color": S.muted,
                                                                                    "size": 11})]],
                  size=15, font=S.body, anchor=MSO_ANCHOR.MIDDLE, line=1.0)]
        for j, (key, _, _, col) in enumerate(cols):
            k = m = 0
            for t in ("sort_one", "cubebox_v1"):
                for p in E1_POLICIES:
                    fr = _frac(f"v2/{t}/e1/{f}/{p}/{key}")
                    if fr:
                        k, m = k + fr[0], m + fr[1]
            totals[key][0] += k
            totals[key][1] += m
            pct = 100 * k / m if m else 0
            bx = nx + nw_ + cw_ * j
            bw = int((cw_ - Inches(1.0)) * pct / 100)
            g.append(box(s, bx, y + Inches(0.14), cw_ - Inches(1.0), rh - Inches(0.28), fill=S.grid))
            g.append(box(s, bx, y + Inches(0.14), max(bw, Pt(2)), rh - Inches(0.28), fill=col))
            g.append(text(s, bx + cw_ - Inches(0.92), y, Inches(0.8), rh, f"{pct:.0f} %", size=14, bold=True,
                          color=S.ink, font=S.body, anchor=MSO_ANCHOR.MIDDLE))
        g.append(line(s, M, y + rh, W - M, y + rh, color=S.grid, weight=Pt(0.5)))
        groups.append(g)
    pct = {k: 100 * a / b for k, (a, b) in totals.items() if b}
    take = box(s, M, y0 + rh * 5 + Inches(0.25), CW, Inches(0.62), fill=S.dark, radius=0.15,
               runs=[[("DRTC got the arm moving again after ", {"color": S.dark_text}),
                      (f"{pct['drtcrec']:.0f} %", {"bold": True, "color": S.dark_accent}),
                      (" of the faults; the patient client after ", {"color": S.dark_text}),
                      (f"{pct['patientrec']:.0f} %", {"bold": True, "color": S.dark_accent}),
                      (". As shipped, LeRobot gave up in ", {"color": S.dark_text}),
                      (f"{pct['ended']:.0f} %", {"bold": True, "color": S.dark_accent}),
                      (" of the runs.", {"color": S.dark_text})]], size=15, font=S.body)
    worse = text(s, M, y0 + rh * 5 + Inches(0.92), CW, Inches(0.5),
                 [[("Where DRTC was worse: ", {"bold": True, "color": S.alert}),
                   ("a rising latency. Every reply 0.4 s slower made DRTC stand still more than the patient client "
                    "(SmolVLA 31 % against 6 %): its cooldown gives up replies that are only slow.",
                    {"color": S.muted})]], size=13, font=S.body)
    builds(s, [[sub] + hdr] + groups + [[take], [worse]])
    notes(s, "E1 on the bench. For every fault type we ask one plain question: after the fault, did the arm move "
             "again? LeRobot as shipped simply ends the run in most faults. The patient version waits, but after "
             "lost replies and outages it never asks again, so it often never recovers. DRTC asks again and "
             "recovers almost every time, for every policy on both tasks.")


def t_e3(prs, n):
    s = S.content(prs, n, "6 · Results", "E3b (bench): the combined stack against DRTC alone")
    fig(s, "fig_v2_stack", M, TOP - Inches(0.15), h=Inches(4.3))
    cap = text(s, M, S.bottom - Inches(0.3), CW, Inches(0.3),
               "Grey: DRTC alone; orange: DRTC + the stack; each line one policy; numbers on the right: how many "
               "times smoother the stack made it.", size=11, color=S.muted, font=S.body, align=PP_ALIGN.CENTER)
    notes(s, "The methods combined into one stack, chosen by a rule written down before the bench ran (next slide). "
             "Against DRTC alone, under the fault mix, it cut the stalls of the slow policies and made every policy "
             "3 to 20 times smoother.")


def t_e4(prs, n):
    s = S.content(prs, n, "Q1 + Q2  ·  robot, 160 trials (E4)", "Faults cut success from 68 % to 13 %; DRTC brings it back to 65 %")
    models = ["diffusion", "grootn17", "pi05", "smolvla"]
    conds = [("upstream_patient__none", "Patient LeRobot\nno faults", S.series["baseline"]),
             ("upstream_patient__mix", "Patient LeRobot\n+ faults", S.series["faults"]),
             ("drtc__mix", "DRTC\n+ faults", S.series["drtc"]),
             ("drtc_stack_nortc__mix", "DRTC + stack\n+ faults", S.series["smooth"])]
    bars = []
    for key, label, colour in conds:
        k = sum(n_of(f"e4_cubebox_v1/{m}/{key}/M1") or 0 for m in models)
        lo, hi = wilson(k, 40)
        pct = int(100 * k / 40 + 0.5)  # half up, as the report (5/40 -> 13 %)
        bars.append((f"{label}\n{k}/40", pct, 100 * lo, 100 * hi, colour))
    static, per = bar_panel(s, M, TOP - Inches(0.3), Inches(7.6), Inches(4.9), "Cube in box, four policies, 160 trials",
                            bars, ink=S.ink, muted=S.muted, grid=S.grid, font=S.body, size=14)
    x = M + Inches(8.1)
    sa = stat(s, x, TOP, Inches(3.4), "68 → 13 %", "the fault mix breaks LeRobot's client, even when patient", color=S.alert)
    sb = stat(s, x, TOP + Inches(1.3), Inches(3.4), "→ 65 %", "DRTC restores it")
    sc = stat(s, x, TOP + Inches(2.45), Inches(3.4), "→ 75 %", "with the stack, and about 3× smoother",
              color=S.series["smooth"])
    cav = text(s, x, TOP + Inches(3.75), Inches(3.4), Inches(0.75),
               "10 trials per policy per bar: only GR00T N1.7's gain (0 \u2192 80 %) is significant on its own after "
               "the Holm correction; the pooled change is exploratory.", size=10, color=S.muted, font=S.body,
               line=1.1)
    builds(s, [per[0], per[1] + sa, per[2] + sb, per[3] + sc, [cav]])
    notes(s, "160 robot trials. With the Holm correction only GR00T N1.7's DRTC-vs-faults difference is significant "
             "on its own; pooled, the effect is large but exploratory. Ten trials per cell give wide intervals.")


def t_e4_video(prs, n):
    s = S.content(prs, n, "Q1 + Q2  ·  one real trial", "The same faults: LeRobot stops, DRTC finishes")
    slot(s, "e4_trials", M, TOP - Inches(0.1), CW, Inches(4.2))
    text(s, M, TOP + Inches(4.2), CW, Inches(0.35),
         "Diffusion Policy, trial 10 of E4: the same faults at the same moments for all three. Patient LeRobot is "
         "ended at 14.3 s; DRTC finishes at 13.0 s, and with the stack at 12.3 s.", size=12, color=S.muted,
         font=S.body, align=PP_ALIGN.CENTER)
    notes(s, "Another real trial, a different policy from the earlier slides: Diffusion Policy, trial 10, so the same "
             "faults at the same moments under each client. With LeRobot's patient client the arm stops after a lost "
             "reply and the run is ended. DRTC asks again and finishes the task; with the smoothing stack it finishes "
             "too, a little sooner here and with smoother motion.")


def t_e4_motion(prs, n):
    s = S.content(prs, n, "Q4  ·  robot (E4)", "On the robot the stack keeps success and cuts jerk about 3\u00d7")
    cols = [S.series["baseline"], S.series["faults"], S.series["drtc"], S.series["smooth"]]
    labs = ["Patient", "+ faults", "DRTC", "+ stack"]
    panels = [("Standing still", "% of the run", [39, 80, 58, 42], 90, "{:.0f}"),
              ("Jerk while moving", "10³ °/s³", [60, 48, 52, 16], 70, "{:.0f}"),
              ("Back in motion after the outage", "seconds", [None, 1.15, 0.90], 1.4, "{:.2f}")]
    pw = (CW - Inches(0.6)) / 3
    groups = []
    for k, (title, unit, vals, vmax, fmt) in enumerate(panels):
        x = M + (pw + Inches(0.3)) * k
        top, bot = TOP + Inches(0.35), TOP + Inches(2.95)
        g = [text(s, x, TOP - Inches(0.3), pw, Inches(0.35), title, size=15, bold=True, color=S.ink, font=S.body),
             text(s, x, TOP + Inches(0.02), pw, Inches(0.3), unit, size=11, color=S.muted, font=S.body),
             line(s, x, bot, x + pw, bot, color=S.muted, weight=Pt(0.75))]
        vs = vals if len(vals) == 4 else [None] + vals[1:] if len(vals) == 3 else vals
        if len(vals) == 3:
            vs = [None, vals[1], vals[2], None]
            vs = [None, None, vals[1], vals[2]]
        slot_w = pw / 4
        for i, v in enumerate(vs):
            cx = x + slot_w * i
            if v is None:
                g.append(text(s, cx, bot - Inches(0.35), slot_w, Inches(0.3), "–" if i == 0 else "few",
                              size=11, color=S.muted, font=S.body, align=PP_ALIGN.CENTER))
            else:
                hgt = int((bot - top) * v / vmax)
                g.append(box(s, int(cx + slot_w * 0.15), bot - hgt, int(slot_w * 0.7), hgt, fill=cols[i]))
                g.append(text(s, cx, bot - hgt - Inches(0.3), slot_w, Inches(0.28), fmt.format(v), size=12,
                              bold=True, color=S.ink, font=S.body, align=PP_ALIGN.CENTER))
            g.append(text(s, cx, bot + Inches(0.05), slot_w, Inches(0.3), labs[i], size=10, color=S.muted,
                          font=S.body, align=PP_ALIGN.CENTER))
        groups.append(g)
    take = text(s, M, TOP + Inches(3.55), CW, Inches(0.8),
                [[("The stack: ", {"bold": True, "color": S.ink}),
                  ("about 3× less jerk than DRTC alone (52 → 16), less standing still (58 → 42 %), and "
                   "faster back in motion after the outage (1.15 → 0.90 s); the cost is 1-2° of drift from "
                   "the policy's own actions.", {"color": S.muted})]], size=14, font=S.body, line=1.2)
    foot = text(s, M, TOP + Inches(4.3), CW, Inches(0.3), "Medians over all four policies, cube in box, 40 trials per "
                "condition. After the outage the patient client got moving again in only a few trials.", size=10,
                color=S.muted, font=S.body)
    builds(s, groups + [[take, foot]])
    notes(s, "How the arm moved in the same 160 trials. Under faults the patient client stood still 80 percent of the "
             "run; DRTC cut that, and the stack cut it further while making the motion about three times smoother. "
             "Jerk is counted only while the arm moves, so a stalled arm does not look smooth.")


def t_pi(prs, n):
    s = S.content(prs, n, "The edge  ·  robot computer", "A Raspberry Pi is enough; the network link is the limit")
    fig(s, "fig_edgertt", M, TOP - Inches(0.15), h=Inches(4.3), centre=False, maxw=Inches(7.3))
    x = M + Inches(7.8)
    a = stat(s, x, TOP - Inches(0.1), Inches(3.7), "29.9 Hz", "the Pi's control loop (median; never below 29.1)")
    b = stat(s, x, TOP + Inches(1.45), Inches(3.7), "0.87 → 0.64 s", "median reply time, WiFi → cable")
    c = stat(s, x, TOP + Inches(2.8), Inches(3.7), "Over WiFi", "DRTC's cooldown gave up every reply of the slower "
             "policies, so E4 ran over a cable", color=S.alert, size=28)
    builds(s, [a, b, c])
    notes(s, "The Pi kept the control loop at 30 Hz in every trial: its CPU is not the limit. The link is: each "
             "request carries 2.9 MB of uncompressed images, and over the Pi's WiFi a reply took 0.87 s at the "
             "median, three to four times what the bench assumed. That was slow enough for DRTC's cooldown to give "
             "up the slower policies' replies as lost, so the robot experiment ran over an Ethernet cable. A cooldown "
             "that adapts to a slowing link is future work.")


def t_limits(prs, n):
    s = S.content(prs, n, "Close", "Limitations and future work")
    lw = Inches(6.2)
    lim_h = text(s, M, TOP - Inches(0.3), lw, Inches(0.3), "LIMITATIONS", size=11, bold=True, color=S.alert,
                 font=S.body, spacing=1.2)
    lims = bullets(s, M, TOP + Inches(0.05), lw, [
        ("Ten trials per condition", "wide intervals; only large differences are significant"),
        ("One task on the robot", "the bench covers two tasks and seven policies"),
        ("Simulated faults", "fixed shapes, not a recorded real network"),
        ("Slow links", "DRTC can give up replies that are only slow (seen over the Pi's WiFi)"),
    ], size=15, gap=Inches(0.85))
    x = M + lw + Inches(0.5)
    fut_h = text(s, x, TOP - Inches(0.3), W - M - x, Inches(0.3), "FUTURE WORK", size=11, bold=True,
                 color=S.accent, font=S.body, spacing=1.2)
    fut = bullets(s, x, TOP + Inches(0.05), W - M - x, [
        ("A cooldown for slow links", "one that follows a slowing network"),
        ("Compressed images", "a shorter round trip over WiFi"),
        ("More trials, more tasks", "20 trials per cell; sorting on the robot"),
    ], size=15, gap=Inches(0.85))
    builds(s, [[lim_h] + sum(lims, []), [fut_h] + sum(fut, [])])
    notes(s, "Be upfront about the limits, and each points at the next step.")


def t_conclusion(prs, n):
    s = S.content(prs, n, "Close", "The four answers")
    answers = [("Q1", "Do network faults hurt?", "Yes: a few seconds of faults cut success from 68 % to 13 %"),
               ("Q2", "Does DRTC win the success back?", "Yes: 65 % on the robot; the arm recovers after 99 % of faults "
                "on the bench"),
               ("Q3", "Which part of DRTC matters?", "Asking again after a lost reply (the cooldown); the rest little"),
               ("Q4", "Smooth without losing success?", "Yes: 75 % and about 3× less jerk; RTC did not pay off")]
    groups = _qrows(s, TOP - Inches(0.2), answers, answer=True)
    foot = text(s, M, TOP + Inches(3.6), CW, Inches(0.5),
                [[("All of it from a Raspberry Pi, ", {"bold": True, "color": S.ink}),
                  ("on a platform that records, trains, serves and tests.", {"color": S.muted})]], size=15,
                font=S.body)
    builds(s, groups + [[foot]])
    notes(s, "The four questions, answered.")


def t_thanks(prs, n):
    S.closing_slide(prs, n, "Thank you", "Questions?")


SLIDES = [t_title, t_hook, t_outline,
          lambda p, n: S.section_slide(p, n, 1, "Robot control with a VLA", "Cameras and joints in, motion out"),
          t_vla_what, t_why_server, t_chunks,
          lambda p, n: S.section_slide(p, n, 2, "The problem", "Control over an unreliable network"),
          t_loop, t_faults, t_fault, t_question, t_project,
          lambda p, n: S.section_slide(p, n, 3, "Our solution", "DRTC and motion smoothing"),
          t_drtc_idea, t_drtc_media, t_jerk, t_stack,
          lambda p, n: S.section_slide(p, n, 4, "What we built", "Hardware, our lab platform, data, training"),
          t_hardware, t_machines, t_lab, t_record, t_first_attempt, t_data, t_trimming, t_models,
          lambda p, n: S.section_slide(p, n, 5, "How we evaluate", "Offline, the bench, live robot trials"),
          t_eval_kinds, t_offline, t_bench, t_robot_eval, t_records,
          lambda p, n: S.section_slide(p, n, 6, "Results", "Four questions, answered"),
          t_questions, t_baselines, t_e4, t_e4_video, t_e4_policy, t_e1, t_e2, t_e3a, t_stack_choice, t_e4_motion,
          t_pi,
          t_conclusion, t_contrib, t_limits, t_thanks]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    prs = new_deck()
    for n, make in enumerate(SLIDES, start=1):
        make(prs, n)
    path = OUT / "talk.pptx"
    prs.save(path)
    print(path, f"({len(SLIDES)} slides)")
    if "--no-preview" not in sys.argv:  # LibreOffice: static, no animations, substituted fonts
        pv = OUT / "talk_preview"
        pv.mkdir(exist_ok=True)
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(pv), str(path)], check=True,
                       capture_output=True)
        subprocess.run(["pdftoppm", "-png", "-r", "50", str(pv / "talk.pdf"), str(pv / "slide")], check=True)
        print(pv / "talk.pdf")


if __name__ == "__main__":
    main()
