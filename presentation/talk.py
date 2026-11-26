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
from talk_draw import (JOINT_COLOURS, JOINTS, chunk_plot, curve, dot, fault_row, first_chunk_curves, first_state,
                       mapper, marker, mini_panel, plot_curve, problem_curves)

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


def slot(s, key, x, y, w, h):
    """A media slot: the file in presentation/media when present, else a dashed box describing it."""
    for ext in (".mp4", ".mov"):
        if (MEDIA / f"{key}{ext}").is_file():
            return movie(s, MEDIA / f"{key}{ext}", x, y, w, h)
    for ext in (".png", ".jpg"):
        if (MEDIA / f"{key}{ext}").is_file():
            return picture(s, MEDIA / f"{key}{ext}", x, y, w=w, h=h)
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
             ("6", "Results", "baselines, the bench studies, the robot experiment")]
    rows = [para(s, M, TOP - Inches(0.1) + Inches(0.75) * i, CW, f"{a}   {b}", c, size=21, h=Inches(0.7))
            for i, (a, b, c) in enumerate(parts)]
    builds(s, rows)
    notes(s, "The order of the talk.")


# Part 1 -----------------------------------------------------------------------------------------------------------

def t_vla_what(prs, n):
    s = S.content(prs, n, "1 · Robot control with a VLA", "What goes in, what comes out")
    ex = json.loads((MEDIA / "data" / "vla_example.json").read_text())
    st = MEDIA / "stills"
    y0, h = TOP - Inches(0.05), Inches(4.3)
    iw, aw, mw = Inches(3.85), Inches(0.55), Inches(2.75)
    ix, mx = M, M + iw + aw
    ox = mx + mw + aw
    ow = W - M - ox
    pad = Inches(0.18)
    eb = dict(size=11, bold=True, color=S.muted, font=S.body, spacing=1.2)
    heads = [text(s, ix, y0 - Inches(0.32), iw, Inches(0.3), "INPUTS  ·  ONE MOMENT", **eb),
             text(s, mx, y0 - Inches(0.32), mw, Inches(0.3), "THE POLICY", **eb),
             text(s, ox, y0 - Inches(0.32), ow, Inches(0.3), "OUTPUT  ·  AN ACTION CHUNK", **eb)]
    # inputs
    inp = [heads[0], box(s, ix, y0, iw, h, fill=S.panel, line=S.panel_line, radius=0.04)]
    pw = (iw - 2 * pad - Inches(0.12)) / 2
    for k, (f, cap) in enumerate((("vla_ep0_wrist.jpg", "wrist camera"), ("vla_ep0_scene.jpg", "scene camera"))):
        x = int(ix + pad + (pw + Inches(0.12)) * k)
        inp.append(picture(s, st / f, x, int(y0 + pad), w=int(pw)))
        inp.append(text(s, x, int(y0 + pad + pw * 0.75 + Inches(0.03)), int(pw), Inches(0.25), cap, size=11,
                        color=S.muted, font=S.body))
    yi = y0 + Inches(1.78)
    inp.append(text(s, ix + pad, yi, iw - 2 * pad, Inches(0.25), "INSTRUCTION", size=10, bold=True, color=S.muted,
                    font=S.body, spacing=1.2))
    inp.append(text(s, ix + pad, yi + Inches(0.22), iw - 2 * pad, Inches(0.5),
                    "“Pick up the green cube and put it in the box”", size=14, italic=True, color=S.ink,
                    font=S.body, line=1.1))
    yj = y0 + Inches(2.62)
    inp.append(text(s, ix + pad, yj, iw - 2 * pad, Inches(0.25), "JOINT POSITIONS NOW", size=10, bold=True,
                    color=S.muted, font=S.body, spacing=1.2))
    labels = ["shoulder pan", "shoulder lift", "elbow", "wrist flex", "wrist roll", "gripper"]
    cw_ = (iw - 2 * pad) / 3
    for k, (lab, v) in enumerate(zip(labels, ex["state"])):
        x = int(ix + pad + cw_ * (k % 3))
        y = int(yj + Inches(0.28) + Inches(0.62) * (k // 3))
        inp.append(text(s, x, y, int(cw_), Inches(0.6), [[(lab, {"size": 10, "color": S.muted})],
                                                          [(f"{v:+.1f}", {"size": 17, "bold": True,
                                                                          "color": S.ink})]],
                        size=12, font=S.body, line=1.05))
    # the policy
    ymid = y0 + h / 2
    a1 = line(s, ix + iw + Inches(0.08), ymid, mx - Inches(0.08), ymid, color=S.ink, weight=Pt(1.75), arrow=True)
    pol = [heads[1], box(s, mx, y0, mw, h, fill=S.dark, radius=0.04),
           text(s, mx + pad, y0 + Inches(0.2), mw - 2 * pad, Inches(0.4), "VLA policy", size=20, bold=True,
                font=S.heading, color=S.dark_text),
           text(s, mx + pad, y0 + Inches(0.6), mw - 2 * pad, Inches(0.3), "e.g. SmolVLA, 0.45 B parameters",
                size=11, color=S.dark_muted, font=S.body)]
    ib = Inches(1.25)
    for k, (t1, t2) in enumerate((("Vision-language model", "images + instruction → tokens"),
                                  ("Action expert", "tokens + joint positions → 50 future actions"))):
        yy = y0 + Inches(1.05) + (ib + Inches(0.35)) * k
        pol.append(box(s, mx + pad, yy, mw - 2 * pad, ib, fill="#232A34", radius=0.08,
                       runs=[[(t1, {"bold": True})], [(t2, {"color": S.dark_muted, "size": 12})]], size=14,
                       font=S.body, color=S.dark_text))
    pol.append(line(s, mx + mw / 2, y0 + Inches(1.05) + ib + Inches(0.04), mx + mw / 2,
                    y0 + Inches(1.05) + ib + Inches(0.31), color=S.dark_muted, weight=Pt(1.5), arrow=True))
    # the output: one row per joint
    a2 = line(s, mx + mw + Inches(0.08), ymid, ox - Inches(0.08), ymid, color=S.ink, weight=Pt(1.75), arrow=True)
    out = [heads[2], box(s, ox, y0, ow, h, fill=S.panel, line=S.panel_line, radius=0.04)]
    chunk = np.array(ex["chunk"])
    lw_ = Inches(0.95)
    gx, gw = ox + pad + lw_, ow - 2 * pad - lw_
    rh = (h - 2 * pad - Inches(0.45)) / 6
    curves = []
    for k, (lab, col) in enumerate(zip(labels, JOINT_COLOURS)):
        ry = y0 + pad + rh * k
        out.append(text(s, ox + pad, ry, lw_, rh, lab, size=11, color=S.ink, font=S.body, anchor=MSO_ANCHOR.MIDDLE))
        out.append(line(s, gx, ry + rh - Inches(0.04), gx + gw, ry + rh - Inches(0.04), color=S.grid,
                        weight=Pt(0.5)))
        v = chunk[:, k]
        lo, hi = float(v.min()), float(v.max())
        if hi - lo < 3:
            lo, hi = lo - 3, hi + 3
        f = mapper(gx, ry + Inches(0.08), gw, rh - Inches(0.18), 0, len(v) - 1, lo, hi)
        curves.append(curve(s, [f(i, a) for i, a in enumerate(v)], col, weight=Pt(2.25)))
    ay = y0 + h - pad - Inches(0.38)
    out += [text(s, gx, ay, Inches(0.8), Inches(0.25), "now", size=10, color=S.muted, font=S.body),
            text(s, gx, ay, gw, Inches(0.25), "50 actions", size=10, color=S.muted, font=S.body,
                 align=PP_ALIGN.CENTER),
            text(s, gx + gw - Inches(0.8), ay, Inches(0.8), Inches(0.25), "+1.7 s", size=10, color=S.muted,
                 font=S.body, align=PP_ALIGN.RIGHT)]
    cap = text(s, M, y0 + h + Inches(0.1), CW, Inches(0.3),
               "One moment of a recorded demonstration (cube in box): the two camera frames, the arm's joint "
               "positions, and the next 50 actions the person made: what the policy learns to predict.", size=11,
               color=S.muted, font=S.body)
    builds(s, [inp, [a1] + pol, [a2] + out + [(c, "wipe_right") for c in curves], [cap]])
    notes(s, "Real data from one moment of a demonstration. In: the wrist and scene camera frames, the instruction, "
             "and the six joint positions (LeRobot's normalised units). The vision-language model turns images and "
             "words into tokens; the action expert turns those and the joint positions into a chunk: the next 50 "
             "actions for every joint, 1.7 s of motion. Shown here: what the person actually did next, which is "
             "exactly what the policy is trained to predict.")


def t_vla_video(prs, n):
    s = S.content(prs, n, "1 · Robot control with a VLA", "From camera to motion")
    slot(s, "vla", M, TOP - Inches(0.1), CW, Inches(4.7))
    notes(s, "Walk through the video: capture, model, chunk, playback, ask again. Thirty times a second.")


def t_chunks(prs, n):
    s = S.content(prs, n, "1 · Robot control with a VLA", "Chunk after chunk: how the arm keeps moving")
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
    s = S.content(prs, n, "1 · Robot control with a VLA", "The model is too big for the robot's computer")
    a = stat(s, M, TOP, Inches(3.5), "0.45-5.6 B", "parameters in the VLAs we trained")
    b = stat(s, M, TOP + Inches(1.5), Inches(3.5), "Raspberry Pi 5", "the robot's computer: no GPU")
    c = stat(s, M, TOP + Inches(3.0), Inches(3.5), "A100 · H200", "the GPU servers that run the models")
    v = slot(s, "why_server", M + Inches(4.2), TOP - Inches(0.1), CW - Inches(4.2), Inches(4.6))
    builds(s, [a, b, c])
    notes(s, "So the model runs on a GPU server, and the robot sends it images over the network. This is remote "
             "inference, and it is where our problem starts.")


# Part 2 -----------------------------------------------------------------------------------------------------------

def t_loop(prs, n):
    s = S.content(prs, n, "2 · The problem", "Every move now depends on the network")
    slot(s, "loop", M, TOP - Inches(0.1), Inches(7.4), Inches(4.6))
    x = M + Inches(7.8)
    rows = bullets(s, x, TOP, Inches(3.8), [
        ("Up:", "2 images + 6 joints, 2.9 MB"),
        ("Down:", "a chunk of moves"),
        ("Round trip:", f"about {R('edge/smolvla/rtt')} (SmolVLA, from the Pi)"),
        ("Queue:", "the moves still waiting to be played"),
    ], size=17, gap=Inches(0.8))
    builds(s, rows)
    notes(s, "The robot computer sends the images and joint angles; the server answers with a chunk; the robot keeps "
             "a queue of moves. As long as the next chunk arrives before the queue empties, the arm moves smoothly.")


def t_fault(prs, n):
    s = S.content(prs, n, "2 · The problem", "When a reply is late, the arm stops")
    slot(s, "fault", M, TOP - Inches(0.1), Inches(7.4), Inches(4.6))
    x = M + Inches(7.8)
    rows = bullets(s, x, TOP, Inches(3.8), [
        ("Late reply", "the queue runs dry"),
        ("Arm stops", "nothing left to play"),
        ("LeRobot as shipped", "ends the run after 1 s without moves"),
        ("Lost reply", "never asked for again"),
    ], size=17, gap=Inches(0.8))
    builds(s, rows)
    notes(s, "LeRobot is the open-source robot-learning library we build on. Its remote client gives up after one "
             "second with an empty queue, and if a reply is lost it never asks again.")


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


def t_drtc_media(prs, n):
    s = S.content(prs, n, "3 · Our solution", "DRTC at work")
    slot(s, "drtc", M, TOP - Inches(0.1), CW, Inches(4.7))
    notes(s, "The same fault, two clients: LeRobot waits and stops; DRTC asks again and keeps going.")


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


def t_record(prs, n):
    s = S.content(prs, n, "4 · What we built", "Teaching by demonstration")
    slot(s, "record", M, TOP - Inches(0.1), Inches(7.4), Inches(4.6))
    x = M + Inches(7.8)
    rows = bullets(s, x, TOP, Inches(3.8), [
        ("Move the leader arm", "by hand; the follower copies it"),
        ("Record", "both cameras + the joints, 30 per second"),
        ("Review", "in Rerun; trim idle starts and ends"),
        ("Hold back", "some episodes for testing"),
    ], size=17, gap=Inches(0.8))
    builds(s, rows)
    notes(s, "Teleoperation: a person drives the leader arm and every frame is saved as training data.")


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
    builds(s, [sum(g, []), [*st, *st2]])
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
                "the Pi's gateway", S.series["baseline"])
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
                     [("SSH tunnel to the policy server", {"color": S.muted})],
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
             "Between them the campus network: every request carries two images and the joints up, and a chunk of "
             "actions comes back. Below: GitHub keeps the code and the registry; the Hugging Face Hub keeps datasets, "
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
    s = S.content(prs, n, "4 · What we built", "Before the robot: offline evaluation")
    a = fig(s, "fig_offline", M, TOP - Inches(0.2), w=Inches(5.7), centre=False)
    b = fig(s, "fig_sizespeed", M + Inches(5.95), TOP - Inches(0.2), w=Inches(5.6), centre=False)
    cap = text(s, M, TOP + Inches(4.15), CW, Inches(0.6),
               [[("Held-out episodes: ", {"bold": True, "color": S.ink}),
                 ("the model predicts every step and we compare with what the person did (error in degrees); "
                  "inference time measured on the H200.", {"color": S.muted})]], size=14, font=S.body)
    builds(s, [[a], [b], [cap]])
    notes(s, "Before any robot time: how close each model's actions are to the human's on held-out episodes, and how "
             "long it takes to answer. Bigger models are slower; that matters on a network.")


def t_bench(prs, n):
    s = S.content(prs, n, "5 · How we evaluate", "The bench: the robot replayed, everything else real")
    bw, gap = Inches(2.0), Inches(0.39)
    y = TOP + Inches(0.1)
    items = [("Recorded episode", "camera frames + joints from a real trial", S.panel_line),
             ("Replay robot", "plays them in place of the arm (our plugin)", S.accent),
             ("Client", "LeRobot or DRTC, with or without smoothing", S.accent),
             ("Fault injector", "this run's fault, seeded", S.alert),
             ("Policy server", "the real trained model on the GPU", S.series["drtc"])]
    nodes, arrows = [], []
    for i, (a_, b_, col) in enumerate(items):
        x = M + (bw + gap) * i
        nodes.append(node(s, x, y, bw, Inches(1.35), a_, b_, edge=col, size=14))
        if i:
            arrows.append(line(s, x - gap + Inches(0.04), y + Inches(0.5), x - Inches(0.04), y + Inches(0.5),
                               color=S.ink, weight=Pt(1.25), arrow=True))
    xr = M + (bw + gap) * 4 + bw / 2
    xc = M + (bw + gap) * 2 + bw / 2
    back = [line(s, xr, y + Inches(1.35), xr, y + Inches(1.75), color=S.ink, weight=Pt(1.25)),
            line(s, xr, y + Inches(1.75), xc, y + Inches(1.75), color=S.ink, weight=Pt(1.25)),
            line(s, xc, y + Inches(1.75), xc, y + Inches(1.38), color=S.ink, weight=Pt(1.25), arrow=True),
            edge_label(s, xc, y + Inches(1.8), xr - xc, "the chunk comes back through the same faults")]
    sy = TOP + Inches(2.45)
    st = [stat(s, M, sy, Inches(3.6), "2,912", "runs per task for E1 + E2: 7 policies × 52 conditions × 8 "
               "runs", size=30),
          stat(s, M + Inches(3.97), sy, Inches(3.6), "14 s", "per run; timing and motion measured, not task success",
               size=30),
          stat(s, M + Inches(7.94), sy, Inches(3.6), "Same seed", "every client meets exactly the same faults",
               size=30)]
    builds(s, [nodes[:2] + arrows[:1], nodes[2:3] + arrows[1:2], nodes[3:] + arrows[2:], back, st[0] + st[1] + st[2]])
    notes(s, "On the bench the arm is replaced by a recording of a real trial: its camera frames and joint states are "
             "played back by our replay robot. Everything else is real: the client, the fault injector, the network "
             "and the trained model on the GPU. So we can run every fault for every policy thousands of times, "
             "identically for each client. It cannot tell us task success; that is what the robot experiment is for.")


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
             ("DRTC inside LeRobot's runtime", "with four fixes found on the real robot"),
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
    slot(s, "robot_eval", M, TOP - Inches(0.1), Inches(7.2), Inches(4.6))
    x = M + Inches(7.6)
    steps = [("1  Set the layout", "taped marks, rotated every round"),
             ("2  Start the trial", "the Pi runs the client; the H200 runs the model"),
             ("3  Faults, live", "the injector applies this trial's seeded faults"),
             ("4  Judge it", "operator presses s (success) or f (fail); 30 s limit"),
             ("5  Save it", "videos, every tick, every chunk, the outcome")]
    rows = [para(s, x, TOP - Inches(0.15) + Inches(0.88) * i, W - M - x, a, b, size=15, h=Inches(0.85))
            for i, (a, b) in enumerate(steps)]
    builds(s, rows)
    notes(s, "This is how every robot trial ran: in real time, on the real arm, from the Raspberry Pi. Trials run in "
             "rounds: each round runs every model under every condition once, in a seeded random order, so slow "
             "drifts (light, motor heat, the operator) affect all conditions alike. A trial with a setup mistake is "
             "voided and re-run, never deleted.")


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
    s = S.content(prs, n, "6 · Results", "E2: which part of DRTC matters?")
    sub = text(s, M, TOP - Inches(0.35), CW, Inches(0.6),
               [[("We took DRTC apart, one piece at a time, and measured ", {"color": S.muted}),
                 ("how much longer the arm stood still", {"bold": True, "color": S.ink}),
                 (" under the fault mix (bench, 7 policies, both tasks).", {"color": S.muted})]], size=15,
               font=S.body)

    def pts(key):
        v = [float(R(f"v2/{t}/e2/{key}/mix/delta")) for t in ("sort_one", "cubebox_v1")]
        return v
    rows = [("a1_no_cooldown", "Remove “ask again”", "the client waits for a lost reply instead of "
             "asking again", "Essential", S.alert),
            ("a4_half_chunk_cap", "Undo our fix", "DRTC's original limit gives up on replies before they can "
             "arrive", "Essential", S.alert),
            ("a3_upstream_trigger", "Ask later", "LeRobot's trigger instead of asking early", "Small", S.muted),
            ("a2_upstream_estimator", "Simpler estimate", "LeRobot's delay estimate instead of ours", "No effect",
             S.muted)]
    cw, gap = Inches(2.65), Inches(0.32)
    groups = []
    for i, (key, title, what, verdict, col) in enumerate(rows):
        x = M + (cw + gap) * i
        v = pts(key)
        lo, hi = min(v), max(v)
        big = f"+{round(lo)}" if abs(hi - lo) < 2 else f"+{max(0, round(lo))}\u2013{round(hi)}"
        if hi < 1:
            big = "≈ 0"
        y = TOP + Inches(0.45)
        g = [box(s, x, y, cw, Inches(3.0), fill=S.panel, line=col if verdict == "Essential" else S.panel_line,
                 weight=Pt(1.5 if verdict == "Essential" else 1), radius=S.card_radius),
             text(s, x + Inches(0.22), y + Inches(0.2), cw - Inches(0.44), Inches(0.35), title, size=16, bold=True,
                  color=S.ink, font=S.body),
             text(s, x + Inches(0.22), y + Inches(0.6), cw - Inches(0.44), Inches(0.8), what, size=12,
                  color=S.muted, font=S.body, line=1.1),
             text(s, x + Inches(0.22), y + Inches(1.4), cw - Inches(0.44), Inches(0.6), big, size=32, bold=True,
                  color=col if verdict == "Essential" else S.ink, font=S.heading),
             text(s, x + Inches(0.22), y + Inches(2.0), cw - Inches(0.44), Inches(0.35), "points more standing still",
                  size=11, color=S.muted, font=S.body),
             box(s, x + Inches(0.22), y + Inches(2.45), Inches(1.15), Inches(0.32), fill=col, radius=0.5,
                 runs=verdict, size=11, bold=True, color="#FFFFFF", font=S.body)]
        groups.append(g)
    take = text(s, M, TOP + Inches(3.7), CW, Inches(0.6),
                [[("Asking again after a lost reply is what makes DRTC work.", {"bold": True, "color": S.ink}),
                  ("  Points = percent of the run the arm stood still, more than with full DRTC.",
                   {"color": S.muted, "size": 12})]], size=16, font=S.body)
    builds(s, [[sub]] + groups + [[take]])
    notes(s, "We removed one piece of DRTC at a time on the bench. Without asking again, the arm stands still about "
             "25 percent more of the run under the faults, for all seven policies. Undoing our first fix, DRTC's "
             "original limit on how long to wait, is almost as bad. Asking later, or a simpler delay estimate, makes "
             "little or no difference. So the single idea that matters is: when a reply is lost, ask again.")


def t_e3a(prs, n):
    s = S.content(prs, n, "6 · Results", "E3a: each smoothing method alone")
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
    s = S.content(prs, n, "6 · Results", "Choosing the stack, and why RTC was dropped")
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


def t_baselines(prs, n):
    s = S.content(prs, n, "6 · Results", "First: how well does each policy do the task?")
    fig(s, "fig_baselines", M, TOP - Inches(0.15), h=Inches(4.6))
    notes(s, f"Normal network, LeRobot as shipped, from the Raspberry Pi. GR00T N1.7 and SmolVLA reach "
             f"{R('base_sort_one/grootn17/baseline/M1')} on sorting; Diffusion Policy "
             f"{R('base_cubebox_v1/diffusion/baseline/M1')} on cube in box.")


def t_e1(prs, n):
    s = S.content(prs, n, "6 · Results", "E1 (bench): DRTC recovers from every fault")
    fig(s, "fig_v2_e1_cubebox_v1", M, TOP - Inches(0.15), h=Inches(4.6))
    notes(s, "E1: DRTC recovered from every injected fault for all seven policies on both tasks, where the waiting "
             "client often did not. E2: removing the cooldown removes almost all of the benefit.")


def t_e3(prs, n):
    s = S.content(prs, n, "6 · Results", "E3b (bench): the stack against DRTC alone")
    fig(s, "fig_v2_stack", M, TOP - Inches(0.15), h=Inches(4.6))
    notes(s, "E3: each smoothing method alone, then the stack against DRTC alone.")


def t_e4(prs, n):
    s = S.content(prs, n, "6 · Results", "E4 (robot): DRTC brings success back")
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
    sa = stat(s, x, TOP, Inches(3.4), "68 → 13 %", "the fault mix breaks LeRobot's client, even set to wait the whole trial", color=S.alert)
    sb = stat(s, x, TOP + Inches(1.5), Inches(3.4), "→ 65 %", "DRTC restores it")
    sc = stat(s, x, TOP + Inches(3.0), Inches(3.4), "→ 75 %", "with the stack, and about 3× smoother",
              color=S.series["smooth"])
    builds(s, [per[0], per[1] + sa, per[2] + sb, per[3] + sc])
    notes(s, "160 robot trials. With the Holm correction only GR00T N1.7's DRTC-vs-faults difference is significant "
             "on its own; pooled, the effect is large but exploratory. Ten trials per cell give wide intervals.")


def t_e4_video(prs, n):
    s = S.content(prs, n, "6 · Results", "Same fault, three clients")
    slot(s, "e4_trials", M, TOP - Inches(0.1), CW, Inches(4.7))
    notes(s, "Three real trials under the same fault mix.")


def t_e4_motion(prs, n):
    s = S.content(prs, n, "6 · Results", "E4 (robot): the stack makes the motion smoother")
    fig(s, "fig_e4_motion", M, TOP - Inches(0.15), h=Inches(4.6))
    notes(s, "Stall, jerk and recovery after the outage, per policy and condition.")


def t_pi(prs, n):
    s = S.content(prs, n, "6 · Results", "The Raspberry Pi can be the robot's computer")
    fig(s, "fig_edgertt", M, TOP - Inches(0.15), h=Inches(4.5), centre=False, maxw=Inches(7.5))
    x = M + Inches(7.9)
    a = stat(s, x, TOP, Inches(3.6), R("edge/hz").replace("--", "-").replace("~", " "),
             "control loop on the Pi (target 30 Hz)")
    b = stat(s, x, TOP + Inches(1.5), Inches(3.6), "0.87 → 0.64 s", "median round trip: WiFi → cable")
    c = stat(s, x, TOP + Inches(3.0), Inches(3.6), "2.9 MB", "of raw images per request: the main cost")
    builds(s, [a, b, c])
    notes(s, "The Pi keeps the loop at 30 Hz; the network, not the Pi, is the bottleneck. A cable helps.")


# Close --------------------------------------------------------------------------------------------------------------

def t_limits(prs, n):
    s = S.content(prs, n, "Close", "What this does not show yet")
    rows = bullets(s, M, TOP, CW, [
        ("Ten trials per condition", "wide intervals; only large differences are significant"),
        ("One task on the robot", "the bench covers two tasks and seven policies"),
        ("Simulated faults", "fixed shapes, not a recorded real network"),
        ("Slow links", "DRTC can give up replies that are only slow (seen over the Pi's WiFi)"),
    ], size=19, gap=Inches(0.8))
    builds(s, rows)
    notes(s, "Be upfront about the limits.")


def t_conclusion(prs, n):
    s = S.content(prs, n, "Close", "Conclusion and future work")
    rows = bullets(s, M, TOP, Inches(6.6), [
        ("Faults break remote control:", "68 → 13 % success"),
        ("DRTC restores it:", "65 %; asking again does the work"),
        ("Smoothing keeps it:", "75 %, about 3× smoother"),
        ("A full platform:", "record, train, serve, test, from a Pi"),
    ], size=18, gap=Inches(0.8))
    x = M + Inches(7.2)
    fut = text(s, x, TOP, Inches(4.3), Inches(3.8),
               [[("Next", {"bold": True, "color": S.accent, "size": 13})],
                [("A cooldown that follows a slowing link")], [("Compressed images")],
                [("More trials, more tasks")]], size=17, font=S.body, line=1.6, color=S.ink)
    builds(s, rows + [[fut]])
    notes(s, "Close on the numbers and what comes next.")


def t_thanks(prs, n):
    S.closing_slide(prs, n, "Thank you", "Questions?")


SLIDES = [t_title, t_hook, t_outline,
          lambda p, n: S.section_slide(p, n, 1, "Robot control with a VLA", "Cameras and joints in, motion out"),
          t_vla_what, t_vla_video, t_chunks, t_why_server,
          lambda p, n: S.section_slide(p, n, 2, "The problem", "Control over an unreliable network"),
          t_loop, t_fault, t_faults, t_question,
          lambda p, n: S.section_slide(p, n, 3, "Our solution", "DRTC and motion smoothing"),
          t_drtc_idea, t_drtc_media, t_jerk, t_stack,
          lambda p, n: S.section_slide(p, n, 4, "What we built", "Hardware, our lab platform, data, training"),
          t_hardware, t_lab, t_machines, t_record, t_first_attempt, t_data, t_trimming, t_models, t_offline,
          lambda p, n: S.section_slide(p, n, 5, "How we evaluate", "Offline, the bench, live robot trials"),
          t_eval_kinds, t_bench, t_robot_eval, t_records, t_map,
          lambda p, n: S.section_slide(p, n, 6, "Results", "Baselines, the bench studies, the robot experiment"),
          t_baselines, t_e1, t_e2, t_e3a, t_stack_choice, t_e3, t_e4, t_e4_video, t_e4_motion, t_pi,
          t_contrib, t_limits, t_conclusion, t_thanks]


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
