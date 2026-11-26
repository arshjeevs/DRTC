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

import subprocess

from build import ASSETS, CW, M, TOP, card, notes, para, photo, stat, table
from deck import FIGS, W, bar_panel, box, builds, line, movie, new_deck, picture, text
from drtc.evals.metrics import wilson
from results import R, title_page
from style import Style
from talk_draw import (JOINTS, chunk_plot, dot, fault_row, first_chunk_curves, first_state, marker,
                       mini_panel, plot_curve, problem_curves)

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
    s = S.content(prs, n, "Outline", "Five steps")
    parts = [("1", "How a robot is driven by an AI model", "cameras and joints in, motion out"),
             ("2", "The problem", "control over a network, and the seven faults"),
             ("3", "Our solution", "DRTC and motion smoothing"),
             ("4", "What we built", "hardware, data, training, testing"),
             ("5", "Results", "on the bench and on the real robot")]
    rows = [para(s, M, TOP + Inches(0.85) * i, CW, f"{a}   {b}", c, size=22, h=Inches(0.8))
            for i, (a, b, c) in enumerate(parts)]
    builds(s, rows)
    notes(s, "The order of the talk.")


# Part 1 -----------------------------------------------------------------------------------------------------------

def t_vla_what(prs, n):
    s = S.content(prs, n, "1 · Robot control with a VLA", "What goes in, what comes out")
    lab = dict(size=12, color=S.muted, font=S.body)
    iw = Inches(1.75)
    ins = [picture(s, ASSETS / "vla_wrist.jpg", M, TOP - Inches(0.15), w=iw),
           picture(s, ASSETS / "vla_scene.jpg", M + iw + Inches(0.12), TOP - Inches(0.15), w=iw)]
    ih = ins[0].height
    ins += [text(s, M, TOP - Inches(0.1) + ih, iw, Inches(0.3), "wrist camera", **lab),
            text(s, M + iw + Inches(0.12), TOP - Inches(0.1) + ih, iw, Inches(0.3), "scene camera", **lab)]
    y2 = TOP + ih + Inches(0.3)
    ins.append(box(s, M, y2, Inches(3.62), Inches(0.62), fill=S.panel, line=S.panel_line, radius=S.card_radius,
                   runs=[[("“Pick up the green cube and put it in the box”", {"italic": True})]], size=14,
                   font=S.body, align=PP_ALIGN.LEFT, margin=Inches(0.15)))
    js = first_state()
    names = ["pan", "lift", "elbow", "wrist", "roll", "grip"]
    vals = "   ".join(f"{nm} {js[j]:.0f}°" if nm != "grip" else f"{nm} {js[j]:.0f}" for nm, j in zip(names, JOINTS))
    ins.append(box(s, M, y2 + Inches(0.75), Inches(3.62), Inches(0.75), fill=S.panel, line=S.panel_line,
                   radius=S.card_radius, runs=[[("six joint angles now", {"bold": True, "size": 12})],
                                                [(vals, {"size": 12, "color": S.muted})]],
                   size=13, font=S.body, align=PP_ALIGN.LEFT, margin=Inches(0.15)))
    # the model
    mx, my, mw, mh = M + Inches(4.45), TOP + Inches(0.2), Inches(3.0), Inches(3.4)
    a1 = [line(s, M + Inches(3.75), my + mh * f, mx - Inches(0.05), my + mh / 2, color=S.ink, weight=Pt(1.25),
               arrow=True) for f in (0.12, 0.62, 0.88)]
    model = [box(s, mx, my, mw, mh, fill=S.dark, radius=S.card_radius),
             text(s, mx + Inches(0.25), my + Inches(0.2), mw - Inches(0.5), Inches(0.4), "VLA policy", size=20,
                  bold=True, font=S.heading, color=S.dark_text),
             box(s, mx + Inches(0.25), my + Inches(0.8), mw - Inches(0.5), Inches(1.05), fill="#232A34",
                 radius=S.card_radius, runs=[[("Vision-language model", {"bold": True})],
                                             [("images + words → tokens", {"color": S.dark_muted, "size": 12})]],
                 size=14, font=S.body, color=S.dark_text),
             box(s, mx + Inches(0.25), my + Inches(2.05), mw - Inches(0.5), Inches(1.05), fill="#232A34",
                 radius=S.card_radius, runs=[[("Action expert", {"bold": True})],
                                             [("tokens + joints → future moves", {"color": S.dark_muted,
                                                                                         "size": 12})]],
                 size=14, font=S.body, color=S.dark_text)]
    tech = text(s, mx, my + mh + Inches(0.12), mw, Inches(0.6),
                "SmolVLA: 0.45 B parameters; the action expert generates the chunk by flow matching", size=11,
                color=S.accent, font=S.body, line=1.15)
    # the output
    ox, ow = mx + mw + Inches(0.75), W - M - (mx + mw + Inches(0.75))
    a2 = line(s, mx + mw + Inches(0.05), my + mh / 2, ox - Inches(0.08), my + mh / 2, color=S.ink, weight=Pt(1.25),
              arrow=True)
    frame = box(s, ox, my, ow, Inches(2.4), fill=S.panel, line=S.panel_line, radius=S.card_radius)
    curves = first_chunk_curves(s, ox + Inches(0.15), my + Inches(0.15), ow - Inches(0.3), Inches(2.1))
    olab = text(s, ox, my + Inches(2.5), ow, Inches(1.1),
                [[("A chunk", {"bold": True, "color": S.ink})],
                 [("50 future positions for each of the 6 joints: the next 1.7 s", {"color": S.muted, "size": 13})]],
                size=16, font=S.body, line=1.15)
    builds(s, [ins, a1 + model + [tech], [a2, frame, olab] + [(c, "wipe_right") for c in curves]])
    notes(s, "Real values from one trial. Two images, the instruction and the six joint angles go in. The VLA turns "
             "images and words into tokens; its action expert turns those and the joint angles into a chunk: the next "
             "50 positions of every joint. Each coloured line is one joint.")


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
    s = S.content(prs, n, "3 · Our solution", "The smoothing stack: one fix for each problem")

    def draw(s, plot, c):
        before, after, _ = c
        return [(plot_curve(s, plot, before, S.muted, dash=True, weight=Pt(1.5)), "fade", 100),
                (plot_curve(s, plot, after, S.accent), "wipe_right", 400)]
    steps = _panels(s, lambda r: r[3], lambda r: f"{r[4]}  (fixes {r[1].lower()})", lambda r: r[5], draw)
    foot = text(s, M, S.bottom - Inches(0.55), CW, Inches(0.5),
                [[("Dashed: before.  Blue: with the method.", {"bold": True, "color": S.ink}),
                  ("   Chosen on the bench by a rule fixed in advance; RTC was tested too but cost more delay than it "
                   "saved.", {"color": S.muted})]], size=13, font=S.body)
    builds(s, steps + [[(foot, "fade")]])
    notes(s, "Each method answers one problem from the previous slide. Together they are the stack we used on the "
             "robot.")


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
    s = S.content(prs, n, "4 · What we built", "Two tasks, three datasets")
    pw = Inches(3.4)
    p1 = photo(s, M, TOP - Inches(0.1), pw, Inches(2.55), "task_cubebox.jpg", "cube in box")
    p2 = photo(s, M + pw + Inches(0.2), TOP - Inches(0.1), pw, Inches(2.55), "task_sort_one.jpg", "sorting")
    l1 = text(s, M, TOP + Inches(2.5), pw, Inches(0.6), [[("Cube in box", {"bold": True})],
                                                        [("pick up the green cube, put it in the box",
                                                          {"color": S.muted, "size": 13})]], size=16, font=S.body)
    l2 = text(s, M + pw + Inches(0.2), TOP + Inches(2.5), pw, Inches(0.6),
              [[("Sorting", {"bold": True})], [("put the cube on the square of its colour", {"color": S.muted,
                                                                                         "size": 13})]],
              size=16, font=S.body)
    x = M + 2 * pw + Inches(0.7)
    g = table(s, x, TOP - Inches(0.1), [Inches(1.9), Inches(1.2), Inches(1.0)], ["DATASET", "EPISODES", "TEST"],
              [["Cube in box v0", R("data/so101_cubebox_v0/episodes"), R("data/so101_cubebox_v0/held")],
               ["Cube in box v1", R("data/so101_cubebox_v1/episodes"), R("data/so101_cubebox_v1/held")],
               ["Sorting", R("data/so101_sort_one_v1/episodes"), R("data/so101_sort_one_v1/held")]],
              size=15, row_h=Inches(0.5))
    builds(s, [[p1, l1], [p2, l2]] + g)
    notes(s, "Two tasks, fixed taped layouts, balanced positions. An episode is one demonstration; the test episodes "
             "are held out from training.")


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


def t_testing(prs, n):
    s = S.content(prs, n, "4 · What we built", "How we test: a bench and the real robot")
    cw, gap = Inches(3.6), Inches(0.37)
    cards = [("Bench", "Replay", "the real model and server; the arm replayed from recorded trials, so every fault "
              "can be run thousands of times"),
             ("Robot", "Real trials", "the arm does the task under the fault mix; success judged by a rule written "
              "down first"),
             ("Faults", "Fault injector", "sits between the robot and the server; seeded, so every client meets the "
              "same faults")]
    groups = [card(s, M + (cw + gap) * i, TOP, cw, Inches(2.6), a, b, c) for i, (a, b, c) in enumerate(cards)]
    foot = text(s, M, TOP + Inches(2.95), CW, Inches(1.4),
                [[("E1-E3 on the bench, E4 on the robot.", {"bold": True, "color": S.ink}),
                  ("   Plans frozen before running, rounds interleaved, every trial's raw record kept.",
                   {"color": S.muted})],
                 [("Measured per trial: success (with a 95 % interval), stalls, jerk, recovery time after a fault.",
                   {"color": S.accent, "size": 13})]], size=16, font=S.body, line=1.4)
    builds(s, groups + [[foot]])
    notes(s, "Bench: E1 DRTC against LeRobot, E2 which part matters, E3 the smoothing. Robot: E4. Differences are "
             "tested with Fisher's exact test (success) and Mann-Whitney (motion), Holm-corrected.")


def t_baselines(prs, n):
    s = S.content(prs, n, "5 · Results", "First: how well does each policy do the task?")
    fig(s, "fig_baselines", M, TOP - Inches(0.15), h=Inches(4.6))
    notes(s, f"Normal network, LeRobot as shipped, from the Raspberry Pi. GR00T N1.7 and SmolVLA reach "
             f"{R('base_sort_one/grootn17/baseline/M1')} on sorting; Diffusion Policy "
             f"{R('base_cubebox_v1/diffusion/baseline/M1')} on cube in box.")


def t_e1(prs, n):
    s = S.content(prs, n, "5 · Results", "Bench: DRTC recovers from every fault")
    fig(s, "fig_v2_e1_cubebox_v1", M, TOP - Inches(0.15), h=Inches(4.6))
    notes(s, "E1: DRTC recovered from every injected fault for all seven policies on both tasks, where the waiting "
             "client often did not. E2: removing the cooldown removes almost all of the benefit.")


def t_e3(prs, n):
    s = S.content(prs, n, "5 · Results", "Bench: the stack smooths the motion")
    fig(s, "fig_v2_stack", M, TOP - Inches(0.15), h=Inches(4.6))
    notes(s, "E3: each smoothing method alone, then the stack against DRTC alone.")


def t_e4(prs, n):
    s = S.content(prs, n, "5 · Results", "On the robot: DRTC brings success back")
    models = ["diffusion", "grootn17", "pi05", "smolvla"]
    conds = [("upstream_patient__none", "LeRobot\nno faults", S.series["baseline"]),
             ("upstream_patient__mix", "LeRobot\n+ faults", S.series["faults"]),
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
    sa = stat(s, x, TOP, Inches(3.4), "68 → 13 %", "the fault mix breaks LeRobot's client", color=S.alert)
    sb = stat(s, x, TOP + Inches(1.5), Inches(3.4), "→ 65 %", "DRTC restores it")
    sc = stat(s, x, TOP + Inches(3.0), Inches(3.4), "→ 75 %", "with the stack, and about 3× smoother",
              color=S.series["smooth"])
    builds(s, [per[0], per[1] + sa, per[2] + sb, per[3] + sc])
    notes(s, "160 robot trials. With the Holm correction only GR00T N1.7's DRTC-vs-faults difference is significant "
             "on its own; pooled, the effect is large but exploratory. Ten trials per cell give wide intervals.")


def t_e4_video(prs, n):
    s = S.content(prs, n, "5 · Results", "Same fault, three clients")
    slot(s, "e4_trials", M, TOP - Inches(0.1), CW, Inches(4.7))
    notes(s, "Three real trials under the same fault mix.")


def t_e4_motion(prs, n):
    s = S.content(prs, n, "5 · Results", "On the robot: the stack makes the motion smoother")
    fig(s, "fig_e4_motion", M, TOP - Inches(0.15), h=Inches(4.6))
    notes(s, "Stall, jerk and recovery after the outage, per policy and condition.")


def t_pi(prs, n):
    s = S.content(prs, n, "5 · Results", "The Raspberry Pi can be the robot's computer")
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
          lambda p, n: S.section_slide(p, n, 4, "What we built", "Hardware, data, training, testing"),
          t_hardware, t_record, t_first_attempt, t_data, t_models, t_testing,
          lambda p, n: S.section_slide(p, n, 5, "Results", "On the bench and on the real robot"),
          t_baselines, t_e1, t_e3, t_e4, t_e4_video, t_e4_motion, t_pi,
          t_limits, t_conclusion, t_thanks]


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
