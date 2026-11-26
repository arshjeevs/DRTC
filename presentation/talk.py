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

from build import CW, M, TOP, card, notes, para, photo, stat, table
from deck import FIGS, W, bar_panel, box, builds, movie, new_deck, picture, text
from drtc.evals.metrics import wilson
from results import R, title_page
from style import Style

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
    s = S.content(prs, n, "1 · Robot control with a VLA", "A model that sees, reads and moves")
    cw, gap = Inches(3.6), Inches(0.37)
    items = [("Sees", "two cameras", "one on the gripper, one watching the whole table", "vla_wrist.jpg"),
             ("Reads", "an instruction", "“Pick up the green cube and put it in the box”", None),
             ("Acts", "six joint angles", "where each servo of the arm should be next", "vla_scene.jpg")]
    groups = []
    for i, (big, title, body, img) in enumerate(items):
        x = M + (cw + gap) * i
        g = card(s, x, TOP, cw, Inches(2.1), big, title, body)
        if img:
            g.append(photo(s, x, TOP + Inches(2.3), cw, Inches(2.2), img, title))
        groups.append(g)
    builds(s, groups)
    notes(s, "A vision-language-action model (VLA) is a neural network trained on demonstrations. It looks at camera "
             "images, reads a sentence, and outputs where the arm's joints should go. No hand-written program.")


def t_vla_video(prs, n):
    s = S.content(prs, n, "1 · Robot control with a VLA", "From camera to motion")
    slot(s, "vla", M, TOP - Inches(0.1), CW, Inches(4.7))
    notes(s, "Walk through the video: capture, model, chunk, playback, ask again. Thirty times a second.")


def t_chunks(prs, n):
    s = S.content(prs, n, "1 · Robot control with a VLA", "Action chunks: the model plans ahead")
    rows = bullets(s, M, TOP, Inches(6.2), [
        ("One request,", "a whole chunk of future moves"),
        ("50 moves", "= 1.7 s of motion at 30 per second (SmolVLA)"),
        ("Played one by one", "from a queue while the next is computed"),
        ("Ask again", "before the queue runs out"),
    ], size=19, gap=Inches(0.75))
    st = stat(s, M + Inches(7.3), TOP + Inches(0.2), Inches(4.2), "30 Hz",
              "the arm needs a new target every 33 ms; the model takes about 0.1 to 0.8 s to answer")
    builds(s, rows + [st])
    notes(s, "The model is too slow to answer every 33 ms, so it predicts a chunk of future moves at once; the robot "
             "plays them while the next chunk is being computed.")


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
    groups = table(s, M, TOP - Inches(0.15), [Inches(1.8), Inches(4.6), Inches(5.1)],
                   ["FAULT", "WHAT HAPPENS", "LIKE"], [list(f) for f in FAULTS], size=16, row_h=Inches(0.53))
    builds(s, groups)
    notes(s, "These are the faults we inject, the same in every trial, shifted by a random 0-4 s. Explain each with "
             "its everyday picture. The mix is what we use on the real robot.")


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
    ideas = [("1", "Ask early", "request the next chunk with about 1 s of moves still left, not 0.5 s"),
             ("2", "Learn the delay", "estimate how long replies take, as the Internet's TCP does"),
             ("3", "Ask again", "if no reply comes in that time, assume it is lost and send a fresh request")]
    groups = [card(s, M + (cw + gap) * i, TOP, cw, Inches(2.6), a, b, c) for i, (a, b, c) in enumerate(ideas)]
    foot = text(s, M, TOP + Inches(3.0), CW, Inches(1.2),
                [[("Distributed Real-Time Chunking (Vial, 2026)", {"bold": True, "color": S.ink}),
                  ("   We ported its scheduling into LeRobot's own runtime; the model and server are unchanged.",
                   {"color": S.muted})]], size=16, font=S.body)
    builds(s, groups + [[foot]])
    notes(s, "DRTC was designed for unreliable networks. Three ideas. The third, the cooldown re-request, turns out to "
             "carry almost all the benefit.")


def t_drtc_media(prs, n):
    s = S.content(prs, n, "3 · Our solution", "DRTC at work")
    slot(s, "drtc", M, TOP - Inches(0.1), CW, Inches(4.7))
    notes(s, "The same fault, two clients: LeRobot waits and stops; DRTC asks again and keeps going.")


def t_fixes(prs, n):
    s = S.content(prs, n, "3 · Our solution", "Four fixes we found on the real robot")
    rows = [["GR00T never received a chunk", "requests given up before the reply could arrive",
             "never give up sooner than the measured delay"],
            ["A storm of re-requests", "replies near the estimate given up; the server slowed further",
             "also wait for the slowest recent reply"],
            ["The robot never started on a slow link", "no reply yet, so no delay estimate", "wait up to 10 s for the first one"],
            ["“Invalid RTC delay” errors", "the delay asked for exceeded the chunk", "cap it at the chunk"]]
    groups = table(s, M, TOP - Inches(0.15), [Inches(3.6), Inches(4.3), Inches(3.6)], ["FAILURE", "CAUSE", "FIX"],
                   rows, size=15, row_h=Inches(0.78))
    builds(s, groups)
    notes(s, "Each was a failure on the real robot, fixed and then checked on the replay bench.")


def t_jerk(prs, n):
    s = S.content(prs, n, "3 · Our solution", "Recovering is not enough: the motion jerks")
    slot(s, "jerk", M, TOP - Inches(0.1), Inches(7.4), Inches(4.6))
    x = M + Inches(7.8)
    rows = bullets(s, x, TOP, Inches(3.8), [
        ("Seams", "a new chunk starts somewhere else"),
        ("Stalls", "the queue runs dry, the arm halts"),
        ("Jitter", "small shakes within a chunk"),
        ("Restarts", "after a fault the arm lurches"),
    ], size=17, gap=Inches(0.8))
    builds(s, rows)
    notes(s, "Four sources of jerky motion. Each smoothing method targets one or more of them.")


def t_stack(prs, n):
    s = S.content(prs, n, "3 · Our solution", "The smoothing stack")
    cw, gap = Inches(2.67), Inches(0.29)
    parts = [("1", "Adaptive playback", "play slower when the queue runs low, so it lasts until the reply"),
             ("2", "Temporal ensembling", "blend overlapping chunks at the seam"),
             ("3", "Butterworth filter", "remove the shakes from the commands"),
             ("4", "Resume ramp", "ease back in after a fault instead of jumping")]
    groups = [card(s, M + (cw + gap) * i, TOP, cw, Inches(2.9), a, b, c) for i, (a, b, c) in enumerate(parts)]
    foot = text(s, M, TOP + Inches(3.3), CW, Inches(1.2),
                [[("Chosen by a rule fixed in advance on the bench.", {"bold": True, "color": S.ink}),
                  ("   RTC was tested too, but cost more in delay than it gained in smoothness on our setup.",
                   {"color": S.muted})]], size=16, font=S.body)
    builds(s, groups + [[foot]])
    notes(s, "Six methods tested; this combination won. Each card can later get its own small before/after curve.")


# Part 4 -----------------------------------------------------------------------------------------------------------

def t_hardware(prs, n):
    s = S.content(prs, n, "4 · What we built", "The hardware")
    key = "hardware"
    if any((MEDIA / f"{key}{e}").is_file() for e in (".mp4", ".png", ".jpg")):
        slot(s, key, M, TOP - Inches(0.1), Inches(7.0), Inches(4.6))
    else:
        photo(s, M, TOP - Inches(0.1), Inches(7.0), Inches(4.6), "setup.jpg", "the setup")
    x = M + Inches(7.4)
    rows = bullets(s, x, TOP - Inches(0.1), Inches(4.2), [
        ("SO-101 arm", "6 servos; a leader arm to teach it"),
        ("Two cameras", "wrist + phone on a mount"),
        ("Raspberry Pi 5", "the robot's computer"),
        ("Laptop", "RTX 4060; recording and the network"),
        ("A100 + H200", "shared GPU servers"),
    ], size=16, gap=Inches(0.72))
    builds(s, rows)
    notes(s, "Everything is low-cost except the GPU servers, which are shared university machines. Later: the photo "
             "gets labels appearing one by one (slot 'hardware').")


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
    cards = [("Bench", "replay recorded trials", "the real model and server; the robot replayed. Thousands of runs, "
              "every fault, timing measured"),
             ("Robot", "real trials", "the arm does the task under the fault mix; success judged by a rule fixed in "
              "advance"),
             ("Fault injector", "the same faults everywhere", "between the robot and the server; seeded, so every "
              "client sees the same fault")]
    groups = [card(s, M + (cw + gap) * i, TOP, cw, Inches(2.7), a, b, c) for i, (a, b, c) in enumerate(cards)]
    foot = text(s, M, TOP + Inches(3.05), CW, Inches(1.2),
                [[("E1-E3 on the bench, E4 on the robot.", {"bold": True, "color": S.ink}),
                  ("   Plans frozen before running, rounds interleaved, every trial's raw record kept.",
                   {"color": S.muted})]], size=16, font=S.body)
    builds(s, groups + [[foot]])
    notes(s, "Bench: E1 DRTC against LeRobot, E2 which part matters, E3 the smoothing. Robot: E4.")


# Part 5 -----------------------------------------------------------------------------------------------------------

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
    sc = stat(s, x, TOP + Inches(3.0), Inches(3.4), "75 %, 3× smoother", "with the stack",
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
          t_drtc_idea, t_drtc_media, t_fixes, t_jerk, t_stack,
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
