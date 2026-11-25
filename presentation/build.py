"""The presentation: one .pptx in the style of the first browser deck (style.py), built from the results.

    cd report && make data            # refresh numbers + figures from results/ (when results change)
    .venv/bin/python presentation/build.py          # -> presentation/out/presentation.pptx (+ .pdf, PNG previews)
    .venv/bin/python presentation/build.py --no-preview

Numbers: report/generated/numbers.tex (results.py), so the deck and the report agree; missing ones read "pending".
Figures: native shapes where a chart is animated (success), the report's slide PNGs elsewhere. Videos: real trials,
embedded. Photos: report/assets/<file> when present, otherwise a still from a trial video or a framed placeholder.
Title-page details: report/config.tex. Each slide has speaker notes. To add a slide: write a function below and list
it in SLIDES.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from deck import FIGS, REPO, W, bar_panel, box, builds, line, movie, new_deck, picture, poster, text
from drtc.evals.metrics import wilson
from results import F, K, R, title_page
from style import Style

S = Style()
OUT = Path(__file__).resolve().parent / "out"
ASSETS = REPO / "report" / "assets"
MAIN = REPO / "results" / "evals" / "sort_one_drtc_main" / "sessions"
MODELS = [("smolvla", "SmolVLA", "smolvla_sort_one_v1_full"), ("groot", "GR00T N1.5", "groot_sort_one_v1_full")]
CONDS = [("baseline", "baseline", "Baseline", "normal network, LeRobot as shipped"),
         ("faults", "baseline_faults", "Baseline + faults", "the fault mix, LeRobot as shipped"),
         ("drtc", "drtc_faults", "DRTC + faults", "the fault mix, DRTC (with our fixes)"),
         ("smooth", "drtc_smooth_faults", "DRTC + smoothing + faults", "as 3, plus the filter and adaptive playback")]
PARTS = ["The problem", "Method", "Experiments", "Results", "Outlook"]

M, TOP = S.margin, S.top
CW = W - 2 * M  # content width


# --- small components -------------------------------------------------------------------------------------------

def notes(s, said: str) -> None:
    s.notes_slide.notes_text_frame.text = said


def para(s, x, y, w, lead, rest="", *, size=18, h=Inches(0.5), marker=None, lead_color=None):
    """One line of text: a bold lead and a grey rest, with an optional colour square before it."""
    shapes = []
    if marker:
        shapes.append(box(s, x, y + Pt(size * 0.42), Inches(0.14), Inches(0.14), fill=marker))
        x, w = x + Inches(0.34), w - Inches(0.34)
    runs = [(lead, {"bold": True, "color": lead_color or S.ink})]
    if rest:
        runs.append(("   " + rest, {"color": S.muted}))
    shapes.append(text(s, x, y, w, h, [runs], size=size, font=S.body, line=1.12))
    return shapes


def stat(s, x, y, w, big, small, *, color=None, size=34, small_size=15):
    a = text(s, x, y, w, Inches(0.7), big, size=size, font=S.heading, color=color or S.accent)
    b = text(s, x, y + Pt(size * 1.25), w, Inches(0.9), small, size=small_size, color=S.muted, font=S.body, line=1.15)
    return [a, b]


def card(s, x, y, w, h, num, title, body, *, num_color=None):
    shapes = [box(s, x, y, w, h, fill=S.panel, line=S.panel_line, weight=Pt(1), radius=S.card_radius)]
    pad = Inches(0.25)
    if num:
        shapes.append(text(s, x + pad, y + pad, w - 2 * pad, Inches(0.5), num, size=24, font=S.heading,
                           color=num_color or S.accent))
    ty = y + pad + (Inches(0.6) if num else 0)
    per_line = max(1, int((w - 2 * pad) / 12700 / (0.58 * 18)))  # characters per line at 18 pt bold
    lines = -(-len(title) // per_line)
    shapes.append(text(s, x + pad, ty, w - 2 * pad, Inches(0.3) * lines + Inches(0.15), title, size=18, bold=True,
                       font=S.body, color=S.ink, line=1.05))
    by = ty + Inches(0.3) * lines + Inches(0.2)
    shapes.append(text(s, x + pad, by, w - 2 * pad, h - (by - y) - Inches(0.1), body, size=14,
                       color=S.muted, font=S.body, line=1.15))
    return shapes


def photo(s, x, y, w, h, asset: str, label: str, fallback: Path | None = None):
    """A photo from report/assets, else a still (fallback), else a framed placeholder naming the file."""
    path = next((p for p in (ASSETS / asset, ASSETS / Path(asset).with_suffix(".png")) if p.is_file()), None)
    path = path or fallback
    if path:
        pic = picture(s, path, x, y, w=w)
        if pic.height > h:  # crop to the frame, centred
            excess = 1 - h / pic.height
            pic.crop_top = pic.crop_bottom = excess / 2
            pic.height = h
        return pic
    return box(s, x, y, w, h, line=S.panel_line, dash=True, weight=Pt(1.25),
               runs=[[(label, {"color": S.muted})], [(f"report/assets/{asset}", {"color": S.muted, "size": 11})]],
               size=15, font=S.body)


def table(s, x, y, widths, header, rows, *, size=15, row_h=Inches(0.48), bold_first=True, highlight=()):
    """A plain table: header in grey caps, hairlines between rows. Returns the shapes of each row (for builds)."""
    def cells(yy, vals, style):
        out, xx = [], x
        for i, (v, wd) in enumerate(zip(vals, widths)):
            out.append(text(s, xx, yy, wd - Inches(0.15), row_h, v, font=S.body, anchor=MSO_ANCHOR.MIDDLE,
                            **{"size": size, "color": S.ink, "line": 1.05, **style(i)}))
            xx += wd
        return out
    total = sum(widths)
    cells(y, header, lambda i: {"size": 11, "color": S.muted, "bold": True})
    line(s, x, y + row_h, x + total, y + row_h, color=S.ink, weight=Pt(0.75))
    groups = []
    for r, vals in enumerate(rows):
        yy = y + row_h * (r + 1) + Inches(0.04)
        hl = r in highlight
        g = cells(yy, vals, lambda i: {"bold": (i == 0 and bold_first) or hl,
                                          "color": S.accent if hl and i == 0 else (S.ink if i == 0 else S.muted)})
        g.append(line(s, x, yy + row_h, x + total, yy + row_h, color=S.grid, weight=Pt(0.75)))
        groups.append(g)
    return groups


def trial_video(cond_dir: str, model_dir: str, want_success: bool | None, prefer: int = 3) -> Path | None:
    """A trial's scene video: trial `prefer` if it has the wanted outcome, else the first one that does."""
    found = []
    for tj in sorted(MAIN.glob(f"*/trials/{cond_dir}/{model_dir}/t*/trial.json")):
        d = json.loads(tj.read_text())
        v = tj.parent / "video" / "scene.mp4"
        if v.is_file() and (want_success is None or bool(d.get("success")) == want_success):
            found.append((d.get("trial") != prefer, str(tj), v))
    return min(found)[2] if found else None


def pending_box(s, x, y, w, h, what):
    return box(s, x, y, w, h, line=S.panel_line, dash=True, weight=Pt(1.25), radius=S.card_radius,
               runs=[[("pending", {"bold": True, "color": S.muted})], [(what, {"color": S.muted, "size": 13})]],
               size=16, font=S.body)


# --- the slides --------------------------------------------------------------------------------------------------

def s_title(prs, n):
    meta = {"kind": "Major project", **title_page()}
    s = S.title_slide(prs, meta)
    notes(s, "Introduce yourselves and the guide. One sentence: we make remote robot inference survive network "
             "faults, on a low-cost arm, and show it with robot trials.")


def s_outline(prs, n):
    s = S.content(prs, n, "Outline", "What this talk covers")
    rows = []
    for i, (part, what) in enumerate(zip(PARTS, [
            "why robots run their policy remotely, and why that breaks",
            "DRTC, our fixes to it, and motion smoothing",
            "the arm, the data, seven policies, and how we evaluate",
            "task success, motion, recovery, each fault alone, offline tests",
            "the redesigned experiment, the Raspberry Pi, limits and next steps"])):
        y = TOP + Inches(0.15) + Inches(0.82) * i
        a = text(s, M, y, Inches(1.0), Inches(0.6), f"{i + 1:02d}", size=28, font=S.heading, color=S.accent)
        b = text(s, M + Inches(1.1), y + Inches(0.02), Inches(4.5), Inches(0.6), part, size=22, font=S.heading,
                 color=S.ink)
        c = text(s, M + Inches(4.6), y + Inches(0.1), Inches(7.2), Inches(0.6), what, size=16, color=S.muted,
                 font=S.body)
        rows.append([a, b, c])
    builds(s, rows)
    notes(s, "Five parts. The core result is in part 4: under the same network faults, the shipped client fails "
             "every run and DRTC recovers.")


def s_section(part):
    leads = {1: "Robot policies have outgrown the robot's computer",
             2: "What DRTC does, what we fixed, and how we smooth the motion",
             3: "The platform, the data, the policies, and the evaluation",
             4: "The first run on the robot, the bench, and the offline tests",
             5: "The redesigned experiment, the edge, and what is next"}

    def make(prs, n):
        s = S.section_slide(prs, n, part, PARTS[part - 1], leads[part])
        notes(s, f"Part {part}: {PARTS[part - 1]}.")
    return make


def s_problem(prs, n):
    s = S.content(prs, n, "The problem", "Robot policies have outgrown the robot")
    w = Inches(5.6)
    items = [("Large models.", "Vision-language-action policies have 0.45 to 3.4 billion parameters."),
             ("Small computers.", "A laptop or a Raspberry Pi cannot run them at 30 actions per second."),
             ("So they run on a GPU server,", "and every action the arm takes crosses the network."),
             ("When the network fails,", "the arm stops, and the task fails.")]
    groups = [para(s, M, TOP + Inches(0.2) + Inches(1.0) * i, w, a, b, size=19, h=Inches(0.9),
                   marker=S.alert if i == 3 else S.accent) for i, (a, b) in enumerate(items)]
    vid = trial_video("baseline_faults", MODELS[0][2], False)
    vw = Inches(5.6)
    vh = int(vw * 3 / 4)
    if vid:
        movie(s, vid, W - M - vw, TOP, vw, vh)
    text(s, W - M - vw, TOP + vh + Inches(0.1), vw, Inches(0.4),
         "SmolVLA, LeRobot as shipped, under a network fault: the arm stops.", size=13, color=S.muted, font=S.body)
    builds(s, groups)
    notes(s, "VLA models are too big for the robot's own computer, so they run on a GPU server. Each step sends the "
             "camera images and gets back a chunk of actions. The video: the arm stops when the network fails.")


def s_setting(prs, n):
    s = S.content(prs, n, "The problem · the setting", "Remote inference, and where it breaks")
    bw, bh = Inches(3.0), Inches(1.5)
    y = TOP + Inches(0.35)
    xs = [M, (W - bw) // 2, W - M - bw]
    names = [("Robot client", "SO-101 arm · laptop, 30 Hz"), ("Network", "Wi-Fi and the local network"),
             ("Policy server", "A100 GPU · the VLA policy")]
    boxes = [box(s, x, y, bw, bh, fill=None if i == 1 else S.panel, line=S.panel_line, dash=i == 1,
                 weight=Pt(1.25), radius=0.08, font=S.body, color=S.panel_ink,
                 runs=[[(a, {"bold": True, "size": 19})], [(b, {"color": S.muted, "size": 13})]])
             for i, (x, (a, b)) in enumerate(zip(xs, names))]
    gap = Inches(0.12)
    flow = []
    for i in range(2):
        x1, x2 = xs[i] + bw + gap, xs[i + 1] - gap
        a = line(s, x1, y + Inches(0.5), x2, y + Inches(0.5), color=S.ink, weight=Pt(1.75), arrow=True)
        b = line(s, x2, y + Inches(1.0), x1, y + Inches(1.0), color=S.ink, weight=Pt(1.75), arrow=True)
        la = text(s, x1, y + Inches(0.12), x2 - x1, Inches(0.3), "observation", size=10, color=S.muted,
                  font=S.body, align=PP_ALIGN.CENTER)
        lb = text(s, x1, y + Inches(1.08), x2 - x1, Inches(0.3), "actions", size=10, color=S.muted,
                  font=S.body, align=PP_ALIGN.CENTER)
        flow += [(a, "wipe_right", 250 * i), (la, "fade", 250 * i), (b, "wipe_left", 500 + 250 * (1 - i)),
                 (lb, "fade", 500 + 250 * (1 - i))]
    pills = []
    pw, pg = Inches(1.0), Inches(0.1)
    px = xs[1] + (bw - 3 * pw - 2 * pg) // 2
    for k, f in enumerate(("spike", "drops", "outage")):
        pills.append((box(s, px + k * (pw + pg), y + bh + Inches(0.22), pw, Inches(0.36), fill=S.alert, radius=0.5,
                          runs=f, size=12, bold=True, color="#FFFFFF", font=S.body, margin=0), "fade", 120 * k))
    sy = y + bh + Inches(1.0)
    stats = []
    for i, (key, name) in enumerate([("smolvla", "SmolVLA"), ("groot", "GR00T N1.5")]):
        stats.append(stat(s, M + Inches(4.2) * i, sy, Inches(3.9), R(f"main/{key}/baseline/M4med"),
                          f"{name} round trip, of which {R(f'main/{key}/baseline/M6')} is inference", size=30,
                          small_size=14))
    stats.append(stat(s, M + Inches(8.4), sy, Inches(3.4), "2.9 MB", "per request: two uncompressed camera images",
                      size=30, small_size=14, color=S.ink))
    builds(s, [[(b, "fade", 150 * i) for i, b in enumerate(boxes)], flow, stats[0] + stats[1] + stats[2], pills])
    notes(s, "Each request carries two camera images to the server and gets back a chunk of actions. The round trip "
             "is about 0.3 s for SmolVLA and 0.45 s for GR00T; most of it is the network, not the model. Then the "
             "three faults we inject: a latency spike, dropped replies, and an outage.")


def s_shipped(prs, n):
    s = S.content(prs, n, "The problem · what LeRobot ships", "One missed second ends the task")
    rules = [("Requests late:", "a new chunk only when about one round trip of actions is left."),
             ("One request in flight:", "a lost reply is never sent again."),
             ("Gives up:", "after 1 s without an action, the run is ended.")]
    groups = [para(s, M, TOP + Inches(0.05) + Inches(0.62) * i, CW, a, b, size=19, marker=S.muted)
              for i, (a, b) in enumerate(rules)]
    # the fault mix on a time axis (first run: the same windows in every trial)
    x0, x1 = M + Inches(1.6), W - M
    ty = TOP + Inches(2.55)
    th = Inches(1.35)

    def tx(t):
        return int(x0 + (x1 - x0) * t / 10)
    axis = [line(s, x0, ty + th, x1, ty + th, color=S.ink, weight=Pt(1))]
    for t in range(0, 11, 2):
        axis.append(line(s, tx(t), ty + th, tx(t), ty + th + Inches(0.07), color=S.ink, weight=Pt(1)))
        axis.append(text(s, tx(t) - Inches(0.3), ty + th + Inches(0.1), Inches(0.6), Inches(0.3), f"{t} s",
                         size=12, color=S.muted, font=S.body, align=PP_ALIGN.CENTER))
    axis.append(text(s, M, ty + Inches(0.4), Inches(1.5), Inches(0.6), [["The fault mix"], [("each trial", {
        "size": 12, "color": S.muted})]], size=15, bold=True, font=S.body, color=S.ink))
    windows = []
    for a, b, name, what in [(2, 3, "spike", "+1 s latency"), (4, 5, "drops", "every reply lost"),
                             (6, 7.5, "outage", "no link, 1.5 s")]:
        r = box(s, tx(a), ty, tx(b) - tx(a), th, fill=S.alert, alpha=0.18)
        t1 = text(s, tx(a), ty + Inches(0.15), tx(b) - tx(a) + Inches(0.8), Inches(0.3), name, size=15, bold=True,
                  color=S.alert, font=S.body, align=PP_ALIGN.LEFT)
        t2 = text(s, tx(a), ty + Inches(0.45), tx(b) - tx(a) + Inches(0.8), Inches(0.5), what, size=12,
                  color=S.muted, font=S.body)
        windows.append([r, t1, t2])
    builds(s, groups + [axis] + windows)
    notes(s, "This is the baseline: LeRobot 0.7's remote inference, unchanged. It asks late, keeps one request in "
             "flight, and ends the run after one second without actions. Our fault mix: a 1 s latency spike at 2 s, "
             "all replies dropped from 4 to 5 s, and a full outage from 6 to 7.5 s. Any one of them is enough to "
             "end the run.")


def s_claim(prs, n):
    s = S.statement_slide(prs, n, "The problem · our claim")
    items = [("Network faults lower the task score", "of client-server robot inference, even when the client "
              "waits instead of giving up."),
             ("DRTC recovers it,", "more than just waiting does."),
             ("Smoothing then improves the motion", "without lowering success.")]
    groups = []
    for i, (a, b) in enumerate(items):
        y = Inches(1.85) + Inches(1.45) * i
        groups.append([text(s, M, y, Inches(0.9), Inches(0.9), str(i + 1), size=40, bold=True, font=S.heading,
                            color=S.on_accent_muted),
                       text(s, M + Inches(0.9), y, CW - Inches(0.9), Inches(1.2),
                            [[(a, {"bold": True, "color": S.on_accent})], [(b, {"color": S.on_accent_muted,
                                                                                 "size": 20})]],
                            size=30, font=S.heading, line=1.15)])
    builds(s, groups)
    notes(s, "Three claims, each tested against a named comparison. The first run tested all three against "
             "LeRobot as shipped; setup v2 tests them against a patient client.")


def s_drtc(prs, n):
    s = S.content(prs, n, "Method · DRTC", "Distributed Real-Time Chunking, in four ideas")
    cw, gap = Inches(2.75), Inches(0.28)
    ch = Inches(2.85)
    items = [("1", "Request early", "when the queue is down to H − s_min actions, not one round trip."),
             ("2", "Estimate the latency", "as TCP does (Jacobson–Karels): a smoothed mean plus k × the deviation."),
             ("3", "Re-request when late", "after a cooldown, send a fresh observation; the freshest reply wins."),
             ("4", "Hold, then carry on", "no 1 s give-up: the arm holds through a stall and resumes.")]
    groups = [card(s, M + (cw + gap) * i, TOP + Inches(0.15), cw, ch, *it) for i, it in enumerate(items)]
    foot = text(s, M, TOP + ch + Inches(0.55), CW, Inches(0.8),
                [[("Our port runs inside LeRobot 0.7's own runtime, ", {"color": S.ink}),
                  ("so every upstream safety check stays. H = the chunk length; s_min = a safety margin in steps.",
                   {"color": S.muted})]], size=16, font=S.body)
    builds(s, groups + [foot])
    notes(s, "DRTC (Vial, 2026) is a scheduler for exactly this setting. It asks for the next chunk early, keeps a "
             "running latency estimate like TCP, and if a reply is late it gives up on it and sends a fresh "
             "observation instead of waiting. And it never ends the run: the arm holds and resumes.")


def s_fixes(prs, n):
    s = S.content(prs, n, "Method · making it work on the arm", "Four fixes, each found on the real setup")
    rows = [["GR00T never got a chunk", "the cooldown was capped at H/2 = 0.27 s; replies take 0.4–0.7 s",
             "the cooldown is never shorter than the measured latency"],
            ["A storm of re-requests", "replies near the estimate were given up; the server slowed further",
             "the cooldown also covers the slowest of the last 10 replies"],
            ["Never started on a slow link", "before the first reply it waited only s_min + ε steps",
             "until the first reply, wait up to the start-up timeout"],
            ["“Invalid RTC delay” errors", "the requested delay could exceed the server's chunk",
             "cap the delay at the server's prediction steps"]]
    groups = table(s, M, TOP + Inches(0.1), [Inches(3.0), Inches(4.6), Inches(4.23)], ["PROBLEM", "CAUSE", "FIX"],
                   rows, size=15, row_h=Inches(0.85))
    builds(s, groups)
    notes(s, "DRTC as published did not run on our setup. The biggest one: GR00T's chunk is only 16 actions, so "
             "DRTC's cooldown cap gave up on every request before its reply arrived. Each fix was found on the "
             "arm and kept as small as possible.")


def s_smoothing(prs, n):
    s = S.content(prs, n, "Method · smoothing", "Smooth motion, even when chunks arrive late")
    half = (CW - Inches(0.5)) // 2
    t1 = text(s, M, TOP, half, Inches(0.3), "IN THE FIRST RUN", size=11, bold=True, color=S.muted, spacing=1.4,
              font=S.body)
    c1 = card(s, M, TOP + Inches(0.4), half, Inches(1.85), "", "Butterworth filter",
              "a causal low-pass per joint (order 2, 3 Hz at 30 Hz); it keeps its state across chunks, which "
              "smooths the seam between them.")
    c2 = card(s, M, TOP + Inches(2.45), half, Inches(1.85), "", "Adaptive playback",
              "when the queue would run dry before the next chunk, play slower instead of stopping; normal speed "
              "with a full queue.")
    x2 = M + half + Inches(0.5)
    t2 = text(s, x2, TOP, half, Inches(0.3), "ADDED IN SETUP V2", size=11, bold=True, color=S.muted, spacing=1.4,
              font=S.body)
    more = [("RTC", "the server continues the actions already running (SmolVLA, GR00T N1.5)."),
            ("Temporal ensembling", "each step averages every chunk that predicted it."),
            ("Zero-lag smoothing", "forward-backward filtering of the queue: no lag."),
            ("Resume ramp", "a minimum-jerk blend out of a stall instead of a jump.")]
    rows = [para(s, x2, TOP + Inches(0.45) + Inches(0.78) * i, half, a, b, size=15, h=Inches(0.75),
                 marker=S.accent) for i, (a, b) in enumerate(more)]
    pick = text(s, x2, TOP + Inches(3.65), half, Inches(0.7),
                "The bench picks a stack by a rule fixed before it ran: lowest jerk, unless the arm drifts from the "
                "plan or stalls get worse.", size=14, color=S.muted, font=S.body, italic=True)
    builds(s, [[t1] + c1, c2, [t2] + rows[0], rows[1], rows[2], rows[3], pick])
    notes(s, "Late chunks make the arm jerk and stall. The first run used two methods: a low-pass filter and "
             "adaptive playback, which slows down instead of stopping. Setup v2 adds four more and picks a "
             "combination on the bench by a rule we fixed in advance.")


def s_platform(prs, n):
    s = S.content(prs, n, "Experiments · platform", "A low-cost arm, a laptop, and a shared GPU")
    rows = [["Arm", "SO-101 follower, 6 servos; the leader arm records demonstrations"],
            ["Cameras", "wrist camera and a phone as the scene camera, 800 × 600 at 30 fps"],
            ["Robot computer", "the laptop (RTX 4060 8 GB) in every experiment"],
            ["Policy server", "A100 80 GB, shared, over Wi-Fi and the local network"],
            ["Training", "H200 and A100; logged to wandb and Trackio"],
            ["Software", "LeRobot 0.7.0 (pinned), PyTorch 2.14; our lab tooling on top"]]
    groups = table(s, M, TOP, [Inches(1.9), Inches(4.9)], ["PART", "WHAT"], rows, size=14, row_h=Inches(0.62))
    still = None
    v = trial_video("baseline", MODELS[0][2], True)
    if v:
        still = poster(v, at=2.0)
    pw = Inches(4.6)
    photo(s, W - M - pw, TOP, pw, int(pw * 3 / 4), "setup.jpg", "Photo: the whole setup", still)
    text(s, W - M - pw, TOP + int(pw * 3 / 4) + Inches(0.1), pw, Inches(0.4), "The scene camera's view of the task.",
         size=12, color=S.muted, font=S.body)
    builds(s, [sum(groups, [])])
    notes(s, "Everything is low-cost: a 3D-printed SO-101 arm, a webcam on the wrist and a phone as the scene "
             "camera. The laptop is the robot computer in every experiment; the policy runs on a shared A100.")


def s_data(prs, n):
    s = S.content(prs, n, "Experiments · data", "Two tasks, recorded by teleoperation")
    pw = Inches(3.2)
    ph = int(pw * 3 / 4)
    v = trial_video("baseline", MODELS[0][2], True)
    photo(s, M, TOP, pw, ph, "task_sort_one.jpg", "Photo: the sorting task", poster(v, at=0.5) if v else None)
    photo(s, M + pw + Inches(0.3), TOP, pw, ph, "task_cubebox.jpg", "Photo: the cube-in-box task")
    for i, cap in enumerate(["Sorting: the cube to the square of its colour", "Cube in box"]):
        text(s, M + (pw + Inches(0.3)) * i, TOP + ph + Inches(0.08), pw, Inches(0.4), cap, size=12, color=S.muted,
             font=S.body)
    x = M + 2 * pw + Inches(0.75)
    w = W - M - x
    ds = [("so101_sort_one_v1", "Sorting", "the DRTC experiment"),
          ("so101_cubebox_v1", "Cube in box v1", "baselines, bench"),
          ("so101_cubebox_v0", "Cube in box v0", "early tests")]
    rows = [[name, R(f"data/{d}/episodes"), R(f"data/{d}/held"), R(f"data/{d}/frames")] for d, name, _ in ds]
    groups = table(s, x, TOP, [Inches(1.5), Inches(0.95), Inches(0.95), w - Inches(3.4)],
                   ["DATASET", "EPISODES", "HELD OUT", "FRAMES"], rows, size=14, row_h=Inches(0.55), highlight=(0,))
    text(s, x, TOP + Inches(2.5), w, Inches(1.4),
         "Recorded with the leader arm at 30 fps; idle starts and ends trimmed automatically. Held-out episodes "
         "are never trained on and are used for the offline test.", size=14, color=S.muted, font=S.body, line=1.2)
    builds(s, [sum(groups, [])])
    notes(s, "Two tasks. Sorting is the main one: put the cube in the square of the same colour. Each dataset keeps "
             "about 10% of the episodes aside for the offline test.")


def s_models(prs, n):
    s = S.content(prs, n, "Experiments · policies", "Seven policies, each with its authors' recipe")
    rows = [["SmolVLA", "action expert only, VLM frozen", "20k × 64", "50"],
            ["GR00T N1.5", "projector + diffusion transformer", "10k × 32", "16"],
            ["GR00T N1.7", "projector + DiT + VLLN; relative actions", "20k × 64", "16"],
            ["π0.5", "full; relative joints", "20k × 64", "50"],
            ["MolmoAct 2", "LoRA 64 on the VLM; connector + action expert", "50k × 16", "30"],
            ["ACT", "full", "100k × 8", "100"],
            ["Diffusion Policy", "full, ResNet-18 from scratch", "600 epochs × 64", "24 of 48"]]
    groups = table(s, M, TOP - Inches(0.1), [Inches(2.4), Inches(5.0), Inches(2.3), Inches(2.13)],
                   ["POLICY", "FINE-TUNING", "STEPS × BATCH", "CHUNK (ACTIONS)"], rows, size=14, row_h=Inches(0.5),
                   highlight=(0, 1))
    text(s, M, S.bottom - Inches(0.35), CW, Inches(0.4),
         [[("Blue: ", {"color": S.accent, "bold": True}),
           ("the two policies in the DRTC experiment. Recipes are fixed, so differences come from the models, not "
            "from our tuning.", {"color": S.muted})]], size=13, font=S.body)
    builds(s, [sum(groups, [])])
    notes(s, "Seven policies, from small (ACT) to 3-billion-parameter VLAs. We never tuned them: each uses its "
             "authors' published recipe. SmolVLA and GR00T N1.5 are the two in the DRTC experiment; note GR00T's "
             "short chunk of 16 actions, about half a second.")


def s_protocol(prs, n):
    s = S.content(prs, n, "Experiments · evaluation", "Every trial planned, logged and judged the same way")
    steps = [("Freeze", "task, success rule, models pinned to their Hub commits, conditions: before trial 1"),
             ("Rounds", "every model × condition once per round, in a seeded random order"),
             ("Judge", "the operator presses s or f within 30 s; a run the system ends is a fault failure"),
             ("Report", "raw files to a private store; 95% intervals and significance tests")]
    bw, gap = Inches(2.6), Inches(0.475)
    groups = []
    for i, (a, b) in enumerate(steps):
        x = M + (bw + gap) * i
        g = card(s, x, TOP + Inches(0.3), bw, Inches(2.9), f"{i + 1}", a, b)
        if i:
            g.insert(0, (line(s, x - gap + Inches(0.08), TOP + Inches(1.75), x - Inches(0.08), TOP + Inches(1.75),
                              color=S.ink, weight=Pt(1.5), arrow=True), "wipe_right"))
        groups.append(g)
    foot = text(s, M, TOP + Inches(3.55), CW, Inches(0.9),
                [[("Metrics v3: ", {"bold": True, "color": S.ink}),
                  ("success, stalls, recovery time, jerk, largest step, turnaround, deviation from the plan. Each "
                   "eval names its primary comparisons (Holm-corrected); the rest are exploratory.",
                   {"color": S.muted})]], size=15, font=S.body, line=1.15)
    builds(s, groups + [foot])
    notes(s, "The evaluation is fixed before the first trial so it cannot drift toward the result. Rounds spread "
             "slow drift like light or motor heat over all conditions. Every trial keeps its raw files and video.")


def s_design(prs, n):
    s = S.content(prs, n, "Experiments · the first run", "Four conditions, two policies, ten trials each")
    groups = []
    for i, (c, _, name, what) in enumerate(CONDS):
        y = TOP + Inches(0.1) + Inches(0.95) * i
        g = [box(s, M, y, Inches(0.62), Inches(0.62), fill=S.series[c], runs=str(i + 1), size=20, bold=True,
                 color="#FFFFFF", font=S.heading)]
        g.append(text(s, M + Inches(0.85), y - Inches(0.02), Inches(6.5), Inches(0.4), name, size=20, bold=True,
                      color=S.ink, font=S.body))
        g.append(text(s, M + Inches(0.85), y + Inches(0.33), Inches(6.5), Inches(0.4), what, size=14,
                      color=S.muted, font=S.body))
        groups.append(g)
    x = M + Inches(8.0)
    big = stat(s, x, TOP + Inches(0.1), Inches(3.8), R("main/n") + " trials", "2 policies × 4 conditions × 10, "
               "sorting task, laptop → A100, 20 November 2026", size=40)
    builds(s, groups + [big])
    notes(s, "The first run compared four conditions: the shipped client with and without faults, DRTC under "
             "faults, and DRTC with smoothing under faults. These colours stay the same in every chart.")


def s_success(prs, n):
    s = S.content(prs, n, "Results · task success", "Network faults break the shipped client; DRTC recovers")
    y = TOP + Inches(0.05)
    ph, pw = Inches(4.6), Inches(3.55)
    bars = {}
    for j, (key, model, _) in enumerate(MODELS):
        data = []
        for c, _, _, _ in CONDS:
            kn = K(f"main/{key}/{c}/M1k") or (0, 0)
            lo, hi = wilson(*kn) if kn[1] else (0, 0)
            label = ["DRTC +", "smooth"] if c == "smooth" else {"baseline": "Baseline", "faults": "Faults",
                                                                 "drtc": "DRTC"}[c]
            data.append((label, 100 * kn[0] / kn[1] if kn[1] else 0, 100 * lo, 100 * hi, S.series[c]))
        _, per = bar_panel(s, M + j * (pw + Inches(0.35)), y, pw, ph, model, data, ink=S.ink, muted=S.muted,
                           grid=S.grid, font=S.body)
        bars[key] = per
    cx = Inches(8.85)
    cw = W - M - cx

    def pair(c):
        return " · ".join(R(f"main/{k}/{c}/M1") for k, _, _ in MODELS)
    p_drtc = R("main/smolvla/p/faults-drtc/M1")
    notes_ = [(pair("baseline"), "Baseline: normal network.", "baseline"),
              (pair("faults"), "Faults: the client ended all 20 runs at the first fault.", "faults"),
              (pair("drtc"), f"DRTC, same faults (SmolVLA {p_drtc} against faults).", "drtc"),
              (pair("smooth"), "DRTC with smoothing.", "smooth")]
    steps = []
    for i, (big, small, c) in enumerate(notes_):
        ny = y + Inches(0.15) + Inches(1.13) * i
        mk = box(s, cx, ny + Inches(0.17), Inches(0.17), Inches(0.17), fill=S.series[c])
        a = text(s, cx + Inches(0.32), ny, cw, Inches(0.5), big, size=26, font=S.heading,
                 color=S.accent if c in ("drtc", "smooth") else S.ink)
        b = text(s, cx + Inches(0.32), ny + Inches(0.5), cw - Inches(0.32), Inches(0.55), small, size=13,
                 color=S.muted, font=S.body, line=1.1)
        step = []
        for j, (key, _, _) in enumerate(MODELS):
            step += [(sh, "wipe_up" if k == 0 else "fade", 150 * j + (0 if k == 0 else 250))
                     for k, sh in enumerate(bars[key][i])]
        steps.append(step + [(mk, "fade", 300), (a, "fade", 300), (b, "fade", 300)])
    builds(s, steps)
    notes(s, "Bars appear one condition at a time; whiskers are Wilson 95% intervals over 10 trials. Without faults: "
             "90% and 70%. With faults the shipped client fails every run: it ends the run at the first fault. DRTC "
             "under the same faults: 60% and 80%, significant against the faulted baseline. With smoothing: 80% and "
             "100%, higher but not significant with 10 trials.")


def s_videos(prs, n):
    s = S.content(prs, n, "Results · the same faults, three ways", "Giving up, recovering, recovering smoothly")
    vw, gap = Inches(3.75), Inches(0.29)
    vh = int(vw * 3 / 4)
    y = TOP - Inches(0.05)
    picks = [("faults", "baseline_faults", False, "Shipped", "stops at the spike; ended 1 s later."),
             ("drtc", "drtc_faults", True, "DRTC", "holds through each fault, then finishes."),
             ("smooth", "drtc_smooth_faults", True, "DRTC + smoothing", "slows instead of stopping.")]
    for i, (c, d, ok, who, what) in enumerate(picks):
        x = M + (vw + gap) * i
        v = trial_video(d, MODELS[0][2], ok)
        if v:
            movie(s, v, x, y, vw, vh)
        para(s, x, y + vh + Inches(0.15), vw, who, what, size=14, h=Inches(0.8), marker=S.series[c])
    notes(s, "SmolVLA, the same fault mix in all three. Left: the shipped client stops at the spike and ends the run. "
             "Middle: DRTC holds through each fault and finishes. Right: with smoothing it slows down instead of "
             "stopping. Click each video to play.")


def s_motion(prs, n):
    s = S.content(prs, n, "Results · motion", "Smoothing makes the motion 5 to 11 times smoother")
    fw = Inches(7.4)
    picture(s, FIGS / "fig_main_jerk.png", M, TOP + Inches(0.1), w=fw)
    x, w = M + fw + Inches(0.5), W - M - (M + fw + Inches(0.5))
    items = [(f"{F('main/smolvla/jerkratio') or 0:.0f}× smoother",
              f"SmolVLA: jerk {R('main/smolvla/drtc/M7')} → {R('main/smolvla/smooth/M7')} "
              f"({R('main/smolvla/p/drtc-smooth/M7')})"),
             (f"{F('main/groot/jerkratio') or 0:.0f}× smoother",
              f"GR00T N1.5: {R('main/groot/drtc/M7')} → {R('main/groot/smooth/M7')} "
              f"({R('main/groot/p/drtc-smooth/M7')})"),
             (f"{R('main/groot/drtc/M2')} → {R('main/groot/smooth/M2')}",
              "of GR00T's trial without new actions: adaptive playback slows down instead of stopping")]
    groups = [stat(s, x, TOP + Inches(0.15) + Inches(1.45) * i, w, a, b, size=28, small_size=14)
              for i, (a, b) in enumerate(items)]
    builds(s, groups)
    notes(s, "Jerk is the third derivative of the commanded joints, measured only while the arm moves; lower is "
             "smoother. Each dot is a trial. Smoothing cuts it about 5 times for SmolVLA and 11 times for GR00T. "
             "GR00T stalls even without faults because its chunk is barely longer than a round trip; adaptive "
             "playback fixes most of that.")


def s_timeline(prs, n):
    s = S.content(prs, n, "Results · one trial each", "Stopping, recovering, and moving smoothly")
    fh = S.bottom - TOP - Inches(0.05)
    pic = picture(s, FIGS / "fig_main_timeline.png", M, TOP - Inches(0.05), h=fh)
    x = M + pic.width + Inches(0.45)
    w = W - M - x
    items = [("faults", "Shipped", "holds at the spike, then the run is ended."),
             ("drtc", "DRTC", "holds during each fault, resumes, finishes."),
             ("smooth", "DRTC + smoothing", "the same, as one smooth curve.")]
    groups = [para(s, x, TOP + Inches(1.6) + Inches(1.0) * i, w, a, b, size=15, h=Inches(0.9), marker=S.series[c])
              for i, (c, a, b) in enumerate(items)]
    builds(s, groups)
    notes(s, "One SmolVLA trial per condition: the commanded shoulder angle over time, with the fault windows "
             "shaded. Flat lines are holds. The shipped client is ended during the drops; DRTC holds and resumes; "
             "with smoothing the same path becomes one smooth curve.")


def s_recovery(prs, n):
    s = S.content(prs, n, "Results · recovery", "DRTC recovered from every fault, in every counted run")
    rows = []
    for c, label in [("faults", "Shipped + faults"), ("drtc", "DRTC + faults"), ("smooth", "DRTC + smoothing")]:
        for key, model, _ in MODELS:
            rows.append([label if key == "smolvla" else "", model] +
                        [R(f"main/{key}/{c}/M11/{f}").replace(", ", "   ") for f in ("spike", "drops", "outage")])
    groups = table(s, M, TOP, [Inches(2.8), Inches(2.2), Inches(2.3), Inches(2.3), Inches(2.23)],
                   ["CONDITION", "POLICY", "SPIKE", "DROPS", "OUTAGE"], rows, size=15, row_h=Inches(0.5))
    text(s, M, TOP + Inches(4.0), CW, Inches(0.6),
         "Recovered runs / runs the fault reached, and the median time from the fault's end until the arm moves "
         "again. A run that had already succeeded is not counted.", size=13, color=S.muted, font=S.body)
    builds(s, [groups[0] + groups[1], groups[2] + groups[3], groups[4] + groups[5]])
    notes(s, "The shipped client never recovered: its first fault ended every run. DRTC recovered from every fault "
             "in every run that reached it. With smoothing, GR00T's recovery after the outage fell from 1.45 to "
             "0.65 seconds.")


def s_bench(prs, n):
    s = S.content(prs, n, "Results · the bench", "One fault alone is enough to end the shipped client's run")
    fh = S.bottom - TOP
    pic = picture(s, FIGS / "fig_bench.png", M, TOP - Inches(0.05), h=fh)
    x = M + pic.width + Inches(0.45)
    w = W - M - x
    items = [("7–8 of 8", "runs of the shipped client ended by a single fault, of any type."),
             ("0 of 8", "under DRTC, for every fault type and both policies."),
             ("lowest jerk", "with smoothing, under every fault type.")]
    groups = [stat(s, x, TOP + Inches(0.3) + Inches(1.45) * i, w, a, b, size=28, small_size=14,
                   color=S.alert if i == 0 else S.accent) for i, (a, b) in enumerate(items)]
    builds(s, groups)
    notes(s, "The bench replays held-out episodes as camera input, with no arm, and a simulated 250 ms network. "
             "Each fault of the mix on its own, 8 runs per cell. Even one 1-second latency spike ends the shipped "
             "client's run. No task success here, only motion and stalls.")


def s_offline(prs, n):
    s = S.content(prs, n, "Results · offline", "Held-out error and inference cost")
    models = [("smolvla", "SmolVLA"), ("groot", "GR00T N1.5"), ("act", "ACT"), ("diffusion", "Diffusion Policy")]
    rows = [[name, R(f"offline/sortone/{k}/M9arm") + "°", R(f"offline/sortone/{k}/M9grip"),
             f"{R(f'offline/sortone/{k}/M10')}  (p95 {R(f'offline/sortone/{k}/M10p95')})",
             R(f"offline/sortone/{k}/M10gb")] for k, name in models]
    groups = table(s, M, TOP, [Inches(2.6), Inches(2.1), Inches(1.9), Inches(3.3), Inches(1.93)],
                   ["POLICY (SORTING)", "ARM ERROR", "GRIPPER", "INFERENCE (MEDIAN)", "GPU MEMORY"], rows, size=16,
                   row_h=Inches(0.58))
    text(s, M, TOP + Inches(3.3), CW, Inches(1.2),
         [[("Lowest error: SmolVLA. Fastest and smallest: ACT. ", {"color": S.ink, "bold": True}),
           ("Error = mean absolute error of the first 15 predicted actions on the held-out episodes. Inference on "
            "the shared H200, 200 requests each. GR00T N1.7, π0.5 and MolmoAct 2 follow when their tests have run.",
            {"color": S.muted})]], size=14, font=S.body, line=1.2)
    builds(s, [sum(groups, [])])
    notes(s, "Offline, without the robot: how close each policy's actions are to the held-out demonstrations, and "
             "what it costs to run. SmolVLA is the most accurate on sorting; ACT is the fastest and smallest.")


def s_redesign(prs, n):
    s = S.content(prs, n, "Outlook · setup v2", "Why we redesigned the experiment")
    half = (CW - Inches(0.6)) // 2
    left = [text(s, M, TOP, half, Inches(0.3), "THE FIRST RUN", size=11, bold=True, color=S.muted, spacing=1.4,
                 font=S.body),
            text(s, M, TOP + Inches(0.45), half, Inches(2.6),
                 [[("Every faulted baseline run was ended by LeRobot's own 1 s give-up.", {"color": S.ink,
                                                                                             "bold": True})],
                  [("So DRTC was compared with a client that gives up by design, and it differed in two things at "
                    "once: its scheduling and its patience.", {"color": S.muted})]], size=18, font=S.body, line=1.2)]
    x2 = M + half + Inches(0.6)
    items = [("Baseline:", "upstream patient, LeRobot's scheduler with both give-up limits raised to the whole "
              "trial."),
             ("Bench E1–E2:", "2 tasks, 5 policies, 7 fault types; DRTC without each of its parts."),
             ("Bench E3:", "each smoothing method alone, then the chosen stack."),
             ("Robot E4:", "patient upstream with and without faults, DRTC, DRTC + stack.")]
    t2 = text(s, x2, TOP, half, Inches(0.3), "SETUP V2", size=11, bold=True, color=S.accent, spacing=1.4,
              font=S.body)
    rows = [para(s, x2, TOP + Inches(0.45) + Inches(0.8) * i, half, a, b, size=15, h=Inches(0.75),
                 marker=S.accent) for i, (a, b) in enumerate(items)]
    found = text(s, M, TOP + Inches(3.85), CW, Inches(0.7),
                 [[("Found while building it: ", {"bold": True, "color": S.ink}),
                   ("upstream never re-sends a lost request, so even a patient client waited to the end of the run. "
                    "Replacing a lost reply is exactly what DRTC's cooldown adds.", {"color": S.muted})]],
                 size=15, font=S.body, line=1.15)
    builds(s, [left, [t2] + rows[0], rows[1], rows[2], rows[3], found])
    notes(s, "A fair critique of the first run: the baseline gave up by design. Setup v2 compares DRTC with a client "
             "that waits as long as the trial lasts, and takes DRTC apart on the bench to see which piece matters.")


def s_v2_results(prs, n):
    s = S.content(prs, n, "Outlook · setup v2 results", "DRTC against a patient client")
    bw, gap = Inches(2.75), Inches(0.28)
    items = [("E1", "DRTC vs patient upstream, per fault: stalls, recovery"),
             ("E2", "DRTC without each part: which piece matters"),
             ("E3", "smoothing methods alone, then the stack"),
             ("E4", "the robot: the three claims, task success")]
    groups = []
    for i, (e, what) in enumerate(items):
        x = M + (bw + gap) * i
        groups.append([text(s, x, TOP + Inches(0.1), bw, Inches(0.5), e, size=26, font=S.heading, color=S.accent),
                       pending_box(s, x, TOP + Inches(0.7), bw, Inches(2.4), what)])
    text(s, M, TOP + Inches(3.5), CW, Inches(0.6), "These panels fill in from the results as the runs finish.",
         size=15, color=S.muted, font=S.body, italic=True)
    builds(s, groups)
    notes(s, "Setup v2 is built and frozen; the runs are in progress. (This slide is replaced by the results.)")


def s_edge(prs, n):
    s = S.content(prs, n, "Outlook · the edge", "A Raspberry Pi 5 as the robot computer")
    items = [(R("edge/hz"), "control rate on the Pi (CPU PyTorch), no video frames dropped", S.accent),
             (R("edge/smolvla/rtt"), f"SmolVLA round trip from the Pi (laptop: {R('main/smolvla/baseline/M4med')})",
              S.alert),
             (R("edge/groot/rtt"), f"GR00T N1.5 round trip (laptop: {R('main/groot/baseline/M4med')})", S.alert)]
    groups = [stat(s, M + Inches(4.0) * i, TOP + Inches(0.2), Inches(3.7), a, b, color=c, size=36)
              for i, (a, b, c) in enumerate(items)]
    foot = text(s, M, TOP + Inches(2.4), CW, Inches(1.2),
                [[("The round trip about doubles, ", {"bold": True, "color": S.ink}),
                  ("so the shipped client starved GR00T even without faults: the case DRTC and image compression are "
                   "for. Set up and tested; the official edge trials have not run yet.", {"color": S.muted})]],
                size=18, font=S.body, line=1.2)
    builds(s, groups + [foot])
    notes(s, "We also set up a Raspberry Pi 5 as the robot computer, with the same commands. It keeps 30 Hz, but "
             "the round trip doubles over Wi-Fi, which is exactly where scheduling matters most.")


def s_limits(prs, n):
    s = S.content(prs, n, "Outlook · limits and next steps", "What these results do not show yet")
    half = (CW - Inches(0.6)) // 2
    lim = [("10 trials per cell:", "wide intervals; the smoothing gain in success is a trend."),
           ("One task, two policies", "on the robot; the bench covers more, without success."),
           ("The operator judges success,", "by a rule written before trial 1."),
           ("Shared GPUs:", "inference times include other users' jobs."),
           ("Uncompressed images:", "about 260 ms of each round trip.")]
    nxt = [("Setup v2", "results: DRTC against a patient client."),
           ("The cube-in-box task", "in the DRTC experiment."),
           ("JPEG images", "to shorten the round trip."),
           ("RTC for GR00T N1.7", "and the later policies."),
           ("An edge fallback:", "a small policy on the robot while the server is away.")]
    t1 = text(s, M, TOP, half, Inches(0.3), "LIMITS", size=11, bold=True, color=S.muted, spacing=1.4, font=S.body)
    t2 = text(s, M + half + Inches(0.6), TOP, half, Inches(0.3), "NEXT", size=11, bold=True, color=S.accent,
              spacing=1.4, font=S.body)
    a = [para(s, M, TOP + Inches(0.45) + Inches(0.78) * i, half, x, y, size=15, h=Inches(0.75), marker=S.muted)
         for i, (x, y) in enumerate(lim)]
    b = [para(s, M + half + Inches(0.6), TOP + Inches(0.45) + Inches(0.78) * i, half, x, y, size=15,
              h=Inches(0.75), marker=S.accent) for i, (x, y) in enumerate(nxt)]
    builds(s, [[t1] + sum(a, []), [t2] + sum(b, [])])
    notes(s, "Be upfront: 10 trials per cell gives wide intervals, so only large effects are significant. The "
             "next steps are mostly running what is already built.")


def s_conclusion(prs, n):
    s = S.content(prs, n, "Conclusion", "Three results")
    items = [("Faults break remote inference.",
              f"With LeRobot as shipped: {R('main/smolvla/baseline/M1')} → {R('main/smolvla/faults/M1')} (SmolVLA), "
              f"{R('main/groot/baseline/M1')} → {R('main/groot/faults/M1')} (GR00T N1.5)."),
             ("DRTC recovers the task.",
              f"{R('main/smolvla/drtc/M1')} and {R('main/groot/drtc/M1')} under the same faults; every fault "
              "recovered."),
             ("Smoothing makes it smooth.",
              f"{F('main/smolvla/jerkratio') or 0:.0f}× and {F('main/groot/jerkratio') or 0:.0f}× lower jerk, "
              "without lowering success.")]
    groups = []
    for i, (a, b) in enumerate(items):
        y = TOP + Inches(0.15) + Inches(1.3) * i
        groups.append([text(s, M, y - Inches(0.08), Inches(0.9), Inches(0.8), str(i + 1), size=40, font=S.heading,
                            color=S.accent),
                       text(s, M + Inches(0.9), y, CW - Inches(0.9), Inches(1.0),
                            [[(a, {"bold": True, "color": S.ink})], [(b, {"color": S.muted, "size": 18})]], size=22,
                            font=S.body, line=1.2)])
    builds(s, groups)
    notes(s, "Three results, one per click. Then: setup v2 will test the same claims against a patient client.")


def s_thanks(prs, n):
    s = S.closing_slide(prs, n, "Thank you", "Questions?")
    notes(s, "Questions.")


SLIDES = [s_title, s_outline,
          s_section(1), s_problem, s_setting, s_shipped, s_claim,
          s_section(2), s_drtc, s_fixes, s_smoothing,
          s_section(3), s_platform, s_data, s_models, s_protocol, s_design,
          s_section(4), s_success, s_videos, s_motion, s_timeline, s_recovery, s_bench, s_offline,
          s_section(5), s_redesign, s_v2_results, s_edge, s_limits, s_conclusion, s_thanks]


def preview(pptx: Path) -> None:
    """A LibreOffice PDF and PNGs (static: no animations, substituted fonts) to check the layout."""
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(pptx)], check=True,
                   capture_output=True)
    for old in OUT.glob("slide-*.png"):
        old.unlink()
    subprocess.run(["pdftoppm", "-png", "-r", "50", str(pptx.with_suffix(".pdf")), str(OUT / "slide")], check=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    prs = new_deck()
    for n, make in enumerate(SLIDES, start=1):
        make(prs, n)
    path = OUT / "presentation.pptx"
    prs.save(path)
    print(path, f"({len(SLIDES)} slides)")
    if "--no-preview" not in sys.argv:
        preview(path)
        print(path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
