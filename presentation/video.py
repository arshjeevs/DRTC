"""The project video (about 2.5 minutes, 1920x1080, 30 fps), built from the real trials and the report's numbers.

    .venv/bin/python presentation/video.py            # -> presentation/out/drtc_video.mp4
    .venv/bin/python presentation/video.py --preview  # one still per scene -> presentation/out/video_preview/

Every clip is a recorded robot trial of the first run (sort_one_drtc_main); every number comes from the report's
generated numbers (results.py), so the video, the slides and the report agree. Frames are drawn with PIL in the deck's
style (style.py: paper and ink slides, Space Grotesk and Inter from presentation/fonts) and piped to ffmpeg. To change
the story, edit SCENES at the bottom.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from results import R, title_page
from style import Style

S = Style()
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "out"
FONTS = HERE / "fonts"
MAIN = REPO / "results" / "evals" / "sort_one_drtc_main" / "sessions"
W, H, FPS = 1920, 1080, 30
M = 128  # margin, as the browser deck
FADE = 0.5  # seconds of cross-fade between scenes


def hexrgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


INK, PAPER, MUTED, LINE = hexrgb(S.ink), hexrgb(S.bg), hexrgb(S.muted), hexrgb(S.grid)
DARK, DTEXT, DMUTED, DACC = hexrgb(S.dark), hexrgb(S.dark_text), hexrgb(S.dark_muted), hexrgb(S.dark_accent)
ACC, ALERT = hexrgb(S.accent), hexrgb(S.alert)
SER = {k: hexrgb(v) for k, v in S.series.items()}

_fonts: dict = {}


def font(kind: str, size: int):
    """kind: h (Space Grotesk Bold), hm (Space Grotesk Medium), b (Inter), bb (Inter Bold), bs (Inter SemiBold)."""
    files = {"h": "SpaceGrotesk-Bold.ttf", "hm": "SpaceGrotesk-Medium.ttf", "b": "Inter-Regular.ttf",
             "bb": "Inter-Bold.ttf", "bs": "Inter-SemiBold.ttf"}
    key = (kind, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(str(FONTS / files[kind]), size)
    return _fonts[key]


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def rise(t: float, start: float, dur: float = 0.5) -> tuple[float, int]:
    """(opacity, upward offset in px) of an element that rises in at `start` seconds."""
    a = ease((t - start) / dur)
    return a, int((1 - a) * 24)


# --- drawing ---------------------------------------------------------------------------------------------------

class Frame:
    def __init__(self, bg):
        self.im = Image.new("RGB", (W, H), bg)
        self.d = ImageDraw.Draw(self.im, "RGBA")

    def text(self, xy, s, f, fill, alpha=1.0, anchor="la", spacing=1.25, width=None):
        if alpha <= 0:
            return 0
        col = (*fill, int(255 * alpha))
        lines = wrap(s, f, width) if width else s.split("\n")
        x, y = xy
        lh = int(f.size * spacing)
        for i, ln in enumerate(lines):
            self.d.text((x, y + i * lh), ln, font=f, fill=col, anchor=anchor)
        return len(lines) * lh

    def eyebrow(self, xy, s, color, alpha=1.0):
        x, y = xy
        f = font("bb", 24)
        for ch in s.upper():  # letter-spaced caps
            self.d.text((x, y), ch, font=f, fill=(*color, int(255 * alpha)))
            x += f.getlength(ch) + 3

    def rect(self, box, fill=None, outline=None, width=2, radius=0, alpha=1.0):
        f = (*fill, int(255 * alpha)) if fill else None
        o = (*outline, int(255 * alpha)) if outline else None
        self.d.rounded_rectangle(box, radius=radius, fill=f, outline=o, width=width)

    def paste(self, img: Image.Image, xy, alpha=1.0, radius=16):
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, *img.size), radius=radius, fill=int(255 * alpha))
        self.im.paste(img, xy, mask)

    def footer(self, n_or_text, dark=False):
        self.text((M, H - 80), S_SHORT, font("b", 22), DMUTED if dark else MUTED)  # no numbers: scenes move


S_SHORT = "Reliable remote inference under network faults"


def wrap(s: str, f, width: int) -> list[str]:
    out = []
    for para in s.split("\n"):
        line = ""
        for w in para.split():
            trial = (line + " " + w).strip()
            if f.getlength(trial) <= width:
                line = trial
            else:
                out.append(line)
                line = w
        out.append(line)
    return out


def header(fr: Frame, t: float, kicker: str, title: str, dark=False):
    a, dy = rise(t, 0.0)
    fr.eyebrow((M, 96 + dy), kicker, DACC if dark else ACC, a)
    size = next((s for s in (64, 58, 52, 46) if font("h", s).getlength(title) <= W - 2 * M), 46)
    fr.text((M, 140 + dy + (64 - size) // 2), title, font("h", size), DTEXT if dark else INK, a, width=W - 2 * M)


# --- trials --------------------------------------------------------------------------------------------------------

def trial_dir(cond: str, model: str, k: int) -> Path:
    hits = sorted(MAIN.glob(f"*/trials/{cond}/{model}/t{k:02d}"))
    if not hits:
        raise SystemExit(f"no trial {cond}/{model}/t{k:02d} under {MAIN}")
    return hits[-1]


class Clip:
    """A trial's scene video, read frame by frame and scaled to `size`; holds its last frame after the end."""

    def __init__(self, d: Path, size: tuple[int, int], cam: str = "scene"):
        self.d, self.size = d, size
        self.cap = cv2.VideoCapture(str(d / "video" / f"{cam}.mp4"))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.i, self.last = -1, None
        self.info = json.loads((d / "trial.json").read_text())
        self.ticks = list(csv.DictReader(open(d / "ticks.csv")))

    @property
    def length(self) -> float:
        return self.n / FPS

    def at(self, t: float) -> Image.Image:
        want = min(int(t * FPS), self.n - 1)
        while self.i < want:
            ok, f = self.cap.read()
            if not ok:
                break
            self.i += 1
            self.last = f
        img = cv2.cvtColor(self.last, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2.resize(img, self.size, interpolation=cv2.INTER_AREA))

    def series(self, col: str) -> list[tuple[float, float]]:
        return [(float(r["t"]), float(r[col])) for r in self.ticks if r.get(col) not in (None, "")]


FAULTS = [("spike", 2.0, 3.0), ("drops", 4.0, 5.0), ("outage", 6.0, 7.5)]  # the first run's mix (no shift)


def timeline(fr: Frame, box, t: float, tmax: float, ended: float | None = None, alpha=1.0, dark=False):
    """The fault mix on the trial's clock with a playhead; `ended` marks where the client ended the run."""
    x0, y0, x1, y1 = box
    ink, mut = (DTEXT, DMUTED) if dark else (INK, MUTED)
    sx = lambda s: x0 + (x1 - x0) * s / tmax  # noqa: E731
    ym = y1 - 34
    for name, a, b in FAULTS:
        on = a <= t
        fr.rect((sx(a), y0 + 26, sx(b), ym - 6), fill=ALERT, alpha=alpha * (0.85 if on else 0.25), radius=6)
        fr.text((sx(a), y0), name, font("bs", 22), ALERT, alpha * (1.0 if on else 0.5))
    fr.d.line((x0, ym, x1, ym), fill=(*mut, int(255 * alpha)), width=2)
    for s in range(0, int(tmax) + 1, 2):
        fr.text((sx(s), ym + 8), f"{s} s", font("b", 20), mut, alpha, anchor="ma")
    px = sx(min(t, tmax))
    fr.d.line((px, y0 + 20, px, ym + 4), fill=(*ink, int(255 * alpha)), width=3)
    if ended is not None and t >= ended:
        ex = sx(ended)
        fr.d.line((ex, y0 + 20, ex, ym), fill=(*ALERT, int(255 * alpha)), width=4)


def queue_plot(fr: Frame, box, clip: Clip, t: float, color, label: str, alpha=1.0):
    """The action queue over time, drawn up to `t`: the arm stalls where it touches zero."""
    x0, y0, x1, y1 = box
    pts = clip.series("queue")
    if not pts:
        return
    hmax = max(v for _, v in pts) or 1
    tmax = 14.0
    fr.text((x0, y0 - 34), label, font("bs", 22), MUTED, alpha)
    fr.d.line((x0, y1, x1, y1), fill=(*LINE, int(255 * alpha)), width=2)
    xy = [(x0 + (x1 - x0) * min(s, tmax) / tmax, y1 - (y1 - y0) * v / hmax) for s, v in pts if s <= t]
    if len(xy) > 1:
        fr.d.line(xy, fill=(*color, int(255 * alpha)), width=4)
    stall = [(s, v) for s, v in pts if s <= t and v <= 0]
    for s, _ in stall:
        x = x0 + (x1 - x0) * min(s, tmax) / tmax
        fr.d.line((x, y1 - 6, x, y1), fill=(*ALERT, int(200 * alpha)), width=3)


# --- scenes: each is (duration in s, draw(t) -> Image) -------------------------------------------------------------

def scene_title():
    meta = title_page()

    def draw(t):
        fr = Frame(DARK)
        a, dy = rise(t, 0.1)
        fr.eyebrow((M, 150 + dy), "Major project", DACC, a)
        a, dy = rise(t, 0.3)
        fr.text((M, 200 + dy), meta["title"], font("h", 84), DTEXT, a, width=W - 2 * M, spacing=1.08)
        a, dy = rise(t, 0.7)
        fr.text((M, 560 + dy), meta["subtitle"], font("hm", 36), DMUTED, a, width=W - 2 * M)
        a, dy = rise(t, 1.1)
        fr.text((M, 800 + dy), meta["authors"], font("b", 32), DTEXT, a)
        fr.text((M, 850 + dy), f"{meta['guide']}  ·  {meta['place']}", font("b", 24), DMUTED, a)
        return fr.im
    return 6.0, draw


def scene_setup():
    clip = Clip(trial_dir("baseline", "groot_sort_one_v1_full", 9), (720, 540))

    def draw(t):
        fr = Frame(PAPER)
        header(fr, t, "The setting", "Big robot models run far from the robot")
        lines = [("A vision-language-action policy", "has 0.45–3.4 billion parameters: too big for the robot's own "
                  "computer at 30 actions per second."),
                 ("So a GPU server runs it.", "The robot sends two camera images and its joints; the server "
                  "answers with a chunk of the next actions."),
                 ("Every action crosses the network.", f"Round trip {R('main/smolvla/baseline/M4med')} for SmolVLA "
                  f"and {R('main/groot/baseline/M4med')} for GR00T N1.5; about 260 ms of it is the network.")]
        y = 330
        for i, (a_, b) in enumerate(lines):
            a, dy = rise(t, 0.6 + 0.5 * i)
            h1 = fr.text((M, y + dy), a_, font("bb", 34), INK, a, width=880)
            h2 = fr.text((M, y + dy + h1 + 4), b, font("b", 30), MUTED, a, width=880)
            y += h1 + h2 + 40
        a, dy = rise(t, 0.4)
        fr.paste(clip.at(t), (W - M - 720, 300 + dy), alpha=a)
        fr.text((W - M - 720, 300 + 540 + 16 + dy), "GR00T N1.5 on the normal network: it sorts the cube.",
                font("b", 24), MUTED, a)
        fr.footer("1")
        return fr.im
    return min(clip.length, 14.0), draw


def scene_failure():
    clip = Clip(trial_dir("baseline_faults", "groot_sort_one_v1_full", 9), (960, 720))
    end = clip.length

    def draw(t):
        fr = Frame(DARK)
        header(fr, t, "What goes wrong", "LeRobot's client gives up after one second", dark=True)
        a, dy = rise(t, 0.3)
        fr.paste(clip.at(min(t, end)), (M, 270 + dy), alpha=a)
        timeline(fr, (M + 1000, 300, W - M, 440), min(t, 9.0), 9.0, ended=end, alpha=a, dark=True)
        notes = [(2.0, "2 s  ·  a reply arrives 1 s late; the queue runs dry"),
                 (end, f"{end:.1f} s  ·  the client ends the run itself")]
        y = 520
        for when, s in notes:
            if t >= when:
                aa, ddy = rise(t, when)
                fr.text((M + 1000, y + ddy), s, font("bs", 30), ALERT if when == end else DTEXT, aa, width=650)
                y += 100
        if t > end + 0.3:
            aa, _ = rise(t, end + 0.3)
            fr.text((M + 1000, 800), f"Every faulted trial ended this way: success {R('main/groot/faults/M1')} "
                                     f"(GR00T N1.5) and {R('main/smolvla/faults/M1')} (SmolVLA).",
                    font("b", 28), DMUTED, aa, width=650)
        fr.footer("2", dark=True)
        return fr.im
    return end + 4.0, draw


def scene_drtc():
    clip = Clip(trial_dir("drtc_faults", "groot_sort_one_v1_full", 9), (880, 660))
    rec = list(csv.DictReader(open(clip.d / "recovery.csv")))
    resends = [float(r["t"]) for r in rec if r.get("action") == "cooldown_resend"]

    def draw(t):
        fr = Frame(PAPER)
        header(fr, t, "Our method · DRTC", "Ask early, hold, re-ask with a fresh view")
        a, dy = rise(t, 0.3)
        fr.paste(clip.at(t), (M, 290 + dy), alpha=a)
        x = M + 940
        timeline(fr, (x, 300, W - M, 430), t, 14.0, alpha=a)
        queue_plot(fr, (x, 520, W - M, 700), clip, t, SER["drtc"], "Actions queued (the arm stalls at zero)", a)
        caps = [(2.0, "Spike: the queue empties, the arm holds; the run goes on."),
                (4.3, "Lost replies: an overdue request is given up and a fresh observation is sent."),
                (7.6, "Outage over: the arm moves again within about a second.")]
        y = 750
        for when, s in caps:
            if t >= when:
                aa, ddy = rise(t, when)
                fr.text((x, y + ddy), s, font("b", 27), INK, aa, width=W - M - x)
                y += 76
        for r in resends:
            if r <= t:
                px = x + (W - M - x) * r / 14.0
                fr.d.ellipse((px - 7, 513, px + 7, 527), fill=(*ACC, 255))
        fr.footer("3")
        return fr.im
    return min(clip.length, 15.0) + 1.0, draw


def scene_three():
    k = 9
    clips = [Clip(trial_dir(c, "groot_sort_one_v1_full", k), (520, 390))
             for c in ("baseline_faults", "drtc_faults", "drtc_smooth_faults")]
    names = [("LeRobot as shipped", "faults"), ("DRTC", "drtc"), ("DRTC + smoothing", "smooth")]
    dur = min(15.0, max(c.length for c in clips)) + 1.0

    def draw(t):
        fr = Frame(PAPER)
        header(fr, t, "The same faults, three ways", "Giving up, recovering, recovering smoothly")
        gap = (W - 2 * M - 3 * 520) // 2
        for i, (c, (nm, key)) in enumerate(zip(clips, names)):
            a, dy = rise(t, 0.3 + 0.2 * i)
            x = M + i * (520 + gap)
            fr.rect((x, 300 + dy, x + 60, 308 + dy), fill=SER[key], alpha=a, radius=4)
            fr.text((x, 322 + dy), nm, font("bb", 30), INK, a)
            fr.paste(c.at(t), (x, 380 + dy), alpha=a)
        timeline(fr, (M, 820, W - M, 950), t, 15.0)
        fr.footer("4")
        return fr.im
    return dur, draw


def scene_motion():
    k = 9
    a_clip = Clip(trial_dir("drtc_faults", "groot_sort_one_v1_full", k), (10, 10))
    b_clip = Clip(trial_dir("drtc_smooth_faults", "groot_sort_one_v1_full", k), (10, 10))
    joint = "cmd.elbow_flex"
    A, B = a_clip.series(joint), b_clip.series(joint)
    lo = min(v for _, v in A + B) - 3
    hi = max(v for _, v in A + B) + 3

    def draw(t):
        fr = Frame(PAPER)
        header(fr, t, "Smoother motion", "Smoothing removes the jerks, not the task")
        x0, x1 = M, W - M - 560
        for j, (pts, key, nm, y0) in enumerate([(A, "drtc", "DRTC alone", 300), (B, "smooth", "DRTC + smoothing", 610)]):
            a, dy = rise(t, 0.3 + 0.3 * j)
            y1 = y0 + 240
            fr.text((x0, y0 - 40 + dy), f"{nm}: elbow command (°)", font("bs", 24), MUTED, a)
            fr.d.line((x0, y1 + dy, x1, y1 + dy), fill=(*LINE, int(255 * a)), width=2)
            for _, s0, s1 in FAULTS:
                fr.rect((x0 + (x1 - x0) * s0 / 15, y0 + dy, x0 + (x1 - x0) * s1 / 15, y1 + dy), fill=ALERT,
                        alpha=0.12 * a)
            prog = ease((t - 0.6) / 6.0) * 15
            xy = [(x0 + (x1 - x0) * s / 15, y1 + dy - (y1 - y0) * (v - lo) / (hi - lo)) for s, v in pts
                  if s <= prog]
            if len(xy) > 1:
                fr.d.line(xy, fill=(*SER[key], int(255 * a)), width=4)
        stats = [(R("main/groot/jerkratio") + "×", "less jerk for GR00T N1.5"),
                 (R("main/smolvla/jerkratio") + "×", "less jerk for SmolVLA"),
                 (f"{R('main/groot/drtc/M2')} → {R('main/groot/smooth/M2')}",
                  "of GR00T's time standing still: adaptive playback slows down instead of stopping")]
        y = 320
        for i, (big, small) in enumerate(stats):
            a, dy = rise(t, 2.0 + 0.6 * i)
            fr.text((W - M - 500, y + dy), big.replace("\\%", "%"), font("h", 60), ACC, a)
            h = fr.text((W - M - 500, y + 74 + dy), small, font("b", 26), MUTED, a, width=500)
            y += 74 + h + 50
        fr.footer("5")
        return fr.im
    return 10.0, draw


def pct(key: str) -> float:
    v = R(key).replace("\\%", "").replace("%", "")
    try:
        return float(v)
    except ValueError:
        return 0.0


def scene_success():
    conds = [("baseline", "Normal network"), ("faults", "Faults, as shipped"), ("drtc", "Faults, DRTC"),
             ("smooth", "Faults, DRTC + smoothing")]

    def draw(t):
        fr = Frame(PAPER)
        header(fr, t, "Results · task success on the robot", "DRTC brings the task back")
        for g, (m, nm) in enumerate([("smolvla", "SmolVLA"), ("groot", "GR00T N1.5")]):
            gx = M + g * 840
            a, dy = rise(t, 0.3 + 0.2 * g)
            fr.text((gx, 290 + dy), nm, font("bb", 32), INK, a)
            base_y, top = 860, 380
            fr.d.line((gx, base_y, gx + 720, base_y), fill=(*LINE, 255), width=2)
            for i, (c, label) in enumerate(conds):
                v = pct(f"main/{m}/{c}/M1")
                grow = ease((t - 0.8 - 0.35 * i) / 0.8)
                bx = gx + i * 180
                hgt = (base_y - top) * v / 100 * grow
                fr.rect((bx, base_y - hgt, bx + 130, base_y), fill=SER[c], radius=0)
                if grow > 0.05:
                    fr.text((bx + 65, base_y - hgt - 44), f"{v * grow:.0f}%", font("h", 34), INK, min(1, grow * 2),
                            anchor="ma")
                fr.text((bx + 65, base_y + 14), label, font("b", 20), MUTED, a, anchor="ma", width=170)
        a, dy = rise(t, 3.0)
        fr.text((M, 960 + dy), f"10 trials per bar, sorting task, {R('main/n')} trials in all. The shipped client "
                               "ended every faulted trial; with DRTC no trial was ended by a fault.",
                font("b", 26), MUTED, a, width=W - 2 * M)
        fr.footer("6")
        return fr.im
    return 8.0, draw


def scene_v2():
    def draw(t):
        fr = Frame(PAPER)
        header(fr, t, "Setup v2 · the bench", "Then we took it apart, on seven policies")
        cards = [("7 policies, 2 tasks", "SmolVLA, GR00T N1.5 and N1.7, π0.5, MolmoAct 2, ACT, Diffusion Policy; "
                  "sorting and cube in box."),
                 ("A replay robot", "Recorded held-out episodes as the cameras, the robot's measured network "
                  "simulated, every fault type, about 8,000 runs."),
                 ("A fair baseline", "LeRobot with its time limits raised to the whole run, so DRTC is compared "
                  "with a client that waits instead of giving up."),
                 ("Smoothing, part by part", "The Butterworth filter cut jerk 4–9×; adaptive playback cut the slow "
                  "policies' stalls by 15–27 points; the final stack lowered jerk 3–20×.")]
        cw = (W - 2 * M - 3 * 32) // 4
        for i, (a_, b) in enumerate(cards):
            a, dy = rise(t, 0.4 + 0.3 * i)
            x = M + i * (cw + 32)
            fr.rect((x, 330 + dy, x + cw, 780 + dy), fill=hexrgb(S.panel), outline=LINE, radius=20, alpha=a)
            fr.text((x + 36, 370 + dy), a_, font("h", 34), INK, a, width=cw - 72)
            fr.text((x + 36, 470 + dy), b, font("b", 26), MUTED, a, width=cw - 72)
        fr.footer("7")
        return fr.im
    return 8.0, draw


def scene_edge():
    def draw(t):
        fr = Frame(DARK)
        header(fr, t, "The edge", "It also runs on a Raspberry Pi 5", dark=True)
        stats = [(R("edge/hz").replace("--", "–"), "control loop on the Pi's CPU, no dropped camera frames"),
                 (R("edge/smolvla/rtt"), "SmolVLA round trip from the Pi (via the laptop's hotspot)"),
                 (R("edge/groot/rtt"), "GR00T N1.5 round trip: exactly the slow-link case DRTC is for")]
        for i, (big, small) in enumerate(stats):
            a, dy = rise(t, 0.4 + 0.4 * i)
            x = M + i * 560
            fr.text((x, 400 + dy), big, font("h", 64), DACC, a)
            fr.text((x, 490 + dy), small, font("b", 28), DMUTED, a, width=480)
        fr.footer("8", dark=True)
        return fr.im
    return 6.0, draw


def scene_close():
    def draw(t):
        fr = Frame(DARK)
        a, dy = rise(t, 0.2)
        fr.text((M, 330 + dy), "Remote inference fails because the client gives up,", font("h", 56), DTEXT, a,
                width=W - 2 * M)
        a, dy = rise(t, 0.8)
        fr.text((M, 410 + dy), "not because the model is wrong.", font("h", 56), DACC, a, width=W - 2 * M)
        a, dy = rise(t, 1.6)
        fr.text((M, 560 + dy), "Ask early, re-ask with a fresh view, smooth what the arm is sent.", font("hm", 34),
                DMUTED, a, width=W - 2 * M)
        a, dy = rise(t, 2.4)
        fr.text((M, 800 + dy), title_page()["authors"], font("b", 30), DTEXT, a)
        return fr.im
    return 6.0, draw


def scene_how():
    """How DRTC works, as a slowed-down simulation: ask early, a lost reply, the cooldown, a fresh request."""
    H, SMIN, RTT, COOL, SLOW = 50, 20, 14, 18, 0.45  # chunk, margin, round trip and cooldown in ticks; playback speed
    ticks, q, pending, events, n_req = [], H, None, [], 0
    for k in range(int(9.0 * 30)):
        if pending is None and q <= H - SMIN:
            n_req += 1
            pending = {"sent": k, "lost": n_req == 3, "id": n_req}
            events.append((k, "send", n_req))
        if pending is not None:
            age = k - pending["sent"]
            if not pending["lost"] and age == RTT:
                q = H - RTT
                events.append((k, "reply", pending["id"]))
                pending = None
            elif pending["lost"] and age > COOL:
                events.append((k, "giveup", pending["id"]))
                pending = None
                n_req += 1
                pending = {"sent": k, "lost": False, "id": n_req}
                events.append((k, "send", n_req))
        ticks.append((q, None if pending is None else dict(pending)))
        q = max(q - 1, 0)

    def draw(t):
        fr = Frame(PAPER)
        header(fr, t, "How DRTC works", "Ask early, and replace a lost reply")
        k = min(int(max(t - 0.8, 0) * 30 * SLOW), len(ticks) - 1)
        q, pend = ticks[k]
        a, dy = rise(t, 0.3)
        rx, sx_, y = M, W - M - 420, 330
        fr.rect((rx, y + dy, rx + 420, y + 170 + dy), fill=hexrgb(S.panel), outline=LINE, radius=20, alpha=a)
        fr.text((rx + 36, y + 36 + dy), "Robot", font("h", 40), INK, a)
        fr.text((rx + 36, y + 96 + dy), "plays one action per tick", font("b", 26), MUTED, a)
        fr.rect((sx_, y + dy, sx_ + 420, y + 170 + dy), fill=hexrgb(S.panel), outline=LINE, radius=20, alpha=a)
        fr.text((sx_ + 36, y + 36 + dy), "GPU server", font("h", 40), INK, a)
        fr.text((sx_ + 36, y + 96 + dy), "returns a chunk of 50", font("b", 26), MUTED, a)
        x0, x1, ym = rx + 440, sx_ - 20, y + 85
        fr.d.line((x0, ym, x1, ym), fill=(*LINE, int(255 * a)), width=3)
        if pend is not None:
            age = k - pend["sent"]
            frac = min(age / RTT, 1.0)
            if pend["lost"] and frac >= 0.5:
                cx = x0 + (x1 - x0) * 0.5
                fr.text((cx, ym - 70), "×  lost", font("h", 40), ALERT, 1.0, anchor="ma")
            else:
                cx = x0 + (x1 - x0) * (frac if frac < 0.5 else 1 - (frac - 0.5) * 2 + 0.0)
                cx = x0 + (x1 - x0) * (2 * frac if frac < 0.5 else 2 - 2 * frac)
                col = ACC if frac < 0.5 else SER["drtc"]
                fr.d.ellipse((cx - 14, ym - 14, cx + 14, ym + 14), fill=(*col, 255))
                fr.text((cx, ym - 56), "observation" if frac < 0.5 else "chunk", font("bs", 24), col, 1.0,
                        anchor="ma")
            # the cooldown: how long the robot waits for this reply before asking again
            cw = 300
            fr.rect((rx, y + 200, rx + cw, y + 214), fill=LINE, radius=7)
            fr.rect((rx, y + 200, rx + cw * min(age / COOL, 1), y + 214), fill=ALERT if pend["lost"] else ACC,
                    radius=7)
            fr.text((rx + cw + 20, y + 194), "cooldown", font("b", 24), MUTED)
        # the action queue, with the point where DRTC asks
        bx, by, cell = M, 690, (W - 2 * M) // H
        fr.text((bx, by - 50), f"Actions queued: {q}", font("bs", 28), INK, a)
        for i in range(H):
            filled = i < q
            fr.rect((bx + i * cell + 2, by, bx + (i + 1) * cell - 2, by + 60),
                    fill=SER["drtc"] if filled else LINE, radius=4, alpha=a)
        mx = bx + (H - SMIN) * cell
        fr.d.line((mx, by - 20, mx, by + 80), fill=(*ACC, int(255 * a)), width=4)
        fr.text((mx, by + 90), "DRTC asks here: H − s_min = 30 actions (1 s) left", font("bs", 24), ACC, a, anchor="ma")
        ux = bx + int(0.5 * 30) * cell
        fr.d.line((ux, by - 20, ux, by + 80), fill=(*MUTED, int(255 * a)), width=2)
        fr.text((ux, by + 130), "upstream asks here: 0.5 s left", font("b", 22), MUTED, a, anchor="ma")
        cap = "It asks for the next chunk early, while plenty of actions are left."
        for kk, kind, rid in events:
            if kk <= k and kind == "giveup":
                cap = ("A reply was lost: after the cooldown the robot gives up on it and sends a fresh observation. "
                       "Upstream would wait for it forever.")
        fr.text((M, 860), cap, font("b", 30), INK, a, width=W - 2 * M)
        fr.footer("3")
        return fr.im
    return 1.0 + len(ticks) / (30 * SLOW) / 1.6, draw


def scene_bench_faults():
    faults = [("spike", "One 1 s spike"), ("drops", "1 s of lost replies"), ("outage", "A 1.5 s outage"),
              ("mix", "All three")]

    def draw(t):
        fr = Frame(PAPER)
        header(fr, t, "The bench · each fault alone", "One fault is enough to end the shipped client's run")
        cw = (W - 2 * M - 3 * 32) // 4
        for i, (f, nm) in enumerate(faults):
            a, dy = rise(t, 0.4 + 0.25 * i)
            x = M + i * (cw + 32)
            fr.rect((x, 300 + dy, x + cw, 820 + dy), fill=hexrgb(S.panel), outline=LINE, radius=20, alpha=a)
            fr.text((x + 36, 336 + dy), nm, font("h", 34), INK, a, width=cw - 72)
            for j, (m, mn) in enumerate([("smolvla", "SmolVLA"), ("groot", "GR00T N1.5")]):
                yy = 440 + j * 180 + dy
                fr.text((x + 36, yy), mn, font("bs", 24), MUTED, a)
                fr.text((x + 36, yy + 36), R(f"bench/{f}/{m}/faults/M1f"), font("h", 56), ALERT, a)
                fr.text((x + 200, yy + 36), R(f"bench/{f}/{m}/drtc/M1f"), font("h", 56), SER["drtc"], a)
        a, dy = rise(t, 2.0)
        fr.text((M, 860 + dy), "Runs ended by the client, out of 8: orange LeRobot as shipped, green DRTC. Replay "
                               "robot, the network measured on the robot (250 ms), 8 runs per cell.",
                font("b", 28), MUTED, a, width=W - 2 * M)
        fr.footer("5")
        return fr.im
    return 8.0, draw


def _medians(path: Path, cond_map: dict) -> dict:
    import statistics
    rows = list(csv.DictReader(open(path)))
    out: dict = {}
    for r in rows:
        c = cond_map.get(r["condition"])
        if c is None:
            continue
        out.setdefault((r["model"].split("_")[0], c), []).append(r)
    med = lambda rs, k: statistics.median(float(x[k]) for x in rs if x.get(k) not in (None, ""))  # noqa: E731
    return {k: (med(v, "M2_stall_pct"), med(v, "M7_jerk_k")) for k, v in out.items()}


def scene_stack():
    bench = REPO / "results" / "bench"
    data = {}
    for name in ("bench_v2_stack_sort_one", "bench_v2_stack_nortc_sort_one"):
        f = bench / name / "metrics.csv"
        if f.is_file():
            data.update(_medians(f, {"drtc__mix": "drtc", "stack__mix": "stack", "stack_nortc__mix": "stack"}))
    names = {"act": "ACT", "smolvla": "SmolVLA", "pi05": "π0.5", "diffusion": "Diffusion Policy",
             "molmoact2": "MolmoAct 2", "groot": "GR00T N1.5", "grootn17": "GR00T N1.7"}
    pols = sorted({m for m, _ in data}, key=lambda m: data.get((m, "drtc"), (0, 0))[0])

    def draw(t):
        fr = Frame(PAPER)
        header(fr, t, "Setup v2 · the smoothing stack", "Smoother for every policy, fewer stalls for the slow ones")
        x0, x1 = M + 330, W - M - 380
        sx = lambda v: x0 + (x1 - x0) * v / 75  # noqa: E731
        fr.text((x0, 290), "time standing still under the fault mix (%)", font("bs", 24), MUTED)
        for v in range(0, 76, 25):
            fr.d.line((sx(v), 330, sx(v), 330 + 74 * len(pols)), fill=(*LINE, 255), width=1)
            fr.text((sx(v), 336 + 74 * len(pols)), f"{v}", font("b", 22), MUTED, anchor="ma")
        fr.text((x1 + 60, 290), "jerk", font("bs", 24), MUTED)
        for i, m in enumerate(pols):
            a, dy = rise(t, 0.4 + 0.15 * i)
            y = 360 + 74 * i
            d, s = data.get((m, "drtc")), data.get((m, "stack"))
            if not d or not s:
                continue
            fr.text((M, y - 18 + dy), names.get(m, m), font("bs", 28), INK, a)
            g = ease((t - 1.2 - 0.15 * i) / 0.8)
            xs = sx(d[0]) + (sx(s[0]) - sx(d[0])) * g
            fr.d.line((sx(d[0]), y + dy, xs, y + dy), fill=(*MUTED, int(200 * a)), width=4)
            fr.d.ellipse((sx(d[0]) - 12, y - 12 + dy, sx(d[0]) + 12, y + 12 + dy), fill=(*MUTED, int(255 * a)))
            fr.d.ellipse((xs - 14, y - 14 + dy, xs + 14, y + 14 + dy), fill=(*SER["smooth"], int(255 * a)))
            if g > 0.6:
                fr.text((x1 + 60, y - 22 + dy), f"÷ {d[1] / s[1]:.0f}", font("h", 40), ACC, min(1, (g - 0.6) * 2.5))
        a, dy = rise(t, 3.0)
        fr.text((M, 930 + dy), "Grey: DRTC alone. Orange: DRTC + the stack (adaptive playback, Butterworth filter, "
                               "resume ramp; RTC or ensembling). Sorting, bench, 8 runs each.",
                font("b", 26), MUTED, a, width=W - 2 * M)
        fr.footer("9")
        return fr.im
    return 9.0, draw


def scene_three_smolvla():
    k = 7
    clips = [Clip(trial_dir(c, "smolvla_sort_one_v1_full", k), (520, 390))
             for c in ("baseline_faults", "drtc_faults", "drtc_smooth_faults")]
    names = [("LeRobot as shipped", "faults"), ("DRTC", "drtc"), ("DRTC + smoothing", "smooth")]
    dur = min(8.0, max(c.length for c in clips)) + 1.5

    def draw(t):
        fr = Frame(PAPER)
        header(fr, t, "The same faults, SmolVLA", "A faster policy, the same story")
        gap = (W - 2 * M - 3 * 520) // 2
        for i, (c, (nm, key)) in enumerate(zip(clips, names)):
            a, dy = rise(t, 0.3 + 0.2 * i)
            x = M + i * (520 + gap)
            fr.rect((x, 300 + dy, x + 60, 308 + dy), fill=SER[key], alpha=a, radius=4)
            fr.text((x, 322 + dy), nm, font("bb", 30), INK, a)
            fr.paste(c.at(t), (x, 380 + dy), alpha=a)
        timeline(fr, (M, 820, W - M, 950), t, 9.0)
        fr.footer("7")
        return fr.im
    return dur, draw


SCENES = [scene_title, scene_setup, scene_failure, scene_how, scene_drtc, scene_bench_faults, scene_three,
          scene_three_smolvla, scene_motion, scene_success, scene_v2, scene_stack, scene_edge, scene_close]


def render(preview: bool = False) -> Path:
    OUT.mkdir(exist_ok=True)
    built = [make() for make in SCENES]
    if preview:
        pdir = OUT / "video_preview"
        pdir.mkdir(exist_ok=True)
        for i, (dur, draw) in enumerate(built, 1):
            draw(min(dur - 0.1, dur * 0.7)).save(pdir / f"scene-{i:02d}.png")
        print(pdir)
        return pdir
    path = OUT / "drtc_video.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf",
                           "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    total = 0.0
    prev_last = None
    for dur, draw in built:
        n = int(dur * FPS)
        for i in range(n):
            img = draw(i / FPS)
            if prev_last is not None and i < FADE * FPS:  # cross-fade from the previous scene's last frame
                img = Image.blend(prev_last, img, (i + 1) / (FADE * FPS))
            ff.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
            last = img
        prev_last = last
        total += dur
    ff.stdin.close()
    ff.wait()
    print(path, f"({total:.0f} s)")
    return path


if __name__ == "__main__":
    render(preview="--preview" in sys.argv)
