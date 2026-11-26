"""Drawn pieces of the talk (native PowerPoint shapes, so each can be animated): curves, small plot panels, the fault
timelines, and the real-trial chunk plot. Used by talk.py."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from deck import REPO, _plain, box, line, rgb, text
from style import Style

S = Style()
# The real trial behind the VLA and chunk slides: SmolVLA, cube in box, baseline, from the Raspberry Pi (2026-11-26).
TRIAL = (REPO / "results/evals/cubebox_v1_baseline/sessions/20261126-1443-pi/trials/baseline/"
         "smolvla_cubebox_v1_full/t01")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_COLOURS = ["#2A78D6", "#1BAF7A", "#EDA100", "#EB6834", "#8C5BD6", "#5E636D"]
CHUNK_COLOURS = ["#2A78D6", "#1BAF7A", "#EDA100", "#8C5BD6", "#EB6834"]


def curve(s, pts, color, weight=Pt(2.25), dash=False):
    """A polyline through absolute EMU points."""
    pts = [(int(x), int(y)) for x, y in pts]
    ff = s.shapes.build_freeform(*pts[0])
    ff.add_line_segments(pts[1:], close=False)
    sh = ff.convert_to_shape()
    _plain(sh)
    sh.fill.background()
    sh.line.color.rgb, sh.line.width = rgb(color), weight
    if dash:
        sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return sh


def mapper(x, y, w, h, t0, t1, v0, v1):
    return lambda t, v: (x + w * (t - t0) / (t1 - t0), y + h * (1 - (v - v0) / (v1 - v0)))


def dot(s, cx, cy, r, color):
    return box(s, int(cx - r), int(cy - r), int(2 * r), int(2 * r), fill=color, radius=0.5)


# --- illustrative joint curves for the smoothing slides (time 0..1, position 0..1) ---------------------------------

T = np.linspace(0, 1, 80)


def _f(t):
    return 0.15 + 0.62 * (0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1)))


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5


def problem_curves():
    """name -> (before, after, marker) on T; the marker is where the problem is (t, kind)."""
    seam_b = _f(T) + 0.14 * (T >= 0.5)
    seam_a = _f(T) + 0.14 * _smoothstep((T - 0.38) / 0.24)
    p_b = np.where(T < 0.4, T, np.where(T < 0.65, 0.4, T - 0.25))
    p_a = np.where(T < 0.3, T, np.where(T < 0.8, 0.3 + 0.55 * (T - 0.3), 0.575 + 0.875 * (T - 0.8)))
    stall_b, stall_a = _f(p_b / 0.75), _f(p_a / 0.75)
    noise = np.random.default_rng(3).normal(0, 0.028, T.size)
    jit_b, jit_a = _f(T) + noise, _f(T - 0.02)
    a = _f(T) + 0.12
    rest_b = np.where(T < 0.5, 0.22, a)
    rest_a = np.where(T < 0.5, 0.22, 0.22 + _smoothstep((T - 0.5) / 0.25) * (a - 0.22))
    return {"seam": (seam_b, seam_a, (0.5, "point")), "stall": (stall_b, stall_a, (0.4, 0.65)),
            "jitter": (jit_b, jit_a, None), "restart": (rest_b, rest_a, (0.5, "point"))}


def mini_panel(s, x, y, w, h, title, sub, detail=None, plot_h=Inches(1.6)):
    """A card with a title, a plot area and text under it. Returns (static shapes, plot box (x, y, w, h))."""
    shapes = [box(s, x, y, w, h, fill=S.panel, line=S.panel_line, weight=Pt(1), radius=S.card_radius)]
    pad = Inches(0.22)
    shapes.append(text(s, x + pad, y + pad, w - 2 * pad, Inches(0.35), title, size=16 if len(title) < 19 else 15,
                       bold=True, font=S.body, color=S.ink))
    py = y + pad + Inches(0.6)
    plot = (x + pad, py, w - 2 * pad, plot_h)
    shapes.append(line(s, plot[0], py + plot_h, plot[0] + plot[2], py + plot_h, color=S.grid, weight=Pt(1)))
    ty = py + plot_h + Inches(0.15)
    paras = [[(sub, {"color": S.muted})]]
    if detail:
        paras.append([(detail, {"color": S.accent, "size": 11})])
    shapes.append(text(s, x + pad, ty, w - 2 * pad, y + h - ty - Inches(0.1), paras, size=13, font=S.body,
                       line=1.15))
    return shapes, plot


def plot_curve(s, plot, values, color, *, dash=False, weight=Pt(2.25)):
    x, y, w, h = plot
    f = mapper(x, y, w, h, 0, 1, 0, 1)
    return curve(s, [f(t, v) for t, v in zip(T, values)], color, weight=weight, dash=dash)


def marker(s, plot, mk, values):
    x, y, w, h = plot
    f = mapper(x, y, w, h, 0, 1, 0, 1)
    if mk is None:
        return None
    if mk[1] == "point":
        i = int(np.searchsorted(T, mk[0]))
        cx, cy = f(T[i], (values[i - 1] + values[i]) / 2)
        r = Inches(0.17)
        return box(s, int(cx - r), int(cy - r), int(2 * r), int(2 * r), line=S.alert, weight=Pt(1.5), radius=0.5)
    x0, _ = f(mk[0], 0)
    x1, _ = f(mk[1], 0)
    return box(s, int(x0), int(y), int(x1 - x0), int(h), fill=S.alert, alpha=0.14)


# --- fault timelines --------------------------------------------------------------------------------------------

def fault_row(s, x, y, w, h, kind):
    """Replies from the server over 10 s (a mark every 0.5 s) and what the fault does to them."""
    out = []
    f = lambda t: int(x + w * t / 10)  # noqa: E731
    mid = y + h // 2
    th = Inches(0.13)
    out.append(line(s, x, mid + th + Inches(0.03), x + w, mid + th + Inches(0.03), color=S.grid, weight=Pt(0.75)))
    kinds = {"mix": ["spike", "drops", "outage"]}.get(kind, [kind])
    gone, late, crossed = set(), {}, set()
    if "spike" in kinds:
        gone.add(4)
        late[4] = 1.0
    if "drops" in kinds:
        crossed |= {8, 9}
    if "outage" in kinds:
        gone |= {12, 13, 14}
        out.append(box(s, f(6.0), mid - th - Inches(0.06), f(7.5) - f(6.0), 2 * th + Inches(0.12), fill=S.alert,
                       alpha=0.18))
    if "rise" in kinds:
        out.append(box(s, f(2.0), mid - th - Inches(0.06), f(7.4) - f(2.0), 2 * th + Inches(0.12), fill=S.alert,
                       alpha=0.08))
        for i in range(4, 14):
            late[i] = 0.4
    if "loss" in kinds:
        crossed |= {6, 15}
    for i in range(21):
        t = 0.5 * i
        if i in crossed:
            out.append(text(s, f(t) - Inches(0.15), mid - Inches(0.2), Inches(0.3), Inches(0.4), "×", size=17,
                            bold=True, color=S.alert, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, font=S.body))
            continue
        if i in late:
            t2 = t + late[i]
            out.append(line(s, f(t), mid, f(t2), mid, color=S.alert, weight=Pt(1.25)))
            out.append(line(s, f(t2), mid - th, f(t2), mid + th, color=S.alert, weight=Pt(2.25)))
            continue
        if i in gone:
            continue
        out.append(line(s, f(t), mid - th, f(t), mid + th, color=S.muted, weight=Pt(2)))
    return out


# --- the real trial ----------------------------------------------------------------------------------------------

def trial_ticks():
    return [r for r in csv.DictReader(open(TRIAL / "ticks.csv")) if r["cmd.elbow_flex"]]


def trial_chunks():
    return [r for r in csv.DictReader(open(TRIAL / "chunks.csv")) if r["accepted"] == "1"]


def first_state():
    r = trial_ticks()[0]
    return {j: float(r[f"meas.{j}"]) for j in JOINTS}


def first_chunk_curves(s, x, y, w, h):
    """The first chunk's 50 commanded positions for every joint, each scaled to its own range."""
    ticks = trial_ticks()[:50]
    shapes = []
    for j, col in zip(JOINTS, JOINT_COLOURS):
        v = np.array([float(r[f"cmd.{j}"]) for r in ticks])
        lo, hi = v.min(), v.max()
        span = (hi - lo) or 1.0
        f = mapper(x, y, w, h, 0, len(v) - 1, lo - 0.15 * span, hi + 0.15 * span)
        shapes.append(curve(s, [f(i, a) for i, a in enumerate(v)], col, weight=Pt(2)))
    return shapes


def chunk_plot(s, joint_box, queue_box, t0, t1, joint="shoulder_lift"):
    """Per accepted chunk inside [t0, t1]: (its joint-curve segment, its queue segment, its arrival time, its request
    time). Segments run from the chunk's arrival to the next arrival (or t1)."""
    ticks = [r for r in trial_ticks() if t0 <= float(r["t"]) <= t1]
    arrivals = [float(c["t_received"]) for c in trial_chunks() if float(c["t_received"]) <= t1]
    sends = [float(c["t_sent"]) for c in trial_chunks()]
    v = [float(r[f"cmd.{joint}"]) for r in ticks]
    lo, hi = min(v), max(v)
    pad = 0.1 * (hi - lo)
    fj = mapper(*joint_box, t0, t1, lo - pad, hi + pad)
    fq = mapper(*queue_box, t0, t1, 0, 55)
    segs = []
    for k, a in enumerate(arrivals):
        b = arrivals[k + 1] if k + 1 < len(arrivals) else t1
        part = [r for r in ticks if a - 1e-6 <= float(r["t"]) <= b + 0.034]
        if len(part) < 2:
            continue
        col = CHUNK_COLOURS[len(segs) % len(CHUNK_COLOURS)]
        jc = curve(s, [fj(float(r["t"]), float(r[f"cmd.{joint}"])) for r in part], col, weight=Pt(2.75))
        qpts = []
        for r in part:
            tx, qy = fq(float(r["t"]), float(r["queue"]))
            if qpts:
                qpts.append((tx, qpts[-1][1]))
            qpts.append((tx, qy))
        qc = curve(s, qpts, col, weight=Pt(2))
        nxt = next((t for t in sends if a < t <= b), None)
        segs.append((jc, qc, a, nxt))
    return segs, fj, fq
