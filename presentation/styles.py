"""The same five slides in every style of themes.py, to choose the deck's look.

    .venv/bin/python presentation/styles.py          # -> presentation/out/styles/

Writes style-<key>-<name>.pptx per style (open in PowerPoint: animations + videos), a LibreOffice PDF and PNGs of each
(static, substituted fonts), styles.pdf with all of them, and overview.png (one row per style).
Slides: title; section divider; a diagram built click by click; the first-run success chart, bar by bar; two videos.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from deck import REPO, W, bar_panel, box, builds, line, movie, new_deck, poster, text
from drtc.evals.metrics import wilson
from themes import THEMES

OUT = Path(__file__).resolve().parent / "out" / "styles"
TRIALS = REPO / "results" / "evals" / "sort_one_drtc_main" / "sessions" / "20261120-1751-laptop" / "trials"
CLIP = TRIALS / "{}" / "smolvla_sort_one_v1_full" / "t03" / "video" / "scene.mp4"
TOTAL = 5

META = {"kind": "Project presentation", "title": ["Recovering remote robot inference", "from network faults"],
        "title3": ["Recovering remote", "robot inference from", "network faults"],
        "subtitle": "Distributed Real-Time Chunking and motion smoothing on a low-cost SO-101 arm",
        "authors": "Name One  ·  Name Two", "affil": "Guide: Prof. Name  ·  Department, Institution  ·  2026",
        "guide": "Prof. Name", "place": "Department, Institution", "date": "2026"}

# first run, sort_one, 10 trials per cell (successes)
SUCCESS = {"SmolVLA": [9, 0, 6, 8], "GR00T N1.5": [7, 0, 8, 10]}
CONDS = [("baseline", "Baseline"), ("faults", "Faults"), ("drtc", "DRTC"), ("smooth", ["DRTC +", "smooth"])]


def title(t, prs):
    meta = dict(META)
    if t.key in "DE":
        meta["title"] = META["title3"]
    if t.key == "D":
        return t.title_slide(prs, meta, still=poster(Path(str(CLIP).format("drtc_smooth_faults")), at=3.0))
    return t.title_slide(prs, meta)


def diagram(t, prs, n):
    s = t.content(prs, n, TOTAL, "Method · the setting", "Remote inference, and where it breaks")
    m, bw, bh = t.margin, Inches(3.2), Inches(1.5)
    y = t.top + Inches(0.45)
    xs = [m, (W - bw) // 2, W - m - bw]
    names = [("Robot client", "SO-101 arm · laptop or Pi"), ("Network", "Wi-Fi or the internet"),
             ("Policy server", "GPU · the VLA policy")]
    boxes = []
    for i, (x, (a, b)) in enumerate(zip(xs, names)):
        boxes.append(box(s, x, y, bw, bh, fill=None if i == 1 else t.panel, line=t.panel_line, dash=i == 1,
                         weight=Pt(1.25), radius=0.08, font=t.body, color=t.panel_ink,
                         runs=[[(a, {"bold": True, "size": 19})], [(b, {"color": t.muted, "size": 13})]]))
    gap = Inches(0.12)
    arrows, labels = [], []
    for i in range(2):
        x1, x2 = xs[i] + bw + gap, xs[i + 1] - gap
        a = line(s, x1, y + Inches(0.5), x2, y + Inches(0.5), color=t.ink, weight=Pt(1.75), arrow=True)
        b = line(s, x2, y + Inches(1.0), x1, y + Inches(1.0), color=t.ink, weight=Pt(1.75), arrow=True)
        la = text(s, x1, y + Inches(0.12), x2 - x1, Inches(0.3), "observation", size=12, color=t.muted,
                  font=t.body, align=PP_ALIGN.CENTER)
        lb = text(s, x1, y + Inches(1.08), x2 - x1, Inches(0.3), "actions", size=12, color=t.muted, font=t.body,
                  align=PP_ALIGN.CENTER)
        arrows += [(a, "wipe_right", 250 * i), (b, "wipe_left", 500 + 250 * (1 - i))]
        labels += [(la, "fade", 250 * i), (lb, "fade", 500 + 250 * (1 - i))]
    pills = []
    pw, pg = Inches(1.0), Inches(0.1)
    px = xs[1] + (bw - 3 * pw - 2 * pg) // 2
    for k, f in enumerate(("spike", "drops", "outage")):
        pills.append((box(s, px + k * (pw + pg), y + bh + Inches(0.22), pw, Inches(0.36), fill=t.alert,
                          radius=0.5, runs=f, size=12, bold=True, color="#FFFFFF", font=t.body,
                          margin=0), "fade", 120 * k))
    rows = []
    for k, (who, what, colour) in enumerate([
            ("LeRobot as shipped", "the action queue runs dry, and after 1 s without actions the run is ended.",
             t.series["faults"]),
            ("DRTC", "requests early, tracks the latency and re-requests a late reply, so the arm keeps going.",
             t.accent)]):
        ry = y + bh + Inches(1.05 + 0.55 * k)
        mk = box(s, m, ry + Inches(0.08), Inches(0.16), Inches(0.16), fill=colour)
        tx = text(s, m + Inches(0.35), ry, W - 2 * m - Inches(0.35), Inches(0.4),
                  [[(who + "   ", {"bold": True, "color": t.ink}), (what, {"color": t.muted})]], size=16,
                  font=t.body)
        rows.append([mk, tx])
    builds(s, [[(b, "fade", 150 * i) for i, b in enumerate(boxes)], arrows + labels, pills, rows[0], rows[1]])
    return s


def chart(t, prs, n):
    s = t.content(prs, n, TOTAL, "First run · task success", "Network faults break the shipped client")
    m, y = t.margin, t.top + Inches(0.05)
    ph, pw = Inches(4.6), Inches(3.55)
    bars = {}
    for j, (model, ks) in enumerate(SUCCESS.items()):
        data = []
        for (c, label), k in zip(CONDS, ks):
            lo, hi = wilson(k, 10)
            data.append((label, 10 * k, 100 * lo, 100 * hi, t.series[c]))
        _, per = bar_panel(s, m + j * (pw + Inches(0.35)), y, pw, ph, model, data, ink=t.ink, muted=t.muted,
                           grid=t.grid, font=t.body)
        bars[model] = per
    cx, cw = Inches(8.85), W - m - Inches(8.85)
    notes = [("90% · 70%", "Baseline: normal network.", "baseline"),
             ("0% · 0%", "Faults: the client ended all 20 runs at the first fault.", "faults"),
             ("60% · 80%", "DRTC, same faults: the arm holds and finishes.", "drtc"),
             ("80% · 100%", "DRTC with smoothing.", "smooth")]
    steps = []
    for i, (big, small, c) in enumerate(notes):
        ny = y + Inches(0.15) + Inches(1.13) * i
        ours = c in ("drtc", "smooth")
        mk = box(s, cx, ny + Inches(0.17), Inches(0.17), Inches(0.17), fill=t.series[c])
        a = text(s, cx + Inches(0.32), ny, cw, Inches(0.5), big, size=26, font=t.heading,
                 color=t.accent if ours else t.ink, bold=t.heading_bold)
        b = text(s, cx + Inches(0.32), ny + Inches(0.5), cw - Inches(0.32), Inches(0.55), small, size=13,
                 color=t.muted, font=t.body, line=1.1)
        step = []
        for j, model in enumerate(bars):
            step += [(sh, "wipe_up" if k == 0 else "fade", 150 * j + (0 if k == 0 else 250))
                     for k, sh in enumerate(bars[model][i])]
        steps.append(step + [(mk, "fade", 300), (a, "fade", 300), (b, "fade", 300)])
    builds(s, steps)
    return s


def videos(t, prs, n):
    s = t.content(prs, n, TOTAL, "First run · the same faults, two clients", "Giving up, or holding and recovering")
    m = t.margin
    vw = Inches(5.5)
    vh = int(vw * 3 / 4)
    y = t.top - Inches(0.05)
    xs = [m, W - m - vw]
    clips = [("baseline_faults", "LeRobot as shipped", "The spike empties the queue; a second later the run ends."),
             ("drtc_faults", "DRTC", "Holds through the spike, the lost replies and the outage, then finishes.")]
    for x, (cond, who, what) in zip(xs, clips):
        movie(s, Path(str(CLIP).format(cond)), x, y, vw, vh)
        text(s, x, y + vh + Inches(0.15), vw, Inches(0.7),
             [[(who + "   ", {"bold": True, "color": t.ink}), (what, {"color": t.muted})]], size=14, line=1.15,
             font=t.body)
    return s


def build(t) -> Path:
    prs = new_deck()
    title(t, prs)
    t.section_slide(prs, 2, TOTAL, 2, 5, "Method", "What breaks remote inference, and how DRTC recovers it")
    diagram(t, prs, 3)
    chart(t, prs, 4)
    videos(t, prs, 5)
    path = OUT / f"style-{t.key}-{t.name.lower().replace(' ', '-')}.pptx"
    prs.save(path)
    return path


def render(pptx: Path) -> list[Path]:
    tmp = OUT / ".render"
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(pptx)], check=True,
                   capture_output=True)
    pdf = tmp / (pptx.stem + ".pdf")
    for old in OUT.glob(pptx.stem + "-*.png"):
        old.unlink()
    subprocess.run(["pdftoppm", "-png", "-r", "60", str(pdf), str(OUT / pptx.stem)], check=True)
    return pdf, sorted(OUT.glob(pptx.stem + "-*.png"))


def overview(rows) -> Path:
    tw, pad, lab = 420, 18, 64
    th = int(tw * 7.5 / 13.333)
    width = pad + 5 * (tw + pad)
    img = Image.new("RGB", (width, len(rows) * (th + lab + pad) + pad), "#E9E8E4")
    d = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype("Carlito-Bold.ttf", 26)
        small = ImageFont.truetype("Carlito-Regular.ttf", 19)
    except OSError:
        big = small = ImageFont.load_default()
    y = pad
    for t, pngs in rows:
        d.text((pad, y), f"{t.key}   {t.name}", font=big, fill="#141414")
        d.text((pad, y + 32), t.blurb, font=small, fill="#55544F")
        for i, p in enumerate(pngs):
            im = Image.open(p).convert("RGB").resize((tw, th), Image.LANCZOS)
            img.paste(im, (pad + i * (tw + pad), y + lab))
        y += th + lab + pad
    out = OUT / "overview.png"
    img.save(out)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    rows, pdfs = [], []
    for t in THEMES:
        if only and t.key not in only:
            continue
        path = build(t)
        pdf, pngs = render(path)
        rows.append((t, pngs))
        pdfs.append(pdf)
        print(path)
    print(overview(rows))
    if len(pdfs) > 1:
        subprocess.run(["pdfunite", *map(str, pdfs), str(OUT / "styles.pdf")], check=True)
        print(OUT / "styles.pdf")
    shutil.rmtree(OUT / ".render", ignore_errors=True)


if __name__ == "__main__":
    main()
