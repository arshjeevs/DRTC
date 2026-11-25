"""Building blocks for the presentation, independent of the visual style (python-pptx + raw slide XML).

Shapes: text boxes, boxes, lines and arrows, pictures, embedded videos with poster frames, and a bar chart drawn from
native shapes (so every bar can be animated on its own and takes the style's colours). Animations are PowerPoint's
own: slide transitions (fade, fade through black, push) and click-by-click builds (rise, fade, wipe up, wipe right), with an
optional stagger inside a click. The style itself lives in style.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGS = REPO / "report" / "generated" / "slides"
CACHE = HERE / ".cache"

W, H = Inches(13.333), Inches(7.5)


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_.lstrip("#").upper())


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def blank(prs, bg: str):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(bg)
    return s


# --- shapes -------------------------------------------------------------------------------------------------------

def text(slide, x, y, w, h, runs, *, size=18, font="Calibri", color="#141414", bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=None, line=1.1, italic=False):
    """A text box. `runs` is a string, or a list of paragraphs, each a string or a list of (text, overrides)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    _fill_text(tb.text_frame, runs, size=size, font=font, color=color, bold=bold, align=align, anchor=anchor,
               spacing=spacing, line=line, italic=italic)
    return tb


def _fill_text(tf, runs, *, size, font, color, bold, align, anchor, spacing, line, italic, margin=0):
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", margin)
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line
        for chunk in ([(para, {})] if isinstance(para, str) else para):
            t, o = chunk if isinstance(chunk, tuple) else (chunk, {})
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size = o.get("font", font), Pt(o.get("size", size))
            f.bold, f.italic = o.get("bold", bold), o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
            sp = o.get("spacing", spacing)
            if sp is not None:
                r._r.get_or_add_rPr().set("spc", str(int(sp * 100)))


def _plain(sh):
    """Drop the default shape style (theme shadow, theme colours): every look is set explicitly."""
    style = sh._element.find(qn("p:style"))
    if style is not None:
        sh._element.remove(style)


def box(slide, x, y, w, h, *, fill=None, line=None, weight=Pt(1), radius=None, dash=False, alpha=None, runs=None,
        size=16, font="Calibri", color="#141414", bold=False, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
        line_spacing=1.1, margin=Inches(0.12)):
    """A rectangle (rounded if `radius`, a fraction of the short side), optionally with text inside."""
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(kind, x, y, w, h)
    if radius:
        sh.adjustments[0] = radius
    _plain(sh)
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
        if alpha is not None:  # 0 = invisible, 1 = opaque
            clr = sh.fill._xPr.find(".//" + qn("a:srgbClr"))
            etree.SubElement(clr, qn("a:alpha"), val=str(int(alpha * 100000)))
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb, sh.line.width = rgb(line), weight
        if dash:
            sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    else:
        sh.line.fill.background()
    if runs is not None:
        _fill_text(sh.text_frame, runs, size=size, font=font, color=color, bold=bold, align=align, anchor=anchor,
                   spacing=None, line=line_spacing, italic=False, margin=margin)
    return sh


def line(slide, x1, y1, x2, y2, *, color="#141414", weight=Pt(1.5), dash=False, arrow=False):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    _plain(ln)
    ln.line.color.rgb, ln.line.width = rgb(color), weight
    if dash:
        ln.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if arrow:
        etree.SubElement(ln.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle", w="med", len="med")
    return ln


def picture(slide, path: Path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def poster(video: Path, at: float = 1.0) -> Path:
    """A still from the video (its poster frame), cached."""
    CACHE.mkdir(exist_ok=True)
    parts = video.parts[-6:-2]  # condition / model / trial / video
    out = CACHE / ("_".join(parts) + f"_{video.stem}_{at:g}.jpg")
    if not out.is_file():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(at), "-i", str(video), "-frames:v", "1",
                        str(out)], check=True)
    return out


def movie(slide, video: Path, x, y, w, h):
    return slide.shapes.add_movie(str(video), x, y, w, h, poster_frame_image=str(poster(video)),
                                  mime_type="video/mp4")


def bar_panel(slide, x, y, w, h, title, bars, *, ink, muted, grid, font, size=13, title_size=15, ymax=115):
    """A bar chart from native shapes: `bars` = [(label, value %, lo %, hi %, colour)].

    Returns (static shapes, [shapes of bar i]) so each bar (with its interval and value label) can be built alone.
    """
    static, per_bar = [], []
    top, bottom = y + Inches(0.55), y + h - Inches(0.75)
    left, right = x + Inches(0.55), x + w

    def ys(v):
        return int(bottom - (bottom - top) * v / ymax)

    static.append(text(slide, left, y, w, Inches(0.35), title, size=title_size, font=font, color=ink, bold=True))
    for v in (0, 25, 50, 75, 100):
        static.append(line(slide, left, ys(v), right, ys(v), color=grid if v else muted, weight=Pt(0.75)))
        static.append(text(slide, x, ys(v) - Inches(0.12), Inches(0.45), Inches(0.25), str(v), size=size - 2,
                           font=font, color=muted, align=PP_ALIGN.RIGHT))
    slot = (right - left) / len(bars)
    bw = int(slot * 0.62)
    for i, (label, v, lo, hi, colour) in enumerate(bars):
        cx = int(left + slot * (i + 0.5))
        shapes = []
        if v > 0:
            shapes.append(box(slide, cx - bw // 2, ys(v), bw, bottom - ys(v), fill=colour))
        else:  # a zero bar: a thin strip on the axis, so the build still shows something
            shapes.append(box(slide, cx - bw // 2, bottom - Pt(3), bw, Pt(3), fill=colour))
        cap = Inches(0.07)
        shapes += [line(slide, cx, ys(hi), cx, ys(lo), color=ink, weight=Pt(1)),
                   line(slide, cx - cap, ys(hi), cx + cap, ys(hi), color=ink, weight=Pt(1)),
                   line(slide, cx - cap, ys(lo), cx + cap, ys(lo), color=ink, weight=Pt(1))]
        shapes.append(text(slide, int(cx - slot / 2), ys(hi) - Inches(0.34), int(slot), Inches(0.3), f"{v:.0f}%",
                           size=size, font=font, color=ink, bold=True, align=PP_ALIGN.CENTER))
        shapes.append(text(slide, int(cx - slot / 2), bottom + Inches(0.08), int(slot), Inches(0.6), label,
                           size=size - 1, font=font, color=muted, align=PP_ALIGN.CENTER, line=1.0))
        per_bar.append(shapes)
    return static, per_bar


# --- PowerPoint animations (raw slide XML; python-pptx has no API for them) ----------------------------------------

def transition(slide, kind: str = "fade", speed: str = "med") -> None:
    """Slide transition: fade | black (fade through black) | push (the new slide pushes up) | none."""
    sld = slide._element
    for old in sld.findall(qn("p:transition")):
        sld.remove(old)
    if kind != "none":
        tr = etree.SubElement(sld, qn("p:transition"), spd=speed)
        if kind == "fade":
            etree.SubElement(tr, qn("p:fade"))
        elif kind == "black":
            etree.SubElement(tr, qn("p:fade"), thruBlk="1")
        elif kind == "push":
            etree.SubElement(tr, qn("p:push"), dir="u")
        else:
            raise ValueError(kind)
    timing = sld.find(qn("p:timing"))  # the transition must come before the timing
    if timing is not None:
        sld.remove(timing)
        sld.append(timing)


# entrance effects: (presetID, presetSubtype, filter); a wipe's filter names the side the shape is revealed from
EFFECTS = {"fade": ("10", "0", "fade"), "wipe_up": ("22", "4", "wipe(down)"), "wipe_right": ("22", "8", "wipe(left)"),
           "wipe_left": ("22", "2", "wipe(right)"),
           "rise": ("42", "0", "fade")}  # PowerPoint's Float In: a fade while the shape moves up a little
RISE = "0.04"  # how far a rising shape starts below its place, as a fraction of the slide height


def _item(item):
    if not isinstance(item, tuple):
        return item, "rise", 0
    return (item + ("rise", 0)[len(item) - 1:])[:3]  # fill in the missing effect / delay


def builds(slide, steps, *, dur: int = 450) -> None:
    """Reveal the slide in clicks. `steps` = one entry per click; an entry is a list of items, each a shape or
    (shape, effect) or (shape, effect, delay ms). Items of one click start together, after their delays."""
    ids = iter(range(3, 100_000))

    def ctn(**kw):
        return etree.Element(qn("p:cTn"), id=str(next(ids)), **kw)

    def cond(parent, delay):
        st = etree.SubElement(parent, qn("p:stCondLst"))
        etree.SubElement(st, qn("p:cond"), delay=str(delay))

    clicks = []
    for step in steps:
        items = step if isinstance(step, list) else [step]
        outer = etree.Element(qn("p:par"))
        c1 = ctn(fill="hold")
        outer.append(c1)
        cond(c1, "indefinite")
        l1 = etree.SubElement(c1, qn("p:childTnLst"))
        mid = etree.SubElement(l1, qn("p:par"))
        c2 = ctn(fill="hold")
        mid.append(c2)
        cond(c2, 0)
        l2 = etree.SubElement(c2, qn("p:childTnLst"))
        for k, item in enumerate(items):
            sh, effect, delay = _item(item)
            pid, sub, filt = EFFECTS[effect]
            par = etree.SubElement(l2, qn("p:par"))
            c3 = ctn(presetID=pid, presetClass="entr", presetSubtype=sub, fill="hold",
                     nodeType="clickEffect" if k == 0 else "withEffect")
            par.append(c3)
            cond(c3, delay)
            l3 = etree.SubElement(c3, qn("p:childTnLst"))
            s = etree.SubElement(l3, qn("p:set"))
            b = etree.SubElement(s, qn("p:cBhvr"))
            c4 = ctn(dur="1", fill="hold")
            b.append(c4)
            cond(c4, 0)
            tg = etree.SubElement(b, qn("p:tgtEl"))
            etree.SubElement(tg, qn("p:spTgt"), spid=str(sh.shape_id))
            al = etree.SubElement(b, qn("p:attrNameLst"))
            etree.SubElement(al, qn("p:attrName")).text = "style.visibility"
            to = etree.SubElement(s, qn("p:to"))
            etree.SubElement(to, qn("p:strVal"), val="visible")
            ef = etree.SubElement(l3, qn("p:animEffect"), transition="in", filter=filt)
            b2 = etree.SubElement(ef, qn("p:cBhvr"))
            b2.append(ctn(dur=str(dur)))
            tg2 = etree.SubElement(b2, qn("p:tgtEl"))
            etree.SubElement(tg2, qn("p:spTgt"), spid=str(sh.shape_id))
            if effect == "rise":
                an = etree.SubElement(l3, qn("p:anim"), calcmode="lin", valueType="num")
                b3 = etree.SubElement(an, qn("p:cBhvr"))
                b3.append(ctn(dur=str(dur + 150), decel="100000", fill="hold"))
                tg3 = etree.SubElement(b3, qn("p:tgtEl"))
                etree.SubElement(tg3, qn("p:spTgt"), spid=str(sh.shape_id))
                al3 = etree.SubElement(b3, qn("p:attrNameLst"))
                etree.SubElement(al3, qn("p:attrName")).text = "ppt_y"
                tav = etree.SubElement(an, qn("p:tavLst"))
                for tm, val in (("0", f"#ppt_y+{RISE}"), ("100000", "#ppt_y")):
                    tv = etree.SubElement(tav, qn("p:tav"), tm=tm)
                    vv = etree.SubElement(tv, qn("p:val"))
                    etree.SubElement(vv, qn("p:strVal"), val=val)
        clicks.append(outer)

    sld = slide._element
    timing = sld.find(qn("p:timing"))
    if timing is None:
        timing = etree.SubElement(sld, qn("p:timing"))
        tnl = etree.SubElement(timing, qn("p:tnLst"))
        root = etree.SubElement(tnl, qn("p:par"))
        rc = etree.SubElement(root, qn("p:cTn"), id="1", dur="indefinite", restart="never", nodeType="tmRoot")
        rl = etree.SubElement(rc, qn("p:childTnLst"))
    else:  # e.g. a slide with a video: add the main sequence next to its media nodes
        rl = timing.find(".//" + qn("p:cTn") + "[@nodeType='tmRoot']/" + qn("p:childTnLst"))
    seq = etree.Element(qn("p:seq"), concurrent="1", nextAc="seek")
    sc = etree.SubElement(seq, qn("p:cTn"), id=str(next(ids)), dur="indefinite", nodeType="mainSeq")
    scl = etree.SubElement(sc, qn("p:childTnLst"))
    for c in clicks:
        scl.append(c)
    for tag, evt in (("p:prevCondLst", "onPrev"), ("p:nextCondLst", "onNext")):
        cl = etree.SubElement(seq, qn(tag))
        cd = etree.SubElement(cl, qn("p:cond"), evt=evt, delay="0")
        tg = etree.SubElement(cd, qn("p:tgtEl"))
        etree.SubElement(tg, qn("p:sldTgt"))
    rl.insert(0, seq)
    for i, el in enumerate(timing.iter(qn("p:cTn")), start=1):  # cTn ids unique on the slide
        el.set("id", str(i))
