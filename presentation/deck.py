"""The presentation's design system: an academic, modern, minimal 16:9 deck built with python-pptx.

One white background, one serif for titles (Georgia), one sans for text (Calibri), ink and grey, and a single accent
colour taken from the report's figures. Animations are PowerPoint's own: a fade between slides, and click-by-click
fade-in builds for the parts of a slide. Videos are embedded, so the file plays anywhere.
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGS = REPO / "report" / "generated" / "slides"
CACHE = HERE / ".cache"

# --- the design system ------------------------------------------------------------------------------------------------
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.75)
SERIF, SANS = "Georgia", "Calibri"
INK = RGBColor(0x14, 0x14, 0x14)
GREY = RGBColor(0x6B, 0x6B, 0x68)
LIGHT = RGBColor(0xD9, 0xD8, 0xD3)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)  # the report's first figure colour
FOOTER = "Distributed Real-Time Chunking on a low-cost arm"


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def text(slide, x, y, w, h, runs, *, size=18, font=SANS, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=None, line=1.1):
    """A text box. `runs` is a string, or a list of paragraphs, each a string or a list of (text, overrides)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
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
            f.bold, f.color.rgb = o.get("bold", bold), o.get("color", color)
            f.italic = o.get("italic", False)
            sp = o.get("spacing", spacing)
            if sp is not None:
                r._r.get_or_add_rPr().set("spc", str(int(sp * 100)))
    return tb


def rule(slide, x, y, w, color=ACCENT, weight=Pt(2.25)):
    ln = slide.shapes.add_connector(1, x, y, x + w, y)
    ln.line.color.rgb, ln.line.width = color, weight
    return ln


def header(slide, kicker: str, title: str):
    rule(slide, MARGIN, Inches(0.62), Inches(0.45))
    text(slide, MARGIN, Inches(0.72), Inches(10), Inches(0.3), kicker.upper(), size=11, color=ACCENT, bold=True,
         spacing=1.6)
    text(slide, MARGIN, Inches(1.0), W - 2 * MARGIN, Inches(0.8), title, size=30, font=SERIF)


def footer(slide, n: int):
    text(slide, MARGIN, H - Inches(0.5), Inches(8), Inches(0.25), FOOTER, size=9, color=GREY)
    text(slide, W - MARGIN - Inches(1), H - Inches(0.5), Inches(1), Inches(0.25), str(n), size=9, color=GREY,
         align=PP_ALIGN.RIGHT)


def picture(slide, path: Path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def poster(video: Path, at: float = 1.0) -> Path:
    """A still from the video (its poster frame), cached."""
    CACHE.mkdir(exist_ok=True)
    out = CACHE / (video.parent.parent.name + "_" + video.parent.parent.parent.name + "_" + video.stem + ".jpg")
    if not out.is_file():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(at), "-i", str(video), "-frames:v", "1",
                        str(out)], check=True)
    return out


def movie(slide, video: Path, x, y, w, h):
    return slide.shapes.add_movie(str(video), x, y, w, h, poster_frame_image=str(poster(video)),
                                  mime_type="video/mp4")


# --- PowerPoint animations (raw slide XML; python-pptx has no API for them) ----------------------------------------
P = "http://schemas.openxmlformats.org/presentationml/2006/main"


def fade_transition(slide, speed: str = "med") -> None:
    sld = slide._element
    for old in sld.findall(qn("p:transition")):
        sld.remove(old)
    tr = etree.SubElement(sld, qn("p:transition"), spd=speed)
    etree.SubElement(tr, qn("p:fade"))
    timing = sld.find(qn("p:timing"))  # the transition must come before the timing
    if timing is not None:
        sld.remove(timing)
        sld.append(timing)


def click_builds(slide, groups) -> None:
    """Fade in each group of shapes on its own click, in order (a group is a shape or a list of shapes)."""
    ids = iter(range(3, 10_000))

    def ctn(**kw):
        return etree.Element(qn("p:cTn"), id=str(next(ids)), **kw)

    def cond(parent, delay):
        st = etree.SubElement(parent, qn("p:stCondLst"))
        etree.SubElement(st, qn("p:cond"), delay=delay)

    clicks = []
    for group in groups:
        shapes = group if isinstance(group, (list, tuple)) else [group]
        outer = etree.Element(qn("p:par"))
        c1 = ctn(fill="hold")
        outer.append(c1)
        cond(c1, "indefinite")
        l1 = etree.SubElement(c1, qn("p:childTnLst"))
        mid = etree.SubElement(l1, qn("p:par"))
        c2 = ctn(fill="hold")
        mid.append(c2)
        cond(c2, "0")
        l2 = etree.SubElement(c2, qn("p:childTnLst"))
        for k, sh in enumerate(shapes):
            par = etree.SubElement(l2, qn("p:par"))
            c3 = ctn(presetID="10", presetClass="entr", presetSubtype="0", fill="hold",
                     nodeType="clickEffect" if k == 0 else "withEffect")
            par.append(c3)
            cond(c3, "0")
            l3 = etree.SubElement(c3, qn("p:childTnLst"))
            s = etree.SubElement(l3, qn("p:set"))
            b = etree.SubElement(s, qn("p:cBhvr"))
            c4 = ctn(dur="1", fill="hold")
            b.append(c4)
            cond(c4, "0")
            tg = etree.SubElement(b, qn("p:tgtEl"))
            etree.SubElement(tg, qn("p:spTgt"), spid=str(sh.shape_id))
            al = etree.SubElement(b, qn("p:attrNameLst"))
            etree.SubElement(al, qn("p:attrName")).text = "style.visibility"
            to = etree.SubElement(s, qn("p:to"))
            etree.SubElement(to, qn("p:strVal"), val="visible")
            ef = etree.SubElement(l3, qn("p:animEffect"), transition="in", filter="fade")
            b2 = etree.SubElement(ef, qn("p:cBhvr"))
            b2.append(ctn(dur="450"))
            tg2 = etree.SubElement(b2, qn("p:tgtEl"))
            etree.SubElement(tg2, qn("p:spTgt"), spid=str(sh.shape_id))
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
    # renumber every cTn id so they are unique on the slide
    for i, el in enumerate(timing.iter(qn("p:cTn")), start=1):
        el.set("id", str(i))
