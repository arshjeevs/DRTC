"""Visual styles for the deck. Each one is a common academic look, set up for click builds and video.

    A  minimal   white, serif titles, one blue accent (the report's colours)          fade
    B  metro     the Metropolis Beamer look: dark title bar, orange accent, flat       fade
    C  campus    university / conference: navy band, gold rule, footer band            fade
    D  dark      robotics lab talk: near-black, light text, bright accents; videos pop  fade through black
    E  swiss     paper-like grid: all Arial, bold titles, greys plus one red            push

A theme draws the frame (background, header, footer, title and section slides) and gives the colours; the slide
contents (styles.py, later the deck) only use the theme's attributes, so the deck can switch style in one line.
Fonts are ones Office ships on Windows and macOS; the LibreOffice previews substitute look-alikes.
"""

from __future__ import annotations

from lxml import etree
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from deck import H, W, blank, box, line, picture, text, transition

SHORT_TITLE = "DRTC on a low-cost arm"


def fade_box(s, x, y, w, h, colour):
    """A box that fades from `colour` (left) to transparent (right): blends a photo into the background."""
    sh = box(s, x, y, w, h, fill=colour)
    spPr = sh.fill._xPr
    solid = spPr.find(qn("a:solidFill"))
    grad = etree.Element(qn("a:gradFill"), rotWithShape="1")
    gs_lst = etree.SubElement(grad, qn("a:gsLst"))
    for pos, a in ((0, 100000), (100000, 0)):
        gs = etree.SubElement(gs_lst, qn("a:gs"), pos=str(pos))
        c = etree.SubElement(gs, qn("a:srgbClr"), val=colour.lstrip("#").upper())
        etree.SubElement(c, qn("a:alpha"), val=str(a))
    etree.SubElement(grad, qn("a:lin"), ang="0", scaled="0")
    solid.addprevious(grad)
    spPr.remove(solid)
    return sh


class Theme:
    key = name = blurb = ""
    bg, ink, muted, grid = "#FFFFFF", "#141414", "#6B6B68", "#E1E0D9"
    accent, alert = "#2A78D6", "#EB6834"  # alert = the faults
    panel, panel_line, panel_ink = "#F4F4F2", "#D9D8D3", "#141414"
    heading, body = "Georgia", "Calibri"
    heading_bold = False
    series = {"baseline": "#2A78D6", "faults": "#EB6834", "drtc": "#1BAF7A", "smooth": "#EDA100"}
    hero = None  # a condition drawn in a stronger colour than the rest, if any
    move = "fade"
    margin = Inches(0.75)
    top = Inches(2.0)  # where slide contents start

    # frame -----------------------------------------------------------------------------------------------------
    def slide(self, prs):
        s = blank(prs, self.bg)
        transition(s, self.move)
        return s

    def content(self, prs, n, total, kicker, title):
        s = self.slide(prs)
        self.header(s, kicker, title)
        self.footer(s, n, total)
        return s

    def header(self, s, kicker, title):
        m = self.margin
        line(s, m, Inches(0.62), m + Inches(0.45), Inches(0.62), color=self.accent, weight=Pt(2.25))
        text(s, m, Inches(0.72), Inches(10), Inches(0.3), kicker.upper(), size=11, color=self.accent, bold=True,
             spacing=1.6, font=self.body)
        text(s, m, Inches(1.0), W - 2 * m, Inches(0.8), title, size=30, font=self.heading, color=self.ink,
             bold=self.heading_bold)

    def footer(self, s, n, total):
        m = self.margin
        text(s, m, H - Inches(0.5), Inches(8), Inches(0.25), SHORT_TITLE, size=9, color=self.muted, font=self.body)
        text(s, W - m - Inches(1), H - Inches(0.5), Inches(1), Inches(0.25), str(n), size=9, color=self.muted,
             align=PP_ALIGN.RIGHT, font=self.body)

    def title_slide(self, prs, meta):
        s, m = self.slide(prs), self.margin
        text(s, m, Inches(1.55), Inches(9), Inches(0.3), meta["kind"].upper(), size=12, color=self.accent,
             bold=True, spacing=2, font=self.body)
        text(s, m, Inches(1.95), Inches(10.5), Inches(1.9), meta["title"], size=44, font=self.heading,
             color=self.ink, line=1.0, bold=self.heading_bold)
        text(s, m, Inches(3.75), Inches(10.5), Inches(0.5), meta["subtitle"], size=20, color=self.muted,
             font=self.body)
        line(s, m, Inches(4.75), m + Inches(0.9), Inches(4.75), color=self.accent, weight=Pt(2.25))
        text(s, m, Inches(5.0), Inches(10), Inches(1.0),
             [[(meta["authors"], {"bold": True})],
              [(meta["affil"], {"color": self.muted, "size": 14})]], size=16, line=1.4, font=self.body,
             color=self.ink)
        return s

    def section_slide(self, prs, n, total, part, parts, title, lead):
        s, m = self.slide(prs), self.margin
        text(s, m, Inches(2.2), Inches(3), Inches(1.4), f"{part:02d}", size=80, font=self.heading,
             color=self.accent)
        line(s, m, Inches(3.75), m + Inches(0.9), Inches(3.75), color=self.accent, weight=Pt(2.25))
        text(s, m, Inches(4.0), Inches(10), Inches(0.9), title, size=40, font=self.heading, color=self.ink,
             bold=self.heading_bold)
        text(s, m, Inches(4.95), Inches(10), Inches(0.6), lead, size=18, color=self.muted, font=self.body)
        self.footer(s, n, total)
        return s


class Minimal(Theme):
    key, name = "A", "Minimal serif"
    blurb = "White, serif titles, one blue accent; the report's own colours. Calm, paper-like."


class Metro(Theme):
    """After Metropolis, the most used modern Beamer theme."""
    key, name = "B", "Metropolis"
    blurb = "The Metropolis Beamer look: dark title bar, orange accent, flat and quiet."
    bg, ink, muted, grid = "#FAFAFA", "#23373B", "#6C7A7D", "#E2E4E4"
    accent, alert = "#EB811B", "#EB811B"
    panel, panel_line, panel_ink = "#FFFFFF", "#C9CFD0", "#23373B"
    heading = body = "Segoe UI"
    series = {"baseline": "#23373B", "faults": "#EB811B", "drtc": "#14B03D", "smooth": "#3E8BC8"}
    top = Inches(1.65)

    def header(self, s, kicker, title):
        box(s, 0, 0, W, Inches(1.0), fill=self.ink)
        text(s, self.margin, 0, W - 2 * self.margin, Inches(1.0), title, size=26, font=self.heading,
             color="#FFFFFF", anchor=MSO_ANCHOR.MIDDLE)

    def footer(self, s, n, total):
        text(s, W - self.margin - Inches(1), H - Inches(0.5), Inches(1), Inches(0.25), str(n), size=10,
             color=self.muted, align=PP_ALIGN.RIGHT, font=self.body)

    def title_slide(self, prs, meta):
        s, m = self.slide(prs), self.margin
        text(s, m, Inches(1.6), Inches(11), Inches(1.7), meta["title"], size=40, font=self.heading, color=self.ink,
             line=1.0, anchor=MSO_ANCHOR.BOTTOM)
        text(s, m, Inches(3.4), Inches(11), Inches(0.5), meta["subtitle"], size=20, color=self.muted,
             font=self.body)
        line(s, m, Inches(4.25), W - m, Inches(4.25), color=self.accent, weight=Pt(1.5))
        text(s, m, Inches(4.55), Inches(11), Inches(1.2),
             [meta["authors"], [(meta["affil"], {"color": self.muted, "size": 14})]], size=17, line=1.45,
             font=self.body, color=self.ink)
        return s

    def section_slide(self, prs, n, total, part, parts, title, lead):
        s, m = self.slide(prs), self.margin
        text(s, m, Inches(2.6), W - 2 * m, Inches(0.9), title, size=36, font=self.heading, color=self.ink,
             anchor=MSO_ANCHOR.BOTTOM)
        y, w = Inches(3.7), W - 2 * m
        line(s, m, y, m + w, y, color=self.grid, weight=Pt(2))
        line(s, m, y, m + int(w * part / parts), y, color=self.accent, weight=Pt(2))  # the progress bar
        text(s, m, Inches(3.95), w, Inches(0.5), lead, size=18, color=self.muted, font=self.body)
        self.footer(s, n, total)
        return s


class Campus(Theme):
    """The university / conference look: a navy band, a gold rule, a footer band."""
    key, name = "C", "Institutional"
    blurb = "University look: navy band, gold rule, footer band with names and page x / n."
    ink, muted, grid = "#1B2A3A", "#5B6573", "#E3E6EA"
    navy, gold = "#1B365D", "#C9A227"
    accent, alert = "#1B365D", "#B23A2E"
    panel, panel_line, panel_ink = "#F2F4F7", "#CDD3DB", "#1B2A3A"
    heading, body = "Cambria", "Calibri"
    series = {"baseline": "#1B365D", "faults": "#B23A2E", "drtc": "#2E7D5B", "smooth": "#C9A227"}
    top = Inches(1.75)

    def header(self, s, kicker, title):
        m = self.margin
        box(s, 0, 0, W, Inches(1.15), fill=self.navy)
        box(s, 0, Inches(1.15), W, Inches(0.05), fill=self.gold)
        text(s, m, Inches(0.2), Inches(10), Inches(0.3), kicker.upper(), size=10.5, color=self.gold, bold=True,
             spacing=1.5, font=self.body)
        text(s, m, Inches(0.45), W - 2 * m, Inches(0.6), title, size=27, font=self.heading, color="#FFFFFF")

    def footer(self, s, n, total, names="Name One · Name Two", place="Department · Institution"):
        y, h = H - Inches(0.42), Inches(0.42)
        box(s, 0, y, W, h, fill="#EEF1F5")
        kw = dict(size=10, color=self.muted, font=self.body, anchor=MSO_ANCHOR.MIDDLE)
        text(s, self.margin, y, Inches(4), h, names, **kw)
        text(s, Inches(4.5), y, Inches(4.3), h, place, align=PP_ALIGN.CENTER, **kw)
        text(s, W - self.margin - Inches(1.5), y, Inches(1.5), h, f"{n} / {total}", align=PP_ALIGN.RIGHT, **kw)

    def title_slide(self, prs, meta):
        s, m = self.slide(prs), self.margin
        box(s, 0, 0, W, Inches(4.3), fill=self.navy)
        box(s, 0, Inches(4.3), W, Inches(0.07), fill=self.gold)
        text(s, m, Inches(0.9), Inches(9), Inches(0.3), meta["kind"].upper(), size=12, color=self.gold, bold=True,
             spacing=2, font=self.body)
        text(s, m, Inches(1.3), Inches(11), Inches(1.9), meta["title"], size=42, font=self.heading,
             color="#FFFFFF", line=1.0)
        text(s, m, Inches(3.3), Inches(11), Inches(0.5), meta["subtitle"], size=19, color="#C8D3E3",
             font=self.body)
        text(s, m, Inches(4.9), Inches(8.5), Inches(1.6),
             [[(meta["authors"], {"bold": True})], [(meta["affil"], {"color": self.muted, "size": 14})]],
             size=17, line=1.5, font=self.body, color=self.ink)
        box(s, W - m - Inches(1.6), Inches(4.85), Inches(1.6), Inches(1.6), line=self.panel_line, dash=True,
            runs="institution\nlogo", size=11, color=self.muted, font=self.body)
        return s

    def section_slide(self, prs, n, total, part, parts, title, lead):
        s, m = self.slide(prs), self.margin
        box(s, 0, 0, Inches(4.2), H, fill=self.navy)
        box(s, Inches(4.2), 0, Inches(0.07), H, fill=self.gold)
        text(s, m, Inches(2.4), Inches(3), Inches(1.6), f"{part:02d}", size=96, font=self.heading,
             color=self.gold)
        text(s, m, Inches(4.1), Inches(3.2), Inches(0.4), f"PART {part} OF {parts}", size=12, color="#C8D3E3",
             bold=True, spacing=2, font=self.body)
        text(s, Inches(5.0), Inches(2.9), Inches(7.6), Inches(1.0), title, size=38, font=self.heading,
             color=self.ink, anchor=MSO_ANCHOR.BOTTOM)
        text(s, Inches(5.0), Inches(4.0), Inches(7.6), Inches(0.8), lead, size=18, color=self.muted,
             font=self.body)
        return s


class Dark(Theme):
    """A robotics lab talk: dark background, light text; trial videos sit well on it."""
    key, name = "D", "Dark lab talk"
    blurb = "Robotics-talk look: near-black, light text, bright accents; videos and photos stand out."
    bg, ink, muted, grid = "#101418", "#ECEFF3", "#98A2AF", "#262D35"
    accent, alert = "#4EA8FF", "#FF7A45"
    panel, panel_line, panel_ink = "#1A2027", "#323B45", "#ECEFF3"
    heading = body = "Segoe UI"
    heading_bold = True
    series = {"baseline": "#4EA8FF", "faults": "#FF7A45", "drtc": "#3DDC97", "smooth": "#FFC53D"}
    move = "black"

    def title_slide(self, prs, meta, still=None):
        s, m = self.slide(prs), self.margin
        if still is not None:  # a photo of the arm on the right, fading into the background
            pw = Inches(6.6)
            pic = picture(s, still, W - pw, 0, h=H)
            crop = max(0.0, (1 - pw / pic.width) / 2)
            pic.crop_left = pic.crop_right = crop
            pic.left, pic.width = W - pw, pw
            fade_box(s, W - pw, 0, Inches(3.2), H, self.bg)
            box(s, W - pw, 0, pw, H, fill=self.bg, alpha=0.2)
        text(s, m, Inches(1.7), Inches(7), Inches(0.3), meta["kind"].upper(), size=12, color=self.accent,
             bold=True, spacing=2, font=self.body)
        text(s, m, Inches(2.1), Inches(7.2), Inches(2.2), meta["title"], size=40, font=self.heading,
             color=self.ink, line=1.0, bold=True)
        text(s, m, Inches(4.25), Inches(6.8), Inches(0.8), meta["subtitle"], size=18, color=self.muted,
             font=self.body)
        text(s, m, Inches(5.4), Inches(7), Inches(1.0),
             [[(meta["authors"], {"bold": True})], [(meta["affil"], {"color": self.muted, "size": 13})]],
             size=16, line=1.4, font=self.body, color=self.ink)
        return s


class Swiss(Theme):
    """Paper-like grid: one sans, bold titles, greys and a single red for the method."""
    key, name = "E", "Swiss grid"
    blurb = "Paper-like grid: all Arial, bold titles, hairlines, greys plus one red for our method."
    ink, muted, grid = "#111111", "#6E6E6E", "#E4E4E4"
    accent, alert = "#D7261E", "#7A7A7A"
    panel, panel_line, panel_ink = "#F3F3F3", "#F3F3F3", "#111111"
    heading = body = "Arial"
    heading_bold = True
    series = {"baseline": "#C4C4C4", "faults": "#8A8A8A", "drtc": "#2B2B2B", "smooth": "#D7261E"}
    hero = "smooth"
    move = "push"
    top = Inches(1.9)

    def header(self, s, kicker, title):
        m = self.margin
        text(s, m, Inches(0.5), Inches(9), Inches(0.3), kicker, size=12, color=self.accent, bold=True,
             font=self.body)
        text(s, m, Inches(0.8), W - 2 * m - Inches(1), Inches(0.7), title, size=30, font=self.heading,
             color=self.ink, bold=True)
        line(s, m, Inches(1.55), W - m, Inches(1.55), color=self.ink, weight=Pt(0.75))

    def footer(self, s, n, total):
        text(s, W - self.margin - Inches(1), Inches(0.5), Inches(1), Inches(0.3), f"{n:02d}", size=12,
             color=self.ink, bold=True, align=PP_ALIGN.RIGHT, font=self.body)

    def title_slide(self, prs, meta):
        s, m = self.slide(prs), self.margin
        box(s, m, Inches(0.75), Inches(0.42), Inches(0.42), fill=self.accent)
        text(s, m, Inches(1.6), Inches(11.5), Inches(2.4), meta["title"], size=54, font=self.heading,
             color=self.ink, bold=True, line=0.95)
        text(s, m, Inches(3.95), Inches(11), Inches(0.5), meta["subtitle"], size=20, color=self.muted,
             font=self.body)
        y = Inches(5.45)
        line(s, m, y, W - m, y, color=self.ink, weight=Pt(0.75))
        cols = [("Presented by", meta["authors"]), ("Guide", meta["guide"]), ("Institution", meta["place"]),
                ("Date", meta["date"])]
        cw = (W - 2 * m) / len(cols)
        for i, (k, v) in enumerate(cols):
            text(s, int(m + cw * i), y + Inches(0.15), int(cw - Inches(0.2)), Inches(0.9),
                 [[(k, {"color": self.muted, "size": 11})], [(v, {"bold": True})]], size=14, line=1.3,
                 font=self.body, color=self.ink)
        return s

    def section_slide(self, prs, n, total, part, parts, title, lead):
        s, m = self.slide(prs), self.margin
        text(s, m - Inches(0.08), Inches(1.3), Inches(5), Inches(3.6), f"{part:02d}", size=210,
             font=self.heading, color=self.accent, bold=True, line=0.9)
        line(s, Inches(5.6), Inches(2.4), Inches(5.6), Inches(5.0), color=self.ink, weight=Pt(0.75))
        text(s, Inches(6.0), Inches(2.4), Inches(6.6), Inches(1.6), title, size=40, font=self.heading,
             color=self.ink, bold=True, anchor=MSO_ANCHOR.BOTTOM, line=1.0)
        text(s, Inches(6.0), Inches(4.15), Inches(6.6), Inches(0.9), lead, size=18, color=self.muted,
             font=self.body)
        self.footer(s, n, total)
        return s


THEMES = [Minimal(), Metro(), Campus(), Dark(), Swiss()]
