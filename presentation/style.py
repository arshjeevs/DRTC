"""The deck's style (locked 2026-11-23): minimal serif.

White, Georgia titles, Calibri text, ink and grey, one blue accent (the report's first figure colour), the report's
four condition colours in charts. Every slide fades in; parts of a slide appear click by click. Slide contents use
only the attributes below, so the look is changed here and nowhere else.
"""

from __future__ import annotations

from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from deck import H, W, blank, line, text, transition

SHORT_TITLE = "Reliable remote inference under network faults"


class Style:
    bg, ink, muted, grid = "#FFFFFF", "#141414", "#6B6B68", "#E1E0D9"
    accent, alert = "#2A78D6", "#EB6834"  # alert = the faults
    panel, panel_line, panel_ink = "#F4F4F2", "#D9D8D3", "#141414"
    heading, body = "Georgia", "Calibri"
    series = {"baseline": "#2A78D6", "faults": "#EB6834", "drtc": "#1BAF7A", "smooth": "#EDA100"}
    margin = Inches(0.75)
    top = Inches(2.0)  # where slide contents start
    bottom = Inches(6.85)  # where they end (the footer is below)

    def slide(self, prs):
        s = blank(prs, self.bg)
        transition(s, "fade")
        return s

    def content(self, prs, n, kicker, title):
        s = self.slide(prs)
        self.header(s, kicker, title)
        self.footer(s, n)
        return s

    def header(self, s, kicker, title):
        m = self.margin
        line(s, m, Inches(0.62), m + Inches(0.45), Inches(0.62), color=self.accent, weight=Pt(2.25))
        text(s, m, Inches(0.72), Inches(10), Inches(0.3), kicker.upper(), size=11, color=self.accent, bold=True,
             spacing=1.6, font=self.body)
        text(s, m, Inches(1.0), W - 2 * m, Inches(0.8), title, size=30, font=self.heading, color=self.ink)

    def footer(self, s, n):
        m = self.margin
        text(s, m, H - Inches(0.5), Inches(8), Inches(0.25), SHORT_TITLE, size=9, color=self.muted, font=self.body)
        text(s, W - m - Inches(1), H - Inches(0.5), Inches(1), Inches(0.25), str(n), size=9, color=self.muted,
             align=PP_ALIGN.RIGHT, font=self.body)

    def title_slide(self, prs, meta):
        s, m = self.slide(prs), self.margin
        text(s, m, Inches(1.25), Inches(9), Inches(0.3), meta["kind"].upper(), size=12, color=self.accent,
             bold=True, spacing=2, font=self.body)
        text(s, m, Inches(1.65), W - 2 * m, Inches(2.0), meta["title"], size=40, font=self.heading,
             color=self.ink, line=1.02)
        text(s, m, Inches(3.75), Inches(11), Inches(0.5), meta["subtitle"], size=20, color=self.muted,
             font=self.body)
        line(s, m, Inches(4.7), m + Inches(0.9), Inches(4.7), color=self.accent, weight=Pt(2.25))
        text(s, m, Inches(4.95), Inches(11), Inches(1.6),
             [[(meta["authors"], {"bold": True})],
              [(meta["guide"], {"color": self.muted, "size": 14})],
              [(meta["place"], {"color": self.muted, "size": 14})]], size=17, line=1.4, font=self.body,
             color=self.ink)
        return s

    def section_slide(self, prs, n, part, title, lead):
        s, m = self.slide(prs), self.margin
        text(s, m, Inches(2.2), Inches(3), Inches(1.4), f"{part:02d}", size=80, font=self.heading,
             color=self.accent)
        line(s, m, Inches(3.75), m + Inches(0.9), Inches(3.75), color=self.accent, weight=Pt(2.25))
        text(s, m, Inches(4.0), Inches(11), Inches(0.9), title, size=40, font=self.heading, color=self.ink)
        text(s, m, Inches(4.95), Inches(11), Inches(0.6), lead, size=18, color=self.muted, font=self.body)
        self.footer(s, n)
        return s
