"""The deck's style (2026-11-25): the look of the first browser deck (claude.ai Slides, 2026-11-20), as a .pptx.

Warm paper slides for content; ink-dark slides for the title, the part openers and the close; one blue statement
slide for the claim. Space Grotesk headings, Inter text (both OFL, in presentation/fonts/: install them on the
computer that presents). A small uppercase blue label above every title, rounded cards with a hairline border, big
numbers. Every slide fades in; parts of a slide rise in click by click. Slide contents use only the attributes below,
so the look is changed here and nowhere else.
"""

from __future__ import annotations

from pptx.util import Inches

from deck import H, W, blank, text, transition

SHORT_TITLE = "Reliable remote inference under network faults"


class Style:
    # paper (content) slides
    bg, ink, muted, grid = "#F6F5F1", "#15191F", "#5E636D", "#DEDBD2"
    accent, alert = "#1C5CAB", "#EB6834"  # accent = text and rules; alert = the faults
    panel, panel_line, panel_ink = "#FCFBF8", "#DEDBD2", "#15191F"
    # dark (title, parts, close) and accent (statement) slides
    dark, dark_text, dark_muted, dark_accent = "#15191F", "#F6F5F1", "#A9AEB8", "#6DA7EC"
    on_accent, on_accent_muted = "#F6F5F1", "#DCE6F5"
    heading, body = "Space Grotesk", "Inter"
    series = {"baseline": "#2A78D6", "faults": "#EB6834", "drtc": "#1BAF7A", "smooth": "#EDA100"}
    card_radius = 0.07
    margin = Inches(0.89)  # 128 px of the browser deck's 1920
    top = Inches(2.0)  # where slide contents start
    bottom = Inches(6.75)  # where they end (the footer is below)

    def slide(self, prs, bg=None):
        s = blank(prs, bg or self.bg)
        transition(s, "fade")
        return s

    def content(self, prs, n, kicker, title):
        s = self.slide(prs)
        self.header(s, kicker, title)
        self.footer(s, n)
        return s

    def eyebrow(self, s, y, words, color, x=None):
        return text(s, x if x is not None else self.margin, y, Inches(11), Inches(0.3), words.upper(), size=12,
                    color=color, bold=True, spacing=1.5, font=self.body)

    def header(self, s, kicker, title):
        m = self.margin
        self.eyebrow(s, Inches(0.62), kicker, self.accent)
        text(s, m, Inches(0.95), W - 2 * m, Inches(0.8), title, size=self.title_size(title, W - 2 * m), bold=True,
             font=self.heading, color=self.ink, line=1.05)

    @staticmethod
    def title_size(title, width, sizes=(32, 30, 28, 26, 24)):
        """The largest size at which the title fits on one line (Space Grotesk bold: about 0.6 em a character)."""
        return next((sz for sz in sizes if len(title) * 0.6 * sz <= width / 12700), sizes[-1])

    def footer(self, s, n, dark=False):
        m = self.margin
        text(s, m, H - Inches(0.62), Inches(10), Inches(0.25), f"{SHORT_TITLE}  ·  {n}", size=11,
             color=self.dark_muted if dark else self.muted, font=self.body)

    def title_slide(self, prs, meta):
        s, m = self.slide(prs, self.dark), self.margin
        self.eyebrow(s, Inches(0.89), f"{meta['kind']}  ·  {meta['place'].split(',')[0]}", self.dark_accent)
        text(s, m, Inches(1.3), W - 2 * m, Inches(2.4), meta["title"], size=44, bold=True, font=self.heading,
             color=self.dark_text, line=1.02)
        text(s, m, Inches(3.85), Inches(10.5), Inches(0.9), meta["subtitle"], size=20, color=self.dark_muted,
             font=self.heading, line=1.25)
        text(s, m, Inches(5.55), Inches(11), Inches(1.1),
             [[(meta["authors"], {"color": self.dark_text})],
              [(f"{meta['guide']}  ·  {meta['place']}", {"color": self.dark_muted, "size": 12})]], size=15,
             line=1.5, font=self.body)
        return s

    def section_slide(self, prs, n, part, title, lead):
        s, m = self.slide(prs, self.dark), self.margin
        self.eyebrow(s, Inches(2.15), f"Part {part}", self.dark_accent)
        text(s, m, Inches(2.5), Inches(3), Inches(1.4), f"{part:02d}", size=72, bold=True, font=self.heading,
             color=self.dark_accent)
        text(s, m, Inches(3.85), Inches(11.5), Inches(0.9), title, size=44, bold=True, font=self.heading,
             color=self.dark_text)
        text(s, m, Inches(4.8), Inches(11), Inches(0.6), lead, size=18, color=self.dark_muted, font=self.body)
        self.footer(s, n, dark=True)
        return s

    def statement_slide(self, prs, n, kicker):
        """The blue statement slide (the claim): the label only; the slide writes its lines in on_accent."""
        s = self.slide(prs, self.accent)
        self.eyebrow(s, Inches(1.2), kicker, self.on_accent_muted)
        text(s, self.margin, H - Inches(0.62), Inches(10), Inches(0.25), f"{SHORT_TITLE}  ·  {n}", size=11,
             color=self.on_accent_muted, font=self.body)
        return s

    def closing_slide(self, prs, n, title, lead):
        s, m = self.slide(prs, self.dark), self.margin
        text(s, m, Inches(2.7), W - 2 * m, Inches(1.2), title, size=60, bold=True, font=self.heading,
             color=self.dark_text)
        text(s, m, Inches(3.95), W - 2 * m, Inches(0.6), lead, size=24, color=self.dark_muted, font=self.heading)
        self.footer(s, n, dark=True)
        return s
