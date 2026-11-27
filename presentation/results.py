"""The report's numbers and title-page details, as plain text for the slides.

Numbers come from report/generated/numbers.tex (written by `make data` in report/ from the results), so the deck
and the report always quote the same values; a missing key reads "pending". The title-page details come from
report/config.tex, so filling them in there fills both.
"""

from __future__ import annotations

import re

from deck import REPO

REPORT = REPO / "report"


def _plain(tex: str) -> str:
    s = tex.replace("{,}", ",").replace("\\%", "%").replace("$^\\circ$", "°").replace("--", "–")
    s = s.replace("~", "\u00a0").replace("\\,", "\u2009")
    s = re.sub(r"\$p\s*([<=])\s*([0-9.]+)\$", lambda mm: f"p {mm.group(1)} {mm.group(2)}", s)
    return s.replace("$", "")


def _load() -> dict[str, str]:
    path = REPORT / "generated" / "numbers.tex"
    if not path.is_file():
        return {}
    return {k: _plain(v) for k, v in re.findall(r"\\setR\{([^}]*)\}\{(.*)\}\s*$", path.read_text(), re.M)}


NUM = _load()


def R(key: str, default: str = "pending") -> str:
    return NUM.get(key, default)


def F(key: str) -> float | None:
    """The leading number of a value (e.g. 43.9 from '43.9', 80 from '80%'), or None."""
    m = re.match(r"-?[0-9.]+", NUM.get(key, ""))
    return float(m.group()) if m else None


def K(key: str) -> tuple[int, int] | None:
    """k, n from 'k/n'."""
    m = re.match(r"(\d+)/(\d+)", NUM.get(key, ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def title_page() -> dict[str, str]:
    tex = (REPORT / "config.tex").read_text()

    def cmd(name):
        m = re.search(r"\\newcommand\{\\" + name + r"\}\{%?\s*(.*?)\}\s*\n\\", tex + "\n\\", re.S)
        return m.group(1).strip() if m else ""

    authors = re.findall(r"#1\{([^}]*)\}\{[^}]*\}", tex)  # \ReportStudents: #1{Name}{Roll number}
    return {"title": cmd("ReportTitle"), "subtitle": cmd("ReportSubtitle"),
            "authors": "  ·  ".join(a for a in authors if a),
            "guide": f"Guide: {cmd('ReportGuideName')}, {cmd('ReportGuideTitle')}",
            "place": f"{cmd('ReportDepartment')}, {cmd('ReportInstitution')}  ·  {cmd('ReportDate')}"}
