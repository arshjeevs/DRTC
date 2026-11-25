"""Assemble the project website into website/site/ (ignored by git): the page and the files it links to.

    python website/build.py [--repo https://github.com/OWNER/REPO]

Needs the built report, paper, slides figures, simulator and video (make -C report; cd paper && latexmk -pdf
paper.tex; python presentation/simulator/build.py; python presentation/video.py). Nothing is published: upload
website/site/ wherever you decide (GitHub Pages, a university page). The viva notes are never copied.
"""

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "website" / "site"
FILES = {
    "report.pdf": REPO / "report" / "report.pdf",
    "paper.pdf": REPO / "paper" / "paper.pdf",
    "simulator.html": REPO / "presentation" / "out" / "drtc_simulator.html",
    "video/drtc_video.mp4": REPO / "presentation" / "out" / "drtc_video.mp4",
    **{f"figures/{n}.png": REPO / "report" / "generated" / "slides" / f"{n}.png"
       for n in ("fig_main_success", "fig_main_timeline", "fig_v2_e1_sort_one", "fig_v2_stack", "fig_bench")},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default="https://github.com/OWNER/REPO", help="the public code link")
    a = ap.parse_args()
    SITE.mkdir(parents=True, exist_ok=True)
    page = (REPO / "website" / "index.html").read_text().replace("https://github.com/OWNER/REPO", a.repo)
    (SITE / "index.html").write_text(page)
    missing = []
    for name, src in FILES.items():
        if not src.is_file():
            missing.append(f"{name} (from {src.relative_to(REPO)})")
            continue
        dst = SITE / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"site: {SITE}")
    for m in missing:
        print(f"  missing: {m}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
