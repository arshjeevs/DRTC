"""Build the DRTC Fault Simulator: page.html with model.js inlined.

    python presentation/simulator/build.py

Writes presentation/out/drtc_simulator.html (a complete page that opens offline in any browser) and
presentation/out/drtc_simulator_artifact.html (the same page without the document skeleton, for publishing as a
Claude artifact). Fonts: Space Grotesk and Inter from Google Fonts when online, else the copies installed with the
deck (presentation/fonts/), else the system font.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"


def main() -> None:
    page = (HERE / "page.html").read_text()
    model = (HERE / "model.js").read_text()
    assert "/*MODEL*/" in page
    body = page.replace("/*MODEL*/", model)
    OUT.mkdir(exist_ok=True)
    (OUT / "drtc_simulator_artifact.html").write_text(body)
    head, sep, rest = body.partition("</style>")
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            + head + sep + "\n</head>\n<body>\n" + rest + "\n</body>\n</html>\n")
    (OUT / "drtc_simulator.html").write_text(full)
    print(f"wrote {OUT / 'drtc_simulator.html'} and {OUT / 'drtc_simulator_artifact.html'}")


if __name__ == "__main__":
    main()
