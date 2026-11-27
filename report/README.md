# The project report (LaTeX)

*The presentation is a separate Slides deck built from the same figures: see [lab/REPORT.md](../lab/REPORT.md), section 12.*

The report we submit. Everything in it that comes from an experiment (numbers in the text, tables, graphs) is
generated from the result files, so it updates when new results arrive. The text is written in `chapters/`.

## Build

```bash
cd report
make          # results -> generated/ (numbers, tables, figures) -> report.pdf
make pdf      # only the PDF, e.g. after editing a chapter
```

Needs: pdflatex and latexmk (TeX Live), and the project's environment with the `report` extra (matplotlib):
`uv pip install matplotlib` in the laptop's venv is already done. The results must be on this machine
(`lab eval report <eval> --pull`; the offline files come with it).

## Working on it from another computer (a teammate)

Everything the report and the slides need is in git, so a fresh clone can edit and build them:
the chapters, the diagrams (`figures/`), the photos (`assets/`), and every generated number, table and chart
(`generated/`, `data/`). The built PDF and the built slides are **not** in git (`report.pdf`, `presentation/out/`):
build them after pulling.

```bash
git clone git@github.com:arshjeevs/DRTC.git drtc-major && cd drtc-major && git checkout major   # once
git pull                                    # every time
```

**Edit the text and rebuild the PDF** - needs only LaTeX (TeX Live with `latexmk`; on Ubuntu:
`sudo apt install texlive-full latexmk`):
```bash
cd report && make pdf                       # -> report/report.pdf
```

**Rebuild the slides** - needs the project's Python environment (it imports the project's own code), set up once as
on the laptop (`lab/docs/SETUP-LAPTOP.md`: `uv sync`), plus python-pptx, and LibreOffice for the PDF and previews:
```bash
uv sync && uv pip install python-pptx       # once
.venv/bin/python presentation/build.py      # -> presentation/out/presentation.pptx
```

**Regenerate the numbers, tables and charts from the results** (`make`, not only `make pdf`) needs the trial
records in `results/`, which are not in git: they are in the private results store on the Hugging Face Hub
(`orangemoon599/drtc-results`), so this step needs the project's environment and access to that dataset. A teammate who
only edits text never needs it: the generated files are committed.

**Send changes back:** `git add report presentation && git commit -m "..." && git push` (pull first if someone else
pushed). Never edit `generated/` by hand: it is rebuilt from the results.

## Layout

| Path | What |
|---|---|
| `main.tex` | the document: front matter, chapters, references, appendices (order of `\include`s) |
| `config.tex` | **title page details** (title, names, roll numbers, guide, department, institution, date) and which front-matter pages to include |
| `preamble.tex` | packages, layout (NIT Calicut format: A4, Times 12 pt, 1.15 spacing), and the macros below |
| `frontmatter/` | title page, certificate, declaration, acknowledgements, abstract, abbreviations |
| `chapters/` | `01_introduction` ... `12_conclusion` (platform, data, policies, control, evaluation, design, results, edge, discussion), appendices `A_recipes`, `B_metrics`, `C_conditions`, `D_reproducibility` |
| `figures/` | diagrams drawn in TikZ (`colors.tex` shared; pipeline, machines, architecture, fault mix, setup v2 design) |
| `assets/` | photos you add (see `assets/README.md`); until a file exists, the report shows a framed placeholder |
| `scripts/build.py`, `scripts/v2_results.py` | read `../results/` and the registry, write `generated/`; setup v2's, E4's and the baselines' tables appear when their results exist |
| `generated/` | **made by the build, never edited**: `numbers.tex`, `tab_*.tex`, `fig_*.pdf`; `slides/fig_*.png` for the presentation |
| `references.bib` | references (entries marked `% check` need their details confirmed) |

## Rules

- **Colour only in figures** (graphs and diagrams). Text, headings, links and tables are black.
- **No number typed by hand** in the text when it comes from a result: write `\R{key}`. The build writes the value; a
  key without a result yet prints *(pending)*. Keys: open `generated/numbers.tex`.
- Generated tables: `\gentable{tab_name}`; generated figures: `\genfigure{fig_name}`; photos:
  `\photo[width]{file.jpg}{what the photo should show}`.

## Adding something later

| To add | Do |
|---|---|
| A new result (e.g. a baseline eval) | pull it (`lab eval report <eval> --pull`); in `scripts/build.py` add a function that sets its numbers (`setn`), writes its table and figure; call it in `main()`; write the text with `\R{...}` in the chapter; `make` |
| A photo | put it in `assets/` under the name the placeholder shows; `make pdf` |
| A chapter or section | a new file in `chapters/`, `\include` it in `main.tex` |
| A reference | add it to `references.bib`, cite with `\cite{key}` |
