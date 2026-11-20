# The project report.
#   make        numbers, tables and figures from ../results/, then report.pdf
#   make data   only the generated files (generated/)
#   make pdf    only the PDF (LaTeX)
#   make clean  remove LaTeX's temporary files (keeps generated/ and report.pdf)
PY ?= ../.venv/bin/python

.PHONY: all data pdf clean
all: data pdf

data:
	$(PY) scripts/build.py

pdf:
	latexmk -pdf -jobname=report -interaction=nonstopmode -halt-on-error -file-line-error main.tex

clean:
	latexmk -c -jobname=report main.tex
