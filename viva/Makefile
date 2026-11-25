# The viva preparation document. Numbers come from ../report/generated/numbers.tex (run `make data` in report/ first
# when results change).
.PHONY: all clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error viva.tex
clean:
	latexmk -c viva.tex
