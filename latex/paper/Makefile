NAME:=thesis
TEX_NAME:=$(NAME).tex
PDF_NAME:=$(NAME).pdf
LATEXMK_BASE_FLAGS:=-lualatex -file-line-error -Werror -emulate-aux-dir -aux-directory=out

.PHONY: pdf
pdf:
	latexmk $(LATEXMK_BASE_FLAGS) -halt-on-error $(TEX_NAME)
	open $(PDF_NAME) || setsid xdg-open $(PDF_NAME)

.PHONY: watch
watch: pdf
	latexmk $(LATEXMK_BASE_FLAGS) -interaction=nonstopmode -pvc -view=none $(TEX_NAME) 

.PHONY: ubuntu
ubuntu:
	@echo "Diegiamas LaTeX (LuaTeX, XeTeX ir kt.)"
	sudo apt-get install texlive-full

.PHONY: wordcount
wordcount:
	texcount -total -utf8 $(TEX_NAME)

.PHONY: check
check:
	chktex $(TEX_NAME)

.PHONY: presentation_1
presentation_1:
	cd presentation_1 && latexmk $(LATEXMK_BASE_FLAGS) -halt-on-error *.tex
	open presentation_1/*.pdf || setsid xdg-open presentation_1/*.pdf

.PHONY: clean
clean:
	rm -rf out/
	rm -f $(wildcard *.out *.run.xml *.log *.blg *.bbl *.aux \
	*.toc *.bcf *.synctex.gz *.fdb_latexmk *.fls *.xdv *.nav *.snm)
