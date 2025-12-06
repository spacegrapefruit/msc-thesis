$pdf_mode = 1; # Keep this as 1 to satisfy VS Code looking for "pdflatex" mode
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';