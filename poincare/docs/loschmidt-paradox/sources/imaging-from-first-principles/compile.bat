@echo off
REM Compile imaging categorical partitions paper

echo Compiling imaging-categorical-partitions.tex...

REM First pass
pdflatex -interaction=nonstopmode imaging-categorical-partitions.tex

REM BibTeX
bibtex imaging-categorical-partitions

REM Second pass (resolve references)
pdflatex -interaction=nonstopmode imaging-categorical-partitions.tex

REM Third pass (ensure all references are correct)
pdflatex -interaction=nonstopmode imaging-categorical-partitions.tex

echo Compilation complete! Output: imaging-categorical-partitions.pdf

REM Clean up auxiliary files
del /Q *.aux *.log *.out *.bbl *.blg *.toc 2>nul

echo Cleaned up auxiliary files.
pause

