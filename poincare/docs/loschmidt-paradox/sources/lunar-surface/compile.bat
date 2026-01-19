@echo off
REM Compilation script for Lunar Surface Imaging paper

echo Compiling Lunar Surface Imaging paper...
echo ==========================================

REM First pass
echo First pass: pdflatex...
pdflatex -interaction=nonstopmode lunar-surface-imaging.tex

REM BibTeX
echo Running BibTeX...
bibtex lunar-surface-imaging

REM Second pass
echo Second pass: pdflatex...
pdflatex -interaction=nonstopmode lunar-surface-imaging.tex

REM Third pass (for cross-references)
echo Third pass: pdflatex...
pdflatex -interaction=nonstopmode lunar-surface-imaging.tex

REM Clean up auxiliary files (optional)
echo Cleaning up auxiliary files...
del *.aux *.log *.out *.bbl *.blg *.toc 2>nul

echo ==========================================
echo Compilation complete! Output: lunar-surface-imaging.pdf

pause

