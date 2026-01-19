@echo off
REM Compilation script for Partition-Based Imaging paper

echo Compiling Partition-Based Imaging paper...
echo ==========================================

REM First pass
echo First pass: pdflatex...
pdflatex -interaction=nonstopmode partitioning-based-imaging-techniques.tex

REM BibTeX
echo Running BibTeX...
bibtex partitioning-based-imaging-techniques

REM Second pass
echo Second pass: pdflatex...
pdflatex -interaction=nonstopmode partitioning-based-imaging-techniques.tex

REM Third pass (for cross-references)
echo Third pass: pdflatex...
pdflatex -interaction=nonstopmode partitioning-based-imaging-techniques.tex

REM Clean up auxiliary files (optional)
echo Cleaning up auxiliary files...
del *.aux *.log *.out *.bbl *.blg *.toc 2>nul

echo ==========================================
echo Compilation complete! Output: partitioning-based-imaging-techniques.pdf

pause

