#!/bin/bash

# Compilation script for Lunar Surface Imaging paper

echo "Compiling Lunar Surface Imaging paper..."
echo "=========================================="

# First pass
echo "First pass: pdflatex..."
pdflatex -interaction=nonstopmode lunar-surface-imaging.tex

# BibTeX
echo "Running BibTeX..."
bibtex lunar-surface-imaging

# Second pass
echo "Second pass: pdflatex..."
pdflatex -interaction=nonstopmode lunar-surface-imaging.tex

# Third pass (for cross-references)
echo "Third pass: pdflatex..."
pdflatex -interaction=nonstopmode lunar-surface-imaging.tex

# Clean up auxiliary files (optional)
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.bbl *.blg *.toc

echo "=========================================="
echo "Compilation complete! Output: lunar-surface-imaging.pdf"

