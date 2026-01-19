#!/bin/bash

# Compilation script for Partition-Based Imaging paper

echo "Compiling Partition-Based Imaging paper..."
echo "=========================================="

# First pass
echo "First pass: pdflatex..."
pdflatex -interaction=nonstopmode partitioning-based-imaging-techniques.tex

# BibTeX
echo "Running BibTeX..."
bibtex partitioning-based-imaging-techniques

# Second pass
echo "Second pass: pdflatex..."
pdflatex -interaction=nonstopmode partitioning-based-imaging-techniques.tex

# Third pass (for cross-references)
echo "Third pass: pdflatex..."
pdflatex -interaction=nonstopmode partitioning-based-imaging-techniques.tex

# Clean up auxiliary files (optional)
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.bbl *.blg *.toc

echo "=========================================="
echo "Compilation complete! Output: partitioning-based-imaging-techniques.pdf"

