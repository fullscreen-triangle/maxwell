#!/bin/bash
# Compile imaging categorical partitions paper

echo "Compiling imaging-categorical-partitions.tex..."

# First pass
pdflatex -interaction=nonstopmode imaging-categorical-partitions.tex

# BibTeX
bibtex imaging-categorical-partitions

# Second pass (resolve references)
pdflatex -interaction=nonstopmode imaging-categorical-partitions.tex

# Third pass (ensure all references are correct)
pdflatex -interaction=nonstopmode imaging-categorical-partitions.tex

echo "Compilation complete! Output: imaging-categorical-partitions.pdf"

# Clean up auxiliary files
rm -f *.aux *.log *.out *.bbl *.blg *.toc

echo "Cleaned up auxiliary files."

