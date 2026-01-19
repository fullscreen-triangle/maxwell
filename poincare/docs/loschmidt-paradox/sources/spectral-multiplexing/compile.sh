#!/bin/bash

# Compilation script for Temporal Resolution through Spectral Multiplexing paper
# Usage: ./compile.sh

set -e  # Exit on error

MAIN="temporal-resolution-spectral-multiplexing"

echo "=========================================="
echo "Compiling: Temporal Super-Resolution"
echo "          through Spectral Multiplexing"
echo "=========================================="
echo ""

# First pass
echo "[1/4] Running pdflatex (first pass)..."
pdflatex -interaction=nonstopmode "$MAIN.tex" > /dev/null 2>&1 || {
    echo "ERROR: First pdflatex pass failed!"
    echo "Check $MAIN.log for details"
    exit 1
}

# BibTeX
echo "[2/4] Running bibtex..."
bibtex "$MAIN" > /dev/null 2>&1 || {
    echo "WARNING: BibTeX failed (may be OK if no citations)"
}

# Second pass
echo "[3/4] Running pdflatex (second pass)..."
pdflatex -interaction=nonstopmode "$MAIN.tex" > /dev/null 2>&1 || {
    echo "ERROR: Second pdflatex pass failed!"
    echo "Check $MAIN.log for details"
    exit 1
}

# Third pass (resolve all references)
echo "[4/4] Running pdflatex (final pass)..."
pdflatex -interaction=nonstopmode "$MAIN.tex" > /dev/null 2>&1 || {
    echo "ERROR: Final pdflatex pass failed!"
    echo "Check $MAIN.log for details"
    exit 1
}

# Clean up auxiliary files
echo ""
echo "Cleaning up auxiliary files..."
rm -f "$MAIN.aux" "$MAIN.bbl" "$MAIN.blg" "$MAIN.log" "$MAIN.out" "$MAIN.toc" 2>/dev/null

echo ""
echo "=========================================="
echo "✓ Compilation successful!"
echo "=========================================="
echo ""
echo "Output: $MAIN.pdf"
echo ""

# Display page count if pdfinfo is available
if command -v pdfinfo &> /dev/null; then
    PAGES=$(pdfinfo "$MAIN.pdf" 2>/dev/null | grep "Pages:" | awk '{print $2}')
    if [ -n "$PAGES" ]; then
        echo "Document: $PAGES pages"
    fi
fi

# Display file size
if [ -f "$MAIN.pdf" ]; then
    SIZE=$(du -h "$MAIN.pdf" | cut -f1)
    echo "File size: $SIZE"
fi

echo ""

