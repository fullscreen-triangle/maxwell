@echo off
REM Compilation script for Temporal Resolution through Spectral Multiplexing paper
REM Usage: compile.bat

setlocal

set MAIN=temporal-resolution-spectral-multiplexing

echo ==========================================
echo Compiling: Temporal Super-Resolution
echo           through Spectral Multiplexing
echo ==========================================
echo.

REM First pass
echo [1/4] Running pdflatex (first pass)...
pdflatex -interaction=nonstopmode "%MAIN%.tex" >nul 2>&1
if errorlevel 1 (
    echo ERROR: First pdflatex pass failed!
    echo Check %MAIN%.log for details
    exit /b 1
)

REM BibTeX
echo [2/4] Running bibtex...
bibtex "%MAIN%" >nul 2>&1
if errorlevel 1 (
    echo WARNING: BibTeX failed (may be OK if no citations^)
)

REM Second pass
echo [3/4] Running pdflatex (second pass)...
pdflatex -interaction=nonstopmode "%MAIN%.tex" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Second pdflatex pass failed!
    echo Check %MAIN%.log for details
    exit /b 1
)

REM Third pass (resolve all references)
echo [4/4] Running pdflatex (final pass)...
pdflatex -interaction=nonstopmode "%MAIN%.tex" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Final pdflatex pass failed!
    echo Check %MAIN%.log for details
    exit /b 1
)

REM Clean up auxiliary files
echo.
echo Cleaning up auxiliary files...
del "%MAIN%.aux" "%MAIN%.bbl" "%MAIN%.blg" "%MAIN%.log" "%MAIN%.out" "%MAIN%.toc" 2>nul

echo.
echo ==========================================
echo * Compilation successful!
echo ==========================================
echo.
echo Output: %MAIN%.pdf
echo.

REM Display file size
if exist "%MAIN%.pdf" (
    for %%A in ("%MAIN%.pdf") do echo File size: %%~zA bytes
)

echo.
pause

endlocal

