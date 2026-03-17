#!/usr/bin/env bash
set -euo pipefail

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex is required to build paper/ieee_paper.tex" >&2
  exit 1
fi

if ! command -v bibtex >/dev/null 2>&1; then
  echo "bibtex is required to build paper/ieee_paper.tex" >&2
  exit 1
fi

cd "$(dirname "$0")"
pdflatex ieee_paper.tex
bibtex ieee_paper
pdflatex ieee_paper.tex
pdflatex ieee_paper.tex
