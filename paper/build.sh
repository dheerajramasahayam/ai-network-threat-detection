#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v tectonic >/dev/null 2>&1; then
  tectonic ieee_paper.tex --outdir .
  exit 0
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "Either tectonic or pdflatex is required to build paper/ieee_paper.tex" >&2
  exit 1
fi

if ! command -v bibtex >/dev/null 2>&1; then
  echo "bibtex is required when building paper/ieee_paper.tex with pdflatex" >&2
  exit 1
fi

pdflatex ieee_paper.tex
bibtex ieee_paper
pdflatex ieee_paper.tex
pdflatex ieee_paper.tex
