# IEEE Paper Source

This directory contains a proper IEEE-style LaTeX paper source for the current benchmark.

## Files

- `ieee_paper.tex`: main manuscript using `IEEEtran`
- `references.bib`: bibliography database
- `build.sh`: local build script for `pdflatex` + `bibtex`

## Build

```bash
cd paper
bash build.sh
```

The paper references figures directly from the repository root:

- `../architecture.png`
- `../results/official_unsw_roc_curve.png`
- `../results/official_unsw_latency_under_load.png`

If you prefer Overleaf, upload this `paper/` directory together with the referenced figures.
