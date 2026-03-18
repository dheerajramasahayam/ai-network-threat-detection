# IEEE Paper Source

This directory contains the release-ready IEEEtran manuscript for the repository.

## Files

- `ieee_paper.tex`: main manuscript source
- `references.bib`: bibliography database
- `build.sh`: local build script that prefers `tectonic` and falls back to `pdflatex` plus `bibtex`
- `deployment_architecture.png`: production deployment figure used in the manuscript
- `ieee_paper.pdf`: built manuscript output

## Build

From the repository root:

```bash
bash paper/build.sh
```

The build script writes `paper/ieee_paper.pdf` and then syncs the same artifact to the repository root as `research_paper.pdf` so the GitHub landing page and the paper source stay aligned.

## Referenced Figures

The manuscript expects these figure assets to remain in place:

- `../architecture.png`
- `deployment_architecture.png`
- `../results/official_unsw_roc_curve.png`
- `../results/official_unsw_latency_under_load.png`
- `../results/transfer_unsw_nsl_to_cicids_drift_timeline.png`
- `../results/realtime_service_case_study.png`

## Packaging Notes

- The authoritative source is `ieee_paper.tex`.
- `ieee_paper.pdf` is the canonical built PDF inside `paper/`.
- `research_paper.pdf` is a root-level mirror for easier GitHub download and citation.
- `RELEASE.md` in the repository root documents the final artifact set and verification steps.

If you prefer Overleaf, upload this directory together with the referenced figures listed above.
