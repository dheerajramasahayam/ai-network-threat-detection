# GitHub Research Release

## Release Summary

This repository is packaged as a paper-first research release for:

`Drift-Adaptive Intrusion Detection for Enterprise Networks`

The release includes:

- an IEEEtran manuscript in `paper/ieee_paper.tex`
- the built paper in `paper/ieee_paper.pdf`
- a root-level PDF mirror in `research_paper.pdf`
- multi-dataset experiment outputs under `results/`
- reproducible training and evaluation entry points under `training/` and `evaluation/`
- citation metadata in `CITATION.cff`

## Key Artifacts

| Artifact | Purpose |
| --- | --- |
| `README.md` | repository landing page and high-level paper summary |
| `paper/ieee_paper.tex` | authoritative manuscript source |
| `paper/ieee_paper.pdf` | authoritative built manuscript |
| `research_paper.pdf` | root-level mirror for easier GitHub download |
| `dataset/README.md` | dataset scope, counts, and preprocessing notes |
| `results/drift_detector_study.md` | formal drift-detector comparison |
| `results/transfer_unsw_nsl_to_cicids_model_comparison.md` | full-corpus transfer benchmark |
| `results/realtime_service_case_study.md` | real-world-style packet replay validation |

## Verified Numbers Included in This Release

- `UNSW-NB15`: static `Drift-Aware Hybrid` weighted `F1 = 90.72%`
- `NSL-KDD`: `LSTM` weighted `F1 = 81.01%`
- `UNSW+NSL -> full CICIDS2017`: `LSTM` weighted `F1 = 71.53%`
- full-stream `Drift-Adaptive Hybrid` on `CICIDS2017`: weighted `F1 = 68.69%`
- drift-detector study: `Isolation Forest` strongest at `70.58%` post-adaptation weighted `F1`
- packet replay case study: `Drift-Adaptive Hybrid` weighted `F1 = 98.05%`

## Release Verification

The release can be revalidated with:

```bash
python3 -m pytest -q tests/test_model.py tests/test_research_upgrade.py
python3 -m compileall src training evaluation demo models tests
bash paper/build.sh
```

The full research suite can be rerun with:

```bash
bash run_training.sh --epochs 1 --batch-size 128 --rf-trees 60 --cicids-sample-size 0
python3 evaluation/run_full_transfer_evaluation.py
python3 evaluation/run_cse_cic_ids2018_transfer_evaluation.py
python3 evaluation/run_drift_detector_study.py
python3 evaluation/run_failure_case_analysis.py
python3 evaluation/realtime_streaming_evaluation.py --source file --chunksize 100000 --max-chunks 5
python3 evaluation/run_realtime_case_study.py
```

## GitHub Publishing Checklist

- confirm `paper/ieee_paper.pdf` and `research_paper.pdf` are identical
- confirm `README.md`, `paper/README.md`, and `CITATION.cff` match the final paper title
- confirm all referenced figures in `paper/` and `results/` exist
- confirm the working tree is clean except for intentionally ignored local raw-data helpers
- push `main` and create a Git tag if you want a versioned GitHub release
