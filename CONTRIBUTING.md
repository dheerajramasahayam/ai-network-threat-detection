# Contributing

This repository is maintained as a reproducible research artifact. Contributions should preserve the paper claims, keep experiments rerunnable, and avoid introducing undocumented result changes.

## Development Setup

```bash
git clone https://github.com/dheerajramasahayam/ai-network-threat-detection.git
cd ai-network-threat-detection

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `dataset/` | dataset notes, preprocessing notebook stubs, and local raw-data conventions |
| `src/` | canonical schema, preprocessing, and shared utilities |
| `training/` | training and benchmark orchestration |
| `models/` | baseline and hybrid detector implementations |
| `evaluation/` | transfer, drift, streaming, latency, failure, and case-study analyses |
| `results/` | generated figures, tables, and summaries used by the paper |
| `paper/` | IEEEtran source, bibliography, build script, and built PDF |
| `demo/` | lightweight inference demo |
| `tests/` | regression coverage for training and research-upgrade behavior |

## Data and Artifact Policy

- Do not commit large raw datasets. Keep local corpora under `dataset/raw/` or other ignored paths.
- Do not overwrite committed result files unless you have rerun the corresponding experiment and updated the paper or README where needed.
- Keep the canonical 41-feature schema in sync with any new dataset loader or model input path.

## Common Verification Commands

```bash
python3 -m pytest -q tests/test_model.py tests/test_research_upgrade.py
python3 -m compileall src training evaluation demo models tests
bash paper/build.sh
```

Use the advanced benchmark entry point when a change affects results:

```bash
bash run_training.sh --epochs 1 --batch-size 128 --rf-trees 60 --cicids-sample-size 0
python3 evaluation/run_full_transfer_evaluation.py
python3 evaluation/run_drift_detector_study.py
python3 evaluation/run_realtime_case_study.py
```

## Pull Request Expectations

1. Describe the motivation and the expected effect on the paper, experiments, or reproducibility story.
2. Keep commits focused. Separate documentation-only changes from experimental reruns when possible.
3. Update `README.md`, `paper/README.md`, `RELEASE.md`, and `CITATION.cff` if the public release surface changes.
4. Rebuild `paper/ieee_paper.pdf` with `bash paper/build.sh` when the manuscript or cited figures change.
5. Include the exact verification commands you ran in the pull request description.

## Adding a Dataset or Model

When adding a dataset:

- map it into the canonical schema defined in `src/preprocessing.py`
- add a loader or iterator in `training/canonical_pipeline.py`
- document row counts, label space, and protocol updates in `dataset/README.md`
- add any new cross-dataset evaluation to `evaluation/`

When adding a model:

- place the implementation under `models/`
- integrate it through the research runners in `training/`
- add or extend tests in `tests/`
- update the README and paper if the model enters the reported benchmark

## Release Discipline

- Treat `paper/ieee_paper.pdf` as the authoritative manuscript output.
- `research_paper.pdf` is a root-level mirror and should be kept in sync via `bash paper/build.sh`.
- If a change affects published numbers, update the relevant `results/*.md`, the manuscript, and any summary tables in `README.md`.
