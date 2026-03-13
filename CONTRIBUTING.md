# Contributing to AI-NIDS

Thank you for your interest in contributing! This project aims to be the most comprehensive open-source ML-based Network Intrusion Detection System. All contributions are welcome.

## Ways to Contribute

- 🐛 **Bug reports** — open a GitHub Issue with steps to reproduce
- 💡 **Feature ideas** — open an Issue describing the proposal first
- 📊 **New datasets** — add a normalizer in `src/dataset_downloader.py`
- 🤖 **New models** — add a model class in `src/model.py`
- 📝 **Documentation** — improve README, docstrings, or architecture notes
- ✅ **Tests** — add test cases in `tests/`

---

## Development Setup

```bash
git clone https://github.com/dheerajramasahayam/ai-network-threat-detection.git
cd ai-network-threat-detection

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify everything works
python -m pytest tests/ -v
```

---

## Project Structure

| File/Dir | Purpose |
|---|---|
| `src/preprocessing.py` | Feature schema, cleaning, encoding, scaling |
| `src/model.py` | `IntrusionDetectionModel` wrapper for RF & XGBoost |
| `src/training.py` | End-to-end training pipeline (`--dataset`, `--tag` flags) |
| `src/detection.py` | `ThreatDetector` for real-time inference |
| `src/anomaly.py` | Zero-day unsupervised ensemble |
| `src/benchmark.py` | Cross-dataset generalization benchmark |
| `src/attack_report.py` | Per-attack-class precision/recall breakdown |
| `src/adversarial.py` | Adversarial robustness tests |
| `src/latency_bench.py` | Inference latency benchmark |
| `src/dataset_downloader.py` | Kaggle dataset downloader + schema normalizer |
| `src/api.py` | FastAPI REST endpoints |
| `tests/test_model.py` | Unit + integration tests |

---

## Adding a New Dataset

1. Add a download + normalize function in `src/dataset_downloader.py`
2. Map your dataset's columns to the 41 canonical `FEATURE_COLUMNS` in `src/preprocessing.py`
3. Add `_source` column with a unique dataset identifier (used by the benchmark)
4. Test it:
   ```bash
   python src/training.py --dataset dataset/your_dataset.csv --label-col label --tag your_tag
   ```

---

## Canonical Feature Schema

All datasets must be normalized to the 41 features in `FEATURE_COLUMNS` (see `src/preprocessing.py`). The `_CICIDS_COL_MAP` dictionary maps CICIDS2017 mixed-case names to canonical lowercase-underscore names. Add an equivalent map for your dataset.

---

## Adding a New Model

1. In `src/model.py`, add a new `elif model_type == "your_model":` branch in `IntrusionDetectionModel.__init__`
2. Ensure your model implements `.fit(X, y)` and `.predict_proba(X)` returning a 1D probability array
3. Run training: `python src/training.py --model your_model`
4. Add test cases in `tests/test_model.py`

---

## Pull Request Guidelines

1. **Branch** from `main`: `git checkout -b feat/your-feature`
2. **Tests pass**: `python -m pytest tests/ -v` — all 21 must pass
3. **No syntax errors**: `flake8 src/ --select=E9,F63,F7,F82`
4. **Descriptive commit message** — follow the existing style
5. **Update README** if you add a major feature
6. Open the PR against `main` and fill in the description template

---

## Code Style

- Python 3.10+ syntax
- Max line length: 120 characters
- Type hints where practical
- Docstrings on all public functions
- `logger.info()` for progress, `logger.warning()` for recoverable issues

---

## Questions?

Open a GitHub Discussion or Issue — we respond within 24 hours.
