# AI-Based Network Threat Detection Using Deep Learning for Enterprise Infrastructure

## Abstract

This repository is organized as a reproducible research project for enterprise network-threat detection across multiple public benchmarks. It aligns `UNSW-NB15`, `NSL-KDD`, and `CICIDS2017` into a shared 41-feature flow representation, then compares five detectors:

- `Signature IDS` as a transparent traditional rule baseline
- `Random Forest`
- `LSTM`
- `Transformer`
- `Drift-Aware Hybrid` as the proposed new method

The upgraded workflow answers three research questions:

1. Can learned models outperform a traditional rule-based IDS baseline on public enterprise traffic datasets?
2. Which model family performs best on full official train/test splits?
3. How much performance survives cross-dataset transfer to external traffic?

The current published run uses the full official `UNSW-NB15` split, the full official `NSL-KDD` split, and a stratified `5,000`-row external `CICIDS2017` holdout for transfer evaluation. On that run, the `Drift-Aware Hybrid` is the best `UNSW-NB15` model, while `LSTM` is the best performer on `NSL-KDD` and the external transfer setting.

## Problem Statement

Traditional IDS engines are strong for deterministic known signatures, but they do not generalize well when traffic distributions shift or when attacks do not match existing rules. This project studies whether AI-based detectors can beat a reproducible rule baseline, remain fast enough for operational scoring, and preserve useful performance when trained on one set of benchmarks and tested on another.

## Dataset

| Dataset | Rows | Split Used in This Repo | Label Space |
| --- | --- | --- | --- |
| `UNSW-NB15` | `257,673` | official `82,332` train / `175,341` test | `10` attack families collapsed to binary attack detection |
| `NSL-KDD` | `148,517` | official `125,973` train / `22,544` test | `40` symbolic labels collapsed to binary attack detection |
| `CICIDS2017` | `2,830,743` | external holdout for transfer evaluation | `15` traffic labels |

Supporting materials:

- `dataset/README.md`
- `dataset/preprocessing.ipynb`
- `notebooks/feature_engineering.ipynb`

## Methodology

All datasets are mapped into a canonical 41-feature flow schema defined in `src/preprocessing.py` and materialized through `training/canonical_pipeline.py`. The feature set includes:

- flow duration
- forward and backward packet counts
- forward and backward byte counts
- flow bytes per second and packets per second
- forward and backward inter-arrival timing
- packet length moments
- TCP flag counts
- initial window sizes
- active data packet counts

This makes the benchmark fair across datasets and allows one rule baseline plus four learning models to operate on the same features.

## Model Architectures

### Signature IDS

A deterministic flow-signature engine over the canonical features. It acts as the traditional IDS baseline for datasets that ship flow records instead of raw packet payloads.

### Random Forest

Classical tabular ensemble baseline and the fastest learned model in the suite.

### LSTM

A recurrent detector that treats the feature vector as an ordered sequence and performs strongly on `NSL-KDD` and the transfer benchmark.

### Transformer

A self-attention detector over the same feature sequence, included as the stronger attention-based deep baseline.

### Drift-Aware Hybrid

The new method introduced in this repo. It fuses the `Signature IDS`, `Random Forest`, `LSTM`, and `Transformer` outputs with an `IsolationForest` drift score, then learns a meta-classifier over those signals.

Source files:

- `models/signature_ids.py`
- `models/random_forest.py`
- `models/lstm_model.py`
- `models/transformer_model.py`
- `models/drift_aware_hybrid.py`

## Experiments

Run the upgraded research pipeline with:

```bash
bash run_training.sh --epochs 1 --batch-size 128 --rf-trees 60 --cicids-sample-size 5000
```

This command:

1. regenerates `architecture.png`
2. trains the rule baseline and all four learning models
3. evaluates full official `UNSW-NB15` and `NSL-KDD` splits
4. evaluates cross-dataset transfer from `UNSW-NB15 + NSL-KDD` to external `CICIDS2017`
5. writes latency-under-load and explainability-ablation artifacts to `results/`
6. generates `research_paper.pdf`

For a larger external holdout, increase `--cicids-sample-size` to `50000` or `200000`. Use `0` to score the full merged `CICIDS2017` file.

## Results

### Official UNSW-NB15

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |
| --- | --- | --- | --- | --- | --- | --- |
| Drift-Aware Hybrid | 90.53% | 91.60% | 90.53% | 90.72% | 0.9816 | 0.1804 |
| Random Forest | 90.46% | 91.55% | 90.46% | 90.65% | 0.9792 | 0.0068 |
| LSTM | 72.84% | 83.67% | 72.84% | 73.60% | 0.9065 | 0.0780 |
| Transformer | 71.15% | 83.42% | 71.15% | 71.87% | 0.8800 | 0.1040 |
| Signature IDS | 68.06% | 46.32% | 68.06% | 55.13% | 0.7418 | 0.0004 |

### Official NSL-KDD

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | 81.08% | 85.23% | 81.08% | 81.01% | 0.9288 | 0.0743 |
| Transformer | 80.61% | 85.04% | 80.61% | 80.52% | 0.9346 | 0.0862 |
| Drift-Aware Hybrid | 79.23% | 84.52% | 79.23% | 79.06% | 0.9508 | 0.1845 |
| Random Forest | 77.71% | 83.75% | 77.71% | 77.44% | 0.9366 | 0.0074 |
| Signature IDS | 54.66% | 51.32% | 54.66% | 49.47% | 0.3541 | 0.0004 |

### Cross-Dataset Transfer: `UNSW-NB15 + NSL-KDD -> CICIDS2017`

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | 80.30% | 64.48% | 80.30% | 71.53% | 0.2901 | 0.0718 |
| Transformer | 80.16% | 64.46% | 80.16% | 71.46% | 0.4974 | 0.0809 |
| Random Forest | 58.48% | 65.82% | 58.48% | 61.59% | 0.4681 | 0.0062 |
| Drift-Aware Hybrid | 57.50% | 67.43% | 57.50% | 61.31% | 0.4677 | 0.1662 |
| Signature IDS | 53.42% | 67.84% | 53.42% | 58.14% | 0.4643 | 0.0004 |

### Interpretation

- `Drift-Aware Hybrid` edges out `Random Forest` on the official `UNSW-NB15` split while staying within a practical latency budget.
- `LSTM` is the best model on `NSL-KDD` and on the external transfer setting, which suggests the recurrent inductive bias is more robust than the current tree and hybrid setup under the chosen canonical mapping.
- All models degrade sharply on cross-dataset transfer, which is useful negative evidence: generalization across benchmark families is still a hard research problem.

Generated figures:

- `results/official_unsw_confusion_matrix.png`
- `results/official_unsw_roc_curve.png`
- `results/official_unsw_feature_importance.png`
- `results/official_unsw_latency_under_load.png`
- `results/official_unsw_explainability_ablation.png`

## Comparison with Traditional IDS

This repository now includes a quantitative rule-based baseline instead of a purely qualitative discussion.

- On `UNSW-NB15`, `Signature IDS` reaches `55.13%` weighted `F1`, versus `90.72%` for the `Drift-Aware Hybrid`.
- On `NSL-KDD`, `Signature IDS` reaches `49.47%` weighted `F1`, versus `81.01%` for `LSTM`.
- On the external `CICIDS2017` transfer holdout, `Signature IDS` reaches `58.14%` weighted `F1`, versus `71.53%` for `LSTM`.
- The rule baseline is still the fastest detector: on `UNSW-NB15` at batch size `1024`, it sustains about `1.28M` flows/s, while `Random Forest` sustains about `71k` flows/s and the hybrid about `5.2k` flows/s.

Because these public benchmarks are distributed as flow records rather than raw packet payloads, the traditional baseline is implemented as a transparent flow-signature IDS instead of direct `Snort` or `Suricata` packet replay.

## Hard Contributions

The repo now goes beyond a basic benchmark in four ways:

- `cross-dataset transfer` from two training corpora into an external holdout
- `drift adaptation` through the new `Drift-Aware Hybrid`
- `online latency under load` across multiple batch sizes
- `explainability validated by ablation` using feature-importance interventions

## Demo

After training, run:

```bash
python3 demo/run_detection.py --sample-index 0
```

The demo loads the best official `UNSW-NB15` model from `models/artifacts/advanced_metadata.json` and scores a real sample from the official test split. If the best model is the rule baseline, it also prints the fired signatures.

## Reproducibility

This repository includes:

- `requirements.txt`
- `Dockerfile`
- `docker/Dockerfile`
- `run_training.sh`
- `research_paper.pdf`
- `CITATION.cff`

Raw data is kept under `dataset/raw/` and can stay out of version control, while the merged experiment assets and notebooks remain reproducible from the scripts in `training/` and `evaluation/`.

## Applications

- enterprise threat-detection research
- IDS benchmarking and reproducibility studies
- cybersecurity ML coursework
- investigations of transfer learning and drift in network defense

## Project Structure

```text
ai-network-threat-detection/
├── dataset/
├── demo/
├── evaluation/
├── models/
├── notebooks/
├── results/
├── src/
├── training/
├── architecture.png
├── research_paper.pdf
├── README.md
├── requirements.txt
└── CITATION.cff
```

## Citation

Use the metadata in `CITATION.cff` or cite the project as:

```bibtex
@article{ramasahayam2026,
  title={AI-Based Network Threat Detection Using Deep Learning for Enterprise Infrastructure},
  author={Ramasahayam, Dheeraj},
  year={2026},
  journal={GitHub Research Repository}
}
```

## License

This repository is distributed under `CC BY 4.0`.
