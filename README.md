# Drift-Adaptive Intrusion Detection for Enterprise Networks

## Abstract

This repository is organized as a reproducible research project for enterprise network-threat detection across multiple public benchmarks. It aligns `UNSW-NB15`, `NSL-KDD`, and `CICIDS2017` into a shared 41-feature flow representation, compares standard IDS and ML baselines, and adds an online drift-adaptive controller on top of a hybrid ensemble.

The benchmark now studies six configurations:

- `Signature IDS`
- `Random Forest`
- `LSTM`
- `Transformer`
- `Drift-Aware Hybrid (static stack)`
- `Drift-Adaptive Hybrid` as the deployment-time upgrade

The key novelty upgrade is not just stacking, but online drift adaptation. On the external `CICIDS2017` holdout, the static hybrid scores `61.33%` weighted `F1`, while the online `Drift-Adaptive Hybrid` improves that to `65.64%` without retraining the base detectors.

## Problem Statement

Traditional IDS engines are strong for deterministic known signatures, but they do not generalize well when traffic distributions shift or when attacks do not match existing rules. Static learned models also degrade when the deployment stream moves away from the source-domain training distribution. This project studies whether a reproducible online adaptation layer can recover part of that loss while keeping the system practical to deploy.

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

All datasets are mapped into a canonical 41-feature flow schema defined in `src/preprocessing.py` and materialized through `training/canonical_pipeline.py`. The feature set includes flow duration, packet counts, byte counts, rate features, inter-arrival timing, packet-length moments, TCP flag counts, and initial window sizes.

The proposed method has two layers:

1. `Drift-Aware Hybrid (static)` learns a stacked meta-classifier over `Signature IDS`, `Random Forest`, `LSTM`, `Transformer`, and a drift signal from `IsolationForest`.
2. `Drift-Adaptive Hybrid` derives stable and stressed ensemble-weight regimes offline, then interpolates between them online as the observed drift score rises in the live stream.

This makes the contribution stronger than a pure stacking benchmark because the system adapts at inference time instead of staying fixed after training.

## Model Architectures

### Signature IDS

A deterministic flow-signature engine over the canonical features. It acts as the traditional IDS baseline for datasets that ship flow records instead of raw packet payloads.

### Random Forest

Classical tabular ensemble baseline and the fastest learned model in the suite.

### LSTM

A recurrent detector that treats the feature vector as an ordered sequence and performs strongly on `NSL-KDD` and the transfer benchmark.

### Transformer

A self-attention detector over the same feature sequence, included as the stronger attention-based deep baseline.

### Drift-Adaptive Hybrid

The new method introduced in this repo. It starts from a static hybrid stack, estimates drift against the training reference distribution, and reweights the ensemble online when the external stream departs from the source-domain regime.

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

The full scripted benchmark trains all base detectors and evaluates:

1. full official `UNSW-NB15`
2. full official `NSL-KDD`
3. external transfer from `UNSW-NB15 + NSL-KDD` into `CICIDS2017`
4. online drift adaptation on the external stream
5. latency-under-load and explainability-ablation artifacts

## Results

### Official UNSW-NB15

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |
| --- | --- | --- | --- | --- | --- | --- |
| Drift-Aware Hybrid (Static) | 90.53% | 91.60% | 90.53% | 90.72% | 0.9816 | 0.1804 |
| Random Forest | 90.46% | 91.55% | 90.46% | 90.65% | 0.9792 | 0.0068 |
| LSTM | 72.84% | 83.67% | 72.84% | 73.60% | 0.9065 | 0.0780 |
| Transformer | 71.15% | 83.42% | 71.15% | 71.87% | 0.8800 | 0.1040 |
| Signature IDS | 68.06% | 46.32% | 68.06% | 55.13% | 0.7418 | 0.0004 |

### Official NSL-KDD

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | 81.08% | 85.23% | 81.08% | 81.01% | 0.9288 | 0.0743 |
| Transformer | 80.61% | 85.04% | 80.61% | 80.52% | 0.9346 | 0.0862 |
| Drift-Aware Hybrid (Static) | 79.23% | 84.52% | 79.23% | 79.06% | 0.9508 | 0.1845 |
| Random Forest | 77.71% | 83.75% | 77.71% | 77.44% | 0.9366 | 0.0074 |
| Signature IDS | 54.66% | 51.32% | 54.66% | 49.47% | 0.3541 | 0.0004 |

### Cross-Dataset Transfer: `UNSW-NB15 + NSL-KDD -> CICIDS2017`

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | 80.30% | 64.48% | 80.30% | 71.53% | 0.2901 | 0.0718 |
| Transformer | 80.16% | 64.46% | 80.16% | 71.46% | 0.4974 | 0.0809 |
| Drift-Adaptive Hybrid | 67.28% | 64.14% | 67.28% | 65.64% | 0.4537 | 0.4343 |
| Random Forest | 58.48% | 65.82% | 58.48% | 61.59% | 0.4681 | 0.0062 |
| Signature IDS | 53.42% | 67.84% | 53.42% | 58.14% | 0.4643 | 0.0004 |

### Online Drift Adaptation

The strongest new result is the deployment-time adaptation ablation on the external stream:

| Variant | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Static Hybrid | 57.52% | 67.44% | 57.52% | 61.33% | 0.4840 |
| Online Drift-Adaptive Hybrid | 67.28% | 64.14% | 67.28% | 65.64% | 0.4537 |

This is a `+4.31` weighted `F1` gain over the static hybrid under external drift, without retraining the already-trained base models.

Measured average inference latency on the transfer sample rises from `0.1662 ms/flow` for the static hybrid to `0.4343 ms/flow` for the online controller, so the gain is robustness under shift rather than raw speed.

Artifacts:

- `results/transfer_unsw_nsl_to_cicids_online_drift_adaptation.csv`
- `results/transfer_unsw_nsl_to_cicids_online_drift_adaptation.md`
- `results/transfer_unsw_nsl_to_cicids_online_drift_adaptation.png`

### Interpretation

- The static `Drift-Aware Hybrid` remains best on the official `UNSW-NB15` split.
- `LSTM` remains the strongest model on `NSL-KDD` and the external transfer benchmark.
- The new novelty result is that online drift adaptation materially improves the hybrid under external shift.
- Cross-dataset generalization is still difficult, but the adaptation layer recovers part of that gap.

## Comparison with Traditional IDS

This repository includes a quantitative rule-based baseline instead of a purely qualitative discussion.

- On `UNSW-NB15`, `Signature IDS` reaches `55.13%` weighted `F1`, versus `90.72%` for the static hybrid.
- On `NSL-KDD`, `Signature IDS` reaches `49.47%` weighted `F1`, versus `81.01%` for `LSTM`.
- On the external `CICIDS2017` transfer holdout, `Signature IDS` reaches `58.14%` weighted `F1`, the static hybrid reaches `61.33%`, the online `Drift-Adaptive Hybrid` reaches `65.64%`, and `LSTM` remains best at `71.53%`.
- The rule baseline is still the fastest detector: on `UNSW-NB15` at batch size `1024`, it sustains about `1.28M` flows/s.

Because these public benchmarks are distributed as flow records rather than raw packet payloads, the traditional baseline is implemented as a transparent flow-signature IDS instead of direct `Snort` or `Suricata` packet replay.

## Hard Contributions

The repo now goes beyond a basic benchmark in four ways:

- `cross-dataset transfer` from two training corpora into an external holdout
- `online drift adaptation` through the new `Drift-Adaptive Hybrid`
- `online latency under load` across multiple batch sizes
- `explainability validated by ablation` using feature-importance interventions

## Demo

After training, run:

```bash
python3 demo/run_detection.py --sample-index 0
```

The demo loads the best official `UNSW-NB15` model from `models/artifacts/advanced_metadata.json` and scores a real sample from the official test split.

## Reproducibility

This repository includes:

- `requirements.txt`
- `Dockerfile`
- `docker/Dockerfile`
- `run_training.sh`
- `paper/ieee_paper.tex`
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
├── paper/
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
  title={Drift-Adaptive Intrusion Detection for Enterprise Networks},
  author={Ramasahayam, Dheeraj},
  year={2026},
  journal={GitHub Research Repository}
}
```

## License

This repository is distributed under `CC BY 4.0`.
