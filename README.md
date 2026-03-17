# AI-Based Network Threat Detection Using Deep Learning for Enterprise Infrastructure

## Abstract

This repository is organized as a reproducible research project for enterprise network-threat detection. It uses the public `UNSW-NB15` benchmark, engineers flow-level features from raw traffic metadata, and compares three detection models:

- `Random Forest`
- `LSTM`
- `Transformer`

The goal is to answer two research questions:

1. Can AI-based models detect network threats accurately enough for practical enterprise monitoring?
2. Which model family performs best for attack detection on a reproducible benchmark?

The bounded CPU benchmark included in this repository uses a stratified `12,000`-row training sample and a `5,000`-row test sample drawn from the official `UNSW-NB15` split. On that run, `Random Forest` achieved the best overall weighted `F1` score and the lowest inference latency, while `Transformer` produced the strongest deep-learning result.

## Problem Statement

Traditional intrusion detection systems are effective for known signatures but struggle when traffic patterns shift or when attack behavior does not match a fixed rule set. This project investigates whether AI-based detection can provide a stronger learned signal from enterprise flow telemetry while remaining practical to reproduce and run on commodity hardware.

## Dataset

The paper workflow is centered on `UNSW-NB15`.

- Full dataset size: `257,673` labeled flows
- Official train split: `82,332`
- Official test split: `175,341`
- Attack families present in the raw corpus: `Analysis`, `Backdoor`, `DoS`, `Exploits`, `Fuzzers`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms`, and `Normal`
- Experiment target: binary `label` column (`Benign` vs `Attack`)

Supporting materials:

- `dataset/README.md`
- `dataset/preprocessing.ipynb`
- `notebooks/feature_engineering.ipynb`

## Methodology

The preprocessing pipeline derives enterprise-relevant flow features from the raw UNSW records, including:

- flow duration
- source and destination bytes
- total packets and total bytes
- average packet size
- byte and packet rates
- inter-packet timing
- TCP round-trip and ACK timing
- TTL and loss counters
- connection-count behavior
- categorical context from `proto`, `service`, and `state`

The implementation lives in `training/data_pipeline.py` and is used directly by the experiment runner so the notebooks and scripted results stay aligned.

## Model Architectures

### Random Forest

Classical ensemble baseline for tabular traffic features. It serves as the strongest low-latency reference model in this repository.

### LSTM

A recurrent model that treats the engineered feature vector as a short ordered sequence, allowing the network to learn interactions across traffic statistics.

### Transformer

A self-attention model with learned positional embeddings over the feature sequence. This is the strongest deep-learning model in the current experiment set.

Source files:

- `models/random_forest.py`
- `models/lstm_model.py`
- `models/transformer_model.py`

## Experiments

Run the full paper pipeline with:

```bash
bash run_training.sh --max-train-rows 12000 --max-test-rows 5000 --epochs 6
```

This will:

1. generate `architecture.png`
2. train all three models
3. write the comparison tables to `results/`
4. create plots for confusion matrix, ROC, feature importance, and training loss
5. generate `research_paper.pdf`

You can also run the core experiment directly:

```bash
python3 training/run_experiments.py --max-train-rows 12000 --max-test-rows 5000 --epochs 6
```

## Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 90.46% | 92.05% | 90.46% | 90.69% | 0.9839 | 0.0094 |
| Transformer | 85.26% | 88.96% | 85.26% | 85.73% | 0.9740 | 0.5292 |
| LSTM | 71.00% | 81.82% | 71.00% | 71.84% | 0.8722 | 0.3965 |

### Interpretation

- `Random Forest` is the best overall attack detector on this bounded reproducible run.
- `Transformer` is the strongest deep-learning model and achieves a high `ROC AUC` of `0.9740`.
- `LSTM` improves over a naive baseline but underperforms the other two models on this dataset and feature representation.

Generated figures:

- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/feature_importance.png`
- `results/training_loss_curves.png`

## Comparison with Traditional IDS

This repository does not benchmark Snort or Suricata directly, so the comparison to traditional IDS is qualitative rather than a strict head-to-head measurement.

- Rule-based IDS remains strong for deterministic known signatures and can be extremely lightweight.
- The AI models here learn from labeled flow behavior and produce calibrated probabilities instead of binary rule hits.
- On the bounded CPU benchmark, the strongest AI model (`Random Forest`) still scores flows quickly enough for practical batch detection while offering broader statistical generalization than a fixed signature list.

## Demo

After training, run:

```bash
python3 demo/run_detection.py --sample-index 0
```

Example output from the current trained artifacts:

```text
Threat detected: Attack
Confidence: 100.00%
Ground truth: Backdoor
Model: Random Forest
```

## Reproducibility

This repository includes:

- `requirements.txt`
- `Dockerfile`
- `docker/Dockerfile`
- `run_training.sh`
- `research_paper.pdf`
- `CITATION.cff`

The raw dataset directory is ignored by Git by default so the repo can track the experiment code and paper assets without forcing large data uploads.

## Applications

- enterprise network monitoring research
- academic cybersecurity projects
- IDS benchmarking and reproducibility studies
- ML and deep-learning coursework in security analytics

## Project Structure

```text
ai-network-threat-detection/
├── dataset/
├── demo/
├── evaluation/
├── models/
├── notebooks/
├── results/
├── training/
├── architecture.png
├── research_paper.pdf
├── README.md
├── requirements.txt
└── CITATION.cff
```

The original `src/` directory from the earlier repository remains available as legacy implementation work, but the research workflow described in this README is driven by the `training/`, `models/`, `evaluation/`, and `demo/` directories.

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
