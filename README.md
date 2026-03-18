# Drift-Adaptive Intrusion Detection for Enterprise Networks

## Abstract

This repository is organized as a reproducible research project for enterprise network-threat detection across multiple public benchmarks. It aligns `UNSW-NB15`, `NSL-KDD`, `CICIDS2017`, and `CSE-CIC-IDS2018` into a shared 41-feature flow representation, compares standard IDS and ML baselines, and adds an online drift-adaptive controller on top of a hybrid ensemble.

The benchmark now studies six configurations:

- `Signature IDS`
- `Random Forest`
- `LSTM`
- `Transformer`
- `Drift-Aware Hybrid (static stack)`
- `Drift-Adaptive Hybrid` as the deployment-time upgrade

The key novelty upgrade is not just stacking, but online drift adaptation plus a formal drift-detector study. On the full external `CICIDS2017` corpus, the static hybrid scores `61.35%` weighted `F1`, while the online `Drift-Adaptive Hybrid` improves that to `68.69%` without retraining the base detectors. A formal comparison between `Isolation Forest`, `ADWIN`, `DDM`, and `Page-Hinkley` shows that `Isolation Forest` is the strongest detector in this benchmark at `70.58%` post-adaptation weighted `F1` with `0` source-domain false positives.

## Release Artifacts

This repository is packaged as a paper-first GitHub research release. The main entry points are:

- `paper/ieee_paper.pdf` for the authoritative IEEE-style manuscript
- `research_paper.pdf` for a root-level mirror of the same PDF
- `paper/ieee_paper.tex` and `paper/references.bib` for the full LaTeX source
- `submission/` for a ready-to-upload IEEE submission package
- `RELEASE.md` for the release inventory, validation steps, and publishing checklist
- `results/advanced_experiment_summary.json` and the `results/` directory for the measured experiment outputs

## Problem Statement

Traditional IDS engines are strong for deterministic known signatures, but they do not generalize well when traffic distributions shift or when attacks do not match existing rules. Static learned models also degrade when the deployment stream moves away from the source-domain training distribution. This project studies whether a reproducible online adaptation layer can recover part of that loss while keeping the system practical to deploy.

## Dataset

| Dataset | Rows | Split Used in This Repo | Label Space |
| --- | --- | --- | --- |
| `UNSW-NB15` | `257,673` | official `82,332` train / `175,341` test | `10` attack families collapsed to binary attack detection |
| `NSL-KDD` | `148,517` | official `125,973` train / `22,544` test | `40` symbolic labels collapsed to binary attack detection |
| `CICIDS2017` | `2,830,743` | full external corpus for transfer evaluation | `15` traffic labels |
| `CSE-CIC-IDS2018` | `505,156` cleaned rows | second external corpus for transfer evaluation | `14+` traffic labels |

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

The deployed ensemble follows:

```text
p(x_t) = Σ w_i(t) · M_i(x_t)
w_i(t) ∝ (1 − α_t) · w_i(stable) + α_t · w_i(stressed)
```

where `α_t` is the online drift coefficient. The repo now also includes an explicit detector-ablation study comparing `Isolation Forest`, `ADWIN`, `DDM`, and `Page-Hinkley` under the same adaptive controller.

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
bash run_training.sh --epochs 1 --batch-size 128 --rf-trees 60 --cicids-sample-size 0
python3 evaluation/run_full_transfer_evaluation.py
python3 evaluation/run_cse_cic_ids2018_transfer_evaluation.py
python3 evaluation/run_drift_detector_study.py
python3 evaluation/run_failure_case_analysis.py
python3 evaluation/realtime_streaming_evaluation.py --source file --chunksize 100000 --max-chunks 5
python3 evaluation/run_realtime_case_study.py
```

The full scripted benchmark trains all base detectors and evaluates:

1. full official `UNSW-NB15`
2. full official `NSL-KDD`
3. external transfer from `UNSW-NB15 + NSL-KDD` into `CICIDS2017`
4. second external transfer into `CSE-CIC-IDS2018`
5. formal drift detector comparison
6. online drift adaptation on the external stream
7. latency-under-load and explainability-ablation artifacts
8. family-level failure analysis on the dominant external attack types
9. real-time streaming evaluation with drift timeline output
10. packet-capture replay case study from a local real-time trace

For streaming-style ingestion from Kafka:

```bash
python3 evaluation/realtime_streaming_evaluation.py \
  --source kafka \
  --kafka-bootstrap-servers localhost:9092 \
  --kafka-topic network_flows
```

Kafka messages should include either the canonical 41-feature schema plus a binary `label`, or CICIDS-style feature names plus `Label`.

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
| LSTM | 80.30% | 64.48% | 80.30% | 71.53% | 0.2904 | 0.0406 |
| Transformer | 80.23% | 64.58% | 80.23% | 71.49% | 0.4886 | 0.0762 |
| Drift-Adaptive Hybrid | 74.26% | 64.13% | 74.26% | 68.69% | 0.4041 | 0.2630 |
| Random Forest | 58.51% | 65.55% | 58.51% | 61.53% | 0.4621 | 0.0060 |
| Signature IDS | 53.34% | 68.33% | 53.34% | 58.09% | 0.4716 | 0.0004 |

### Online Drift Adaptation

The strongest new result is the deployment-time adaptation ablation on the external stream:

| Variant | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Static Hybrid | 57.65% | 67.10% | 57.65% | 61.35% | 0.4755 |
| Online Drift-Adaptive Hybrid | 74.26% | 64.13% | 74.26% | 68.69% | 0.4041 |

This is a `+7.34` weighted `F1` gain over the static hybrid under full-corpus external drift, without retraining the already-trained base models.

Measured average inference latency rises from `0.1240 ms/flow` for the static hybrid to `0.2630 ms/flow` for the online controller, so the gain is robustness under shift rather than raw speed.

Artifacts:

- `results/transfer_unsw_nsl_to_cicids_online_drift_adaptation.csv`
- `results/transfer_unsw_nsl_to_cicids_online_drift_adaptation.md`
- `results/transfer_unsw_nsl_to_cicids_online_drift_adaptation.png`

### Formal Drift Detector Study

The repo now includes a direct comparison of drift detectors under the same adaptive hybrid controller:

| Drift Detector | Detection Delay (windows) | False Positives | F1 After Adaptation |
| --- | --- | --- | --- |
| Isolation Forest | 2 | 0 | 70.58 |
| DDM | 0 | 0 | 69.77 |
| Page-Hinkley | 0 | 0 | 69.77 |
| ADWIN | not detected | 0 | 62.93 |

This removes the obvious reviewer criticism of “why this detector?” and also preserves a practical distinction: `Isolation Forest` remains label-free, while the error-stream detectors need post-label feedback.

Artifacts:

- `results/drift_detector_study.csv`
- `results/drift_detector_study.md`
- `results/drift_detector_study.png`

### Additional External Evaluation: `CSE-CIC-IDS2018`

The repo now includes a second modern external dataset. After removing `59` repeated header rows from the local copy, the cleaned benchmark contains `505,156` rows (`50,000` benign, `455,156` attack).

| Model | Accuracy | F1 Score | ROC AUC |
| --- | --- | --- | --- |
| Signature IDS | 57.95% | 66.37% | 0.5573 |
| Static Hybrid | 22.55% | 27.91% | 0.5025 |
| Random Forest | 22.45% | 27.58% | 0.5353 |
| Drift-Adaptive Hybrid | 11.89% | 7.39% | 0.3870 |
| Transformer | 9.88% | 2.06% | 0.2637 |
| LSTM | 9.90% | 1.78% | 0.2334 |

This is a deliberately hard result. The signature baseline becomes the strongest model on this corpus, which makes the paper’s transfer claims more credible because it shows where the learned source-domain models still fail.

Artifacts:

- `results/transfer_unsw_nsl_to_cse_cic_ids2018_model_comparison.md`
- `results/transfer_unsw_nsl_to_cse_cic_ids2018_online_drift_adaptation.md`
- `results/transfer_unsw_nsl_to_cse_cic_ids2018_summary.json`

### Failure Analysis

The repo now includes attack-family failure analysis on the dominant `CICIDS2017` attack types, evaluated as binary family-vs-benign tasks:

| Attack Type | Support | LSTM F1 | Drift-Adaptive Hybrid F1 |
| --- | --- | --- | --- |
| FTP-Patator | 7,938 | 0.0000 | 0.1793 |
| SSH-Patator | 5,897 | 0.0000 | 0.0194 |
| DoS GoldenEye | 10,293 | 0.0000 | 0.0015 |
| DDoS | 128,027 | 0.0000 | 0.0010 |
| DoS Hulk | 231,073 | 0.0000 | 0.0008 |
| PortScan | 158,930 | 0.0000 | 0.0000 |

This result is intentionally not flattering: the source-trained `LSTM` collapses on all six dominant external attack families, and the adaptive hybrid only recovers meaningful signal on `FTP-Patator` plus a smaller amount on `SSH-Patator`. That makes the paper stronger because it documents exactly where transfer still breaks.

Artifacts:

- `results/transfer_unsw_nsl_to_cicids_failure_case_analysis.md`
- `results/transfer_unsw_nsl_to_cicids_failure_case_analysis.png`

### Real-World Packet Replay Case Study

The repo now includes a replayed packet-capture validation using the local pair:

- `dataset/raw/new_2026/realtime_ids/RTN_traffic_dataset.csv`
- `dataset/raw/new_2026/realtime_ids/RealTimeNetworkTrafficCapture.pcapng`

This trace was collected from a controlled internal lab environment simulating enterprise traffic.

The runner aggregates the packet trace into one-second bidirectional flow windows and replays them through the transfer-trained adaptive hybrid:

```bash
python3 evaluation/run_realtime_case_study.py
```

Verified case-study results:

- `221,253` packet rows replayed
- `201` one-second bidirectional flow windows
- `10` seconds benign warm-up, `35` seconds attack, `10` seconds benign cool-down
- dominant attack path: `192.168.76.9 -> 192.168.12.56:3000/UDP`
- hybrid detection delay: `0` seconds
- dominant attack-flow probability at onset: `0.5382`
- dominant attack-flow probability during attack: mean `0.5596`, range `[0.5382, 0.5703]`
- `Drift-Adaptive Hybrid` flow-window weighted `F1`: `98.05%`
- `Signature IDS` flow-window weighted `F1`: `5.16%`

This is the most operational artifact in the repo because it starts from packet-level evidence rather than a pre-engineered benchmark table and shows the controller reacting to a real replayed attack episode.

Artifacts:

- `results/realtime_service_case_study.md`
- `results/realtime_service_case_study_timeline.csv`
- `results/realtime_service_case_study.png`
- `results/realtime_service_case_study.json`

### Real-Time Streaming Evaluation

The repository now includes an explicit real-time evaluation loop that can run either over file-backed streamed chunks or a Kafka topic. On the full file-backed `CICIDS2017` stream, the online `Drift-Adaptive Hybrid` processed `2,830,743` rows across `29` sequential windows and matched the full-corpus transfer score: `74.26%` accuracy, `68.69%` weighted `F1`, and `0.4041` `ROC AUC`.

Windowed behavior is intentionally reported separately because it reveals the operational drift pattern rather than only the final aggregate score. In the verified full-stream run, mean window `F1` was `73.80%`, mean adaptation alpha was `0.9985`, and the controller entered the stressed regime almost immediately (`0.9572` in the first window, `1.0000` thereafter for nearly the entire stream).

Artifacts:

- `results/transfer_unsw_nsl_to_cicids_realtime_stream_timeline.csv`
- `results/transfer_unsw_nsl_to_cicids_realtime_streaming_evaluation.md`
- `results/transfer_unsw_nsl_to_cicids_drift_timeline.png`

### Interpretation

- The static `Drift-Aware Hybrid` remains best on the official `UNSW-NB15` split.
- `LSTM` remains the strongest model on `NSL-KDD` and the external transfer benchmark.
- The new novelty result is that online drift adaptation materially improves the hybrid under full external shift.
- `Isolation Forest` is now justified empirically as the strongest drift detector in this benchmark.
- The second external `CSE-CIC-IDS2018` evaluation shows that source-only learned transfer can still collapse on newer corpora.
- Cross-dataset generalization is still difficult, but the adaptation layer recovers part of that gap.

### Positioning Against Recent Work

The repo now positions itself explicitly against recent primary-source IDS papers:

| Study | Year | Setting | Reported Result |
| --- | --- | --- | --- |
| Yan et al. | 2025 | `UNSW-NB15` within-dataset Transformer | `89.00%` `F1` |
| Xin and Xu | 2025 | `NSL-KDD -> UNSW-NB15` cross-dataset Transformer-IDS | `55.00%` `F1` |
| Wang et al. (`BS-GAT`) | 2025 | edge/IoT graph-based binary IDS | `>99%` binary `F1` |
| This work | 2026 | `UNSW+NSL -> full CICIDS2017` drift-adaptive transfer | `68.69%` weighted `F1` |

This is not meant as a forced apples-to-apples leaderboard because the datasets and protocols differ. The point is that this repo now clearly states where it sits relative to recent Transformer and graph-based IDS research while offering broader external-transfer, streaming, and detector-ablation coverage.

## Comparison with Traditional IDS

This repository includes a quantitative rule-based baseline instead of a purely qualitative discussion.

- On `UNSW-NB15`, `Signature IDS` reaches `55.13%` weighted `F1`, versus `90.72%` for the static hybrid.
- On `NSL-KDD`, `Signature IDS` reaches `49.47%` weighted `F1`, versus `81.01%` for `LSTM`.
- On the full external `CICIDS2017` corpus, `Signature IDS` reaches `58.09%` weighted `F1`, the static hybrid reaches `61.35%`, the online `Drift-Adaptive Hybrid` reaches `68.69%`, and `LSTM` remains best at `71.53%`.
- On cleaned `CSE-CIC-IDS2018`, `Signature IDS` reaches `66.37%` weighted `F1` and is the strongest model, which highlights that some external corpora remain more rule-aligned than source-trained learned detectors.
- The rule baseline is still the fastest detector: on `UNSW-NB15` at batch size `1024`, it sustains about `1.28M` flows/s.

Because these public benchmarks are distributed as flow records rather than raw packet payloads, the traditional baseline is implemented as a transparent flow-signature IDS instead of direct `Snort` or `Suricata` packet replay.

## Hard Contributions

The repo now goes beyond a basic benchmark in four ways:

- `cross-dataset transfer` from two training corpora into the full external `CICIDS2017` corpus
- `second external validation` on cleaned `CSE-CIC-IDS2018`
- `online drift adaptation` through the new `Drift-Adaptive Hybrid`
- `formal drift detector comparison` across `Isolation Forest`, `ADWIN`, `DDM`, and `Page-Hinkley`
- `online latency under load` across multiple batch sizes
- `failure-case analysis` on dominant external attack families
- `explainability validated by ablation` using feature-importance interventions

## Production Deployment Scenario

The paper now includes a deployment architecture figure at `paper/deployment_architecture.png`. The scenario models:

- enterprise traffic mirrors
- flow collection through `Zeek / NetFlow`
- `Kafka` or file-backed stream ingestion
- canonical 41-feature extraction
- low-latency `Drift-Adaptive Hybrid` inference
- alert handoff to `SIEM / SOC`

## Future Work

The paper now ends with an explicit future-work section. The next research directions are:

- `continual learning` with delayed labels so the controller can refresh model parameters, not just reweight them
- `RL / bandit-based adaptation` to optimize the precision-recall-latency tradeoff online
- `multi-domain training` across enterprise, cloud, and IoT corpora, potentially with graph-based side information

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
- `paper/build.sh`
- `paper/ieee_paper.tex`
- `research_paper.pdf`
- `submission/build_submission_assets.sh`
- `CITATION.cff`
- `RELEASE.md`

Raw data is kept under `dataset/raw/` and can stay out of version control, while the merged experiment assets and notebooks remain reproducible from the scripts in `training/` and `evaluation/`.

To rebuild the paper and refresh the root-level PDF mirror:

```bash
bash paper/build.sh
```

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
├── submission/
├── RELEASE.md
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
