# System Architecture — AI Network Intrusion Detection System

## Overview

This document describes the end-to-end technical architecture of the AI-based Network Intrusion Detection System (AI-NIDS). The system transforms raw network traffic into structured threat classifications using a supervised machine learning pipeline.

---

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NETWORK PERIMETER                           │
│                                                                     │
│  Switches / Taps / Mirror Ports / NFV probes                        │
│          │                                                          │
│          ▼                                                          │
│  ┌───────────────────┐                                              │
│  │  Traffic Capture  │  pcap / NetFlow / IPFIX                      │
│  └────────┬──────────┘                                              │
│           │                                                         │
└───────────┼─────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Feature Extraction   │  CICFlowMeter / nfstream / custom eBPF     │
│  (per bidirectional   │  → 78 statistical flow features            │
│   IP flow)            │                                             │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Preprocessing        │  src/preprocessing.py
│  • Clean (inf / NaN)  │
│  • Encode labels      │  BENIGN (0) / ATTACK (1)
│  • StandardScaler     │  zero-mean, unit-variance
│  • Class balance      │  random undersample majority
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Model Training       │  src/training.py
│  ┌────────────────┐   │
│  │ RandomForest   │   │  sklearn RandomForestClassifier
│  └────────────────┘   │  n_estimators=200, balanced weights
│  ┌────────────────┐   │
│  │ XGBoost        │   │  xgboost XGBClassifier
│  └────────────────┘   │  n_estimators=300, lr=0.05
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Model Artifacts      │  models/{random_forest,xgboost}.joblib
│  • Serialised model   │
│  • Feature names      │
│  • Training config    │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Threat Detection     │  src/detection.py
│  ThreatDetector       │
│  • Load artifact      │
│  • Scale features     │
│  • Predict proba      │  [0 BENIGN … 1 ATTACK]
│  • Apply threshold    │  default 0.5
│  • Fire alert         │  pluggable callback
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Threat Classification│
│  BENIGN / ATTACK      │
│  + confidence score   │
│  + latency (ms)       │
└───────────────────────┘
```

---

## Component Breakdown

### 1. Feature Extraction Layer
- **Tool**: CICFlowMeter or nfstream
- **Output**: One CSV row per completed bidirectional IP flow
- **Features (78 total)**: packet-length statistics, IAT distributions, flag counts, byte/packet rates, window sizes

### 2. Preprocessing Module (`src/preprocessing.py`)
| Step | Method | Purpose |
|---|---|---|
| NaN / Inf removal | `dropna()` + `replace(inf, NaN)` | Remove corrupted measurements |
| Label encoding | Binary (BENIGN=0, ATTACK=1) | Simplify multi-class to binary threat detection |
| Feature scaling | `StandardScaler` | Normalise features for distance-sensitive algorithms |
| Class balancing | Random undersampling | Prevent majority-class bias during training |
| Train/Test split | Stratified 80/20 | Preserve class ratio in both splits |

### 3. Model Layer (`src/model.py`)

#### RandomForest
- Bootstrap aggregation of 200 decision trees
- Feature subset at each split: `sqrt(n_features)`
- Eliminates variance through ensemble averaging
- Produces inherent feature importance scores

#### XGBoost
- Sequential gradient boosting on decision trees
- Regularisation: L1 (alpha) + L2 (lambda) penalties on leaf weights
- Sub-sampling at tree and column level for robustness
- Faster inference than RandomForest at equivalent complexity

### 4. Training Script (`src/training.py`)
- Orchestrates preprocessing → model fitting → evaluation
- Exports confusion matrices and ROC curves to `results/`
- Logs all metrics to `results/logs.txt`
- Saves model artifacts to `models/`

### 5. Detection Engine (`src/detection.py`)
- Loads `.joblib` model artifact at startup
- Exposes `inspect(features)` for single-flow classification
- `inspect_batch(rows)` for bulk / offline analysis
- `tune_threshold()` grid-searches decision boundary on a validation set
- Pluggable alert callback system for SIEM integration

---

## Data Flow Diagram

```
[Raw Packets]  →  [Flow Meter]  →  [Feature CSV]
                                        │
                              ┌─────────┘
                              │
                      [Preprocessing]
                         │         │
                    [Training]  [Detection]
                         │         │
                    [Artifacts] [Alerts / SIEM]
```

---

## Scalability Considerations

| Concern | Approach |
|---|---|
| **Throughput** | Batch inference via `inspect_batch`; model is stateless and thread-safe |
| **Latency** | Median inference latency < 1 ms per flow on a modern CPU |
| **Model updates** | Artifacts are swappable at runtime; no restart required |
| **Class drift** | Periodic retraining with fresh labelled data; threshold auto-tuning |
| **Horizontal scaling** | Stateless design enables multi-process / containerised deployments |

---

## Security Considerations

- Models are serialised via `joblib`; validate artifact checksums before loading in production
- Avoid exposing the detection API to untrusted networks without authentication
- Feature extraction should happen inside a network tap, not on the host being monitored
- Regularly audit the training set for poisoning attacks (adversarial samples)

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Framework | scikit-learn 1.4, XGBoost 2.0 |
| Data Processing | pandas 2.2, numpy 1.26 |
| Visualisation | matplotlib 3.8, seaborn 0.13 |
| Serialisation | joblib |
| Containerisation | Docker (python:3.11-slim) |
| Testing | unittest (stdlib) |
