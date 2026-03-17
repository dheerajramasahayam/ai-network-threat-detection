# Dataset

This repository uses `UNSW-NB15` as the primary reproducible dataset for the research paper workflow.

## Selected Dataset

- Dataset: `UNSW-NB15`
- Raw files:
  - `dataset/raw/unsw_nb15/UNSW_NB15_training-set.csv`
  - `dataset/raw/unsw_nb15/UNSW_NB15_testing-set.csv`
- Total size: `257,673` labeled flows
- Official split:
  - Training rows: `82,332`
  - Testing rows: `175,341`
- Number of classes: `10`
- Threat classes:
  - `Normal`
  - `Analysis`
  - `Backdoor`
  - `DoS`
  - `Exploits`
  - `Fuzzers`
  - `Generic`
  - `Reconnaissance`
  - `Shellcode`
  - `Worms`

## Features Used

The research pipeline derives a compact enterprise-traffic feature set from the original UNSW flow records:

- Flow duration
- Source packets
- Destination packets
- Source bytes
- Destination bytes
- Total packets
- Total bytes
- Average packet size
- Mean source packet size
- Mean destination packet size
- Bytes per second
- Packets per second
- Source bytes per second
- Destination bytes per second
- Source inter-packet time
- Destination inter-packet time
- TCP round-trip time
- SYN/ACK latency
- ACK latency
- Source TTL
- Destination TTL
- Source loss
- Destination loss
- Service connection count
- Destination connection count
- Source-destination flow count
- Source port reuse count
- Destination port reuse count
- Same IP/port flag
- State-TTL count

Categorical context is preserved through:

- `proto`
- `service`
- `state`

## Why UNSW-NB15

`UNSW-NB15` is a strong fit for an enterprise-network threat paper because it includes modern attack categories, a public official train/test split, and flow-level metadata that supports both classical ML and deep learning.

## Reproducibility Notes

- Raw datasets are kept under `dataset/raw/` and excluded from Git by default because of size.
- The experiment runner uses the official split directly instead of inventing a custom one.
- The paper experiments use the official binary `label` target for attack-vs-benign detection, while preserving `attack_cat` for class distribution analysis and demo examples.
- The notebook in `dataset/preprocessing.ipynb` mirrors the same preprocessing implemented in `training/data_pipeline.py`.
