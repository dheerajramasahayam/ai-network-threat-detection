# Dataset

This repository now uses three public network-security datasets in a shared research workflow.

## Included Benchmarks

| Dataset | Raw Files | Total Rows | Split Strategy | Original Label Space |
| --- | --- | --- | --- | --- |
| `UNSW-NB15` | `dataset/raw/unsw_nb15/UNSW_NB15_training-set.csv`, `dataset/raw/unsw_nb15/UNSW_NB15_testing-set.csv` | `257,673` | official train/test split | `10` traffic categories |
| `NSL-KDD` | `dataset/raw/nsl_kdd/KDDTrain+.txt`, `dataset/raw/nsl_kdd/KDDTest+.txt` | `148,517` | official train/test split | `40` symbolic labels |
| `CICIDS2017` | `dataset/cicids2017.csv` | `2,830,743` | external holdout for transfer evaluation | `15` labels |

## Official Split Details

### UNSW-NB15

- Training rows: `82,332`
- Testing rows: `175,341`
- Families: `Analysis`, `Backdoor`, `DoS`, `Exploits`, `Fuzzers`, `Generic`, `Normal`, `Reconnaissance`, `Shellcode`, `Worms`

### NSL-KDD

- Training rows: `125,973`
- Testing rows: `22,544`
- Original labels: `40` attack or normal categories, collapsed to binary attack detection for this repo

### CICIDS2017

- Total rows: `2,830,743`
- Labels: `BENIGN`, `DDoS`, `DoS Hulk`, `PortScan`, `FTP-Patator`, `SSH-Patator`, `Bot`, `Infiltration`, `Heartbleed`, and other web or DoS variants
- Usage in this repo: external dataset for cross-dataset transfer from `UNSW-NB15 + NSL-KDD`

## Features Used

The advanced research pipeline maps all datasets to the same canonical 41-feature flow schema from `src/preprocessing.py`. Representative features include:

- `flow_duration`
- `total_fwd_packets`
- `total_bwd_packets`
- `total_len_fwd_packets`
- `total_len_bwd_packets`
- `flow_bytes_s`
- `flow_packets_s`
- `flow_iat_mean`
- `packet_len_mean`
- `packet_len_std`
- `syn_flag_cnt`
- `ack_flag_cnt`
- `avg_packet_size`
- `init_win_bytes_fwd`
- `init_win_bytes_bwd`

This shared schema is what makes the rule baseline, the classical ML model, the deep models, and the hybrid detector directly comparable.

## Experimental Protocol

- `UNSW-NB15`: evaluated on the full official split
- `NSL-KDD`: evaluated on the full official split
- `CICIDS2017`: used as an external holdout for transfer evaluation
- Binary target: all datasets are mapped to `Benign` vs `Attack`
- External holdout size: configurable through `--cicids-sample-size`

The paper-facing runner is `training/run_advanced_research.py`, and the notebook in `dataset/preprocessing.ipynb` can be used to inspect the same preprocessing path interactively.

## Reproducibility Notes

- Raw datasets remain under `dataset/raw/` or `dataset/` because of size and licensing constraints.
- `training/canonical_pipeline.py` handles dataset alignment, sanitization, and official split loading.
- `results/advanced_experiment_summary.json` records the exact row counts, feature list, and per-model metrics for the advanced benchmark.
