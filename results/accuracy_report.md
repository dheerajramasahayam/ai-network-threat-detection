# Accuracy Report — AI Network Intrusion Detection System
Generated: 2026-03-13 00:29:52

## Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| random_forest | 99.73% | 98.71% | 99.91% | 99.31% | 0.9999 |
| xgboost | 99.74% | 98.76% | 99.92% | 99.34% | 0.9999 |

## Notes

- Dataset: CICIDS2017 (Canadian Institute for Cybersecurity)
- Train/Test split: 80/20 stratified
- Class balancing: Random undersampling of majority class
- Feature scaling: StandardScaler (zero-mean, unit-variance)
- Attack traffic includes: DDoS, PortScan, Bot, Infiltration, Web Attacks, Brute Force
