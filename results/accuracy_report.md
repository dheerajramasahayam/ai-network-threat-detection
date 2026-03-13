# Accuracy Report — AI Network Intrusion Detection System

Generated: 2024-03-12 23:48:31

## Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| random_forest | 95.82% | 92.14% | 93.76% | 92.94% | 0.9842 |
| xgboost | 97.13% | 95.61% | 94.22% | 94.91% | 0.9914 |

## Notes

- Dataset: CICIDS2017 (Canadian Institute for Cybersecurity)
- Train/Test split: 80/20 stratified
- Class balancing: Random undersampling of majority class
- Feature scaling: StandardScaler (zero-mean, unit-variance)
- Attack traffic includes: DDoS, PortScan, Bot, Infiltration, Web Attacks, Brute Force
