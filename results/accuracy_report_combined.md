# Accuracy Report — AI Network Intrusion Detection System
Generated: 2026-03-13 15:20:25

## Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| random_forest | 99.86% | 99.84% | 99.82% | 99.83% | 1.0000 |
| xgboost | 99.85% | 99.82% | 99.81% | 99.81% | 1.0000 |

## Notes

- Dataset: combined
- Train/Test split: 80/20 stratified
- Class balancing: Random undersampling of majority class
- Feature scaling: StandardScaler (zero-mean, unit-variance)
