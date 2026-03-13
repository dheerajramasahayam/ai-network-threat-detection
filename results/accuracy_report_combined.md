# Accuracy Report — AI Network Intrusion Detection System
Generated: 2026-03-13 15:14:31

## Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| random_forest | 98.62% | 97.55% | 99.12% | 98.33% | 0.9993 |
| xgboost | 98.51% | 97.50% | 98.89% | 98.19% | 0.9991 |

## Notes

- Dataset: combined
- Train/Test split: 80/20 stratified
- Class balancing: Random undersampling of majority class
- Feature scaling: StandardScaler (zero-mean, unit-variance)
