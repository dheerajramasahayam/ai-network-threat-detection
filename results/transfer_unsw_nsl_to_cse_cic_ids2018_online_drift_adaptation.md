# UNSW+NSL -> CSE-CIC-IDS2018 Online Drift Adaptation

Evaluation on `505156` cleaned rows from the CSE-CIC-IDS2018 external corpus.

| Variant | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |
| --- | --- | --- | --- | --- | --- | --- |
| Static Hybrid | 22.55% | 74.92% | 22.55% | 27.91% | 0.5025 | 0.1631 |
| Online Drift-Adaptive Hybrid | 11.89% | 68.53% | 11.89% | 7.39% | 0.3870 | 0.3420 |

Adaptive batches summarize mean drift score, adaptation alpha, and the online ensemble weights.

Average adaptation alpha: `0.9919`

Peak adaptation alpha: `1.0000`
