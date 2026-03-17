# UNSW+NSL -> CICIDS2017 Online Drift Adaptation

Batch size for online adaptation: `1024`

| Variant | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Static Hybrid | 57.52% | 67.44% | 57.52% | 61.33% | 0.4840 |
| Online Drift-Adaptive Hybrid | 67.28% | 64.14% | 67.28% | 65.64% | 0.4537 |

Adaptive batches summarize mean drift score, adaptation alpha, and the online ensemble weights.

Average model latency rises from `0.1662 ms/flow` for the static hybrid to `0.4343 ms/flow` for the online controller on the transfer evaluation sample.

Average adaptation alpha: `0.7862`

Peak adaptation alpha: `1.0000`
