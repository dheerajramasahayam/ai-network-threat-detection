# UNSW+NSL -> CICIDS2017 Online Drift Adaptation

Evaluation on the full CICIDS2017 external corpus.

| Variant | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms/flow) |
| --- | --- | --- | --- | --- | --- | --- |
| Static Hybrid | 57.65% | 67.10% | 57.65% | 61.35% | 0.4755 | 0.1240 |
| Online Drift-Adaptive Hybrid | 74.26% | 64.13% | 74.26% | 68.69% | 0.4041 | 0.2630 |

Adaptive batches summarize mean drift score, adaptation alpha, and the online ensemble weights.

Average adaptation alpha: `0.9985`

Peak adaptation alpha: `1.0000`
