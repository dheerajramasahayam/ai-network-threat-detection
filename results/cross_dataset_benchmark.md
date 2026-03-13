# Cross-Dataset Generalization Benchmark

**Protocol**: Leave-one-out — trained on 3 sources, tested on unseen 4th source.
This tests true generalization to novel network environments.

| Hold-Out Dataset | Test Rows | Accuracy | Precision | Recall | F1 | AUC | Train (s) |
|---|---|---|---|---|---|---|---|
| **cicids2017** | 2,830,743.0 | 19.66% | 18.44% | 89.92% | 30.6% | 0.3784 | 0.6s |
| **unsw_nb15** | 500,000.0 | 93.01% | 100.0% | 93.01% | 96.38% | 0.5 | 4.7s |
| **nsl_kdd** | 37,042.0 | 57.88% | 57.88% | 100.0% | 73.32% | 0.1332 | 3.8s |
| **nf_ton_iot** | 500,000.0 | 100.0% | 100.0% | 100.0% | 100.0% | 0.5 | 4.0s |

**Average across all hold-outs**: acc=67.64%  prec=69.08%  rec=95.73%  f1=75.07%  auc=0.3779

## Research significance

Most published models report single-dataset accuracy. Cross-dataset generalization directly measures how well the model transfers to unseen network environments — a critical practical metric ignored by virtually all existing open-source NIDS projects.
