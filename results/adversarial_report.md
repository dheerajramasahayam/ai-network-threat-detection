# Adversarial Robustness Report

Tests how well the model withstands adversarially crafted flows.
**Evasion rate** = % of true attack flows re-classified as BENIGN after perturbation.
Lower evasion rate = more robust model.

| Method | ε | Evasion Rate | Avg Perturbation |δ| | n Tested |
|---|---|---|---|---|
| 🔴 Random Noise | 0.05 | **96.59%** | 0.1860 | 88 |
| 🔴 Sign-Gradient (FGSM) | 0.05 | **100.00%** | 0.0396 | 88 |
| 🟢 Feature Masking | 0.05 | **1.14%** | 5.1708 | 88 |
| 🔴 Boundary Walk | 0.05 | **100.00%** | 0.0154 | 88 |
| 🔴 Random Noise | 0.1 | **100.00%** | 0.3672 | 88 |
| 🔴 Sign-Gradient (FGSM) | 0.1 | **100.00%** | 0.0764 | 88 |
| 🟢 Feature Masking | 0.1 | **1.14%** | 5.1708 | 88 |
| 🔴 Boundary Walk | 0.1 | **100.00%** | 0.0238 | 88 |
| 🔴 Random Noise | 0.2 | **100.00%** | 0.7337 | 88 |
| 🔴 Sign-Gradient (FGSM) | 0.2 | **100.00%** | 0.1511 | 88 |
| 🟢 Feature Masking | 0.2 | **1.14%** | 5.1708 | 88 |
| 🔴 Boundary Walk | 0.2 | **100.00%** | 0.0457 | 88 |
| 🔴 Random Noise | 0.5 | **100.00%** | 1.8386 | 88 |
| 🔴 Sign-Gradient (FGSM) | 0.5 | **100.00%** | 0.3852 | 88 |
| 🟢 Feature Masking | 0.5 | **1.14%** | 5.1708 | 88 |
| 🔴 Boundary Walk | 0.5 | **100.00%** | 0.1065 | 88 |
| 🔴 Random Noise | 1.0 | **100.00%** | 3.6997 | 88 |
| 🔴 Sign-Gradient (FGSM) | 1.0 | **100.00%** | 0.7948 | 88 |
| 🟢 Feature Masking | 1.0 | **1.14%** | 5.1708 | 88 |
| 🔴 Boundary Walk | 1.0 | **100.00%** | 0.2153 | 88 |

🟢 Evasion ≤ 10% (robust)  🟡 10–30%  🔴 > 30% (vulnerable)

## Methods

- **Random Noise**: uniform noise ±ε (baseline)
- **Sign-Gradient**: FGSM-inspired finite-difference gradient attack
- **Feature Masking**: zero out top-k most discriminative features
- **Boundary Walk**: gradient-free random walk to decision boundary
