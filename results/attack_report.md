# Per-Attack-Type Detection Report

This report exposes per-class performance — the metric that 99%+ overall
accuracy hides. Rare attacks often have detection rates << overall accuracy.

| Attack Type | Samples | Detection Rate | Miss Rate | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 🟢 **Heartbleed** | 11 | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% |
| 🔴 **Web Attack � Sql Injection** | 21 | 4.76% | 95.24% | 100.00% | 4.76% | 9.09% |
| 🔴 **Infiltration** | 36 | 75.00% | 25.00% | 100.00% | 75.00% | 85.71% |
| 🔴 **Web Attack � XSS** | 652 | 2.91% | 97.09% | 100.00% | 2.91% | 5.66% |
| 🔴 **Web Attack � Brute Force** | 1,507 | 6.97% | 93.03% | 100.00% | 6.97% | 13.03% |
| 🔴 **Bot** | 1,956 | 5.67% | 94.33% | 100.00% | 5.67% | 10.74% |
| 🔴 **DDoS** | 5,000 | 62.82% | 37.18% | 100.00% | 62.82% | 77.16% |
| 🔴 **DoS GoldenEye** | 5,000 | 68.30% | 31.70% | 100.00% | 68.30% | 81.16% |
| 🔴 **DoS Hulk** | 5,000 | 69.06% | 30.94% | 100.00% | 69.06% | 81.70% |
| 🔴 **DoS Slowhttptest** | 5,000 | 40.16% | 59.84% | 100.00% | 40.16% | 57.31% |
| 🔴 **DoS slowloris** | 5,000 | 38.14% | 61.86% | 100.00% | 38.14% | 55.22% |
| 🔴 **FTP-Patator** | 5,000 | 49.94% | 50.06% | 100.00% | 49.94% | 66.61% |
| 🔴 **PortScan** | 5,000 | 0.10% | 99.90% | 100.00% | 0.10% | 0.20% |
| 🔴 **SSH-Patator** | 5,000 | 50.10% | 49.90% | 100.00% | 50.10% | 66.76% |
| 🟢 **BENIGN** | 5,000 | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |

**Legend:** 🟢 Miss rate ≤ 5%  🟡 5–20%  🔴 > 20%

## Key findings

Models that achieve 99%+ overall accuracy often miss rare attack categories.
This breakdown is essential for real-world deployment decisions.
