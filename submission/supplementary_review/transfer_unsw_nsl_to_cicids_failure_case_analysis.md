# CICIDS2017 Failure Case Analysis

Each attack family is evaluated as a binary task against benign traffic only. Other attack families are excluded from the family-specific score so the table reflects detection quality for that family rather than multi-class attribution.

| Attack Type | Support | LSTM Precision | LSTM Recall | LSTM F1 | Drift-Adaptive Hybrid Precision | Drift-Adaptive Hybrid Recall | Drift-Adaptive Hybrid F1 | Hybrid Gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FTP-Patator | 7938 | 0.0000 | 0.0000 | 0.0000 | 0.1092 | 0.5005 | 0.1793 | +0.1793 |
| SSH-Patator | 5897 | 0.0000 | 0.0000 | 0.0000 | 0.0113 | 0.0685 | 0.0194 | +0.0194 |
| DoS GoldenEye | 10293 | 0.0000 | 0.0000 | 0.0000 | 0.0010 | 0.0028 | 0.0015 | +0.0015 |
| DDoS | 128027 | 0.0000 | 0.0000 | 0.0000 | 0.0141 | 0.0005 | 0.0010 | +0.0010 |
| DoS Hulk | 231073 | 0.0000 | 0.0000 | 0.0000 | 0.0173 | 0.0004 | 0.0008 | +0.0008 |
| PortScan | 158930 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
