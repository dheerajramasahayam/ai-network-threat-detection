# UNSW+NSL -> CICIDS2017 Latency Under Load

| Model | Batch Size | Latency (ms/flow) | Throughput (flows/s) |
| --- | --- | --- | --- |
| Signature IDS | 1 | 0.8504 | 1175.85 |
| Signature IDS | 16 | 0.0487 | 20524.66 |
| Signature IDS | 64 | 0.0122 | 82156.57 |
| Signature IDS | 256 | 0.0031 | 325844.76 |
| Signature IDS | 1024 | 0.0008 | 1257264.26 |
| Random Forest | 1 | 15.9234 | 62.80 |
| Random Forest | 16 | 0.9967 | 1003.34 |
| Random Forest | 64 | 0.2483 | 4027.05 |
| Random Forest | 256 | 0.0529 | 18890.10 |
| Random Forest | 1024 | 0.0125 | 80031.47 |
| LSTM | 1 | 0.7996 | 1250.64 |
| LSTM | 16 | 0.1536 | 6511.10 |
| LSTM | 64 | 0.1170 | 8546.03 |
| LSTM | 256 | 0.0737 | 13570.09 |
| LSTM | 1024 | 0.0722 | 13858.11 |
| Transformer | 1 | 0.4872 | 2052.58 |
| Transformer | 16 | 0.1036 | 9649.65 |
| Transformer | 64 | 0.0789 | 12671.85 |
| Transformer | 256 | 0.0745 | 13418.01 |
| Transformer | 1024 | 0.0736 | 13580.22 |
| Drift-Aware Hybrid | 1 | 23.2334 | 43.04 |
| Drift-Aware Hybrid | 16 | 1.6137 | 619.71 |
| Drift-Aware Hybrid | 64 | 0.5094 | 1962.91 |
| Drift-Aware Hybrid | 256 | 0.2458 | 4068.28 |
| Drift-Aware Hybrid | 1024 | 0.1872 | 5341.95 |
