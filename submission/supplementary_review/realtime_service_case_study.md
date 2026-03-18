# Real-World Packet Replay Case Study

This case study replays the local `RTN_traffic_dataset.csv` export from the paired packet capture `RealTimeNetworkTrafficCapture.pcapng`. This trace was collected from a controlled internal lab environment simulating enterprise traffic.

- Packet rows: `221253`
- Aggregated one-second bidirectional flow windows: `201`
- Benign warm-up seconds: `10`
- Attack seconds: `35`
- Benign cool-down seconds: `10`
- Attack onset: `t=10s`
- Attack end: `t=44s`
- Dominant attack path: `192.168.76.9 -> 192.168.12.56:3000/UDP`
- Hybrid detection delay: `0` seconds
- Signature detection delay: `0` seconds
- Dominant attack-flow hybrid probability at onset: `0.5382`
- Dominant attack-flow hybrid probability during attack: mean `0.5596`, range `[0.5382, 0.5703]`
- Mean max hybrid probability during benign warm-up: `0.4079`
- Strongest signature explanation on the dominant attack flow: `volumetric_flood, asymmetric_scan, microburst, handshake_failure`

## Flow-Window Metrics

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Signature IDS | 17.41% | 3.03% | 17.41% | 5.16% | 1.0000 |
| Drift-Adaptive Hybrid | 98.01% | 98.21% | 98.01% | 98.05% | 0.9993 |
