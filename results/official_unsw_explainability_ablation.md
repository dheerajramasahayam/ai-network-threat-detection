# UNSW-NB15 Official Explainability Ablation

Top RF importance features were replaced with benign-reference values and compared against random feature ablation.

Top features: total_len_fwd_packets, flow_bytes_s, packet_len_mean, flow_duration, bwd_packets_s

Random features: rst_flag_cnt, bwd_psh_flags, packet_len_variance, total_len_bwd_packets, min_seg_size_fwd

Baseline weighted F1 on the sampled evaluation set: 0.9159

| Ablation | Weighted F1 drop |
| --- | --- |
| Top feature importance | 0.1284 |
| Random features | 0.0165 |
