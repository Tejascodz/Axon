# Round 4: Competitive MaxSim, localization, lead-in confound, latency (training-free)
## lightonai/LateOn

| Scoring | T1 mean | T1 k=10 | T2 monotonicity | T3 pos>HN1 |
|---|---|---|---|---|
| ColBERT | 0.691 | 0.427 | 0.685 | 0.409 |
| ColBERT, lead-in removed | 0.735 | 0.483 | 0.715 | 0.420 |
| clause grouping | 0.737 | 0.490 | 0.725 | 0.442 |
| Competitive MaxSim (τ 0.1) | 0.660 | 0.456 | 0.637 | 0.417 |
| conditions encoded apart, token-weighted | 0.740 | 0.494 | 0.729 | 0.450 |
| Competitive MaxSim (τ 0.03) | 0.627 | 0.432 | 0.613 | 0.400 |
| Competitive MaxSim (τ 0.3) | 0.726 | 0.497 | 0.700 | 0.449 |

Paired differences over documents (95% bootstrap interval):

| Comparison | T1 mean | T1 k=10 | T2 monotonicity | T3 pos>HN1 |
|---|---|---|---|---|
| Q4 Competitive MaxSim − grouping | -0.077 [-0.083, -0.070] | -0.036 [-0.055, -0.017] | -0.088 [-0.095, -0.081] | -0.028 [-0.049, -0.007] |
| grouping − ColBERT | +0.045 [+0.039, +0.051] | +0.062 [+0.043, +0.079] | +0.039 [+0.032, +0.045] | +0.032 [+0.015, +0.048] |
| Q6 lead-in removed − ColBERT | +0.046 [+0.041, +0.050] | +0.058 [+0.043, +0.074] | +0.030 [+0.025, +0.036] | +0.011 [+0.001, +0.022] |
| Q6 grouping − lead-in removed | -0.001 [-0.006, +0.004] | +0.003 [-0.013, +0.020] | +0.008 [+0.002, +0.014] | +0.020 [+0.004, +0.036] |
| Q7 separate encoding (token-weighted) − lead-in removed | +0.003 [-0.002, +0.008] | +0.007 [-0.010, +0.023] | +0.012 [+0.007, +0.018] | +0.029 [+0.012, +0.044] |
| Q7 equal weight per condition − token-weighted | -0.003 [-0.005, -0.001] | -0.003 [-0.010, +0.003] | -0.004 [-0.007, -0.002] | -0.008 [-0.016, -0.001] |
| Q4 sensitivity τ 0.03 − grouping | -0.111 [-0.118, -0.104] | -0.061 [-0.079, -0.041] | -0.112 [-0.120, -0.105] | -0.046 [-0.068, -0.024] |
| Q4 sensitivity τ 0.3 − grouping | -0.011 [-0.016, -0.006] | +0.007 [-0.008, +0.021] | -0.026 [-0.031, -0.020] | +0.007 [-0.009, +0.023] |

Q5 Which condition does a near-miss break? Accuracy with 95% interval (chance 0.100):

| Method | n | comparative (sees the positive) | absolute (near-miss only) |
|---|---|---|---|
| word-overlap baseline (no model) | 1455 | – | 0.256 [0.234, 0.278] |
| clause grouping | 1455 | 0.535 [0.510, 0.561] | 0.249 [0.227, 0.271] |
| Competitive MaxSim (τ 0.1) | 1455 | 0.379 [0.354, 0.405] | 0.183 [0.163, 0.203] |

Q8 Latency, re-ranking 100 candidates (documents pre-encoded):

| Scoring | device | docs/s to encode | ms per query (median) | ms per query (p95) |
|---|---|---|---|---|
| ColBERT | cuda:0 | 94.7 | 29.39 | 34.93 |
| clause grouping | cuda:0 | 95.3 | 27.66 | 29.13 |
| Competitive MaxSim (τ 0.1) | cuda:0 | 95.7 | 31.21 | 33.21 |

## lightonai/GTE-ModernColBERT-v1

| Scoring | T1 mean | T1 k=10 | T2 monotonicity | T3 pos>HN1 |
|---|---|---|---|---|
| ColBERT | 0.712 | 0.471 | 0.684 | 0.436 |
| ColBERT, lead-in removed | 0.738 | 0.505 | 0.702 | 0.428 |
| clause grouping | 0.763 | 0.540 | 0.733 | 0.505 |
| Competitive MaxSim (τ 0.1) | 0.642 | 0.406 | 0.590 | 0.391 |

Paired differences over documents (95% bootstrap interval):

| Comparison | T1 mean | T1 k=10 | T2 monotonicity | T3 pos>HN1 |
|---|---|---|---|---|
| Q4 Competitive MaxSim − grouping | -0.121 [-0.127, -0.115] | -0.135 [-0.153, -0.117] | -0.144 [-0.152, -0.137] | -0.117 [-0.137, -0.098] |
| grouping − ColBERT | +0.050 [+0.045, +0.055] | +0.067 [+0.051, +0.084] | +0.049 [+0.043, +0.055] | +0.068 [+0.051, +0.085] |
| Q6 lead-in removed − ColBERT | +0.025 [+0.021, +0.029] | +0.035 [+0.023, +0.046] | +0.018 [+0.014, +0.022] | -0.009 [-0.019, +0.002] |
| Q6 grouping − lead-in removed | +0.025 [+0.020, +0.029] | +0.033 [+0.018, +0.048] | +0.031 [+0.026, +0.037] | +0.077 [+0.060, +0.094] |

Q5 Which condition does a near-miss break? Accuracy with 95% interval (chance 0.100):

| Method | n | comparative (sees the positive) | absolute (near-miss only) |
|---|---|---|---|
| word-overlap baseline (no model) | 1455 | – | 0.256 [0.234, 0.278] |
| clause grouping | 1455 | 0.544 [0.519, 0.570] | 0.377 [0.352, 0.402] |
| Competitive MaxSim (τ 0.1) | 1455 | 0.337 [0.313, 0.362] | 0.258 [0.235, 0.281] |

## lightonai/mLateOn

| Scoring | T1 mean | T1 k=10 | T2 monotonicity | T3 pos>HN1 |
|---|---|---|---|---|
| ColBERT | 0.813 | 0.670 | 0.782 | 0.650 |
| ColBERT, lead-in removed | 0.842 | 0.750 | 0.801 | 0.650 |
| clause grouping | 0.847 | 0.790 | 0.805 | 0.744 |
| Competitive MaxSim (τ 0.1) | 0.741 | 0.663 | 0.681 | 0.641 |

Paired differences over documents (95% bootstrap interval):

| Comparison | T1 mean | T1 k=10 | T2 monotonicity | T3 pos>HN1 |
|---|---|---|---|---|
| Q4 Competitive MaxSim − grouping | -0.106 [-0.113, -0.100] | -0.127 [-0.148, -0.106] | -0.126 [-0.133, -0.118] | -0.104 [-0.126, -0.081] |
| grouping − ColBERT | +0.031 [+0.026, +0.036] | +0.112 [+0.092, +0.132] | +0.022 [+0.017, +0.028] | +0.090 [+0.071, +0.108] |
| Q6 lead-in removed − ColBERT | +0.027 [+0.023, +0.031] | +0.076 [+0.060, +0.093] | +0.018 [+0.014, +0.022] | -0.001 [-0.010, +0.008] |
| Q6 grouping − lead-in removed | +0.004 [+0.000, +0.008] | +0.036 [+0.020, +0.051] | +0.004 [-0.001, +0.009] | +0.091 [+0.072, +0.109] |

Q5 Which condition does a near-miss break? Accuracy with 95% interval (chance 0.100):

| Method | n | comparative (sees the positive) | absolute (near-miss only) |
|---|---|---|---|
| word-overlap baseline (no model) | 1455 | – | 0.256 [0.234, 0.278] |
| clause grouping | 1455 | 0.821 [0.801, 0.840] | 0.241 [0.219, 0.264] |
| Competitive MaxSim (τ 0.1) | 1455 | 0.576 [0.551, 0.601] | 0.124 [0.107, 0.140] |

All jobs finished.
