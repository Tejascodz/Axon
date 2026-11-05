# Round 6b + 7

## serving_mixed_chunked512

```
{
 "concurrency_16": {
  "requests": 240,
  "failed": 0,
  "wall_s": 224.7,
  "output_tokens_per_s": 230.7,
  "requests_per_s": 1.068,
  "gpu_util_mean_pct": 94.3,
  "gpu_mem_peak_mib": 11401.0,
  "gpu_sm_clock_mean_mhz": 825,
  "gpu_temp_max_c": 81.0,
  "gpu_power_mean_w": 66.2,
  "prompt=short": {
   "n": 104,
   "ttft_ms_p50": 148.4,
   "ttft_ms_p95": 11146.5,
   "tpot_ms_p50": 60.6,
   "e2e_ms_p50": 7979.7,
   "e2e_ms_p95": 33664.6
  },
  "prompt=long": {
   "n": 136,
   "ttft_ms_p50": 434.0,
   "ttft_ms_p95": 10769.6,
   "tpot_ms_p50": 59.4,
   "e2e_ms_p50": 8012.9,
   "e2e_ms_p95": 32351.6
  },
  "max_tokens=32": {
   "n": 88,
   "ttft_ms_p50": 381.1,
   "ttft_ms_p95": 10769.6,
   "tpot_ms_p50": 58.0,
   "e2e_ms_p50": 2156.8,
   "e2e_ms_p95": 14133.5
  },
  "max_tokens=128": {
   "n": 75,
   "ttft_ms_p50": 402.3,
   "ttft_ms_p95": 10953.3,
   "tpot_ms_p50": 59.9,
   "e2e_ms_p50": 7979.6,
   "e2e_ms_p95": 20423.2
  },
  "max_tokens=512": {
   "n": 77,
   "ttft_ms_p50": 354.2,
   "ttft_ms_p95": 10973.9,
   "tpot_ms_p50": 61.1,
   "e2e_ms_p50": 31643.7,
   "e2e_ms_p95": 42519.1
  },
  "itl_ms_p50": 49.0,
  "itl_ms_p99": 431.0
 }
}
```

## soak

```
{
 "minutes": 30,
 "clients": 8,
 "requests": 2958,
 "status": {
  "200": 2952,
  "502": 6
 },
 "error_rate": 0.00203,
 "latency_ms": {
  "50": 4517.8,
  "95": 8584.8,
  "99": 11866.4
 },
 "ttft_ms": {
  "50": 473.6,
  "95": 1071.6
 },
 "abstained": 0.4502,
 "cross_tenant_results": 0,
 "gpu0": {
  "gpu_util_mean_pct": 99.2,
  "gpu_mem_peak_mib": 11631.0,
  "gpu_sm_clock_mean_mhz": 754,
  "gpu_temp_max_c": 81.0,
  "gpu_power_mean_w": 67.8
 },
 "update": {
  "start": 1793888773.2784402,
  "update_seconds": 19.0,
  "rc": 0,
  "report": null,
  "version_before": 1.0,
  "version_after": 2.0,
  "served_after_seconds": 2.3,
  "deleted": 100
 },
 "during_update": {
  "requests": 192,
  "status": {
   "200": 192
  },
  "error_rate": 0.0,
  "latency_ms": {
   "50": 4682.8,
   "95": 10116.1,
   "99": 13014.4
  },
  "ttft_ms": {
   "50": 570.3,
   "95": 1332.9
  },
  "abstained": 0.4792,
  "cross_tenant_results": 0
 },
 "drift": {
  "first_5_min_p50_ms": 4492.1,
  "last_5_min_p50_ms": 4497.4,
  "first_5_min_p99_ms": 12223.3,
  "last_5_min_p99_ms": 10015.9
 },
 "memory": {
  "rss_start_mb": 1152,
  "rss_end_mb": 2652,
  "rss_max_mb": 2652,
  "slope_mb_per_hour": 482.7
 }
}
```

## burst

```
{
 "burst": {
  "requests": 19417,
  "status": {
   "503": 19337,
   "200": 80
  },
  "error_rate": 0.99588,
  "latency_ms": {
   "50": 13888.9,
   "95": 15614.2,
   "99": 16558.5
  },
  "ttft_ms": {
   "50": 1920.7,
   "95": 3560.2
  },
  "abstained": 0.4,
  "cross_tenant_results": 0,
  "clients": 48,
  "max_inflight": 16,
  "rejected_503": 19337,
  "rejection_ms_p99": 197.2,
  "accepted_p99_ms": 16558.5,
  "other_errors": 0
 },
 "recovery": {
  "requests": 101,
  "status": {
   "200": 101
  },
  "error_rate": 0.0,
  "latency_ms": {
   "50": 4552.8,
   "95": 7661.6,
   "99": 8858.7
  },
  "ttft_ms": {
   "50": 602.3,
   "95": 1736.7
  },
  "abstained": 0.4158,
  "cross_tenant_results": 0
 }
}
```

## soak_verify

```
{
 "nli": {
  "requests": 490,
  "status": {
   "200": 489,
   "502": 1
  },
  "error_rate": 0.00204,
  "latency_ms": {
   "50": 4597.0,
   "95": 8227.1,
   "99": 11870.9
  },
  "ttft_ms": {
   "50": 489.6,
   "95": 1100.5
  },
  "abstained": 0.6708,
  "cross_tenant_results": 0,
  "threshold": 0.0536,
  "verify_ms_mean": 114.2,
  "withheld_unsupported": 112.0,
  "answered": 161.0
 },
 "llm": {
  "requests": 447,
  "status": {
   "200": 447
  },
  "error_rate": 0.0,
  "latency_ms": {
   "50": 5125.9,
   "95": 9406.7,
   "99": 12936.6
  },
  "ttft_ms": {
   "50": 475.7,
   "95": 1120.7
  },
  "abstained": 0.6532,
  "cross_tenant_results": 0,
  "threshold": null,
  "verify_ms_mean": 403.3,
  "withheld_unsupported": 96.0,
  "answered": 155.0
 }
}
```

## judge_real

```
{
 "base": {
  "answer": {
   "answers": 500,
   "answers_given": 282,
   "claims": 288,
   "claims_supported": 0.8021,
   "claims_uncited": 5,
   "answers_supported": 193,
   "answers_unsupported": 89,
   "reasons_unsupported": {
    "no evidence sentence, and the answer is not in a cited source": 12,
    "answer not in cited sources": 22,
    "uncited or unsupported claim": 55
   },
   "seconds_per_answer": 0.104,
   "em_if_supported": 0.6218,
   "em_if_unsupported": 0.4719,
   "before": {
    "answered": 282,
    "em_among_answered": 0.5745,
    "em_overall": 0.324,
    "wrong_answers": 120
   },
   "after_withholding_unsupported": {
    "answered": 193,
    "em_among_answered": 0.6218,
    "em_overall": 0.24,
    "wrong_answers": 73,
    "wrong_removed": 47,
    "right_removed": 42
   }
  },
  "unanswerable": {
   "answers": 200,
   "answers_given": 10,
   "claims": 8,
   "claims_supported": 1.0,
   "claims_uncited": 0,
   "answers_supported": 8,
   "answers_unsupported": 2,
   "reasons_unsupported": {
    "no evidence sentence, and the answer is not in a cited source": 2
   },
   "seconds_per_answer": 0.0082,
   "answered_without_gold": 10,
   "caught_by_check": 2
  }
 },
 "ft": {
  "answer": {
   "answers": 500,
   "answers_given": 420,
   "claims": 729,
   "claims_supported": 0.915,
   "claims_uncited": 0,
   "answers_supported": 357,
   "answers_unsupported": 63,
   "reasons_unsupported": {
    "answer not in cited sources": 8,
    "uncited or unsupported claim": 55
   },
   "seconds_per_answer": 0.2393,
   "em_if_supported": 0.6106,
   "em_if_unsupported": 0.5556,
   "before": {
    "answered": 420,
    "em_among_answered": 0.6024,
    "em_overall": 0.506,
    "wrong_answers": 167
   },
   "after_withholding_unsupported": {
    "answered": 357,
    "em_among_answered": 0.6106,
    "em_overall": 0.436,
    "wrong_answers": 139,
    "wrong_removed": 28,
    "right_removed": 35
   }
  },
  "unanswerable": {
   "answers": 200,
   "answers_given": 17,
   "claims": 24,
   "claims_supported": 0.8333,
   "claims_uncited": 0,
   "answers_supported": 11,
   "answers_unsupported": 6,
   "reasons_unsupported": {
    "answer not in cited sources": 2,
    "uncited or unsupported claim": 4
   },
   "seconds_per_answer": 0.0223,
   "answered_without_gold": 17,
   "caught_by_check": 6
  }
 }
}
```

## soak_index

```
{
 "passages": 29476,
 "seconds": 194.79,
 "tenants": 4,
 "extra_files": 200
}
```

## sglang

```
{
 "flags": [],
 "mem_fraction_static": 0.7
}
```

