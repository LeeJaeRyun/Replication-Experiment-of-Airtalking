# Adaptive semantic compression follow-up research

## What changed

This experiment moves beyond the earlier SINR-only probe. The adaptive compression policy is now inside the UAV scheduler, so each candidate UAV relay action is evaluated with the payload ratio selected for that candidate link condition.

## Data and settings

- Public semantic data: Cityscapes gtFine train/val label maps.
- Base simulator: the calibrated AirTalking reproduction parameters from `outputs/airtalking_cityscapes_calibrated_final_p012/run_metadata.json`.
- Repeats: 10; simulation slots per repeat: 1000.
- Compared modes: nonsemantic raw payload, fixed Cityscapes paper-like semantic payload, and channel-aware adaptive semantic payload.

## Compression table

| mode | payload ratio | mean IoU |
|---|---:|---:|
| emergency | 0.005208 | 0.814 |
| low | 0.020833 | 0.892 |
| medium | 0.046875 | 0.935 |
| paper_like | 0.104464 | 0.951 |
| high | 0.187500 | 0.969 |

## 300 x 300 m result table

| policy | mode | finished | avg time (s) | flight J/req | semantic quality | payload ratio |
|---|---|---:|---:|---:|---:|---:|
| Stochastic | nonsemantic | 66.5 | 78.61 | 17885.3 | 1.000 | 1.000 |
| Stochastic | fixed_paper_like | 75.0 | 70.29 | 15931.2 | 0.987 | 0.769 |
| Stochastic | adaptive_semantic | 78.6 | 65.83 | 14937.8 | 0.965 | 0.742 |
| LinUCB | nonsemantic | 66.5 | 81.98 | 18569.2 | 1.000 | 1.000 |
| LinUCB | fixed_paper_like | 127.5 | 29.53 | 6489.2 | 0.967 | 0.398 |
| LinUCB | adaptive_semantic | 155.5 | 19.96 | 4283.5 | 0.922 | 0.385 |
| SA | nonsemantic | 68.3 | 76.57 | 17400.8 | 1.000 | 1.000 |
| SA | fixed_paper_like | 90.1 | 54.86 | 12394.8 | 0.974 | 0.529 |
| SA | adaptive_semantic | 103.2 | 44.41 | 10012.7 | 0.929 | 0.488 |
| Greedy | nonsemantic | 75.9 | 66.19 | 15033.2 | 1.000 | 1.000 |
| Greedy | fixed_paper_like | 187.4 | 11.18 | 2416.0 | 0.961 | 0.301 |
| Greedy | adaptive_semantic | 211.7 | 6.26 | 1276.6 | 0.940 | 0.371 |
| MCTS | nonsemantic | 77.0 | 63.76 | 14465.3 | 1.000 | 1.000 |
| MCTS | fixed_paper_like | 158.5 | 19.33 | 4244.8 | 0.962 | 0.318 |
| MCTS | adaptive_semantic | 183.6 | 11.98 | 2529.0 | 0.927 | 0.360 |

## Adaptive vs fixed semantic at 300 x 300 m

| policy | finished change | avg time change | quality change | payload-ratio change |
|---|---:|---:|---:|---:|
| Stochastic | 4.8% | -6.3% | -0.022 | -3.5% |
| LinUCB | 22.0% | -32.4% | -0.045 | -3.3% |
| SA | 14.5% | -19.0% | -0.045 | -7.8% |
| Greedy | 13.0% | -44.0% | -0.021 | 23.5% |
| MCTS | 15.8% | -38.0% | -0.036 | 13.1% |

## Main interpretation

- Greedy at 300 m: adaptive finished 211.7 requests vs fixed 187.4; average time changed from 11.18s to 6.26s.
- MCTS at 300 m: adaptive finished 183.6 requests vs fixed 158.5; average time changed from 19.33s to 11.98s.
- The result should be read as a trade-off study, not as a claim that the original AirTalking paper is fully reproduced. The encoder/decoder network is still not public; Cityscapes label maps are used as a public semantic-feature proxy.
- Adaptive compression improves scheduling latency by changing the payload-quality choice per candidate link. In weak links it can choose a smaller payload than the fixed paper-like mode; in stronger links it may spend more payload to preserve semantic quality.
- Therefore the key claim is not that adaptive always lowers the average payload ratio. The measured gain is a latency/completion improvement with a controlled semantic-quality drop.

## Generated artifacts

- `summary_metrics.csv`: final numeric comparison for all modes, areas, and policies.
- `compression_mode_usage.csv`: selected adaptive mode counts.
- `timeseries_and_sinr_samples.npz`: time series and SINR samples for later plotting.
- `figures/`: result figures for area scaling, latency-quality trade-off, and adaptive mode usage.

Elapsed wall time: 60987.3 seconds.
