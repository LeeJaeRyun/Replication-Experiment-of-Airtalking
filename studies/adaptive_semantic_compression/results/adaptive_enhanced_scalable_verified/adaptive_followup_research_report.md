# Adaptive semantic compression follow-up research

## What changed

This experiment moves beyond the earlier SINR-only probe. The adaptive compression policy is now inside the UAV scheduler, so each candidate UAV relay action is evaluated with the payload ratio selected for that candidate link condition.

## Data and settings

- Public semantic data: Cityscapes gtFine train/val label maps.
- Base simulator parameters: `C:\Users\firep\OneDrive\바탕 화면\Replication-Experiment-of-Airtalking\studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified\run_metadata.json`.
- Repeats: 10; simulation slots per repeat: 1000.
- Compared modes: nonsemantic raw payload, fixed Cityscapes paper-like semantic payload, and channel-aware adaptive semantic payload.
- Neural encoder/decoder reflection: all five adaptive levels use measured payload and mIoU from one trained scalable neural codec for mode selection.

## Compression table

| mode | payload ratio | mean IoU |
|---|---:|---:|
| emergency | 0.026042 | 0.305 |
| low | 0.052083 | 0.306 |
| medium | 0.078125 | 0.305 |
| paper_like | 0.104167 | 0.305 |
| high | 0.156250 | 0.306 |

## Neural encoder/decoder anchor

- rho_c: 0.104167
- pixel accuracy: 0.8286
- mIoU: 0.3054
- encode/decode median time: 1.87 ms / 3.01 ms
- adaptive quality mode: selection; measured neural mIoU is used both for recording and mode selection.
- SINR-to-quality threshold rule: measured_ordered (measured_multi_rate_neural_miou_ascending); actual thresholds are preserved in run metadata.

## 300 x 300 m result table

| policy | mode | finished | avg time (s) | flight J/req | semantic quality | payload ratio |
|---|---|---:|---:|---:|---:|---:|
| Stochastic | nonsemantic | 66.5 | 78.61 | 17885.3 | 1.000 | 1.000 |
| Stochastic | fixed_paper_like | 72.2 | 68.05 | 15452.5 | 0.806 | 0.749 |
| Stochastic | adaptive_semantic | 80.0 | 59.23 | 13413.9 | 0.795 | 0.717 |
| LinUCB | nonsemantic | 66.5 | 81.98 | 18569.2 | 1.000 | 1.000 |
| LinUCB | fixed_paper_like | 138.3 | 25.81 | 5645.7 | 0.529 | 0.393 |
| LinUCB | adaptive_semantic | 149.4 | 20.83 | 4492.5 | 0.537 | 0.364 |
| SA | nonsemantic | 68.3 | 76.57 | 17400.8 | 1.000 | 1.000 |
| SA | fixed_paper_like | 100.3 | 46.41 | 10467.1 | 0.616 | 0.504 |
| SA | adaptive_semantic | 108.7 | 38.52 | 8660.8 | 0.610 | 0.462 |
| Greedy | nonsemantic | 75.9 | 66.19 | 15033.2 | 1.000 | 1.000 |
| Greedy | fixed_paper_like | 190.7 | 10.52 | 2234.6 | 0.316 | 0.118 |
| Greedy | adaptive_semantic | 210.6 | 6.27 | 1278.9 | 0.331 | 0.125 |
| MCTS | nonsemantic | 77.0 | 63.76 | 14465.3 | 1.000 | 1.000 |
| MCTS | fixed_paper_like | 164.4 | 19.04 | 4154.0 | 0.426 | 0.260 |
| MCTS | adaptive_semantic | 184.1 | 12.98 | 2771.8 | 0.470 | 0.279 |

## Adaptive vs fixed semantic at 300 x 300 m

| policy | finished change | avg time change | quality change | payload-ratio change |
|---|---:|---:|---:|---:|
| Stochastic | 10.8% | -13.0% | -0.011 | -4.3% |
| LinUCB | 8.0% | -19.3% | +0.008 | -7.3% |
| SA | 8.4% | -17.0% | -0.006 | -8.3% |
| Greedy | 10.4% | -40.4% | +0.015 | 6.2% |
| MCTS | 12.0% | -31.8% | +0.043 | 7.4% |

## Main interpretation

- Greedy at 300 m: adaptive finished 210.6 requests vs fixed 190.7; average time changed from 10.52s to 6.27s.
- MCTS at 300 m: adaptive finished 184.1 requests vs fixed 164.4; average time changed from 19.04s to 12.98s.
- The result is a trade-off study, not a claim that the original AirTalking neural network was reproduced. Its weights and training recipe are not public; this run uses the explicitly specified paper-inspired scalable codec.
- Adaptive compression improves scheduling latency by changing the payload-quality choice per candidate link. In weak links it can choose a smaller payload than the fixed paper-like mode; in stronger links it may spend more payload to preserve semantic quality.
- Therefore the key claim is not that adaptive always lowers the average payload ratio. The measured gain is a latency/completion improvement with a controlled semantic-quality drop.

## Generated artifacts

- `summary_metrics.csv`: final numeric comparison for all modes, areas, and policies.
- `compression_mode_usage.csv`: selected adaptive mode counts.
- `timeseries_and_sinr_samples.npz`: time series and SINR samples for later plotting.
- `figures/`: result figures for area scaling, latency-quality trade-off, and adaptive mode usage.

Elapsed wall time: 197.0 seconds.