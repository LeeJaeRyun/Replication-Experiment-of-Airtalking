# AirTalking reproduction verification against paper figures_calibrated_final_p012

## Conclusion

The current reproduction does not quantitatively match the paper figures. It preserves a few qualitative directions, especially that semantic processing outperforms the non-semantic baseline at 300 x 300 m2, but several key policy rankings and magnitudes differ.

## Quantitative check summary

- Match: 18
- Partial: 15
- Mismatch: 40

A `match` means the reproduction is within 25% of the paper visual estimate; `partial` is within 50%; `mismatch` is outside 50%. Because the paper provides plots but not raw data, the paper-side numbers are visual estimates from rendered Figure 3, Figure 4, and Figure 6.

## Qualitative checks

| Check | Status | Evidence |
|---|---|---|
| Finished requests increase with area for Stochastic | match | 100:28.3, 200:51.0, 300:75.0, 400:82.2, 500:82.7 |
| Finished requests increase with area for LinUCB | partial | 100:67.7, 200:123.4, 300:127.5, 400:128.7, 500:125.8 |
| Finished requests increase with area for SA | partial | 100:40.4, 200:78.7, 300:90.1, 400:111.5, 500:102.2 |
| Finished requests increase with area for Greedy | partial | 100:127.6, 200:170.8, 300:187.4, 400:194.9, 500:188.5 |
| Finished requests increase with area for MCTS | partial | 100:111.9, 200:157.0, 300:158.5, 400:160.9, 500:149.5 |
| Paper statement: LinUCB outperforms SA as area enlarges | match | 400m LinUCB=128.7, SA=111.5; 500m LinUCB=125.8, SA=102.2 |
| Figure 6 direction: semantic beats non-semantic on finished requests | match | LinUCB: sem=127.5, ns=66.5; SA: sem=90.1, ns=68.3; Greedy: sem=187.4, ns=75.9; MCTS: sem=158.5, ns=77.0 |

## Largest numeric deviations

| Check | Area | Policy | Metric | Paper estimate | Reproduction | Relative error |
|---|---:|---|---|---:|---:|---:|
| Figure 4 avg_time | 400 | SA | avg_time | 7.000 | 40.135 | 4.734 |
| Figure 4 avg_time | 200 | SA | avg_time | 12.000 | 65.917 | 4.493 |
| Figure 3 finished | 300 | Stochastic | finished | 15.000 | 75.000 | 4.000 |
| Figure 3 finished | 500 | Stochastic | finished | 18.000 | 82.700 | 3.594 |
| Figure 4 avg_time | 300 | SA | avg_time | 12.000 | 54.860 | 3.572 |
| Figure 3 finished | 400 | Stochastic | finished | 18.000 | 82.200 | 3.567 |
| Figure 4 avg_time | 500 | MCTS | avg_time | 5.000 | 21.118 | 3.224 |
| Figure 4 avg_time | 400 | LinUCB | avg_time | 8.000 | 32.311 | 3.039 |

## Main mismatch types

- Magnitude mismatch: several reproduced metrics are outside the 50% tolerance against visually estimated paper values.
- Hidden-parameter sensitivity: request probability, workload distribution, propulsion/hover power, and detailed interference scheduling are not numerically disclosed in the paper.
- Dataset/profile sensitivity: substitute semantic payload profiles can improve some metrics but shift completed-request counts and energy/latency trade-offs.

## Likely causes

- The paper does not publish raw source code, request probability, workload distribution, propulsion/hover power, codec power, or full interference scheduling details.
- The reproduction uses assumed values for those hidden parameters, and those assumptions materially change latency, energy, and policy ranking.
- The density interference correction added to mimic small-area interference improves one trend but inflates latency and flight energy relative to the paper.

## Detailed CSV

See `studies\airtalking_reproduction\results\airtalking_cityscapes_calibrated_final_p012\verification_against_paper_calibrated_final_p012.csv` for row-level expected vs. reproduced values.
