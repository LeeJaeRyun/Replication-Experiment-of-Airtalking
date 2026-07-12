# AirTalking reproduction verification against paper figures (enhanced_scalable_verified)

## Conclusion

The current reproduction does not quantitatively match the paper figures. It preserves a few qualitative directions, especially that semantic processing outperforms the non-semantic baseline at 300 x 300 m2, but several key policy rankings and magnitudes differ.

## Quantitative check summary

- Match: 17
- Partial: 17
- Mismatch: 39

A `match` means the reproduction is within 25% of the paper visual estimate; `partial` is within 50%; `mismatch` is outside 50%. Because the paper provides plots but not raw data, the paper-side numbers are visual estimates from rendered Figure 3, Figure 4, and Figure 6.

## Qualitative checks

| Check | Status | Evidence |
|---|---|---|
| Finished requests increase with area for Stochastic | match | 100:31.5, 200:55.3, 300:72.2, 400:87.1, 500:88.1 |
| Finished requests increase with area for LinUCB | partial | 100:65.3, 200:119.5, 300:138.3, 400:126.2, 500:130.3 |
| Finished requests increase with area for SA | match | 100:44.2, 200:81.0, 300:100.3, 400:107.1, 500:107.3 |
| Finished requests increase with area for Greedy | partial | 100:122.4, 200:174.0, 300:190.7, 400:185.2, 500:194.6 |
| Finished requests increase with area for MCTS | partial | 100:123.9, 200:154.4, 300:164.4, 400:163.5, 500:158.5 |
| Paper statement: LinUCB outperforms SA as area enlarges | match | 400m LinUCB=126.2, SA=107.1; 500m LinUCB=130.3, SA=107.3 |
| Figure 6 direction: semantic beats non-semantic on finished requests | match | LinUCB: sem=138.3, ns=66.5; SA: sem=100.3, ns=68.3; Greedy: sem=190.7, ns=75.9; MCTS: sem=164.4, ns=77.0 |

## Largest numeric deviations

| Check | Area | Policy | Metric | Paper estimate | Reproduction | Relative error |
|---|---:|---|---|---:|---:|---:|
| Figure 4 avg_time | 400 | SA | avg_time | 7.000 | 42.849 | 5.121 |
| Figure 3 finished | 500 | Stochastic | finished | 18.000 | 88.100 | 3.894 |
| Figure 3 finished | 400 | Stochastic | finished | 18.000 | 87.100 | 3.839 |
| Figure 3 finished | 300 | Stochastic | finished | 15.000 | 72.200 | 3.813 |
| Figure 4 avg_time | 200 | SA | avg_time | 12.000 | 55.745 | 3.645 |
| Figure 4 avg_time | 400 | LinUCB | avg_time | 8.000 | 32.189 | 3.024 |
| Figure 4 avg_time | 300 | SA | avg_time | 12.000 | 46.413 | 2.868 |
| Figure 4 avg_time | 300 | MCTS | avg_time | 5.000 | 19.042 | 2.808 |

## Main mismatch types

- Magnitude mismatch: several reproduced metrics are outside the 50% tolerance against visually estimated paper values.
- Hidden-parameter sensitivity: request probability, workload distribution, propulsion/hover power, and detailed interference scheduling are not numerically disclosed in the paper.
- Dataset/profile sensitivity: substitute semantic payload profiles can improve some metrics but shift completed-request counts and energy/latency trade-offs.

## Likely causes

- The paper does not publish raw source code, request probability, workload distribution, propulsion/hover power, codec power, or full interference scheduling details.
- The reproduction uses assumed values for those hidden parameters, and those assumptions materially change latency, energy, and policy ranking.
- The density interference correction added to mimic small-area interference improves one trend but inflates latency and flight energy relative to the paper.

## Detailed CSV

See `studies\airtalking_reproduction\results\airtalking_enhanced_scalable_verified\verification_against_paper_enhanced_scalable_verified.csv` for row-level expected vs. reproduced values.
