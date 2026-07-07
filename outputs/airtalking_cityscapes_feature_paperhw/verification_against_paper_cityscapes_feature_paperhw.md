# AirTalking reproduction verification against paper figures_cityscapes_feature_paperhw

## Conclusion

The current reproduction does not quantitatively match the paper figures. It preserves a few qualitative directions, especially that semantic processing outperforms the non-semantic baseline at 300 x 300 m2, but several key policy rankings and magnitudes differ.

## Quantitative check summary

- Match: 6
- Partial: 14
- Mismatch: 53

A `match` means the reproduction is within 25% of the paper visual estimate; `partial` is within 50%; `mismatch` is outside 50%. Because the paper provides plots but not raw data, the paper-side numbers are visual estimates from rendered Figure 3, Figure 4, and Figure 6.

## Qualitative checks

| Check | Status | Evidence |
|---|---|---|
| Finished requests increase with area for Stochastic | match | 100:19.1, 200:36.7, 300:50.4, 400:61.0, 500:63.4 |
| Finished requests increase with area for LinUCB | match | 100:17.0, 200:40.2, 300:78.6, 400:109.6, 500:114.1 |
| Finished requests increase with area for SA | match | 100:30.5, 200:82.4, 300:104.8, 400:133.0, 500:133.1 |
| Finished requests increase with area for Greedy | match | 100:33.0, 200:74.0, 300:118.6, 400:151.7, 500:178.8 |
| Finished requests increase with area for MCTS | match | 100:30.6, 200:66.2, 300:98.7, 400:129.8, 500:146.3 |
| Paper statement: LinUCB outperforms SA as area enlarges | mismatch | 400m LinUCB=109.6, SA=133.0; 500m LinUCB=114.1, SA=133.1 |
| Figure 6 direction: semantic beats non-semantic on finished requests | match | LinUCB: sem=78.6, ns=21.0; SA: sem=104.8, ns=23.0; Greedy: sem=118.6, ns=24.5; MCTS: sem=98.7, ns=25.5 |

## Largest numeric deviations

| Check | Area | Policy | Metric | Paper estimate | Reproduction | Relative error |
|---|---:|---|---|---:|---:|---:|
| Figure 3 flight_energy | 200 | Greedy | flight_energy_per_req | 8000.000 | 107353.061 | 12.419 |
| Figure 3 flight_energy | 200 | LinUCB | flight_energy_per_req | 10000.000 | 129517.354 | 11.952 |
| Figure 4 avg_time | 200 | Greedy | avg_time | 7.000 | 88.561 | 11.652 |
| Figure 4 avg_time | 300 | MCTS | avg_time | 5.000 | 60.609 | 11.122 |
| Figure 3 flight_energy | 300 | MCTS | flight_energy_per_req | 6000.000 | 72652.194 | 11.109 |
| Figure 4 avg_time | 200 | MCTS | avg_time | 7.000 | 83.680 | 10.954 |
| Figure 3 flight_energy | 200 | MCTS | flight_energy_per_req | 8500.000 | 101194.803 | 10.905 |
| Figure 4 avg_time | 200 | LinUCB | avg_time | 9.000 | 106.652 | 10.850 |

## Main mismatch types

- Magnitude mismatch: several reproduced metrics are outside the 50% tolerance against visually estimated paper values.
- Hidden-parameter sensitivity: request probability, workload distribution, propulsion/hover power, and detailed interference scheduling are not numerically disclosed in the paper.
- Dataset/profile sensitivity: substitute semantic payload profiles can improve some metrics but shift completed-request counts and energy/latency trade-offs.

## Likely causes

- The paper does not publish raw source code, request probability, workload distribution, propulsion/hover power, codec power, or full interference scheduling details.
- The reproduction uses assumed values for those hidden parameters, and those assumptions materially change latency, energy, and policy ranking.
- The density interference correction added to mimic small-area interference improves one trend but inflates latency and flight energy relative to the paper.

## Detailed CSV

See `outputs\airtalking_cityscapes_feature_paperhw\verification_against_paper_cityscapes_feature_paperhw.csv` for row-level expected vs. reproduced values.
