# AirTalking reproduction verification against paper figures_neural_encoder_decoder_timed

## Conclusion

The current reproduction does not quantitatively match the paper figures. It preserves a few qualitative directions, especially that semantic processing outperforms the non-semantic baseline at 300 x 300 m2, but several key policy rankings and magnitudes differ.

## Quantitative check summary

- Match: 20
- Partial: 12
- Mismatch: 41

A `match` means the reproduction is within 25% of the paper visual estimate; `partial` is within 50%; `mismatch` is outside 50%. Because the paper provides plots but not raw data, the paper-side numbers are visual estimates from rendered Figure 3, Figure 4, and Figure 6.

## Qualitative checks

| Check | Status | Evidence |
|---|---|---|
| Finished requests increase with area for Stochastic | match | 100:30.7, 200:54.8, 300:74.4, 400:77.3, 500:84.7 |
| Finished requests increase with area for LinUCB | partial | 100:69.3, 200:110.9, 300:127.9, 400:126.7, 500:122.8 |
| Finished requests increase with area for SA | partial | 100:42.5, 200:77.6, 300:101.4, 400:114.1, 500:102.3 |
| Finished requests increase with area for Greedy | match | 100:126.4, 200:177.9, 300:184.0, 400:187.0, 500:195.7 |
| Finished requests increase with area for MCTS | partial | 100:107.5, 200:153.6, 300:160.8, 400:163.4, 500:158.2 |
| Paper statement: LinUCB outperforms SA as area enlarges | match | 400m LinUCB=126.7, SA=114.1; 500m LinUCB=122.8, SA=102.3 |
| Figure 6 direction: semantic beats non-semantic on finished requests | match | LinUCB: sem=127.9, ns=66.5; SA: sem=101.4, ns=68.3; Greedy: sem=184.0, ns=75.9; MCTS: sem=160.8, ns=77.0 |

## Largest numeric deviations

| Check | Area | Policy | Metric | Paper estimate | Reproduction | Relative error |
|---|---:|---|---|---:|---:|---:|
| Figure 4 avg_time | 200 | SA | avg_time | 12.000 | 67.401 | 4.617 |
| Figure 4 avg_time | 400 | SA | avg_time | 7.000 | 38.114 | 4.445 |
| Figure 3 finished | 300 | Stochastic | finished | 15.000 | 74.400 | 3.960 |
| Figure 3 finished | 500 | Stochastic | finished | 18.000 | 84.700 | 3.706 |
| Figure 3 finished | 400 | Stochastic | finished | 18.000 | 77.300 | 3.294 |
| Figure 4 avg_time | 200 | LinUCB | avg_time | 9.000 | 36.535 | 3.059 |
| Figure 4 avg_time | 400 | LinUCB | avg_time | 8.000 | 32.446 | 3.056 |
| Figure 4 avg_time | 300 | MCTS | avg_time | 5.000 | 19.856 | 2.971 |

## Main mismatch types

- Magnitude mismatch: several reproduced metrics are outside the 50% tolerance against visually estimated paper values.
- Hidden-parameter sensitivity: request probability, workload distribution, propulsion/hover power, and detailed interference scheduling are not numerically disclosed in the paper.
- Dataset/profile sensitivity: substitute semantic payload profiles can improve some metrics but shift completed-request counts and energy/latency trade-offs.

## Likely causes

- The paper does not publish raw source code, request probability, workload distribution, propulsion/hover power, codec power, or full interference scheduling details.
- The reproduction uses assumed values for those hidden parameters, and those assumptions materially change latency, energy, and policy ranking.
- The density interference correction added to mimic small-area interference improves one trend but inflates latency and flight energy relative to the paper.

## Detailed CSV

See `outputs\airtalking_neural_encoder_decoder_timed\verification_against_paper_neural_encoder_decoder_timed.csv` for row-level expected vs. reproduced values.
