# Adaptive Semantic Compression Study

This folder contains a follow-up study for a proposed research topic:

> Channel-aware adaptive semantic compression for multi-UAV D2D communication.

The experiment does not train a new neural network. Instead, it uses Cityscapes semantic label maps as a public proxy for semantic feature maps and measures the trade-off between:

- semantic payload size,
- semantic map restoration quality,
- estimated communication latency under UAV channel samples,
- actual scheduling outcomes inside the AirTalking reproduction simulator.

## Inputs

Expected local inputs:

```text
dataset/gtFine_trainvaltest/gtFine/{train,val}/**/*_gtFine_labelIds.png
outputs/airtalking_cityscapes_calibrated_final_p012/timeseries_and_sinr_samples.npz
outputs/airtalking_cityscapes_calibrated_final_p012/run_metadata.json
```

Data sources:

- Cityscapes `gtFine_trainvaltest.zip`, official Cityscapes download site. The original dataset files are not included in this repository.
- AirTalking reproduction outputs generated locally from the public paper equations and the calibrated reproduction script. Raw paper data and official AirTalking source code were not available.

## Run

```bash
# Pilot SINR-only probe.
python adaptive_semantic_compression_study/run_adaptive_probe.py --sample-limit 400
python adaptive_semantic_compression_study/run_adaptive_probe.py --reuse-quality

# Full scheduler-integrated follow-up experiment.
python adaptive_semantic_compression_study/run_full_adaptive_research.py

# Proposal document builder.
python adaptive_semantic_compression_study/build_research_proposal.py
```

## Outputs

```text
# Pilot outputs.
adaptive_semantic_compression_study/outputs/compression_quality.csv
adaptive_semantic_compression_study/outputs/policy_summary.json
adaptive_semantic_compression_study/outputs/figures/quality_vs_payload.png
adaptive_semantic_compression_study/outputs/figures/delivery_time_by_policy.png
adaptive_semantic_compression_study/outputs/figures/adaptive_mode_usage.png

# Full follow-up outputs.
adaptive_semantic_compression_study/full_adaptive_results/summary_metrics.csv
adaptive_semantic_compression_study/full_adaptive_results/compression_mode_usage.csv
adaptive_semantic_compression_study/full_adaptive_results/run_metadata.json
adaptive_semantic_compression_study/full_adaptive_results/figures/finished_by_area_greedy_mcts.png
adaptive_semantic_compression_study/full_adaptive_results/figures/latency_quality_tradeoff_300m.png
adaptive_semantic_compression_study/full_adaptive_results/figures/adaptive_mode_usage_300m.png

# Local-only document/report outputs.
adaptive_semantic_compression_study/Adaptive_Semantic_Compression_Research_Proposal_KR.docx
adaptive_semantic_compression_study/full_adaptive_results/adaptive_followup_research_report.md
```

The full experiment compares three modes across all five AirTalking area settings and all five policies:

- `nonsemantic`: raw payload without semantic compression.
- `fixed_paper_like`: fixed Cityscapes paper-like semantic payload ratio.
- `adaptive_semantic`: channel-aware selection among emergency, low, medium, paper-like, and high-quality semantic payload modes.
