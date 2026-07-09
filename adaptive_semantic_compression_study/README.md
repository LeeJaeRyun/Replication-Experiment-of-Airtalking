# Adaptive Semantic Compression Study

This folder contains a small proof-of-concept study for a proposed research topic:

> Channel-aware adaptive semantic compression for multi-UAV D2D communication.

The experiment does not train a new neural network. Instead, it uses Cityscapes semantic label maps as a proxy for semantic feature maps and measures the trade-off between:

- semantic payload size,
- semantic map restoration quality,
- estimated communication latency under UAV channel samples.

## Inputs

Expected local inputs:

```text
dataset/gtFine_trainvaltest/gtFine/{train,val}/**/*_gtFine_labelIds.png
outputs/airtalking_cityscapes_calibrated_final_p012/timeseries_and_sinr_samples.npz
```

The original Cityscapes dataset is not included in the repository.

## Run

```bash
python adaptive_semantic_compression_study/run_adaptive_probe.py --sample-limit 400
python adaptive_semantic_compression_study/run_adaptive_probe.py --reuse-quality
python adaptive_semantic_compression_study/build_research_proposal.py
```

## Outputs

```text
adaptive_semantic_compression_study/outputs/compression_quality.csv
adaptive_semantic_compression_study/outputs/policy_summary.json
adaptive_semantic_compression_study/outputs/figures/quality_vs_payload.png
adaptive_semantic_compression_study/outputs/figures/delivery_time_by_policy.png
adaptive_semantic_compression_study/outputs/figures/adaptive_mode_usage.png
adaptive_semantic_compression_study/Adaptive_Semantic_Compression_Research_Proposal_KR.docx
```
