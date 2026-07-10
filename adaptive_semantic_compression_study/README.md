# Adaptive Semantic Compression Study

This folder contains a follow-up study for a proposed research topic:

> Channel-aware adaptive semantic compression for multi-UAV D2D communication.

The experiment does not train a new neural network. Instead, it uses Cityscapes semantic label maps as a public proxy for semantic feature maps and measures the trade-off between:

- semantic payload size,
- semantic map restoration quality,
- estimated communication latency under UAV channel samples,
- actual scheduling outcomes inside the AirTalking reproduction simulator.

A separate lightweight neural baseline is also included:

```text
RGB image -> neural encoder -> compressed latent feature -> neural decoder -> semantic segmentation map
```

This baseline is a real trainable encoder/decoder, but it is intentionally small so it can run on CPU. It is not yet a paper-grade semantic codec.

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

# Real neural encoder/decoder smoke training.
python adaptive_semantic_compression_study/train_semantic_encoder_decoder.py --epochs 1 --train-limit 16 --val-limit 8

# Earlier CPU baseline.
python adaptive_semantic_compression_study/train_semantic_encoder_decoder.py --out adaptive_semantic_compression_study/encoder_decoder_results/cpu_baseline_latent8 --epochs 20 --train-limit 256 --val-limit 128 --image-width 128 --image-height 64 --batch-size 8 --width 16 --latent-channels 8 --timing-runs 10 --save-checkpoint

# Paper-like neural encoder/decoder result.
python adaptive_semantic_compression_study/train_semantic_encoder_decoder.py --out adaptive_semantic_compression_study/encoder_decoder_results/paperlike_tiny_latent20 --model tiny --epochs 30 --train-limit 512 --val-limit 256 --image-width 128 --image-height 64 --batch-size 8 --width 24 --latent-channels 20 --class-balanced-loss --timing-runs 10 --save-checkpoint

# Table-III-timed neural encoder/decoder result.
python adaptive_semantic_compression_study/train_semantic_encoder_decoder.py --out adaptive_semantic_compression_study/encoder_decoder_results/paperlike_timed_latent20 --model paperlite --epochs 30 --train-limit 512 --val-limit 256 --image-width 128 --image-height 64 --batch-size 8 --width 8 --latent-channels 20 --class-balanced-loss --timing-runs 10 --save-checkpoint

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

# Neural encoder/decoder outputs.
adaptive_semantic_compression_study/encoder_decoder_results/cpu_baseline_latent8/result_summary.json
adaptive_semantic_compression_study/encoder_decoder_results/cpu_baseline_latent8/training_history.csv
adaptive_semantic_compression_study/encoder_decoder_results/paperlike_tiny_latent20/result_summary.json
adaptive_semantic_compression_study/encoder_decoder_results/paperlike_tiny_latent20/airtalking_semantic_summary.json
adaptive_semantic_compression_study/encoder_decoder_results/paperlike_tiny_latent20/training_history.csv
adaptive_semantic_compression_study/encoder_decoder_results/paperlike_timed_latent20/result_summary.json
adaptive_semantic_compression_study/encoder_decoder_results/paperlike_timed_latent20/airtalking_semantic_summary.json

# Local-only document/report outputs.
adaptive_semantic_compression_study/Adaptive_Semantic_Compression_Research_Proposal_KR.docx
adaptive_semantic_compression_study/full_adaptive_results/adaptive_followup_research_report.md
adaptive_semantic_compression_study/encoder_decoder_results/**/semantic_encoder_decoder.pt
```

The full experiment compares three modes across all five AirTalking area settings and all five policies:

- `nonsemantic`: raw payload without semantic compression.
- `fixed_paper_like`: fixed Cityscapes paper-like semantic payload ratio.
- `adaptive_semantic`: channel-aware selection among emergency, low, medium, paper-like, and high-quality semantic payload modes.

Current CPU neural baseline result:

- Input size: 128 x 64 RGB image.
- Latent: 8 channels at 1/8 spatial resolution.
- Assumed latent quantization: 8-bit.
- Payload ratio: 0.0417 of raw RGB.
- Training subset: 256 train images.
- Validation subset: 128 val images.
- Epochs: 20.
- Validation pixel accuracy: 0.7567.
- Validation mIoU: 0.1881.
- Median encode/decode/full inference time on CPU: 0.64 ms / 1.83 ms / 2.57 ms.

Current paper-like neural encoder/decoder result:

- Input size: 128 x 64 RGB image.
- Latent: 20 channels at 1/8 spatial resolution.
- Assumed latent quantization: 8-bit.
- Payload ratio: 0.10417, close to the AirTalking Table III value 0.104.
- Training subset: 512 train images.
- Validation subset: 256 val images.
- Epochs: 30.
- Validation pixel accuracy: 0.7377.
- Validation mIoU: 0.2219.
- Median encode/decode/full inference time on CPU: 0.78 ms / 2.71 ms / 3.20 ms.
- Restoration-throughput-based decode bitrate: 22.65 Mbps, close to the AirTalking Table III value 23.23 Mbps.

Current Table-III-timed neural encoder/decoder result:

- Input size: 128 x 64 RGB image.
- Latent: 20 channels at 1/8 spatial resolution.
- Payload ratio: 0.10417, close to the AirTalking Table III value 0.104.
- Training subset: 512 train images.
- Validation subset: 256 val images.
- Best epoch reached before timeout: 27.
- Validation pixel accuracy: 0.7408.
- Validation mIoU: 0.2135.
- Median encode/decode/full inference time on CPU: 1.75 ms / 2.53 ms / 4.48 ms.
- Encoding bitrate: 112.40 Mbps, higher than the AirTalking Table III value 91.30 Mbps.
- Restoration-throughput-based decode bitrate: 24.30 Mbps, close to the AirTalking Table III value 23.23 Mbps.
