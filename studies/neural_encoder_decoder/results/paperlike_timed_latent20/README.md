# Table-III-timed neural semantic encoder/decoder result

## Purpose

This run is the closest current neural encoder/decoder approximation to the AirTalking Table III encoder/decoder settings.

```text
Cityscapes RGB image -> encoder -> compressed semantic latent -> decoder -> semantic segmentation map
```

Unlike the earlier proxy experiment, this is a real trained neural encoder/decoder.

## Model and Data

- Dataset: Cityscapes `leftImg8bit` RGB images and `gtFine` label maps.
- Train subset: 512 images.
- Validation subset: 256 images.
- Input resolution: 128 x 64.
- Model: `paperlite` convolutional semantic encoder/decoder.
- Latent: 20 channels at 1/8 spatial resolution.
- Quantization assumption: 8-bit latent values.
- Training: class-balanced cross entropy on CPU.
- Best checkpoint: epoch 27. The 30-epoch run timed out before final summary writing, so the best checkpoint was reloaded and evaluated.

## Encoder/Decoder Table III Comparison

| Metric | This run | AirTalking Table III | Difference |
|---|---:|---:|---:|
| Semantic compression ratio rho_c | 0.10417 | 0.104 | +0.16% |
| Restoration ratio rho_r | 3.0 | 3.0 | 0.0% |
| Encoding throughput | 112.40 Mbps | 91.30 Mbps | +23.11% |
| Decoding throughput, restoration basis | 24.30 Mbps | 23.23 Mbps | +4.59% |
| Validation pixel accuracy | 0.7408 | Not reported | Extra metric |
| Validation mIoU | 0.2135 | Not reported | Extra metric |
| Encode/decode/full time | 1.75 / 2.53 / 4.48 ms | Not directly reported | CPU measurement |

## AirTalking Simulation With This Model

The generated `airtalking_semantic_summary.json` was used as the semantic profile for the AirTalking reproduction simulator.

At 300 x 300 m:

| Policy | Semantic finished | Non-semantic finished | Semantic avg time |
|---|---:|---:|---:|
| LinUCB | 127.9 | 66.5 | 26.45 s |
| SA | 101.4 | 68.3 | 46.81 s |
| Greedy | 184.0 | 75.9 | 10.63 s |
| MCTS | 160.8 | 77.0 | 19.86 s |

Paper-figure verification count:

- Match: 20
- Partial: 12
- Mismatch: 41

The encoder/decoder Table III values are close, especially `rho_c` and decoding bitrate. Encoding is faster than the Table III value under the current CPU/resolution measurement. The full paper figures still do not fully match because the paper does not disclose all simulator constants, workload generation details, interference scheduling, or source code.

## Files

- `result_summary.json`: quality, timing, and paper comparison.
- `training_history.csv`: evaluation row for the recovered best checkpoint.
- `airtalking_semantic_summary.json`: profile consumed by `airtalking_reproduction.py`.
- `best_semantic_encoder_decoder.pt`: local checkpoint, ignored by git.
