from __future__ import annotations

import argparse
import csv
import json
import time
import zlib
from collections import Counter
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image


def collect_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = root / split
    label_dir = root / f"{split}annot"
    pairs = []
    for image_path in sorted(image_dir.glob("*.png")):
        label_path = label_dir / image_path.name
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def build_palette(label_paths: list[Path]) -> dict[tuple[int, int, int], int]:
    colors: set[tuple[int, int, int]] = set()
    for path in label_paths:
        arr = np.asarray(Image.open(path).convert("RGB"))
        flat = arr.reshape(-1, 3)
        for color in np.unique(flat, axis=0):
            colors.add(tuple(int(v) for v in color))
    return {color: idx for idx, color in enumerate(sorted(colors))}


def label_to_index(label_rgb: np.ndarray, palette: dict[tuple[int, int, int], int]) -> np.ndarray:
    packed = (
        label_rgb[:, :, 0].astype(np.uint32) << 16
        | label_rgb[:, :, 1].astype(np.uint32) << 8
        | label_rgb[:, :, 2].astype(np.uint32)
    )
    mapping = {((r << 16) | (g << 8) | b): idx for (r, g, b), idx in palette.items()}
    out = np.zeros(label_rgb.shape[:2], dtype=np.uint8)
    for packed_color, idx in mapping.items():
        out[packed == packed_color] = idx
    return out


def index_to_rgb(label_idx: np.ndarray, reverse_palette: list[tuple[int, int, int]]) -> np.ndarray:
    palette_arr = np.asarray(reverse_palette, dtype=np.uint8)
    return palette_arr[label_idx]


def measure_pair(
    image_path: Path,
    label_path: Path,
    palette: dict[tuple[int, int, int], int],
    repeats: int,
    feature_scale: float,
) -> dict[str, float | str | int]:
    image = Image.open(image_path).convert("RGB")
    label = Image.open(label_path).convert("RGB")
    image_arr = np.asarray(image)
    label_arr = np.asarray(label)
    reverse_palette = [color for color, _ in sorted(palette.items(), key=lambda item: item[1])]

    raw_uncompressed_bytes = int(image_arr.size)
    raw_png_bytes = image_path.stat().st_size
    label_png_bytes = label_path.stat().st_size

    encode_times = []
    decode_times = []
    feature_encode_times = []
    feature_decode_times = []
    compressed_payload = b""
    label_idx = None
    restored_rgb = None
    feature_payload = b""
    feature_restored_rgb = None
    for _ in range(repeats):
        start = time.perf_counter()
        label_idx = label_to_index(label_arr, palette)
        compressed_payload = zlib.compress(label_idx.tobytes(), level=6)
        encode_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        decoded = np.frombuffer(zlib.decompress(compressed_payload), dtype=np.uint8).reshape(label_idx.shape)
        restored_rgb = index_to_rgb(decoded, reverse_palette)
        decode_times.append(time.perf_counter() - start)

        feature_w = max(1, int(round(image.width * feature_scale)))
        feature_h = max(1, int(round(image.height * feature_scale)))
        start = time.perf_counter()
        if label_idx is None:
            label_idx = label_to_index(label_arr, palette)
        feature_map = np.asarray(
            Image.fromarray(label_idx, mode="L").resize((feature_w, feature_h), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        feature_payload = feature_map.tobytes()
        feature_encode_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        feature_decoded = np.frombuffer(feature_payload, dtype=np.uint8).reshape((feature_h, feature_w))
        upsampled = np.asarray(
            Image.fromarray(feature_decoded, mode="L").resize((image.width, image.height), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        feature_restored_rgb = index_to_rgb(upsampled, reverse_palette)
        feature_decode_times.append(time.perf_counter() - start)

    assert label_idx is not None
    assert restored_rgb is not None
    assert feature_restored_rgb is not None
    exact_restore = bool(np.array_equal(restored_rgb, label_arr))
    semantic_bytes = len(compressed_payload)
    feature_bytes = len(feature_payload)
    return {
        "image": str(image_path),
        "label": str(label_path),
        "width": image.width,
        "height": image.height,
        "raw_uncompressed_bytes": raw_uncompressed_bytes,
        "raw_png_bytes": raw_png_bytes,
        "label_png_bytes": label_png_bytes,
        "semantic_zlib_bytes": semantic_bytes,
        "semantic_feature_bytes": feature_bytes,
        "rho_c_uncompressed": semantic_bytes / raw_uncompressed_bytes,
        "rho_c_feature_uncompressed": feature_bytes / raw_uncompressed_bytes,
        "rho_c_png": semantic_bytes / raw_png_bytes,
        "rho_c_feature_png": feature_bytes / raw_png_bytes,
        "label_png_to_raw_ratio": label_png_bytes / raw_uncompressed_bytes,
        "encode_seconds_median": median(encode_times),
        "decode_seconds_median": median(decode_times),
        "feature_encode_seconds_median": median(feature_encode_times),
        "feature_decode_seconds_median": median(feature_decode_times),
        "encode_bitrate_mbps": (raw_uncompressed_bytes * 8) / median(encode_times) / 1e6,
        "decode_bitrate_mbps": (semantic_bytes * 8) / median(decode_times) / 1e6,
        "feature_encode_bitrate_mbps": (raw_uncompressed_bytes * 8) / median(feature_encode_times) / 1e6,
        "feature_decode_bitrate_mbps": (feature_bytes * 8) / median(feature_decode_times) / 1e6,
        "exact_restore": int(exact_restore),
    }


def write_sample_visuals(rows: list[dict[str, float | str | int]], out_dir: Path, max_samples: int = 6) -> None:
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows[:max_samples]):
        image = Image.open(str(row["image"])).convert("RGB").resize((360, 270))
        label = Image.open(str(row["label"])).convert("RGB").resize((360, 270), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (720, 300), "white")
        canvas.paste(image, (0, 24))
        canvas.paste(label, (360, 24))
        canvas.save(sample_dir / f"camvid_semantic_pair_{idx+1}.png")


def summarize(rows: list[dict[str, float | str | int]], palette_size: int, source: str) -> dict[str, object]:
    numeric_keys = [
        "raw_uncompressed_bytes",
        "raw_png_bytes",
        "label_png_bytes",
        "semantic_zlib_bytes",
        "semantic_feature_bytes",
        "rho_c_uncompressed",
        "rho_c_feature_uncompressed",
        "rho_c_png",
        "rho_c_feature_png",
        "label_png_to_raw_ratio",
        "encode_seconds_median",
        "decode_seconds_median",
        "feature_encode_seconds_median",
        "feature_decode_seconds_median",
        "encode_bitrate_mbps",
        "decode_bitrate_mbps",
        "feature_encode_bitrate_mbps",
        "feature_decode_bitrate_mbps",
    ]
    summary = {
        "source": source,
        "num_samples": len(rows),
        "palette_classes": palette_size,
        "exact_restore_count": int(sum(int(row["exact_restore"]) for row in rows)),
        "measure_definition": {
            "raw_payload": "uncompressed RGB image bytes",
            "semantic_payload": "zlib-compressed 8-bit semantic label-index map from CamVid annotation",
            "semantic_feature_payload": "uncompressed downsampled 8-bit semantic label-index tensor",
            "decoder_output": "palette RGB reconstruction of the semantic map",
            "rho_c": "semantic_payload/raw_payload",
            "rho_r_proxy": "raw RGB channels per semantic index channel = 3",
        },
        "rho_r_proxy": 3.0,
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_median"] = median(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure CamVid semantic payload compression for AirTalking reproduction.")
    parser.add_argument("--root", default="datasets/SegNet-Tutorial/CamVid")
    parser.add_argument("--out", default="outputs/camvid_semantic_measurement")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all pairs.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--feature-scale", type=float, default=0.56)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    pairs: list[tuple[Path, Path]] = []
    split_counts = Counter()
    for split in splits:
        split_pairs = collect_pairs(root, split)
        pairs.extend(split_pairs)
        split_counts[split] = len(split_pairs)
    if args.max_samples and args.max_samples < len(pairs):
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise SystemExit(f"No CamVid pairs found under {root}")

    palette = build_palette([label for _, label in pairs])
    rows = [measure_pair(image, label, palette, args.repeats, args.feature_scale) for image, label in pairs]
    csv_path = out_dir / "camvid_semantic_measurements.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, len(palette), str(root))
    summary["split_counts"] = dict(split_counts)
    summary["feature_scale"] = args.feature_scale
    summary_path = out_dir / "camvid_semantic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_sample_visuals(rows, out_dir)

    print(json.dumps({"csv": str(csv_path), "summary": str(summary_path), "summary_values": summary}, indent=2))


if __name__ == "__main__":
    main()
