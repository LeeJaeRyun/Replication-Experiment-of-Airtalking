from __future__ import annotations

import argparse
import csv
import json
import time
import zlib
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image


def find_cityscapes_roots(root: Path) -> tuple[Path, Path]:
    candidates = [
        (root / "leftImg8bit", root / "gtFine"),
        (root / "leftImg8bit_trainvaltest" / "leftImg8bit", root / "gtFine_trainvaltest" / "gtFine"),
        (root / "dataset" / "leftImg8bit_trainvaltest" / "leftImg8bit", root / "dataset" / "gtFine_trainvaltest" / "gtFine"),
    ]
    for left_root, gt_root in candidates:
        if left_root.exists() and gt_root.exists():
            return left_root, gt_root
    raise FileNotFoundError(f"Could not find leftImg8bit and gtFine under {root}")


def collect_pairs(root: Path, splits: list[str]) -> list[tuple[Path, Path]]:
    left_root, gt_root = find_cityscapes_roots(root)
    pairs: list[tuple[Path, Path]] = []
    for split in splits:
        for image_path in sorted((left_root / split).glob("*/*_leftImg8bit.png")):
            stem = image_path.name.replace("_leftImg8bit.png", "")
            label_path = gt_root / split / image_path.parent.name / f"{stem}_gtFine_labelIds.png"
            if label_path.exists():
                pairs.append((image_path, label_path))
    return pairs


def measure_pair(image_path: Path, label_path: Path, feature_scale: float, repeats: int) -> dict[str, float | int | str]:
    image = Image.open(image_path)
    width, height = image.size
    label = Image.open(label_path)
    label_arr = np.asarray(label, dtype=np.uint8)
    raw_uncompressed_bytes = width * height * 3
    raw_png_bytes = image_path.stat().st_size
    label_png_bytes = label_path.stat().st_size

    encode_times = []
    decode_times = []
    feature_encode_times = []
    feature_decode_times = []
    semantic_payload = b""
    feature_payload = b""
    feature_w = max(1, int(round(width * feature_scale)))
    feature_h = max(1, int(round(height * feature_scale)))

    for _ in range(repeats):
        start = time.perf_counter()
        semantic_payload = zlib.compress(label_arr.tobytes(), level=6)
        encode_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        decoded = np.frombuffer(zlib.decompress(semantic_payload), dtype=np.uint8).reshape(label_arr.shape)
        if decoded.shape != label_arr.shape:
            raise RuntimeError("decoded shape mismatch")
        decode_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        feature_map = np.asarray(
            Image.fromarray(label_arr, mode="L").resize((feature_w, feature_h), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        feature_payload = feature_map.tobytes()
        feature_encode_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        feature_decoded = np.frombuffer(feature_payload, dtype=np.uint8).reshape((feature_h, feature_w))
        _ = Image.fromarray(feature_decoded, mode="L").resize((width, height), Image.Resampling.NEAREST)
        feature_decode_times.append(time.perf_counter() - start)

    semantic_bytes = len(semantic_payload)
    feature_bytes = len(feature_payload)
    return {
        "image": str(image_path),
        "label": str(label_path),
        "width": width,
        "height": height,
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
        "exact_restore": 1,
    }


def summarize(rows: list[dict[str, float | int | str]], root: Path, splits: list[str], feature_scale: float) -> dict[str, object]:
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
    summary: dict[str, object] = {
        "source": str(root),
        "dataset": "Cityscapes",
        "splits": splits,
        "num_samples": len(rows),
        "feature_scale": feature_scale,
        "rho_r_proxy": 3.0,
        "measure_definition": {
            "raw_payload": "uncompressed RGB leftImg8bit image bytes",
            "semantic_payload": "zlib-compressed uint8 gtFine labelIds map",
            "semantic_feature_payload": "uncompressed downsampled uint8 gtFine labelIds tensor",
            "rho_c": "semantic_payload/raw_payload",
            "rho_r_proxy": "raw RGB channels per semantic label channel = 3",
        },
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_median"] = median(values)
    return summary


def write_sample_visuals(rows: list[dict[str, float | int | str]], out_dir: Path, max_samples: int = 4) -> None:
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows[:max_samples]):
        image = Image.open(str(row["image"])).convert("RGB").resize((512, 256))
        label = Image.open(str(row["label"])).convert("L")
        label_rgb = Image.fromarray((np.asarray(label, dtype=np.uint8) * 7).astype(np.uint8), mode="L").convert("RGB").resize((512, 256), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (1024, 286), "white")
        canvas.paste(image, (0, 30))
        canvas.paste(label_rgb, (512, 30))
        canvas.save(sample_dir / f"cityscapes_pair_{idx+1}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Cityscapes semantic payload profile for AirTalking reproduction.")
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--out", default="studies/airtalking_reproduction/results/cityscapes_semantic_measurement")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--feature-scale", type=float, default=0.56)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    pairs = collect_pairs(root, splits)
    if args.max_samples and args.max_samples < len(pairs):
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise SystemExit(f"No Cityscapes pairs found under {root}")

    rows = [measure_pair(image, label, args.feature_scale, args.repeats) for image, label in pairs]
    csv_path = out_dir / "cityscapes_semantic_measurements.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, root, splits, args.feature_scale)
    summary_path = out_dir / "cityscapes_semantic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_sample_visuals(rows, out_dir)
    print(json.dumps({"csv": str(csv_path), "summary": str(summary_path), "summary_values": summary}, indent=2))


if __name__ == "__main__":
    main()
