from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_ROOT = WORKSPACE_ROOT / "dataset" / "gtFine_trainvaltest" / "gtFine"
DEFAULT_SINR = (
    WORKSPACE_ROOT
    / "studies"
    / "airtalking_reproduction"
    / "results"
    / "airtalking_cityscapes_calibrated_final_p012"
    / "timeseries_and_sinr_samples.npz"
)
OUT_DIR = STUDY_ROOT / "results" / "probe_outputs"
FIG_DIR = OUT_DIR / "figures"

RAW_RGB_BYTES = 2048 * 1024 * 3
PAPER_ENCODER_BPS = 91.30e6
PAPER_DECODER_BPS = 23.23e6
BANDWIDTH_HZ = 80e6
IGNORE_LABEL = 255


@dataclass(frozen=True)
class CompressionMode:
    name: str
    scale: float
    description: str


MODES = [
    CompressionMode("emergency", 0.125, "very small payload for poor links"),
    CompressionMode("low", 0.25, "small payload when latency is critical"),
    CompressionMode("medium", 0.375, "balanced compression"),
    CompressionMode("paper_like", 0.56, "close to AirTalking rho_c=0.104"),
    CompressionMode("high", 0.75, "higher semantic quality when channel is good"),
]


def find_label_files(gt_root: Path, splits: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for split in splits:
        files.extend(sorted((gt_root / split).rglob("*_gtFine_labelIds.png")))
    return files


def evenly_spaced_subset(files: list[Path], limit: int | None) -> list[Path]:
    if limit is None or limit <= 0 or len(files) <= limit:
        return files
    indices = np.linspace(0, len(files) - 1, limit, dtype=int)
    return [files[int(i)] for i in indices]


def resized_label_metrics(label: np.ndarray, scale: float) -> dict[str, float]:
    h, w = label.shape
    small_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    original = Image.fromarray(label.astype(np.uint8), mode="L")
    down = original.resize(small_size, Image.Resampling.NEAREST)
    restored = down.resize((w, h), Image.Resampling.NEAREST)
    restored_arr = np.asarray(restored, dtype=np.uint8)

    valid = label != IGNORE_LABEL
    valid_count = int(np.count_nonzero(valid))
    correct = int(np.count_nonzero((label == restored_arr) & valid))
    pixel_acc = correct / max(valid_count, 1)

    labels = np.union1d(np.unique(label[valid]), np.unique(restored_arr[valid]))
    ious: list[float] = []
    for cls in labels:
        pred = restored_arr == cls
        truth = label == cls
        intersection = int(np.count_nonzero(pred & truth & valid))
        union = int(np.count_nonzero((pred | truth) & valid))
        if union:
            ious.append(intersection / union)
    miou = float(np.mean(ious)) if ious else 0.0

    down_bytes = np.asarray(down, dtype=np.uint8).nbytes
    zlib_bytes = len(zlib.compress(np.asarray(down, dtype=np.uint8).tobytes(), level=6))
    return {
        "feature_bytes": float(down_bytes),
        "zlib_bytes": float(zlib_bytes),
        "feature_ratio": float(down_bytes / RAW_RGB_BYTES),
        "zlib_ratio": float(zlib_bytes / RAW_RGB_BYTES),
        "pixel_accuracy": float(pixel_acc),
        "mean_iou": float(miou),
    }


def aggregate_quality(label_files: list[Path]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    accum: dict[str, dict[str, list[float]]] = {
        mode.name: {
            "feature_bytes": [],
            "zlib_bytes": [],
            "feature_ratio": [],
            "zlib_ratio": [],
            "pixel_accuracy": [],
            "mean_iou": [],
        }
        for mode in MODES
    }
    for path in label_files:
        label = np.asarray(Image.open(path), dtype=np.uint8)
        for mode in MODES:
            metrics = resized_label_metrics(label, mode.scale)
            for key, value in metrics.items():
                accum[mode.name][key].append(float(value))

    for mode in MODES:
        values = accum[mode.name]
        row: dict[str, float | str] = {
            "mode": mode.name,
            "scale": mode.scale,
            "description": mode.description,
        }
        for key, series in values.items():
            arr = np.array(series, dtype=float)
            row[f"{key}_mean"] = float(np.mean(arr))
            row[f"{key}_median"] = float(np.median(arr))
        rows.append(row)
    return rows


def load_sinr_samples(path: Path) -> np.ndarray:
    if not path.exists():
        return np.linspace(-20.0, 20.0, 1000)
    data = np.load(path, allow_pickle=True)
    samples: list[np.ndarray] = []
    for key in data.files:
        if key.endswith("_sinr"):
            arr = np.asarray(data[key], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                samples.append(arr)
    if not samples:
        return np.linspace(-20.0, 20.0, 1000)
    out = np.concatenate(samples)
    return out[(out > -60.0) & (out < 80.0)]


def shannon_rate_bps(sinr_db: np.ndarray) -> np.ndarray:
    sinr_linear = np.power(10.0, sinr_db / 10.0)
    rate = BANDWIDTH_HZ * np.log2(1.0 + sinr_linear)
    return np.maximum(rate, 1e5)


def delivery_time(raw_bits: float, feature_ratio: float, rate_bps: np.ndarray) -> np.ndarray:
    semantic_bits = raw_bits * feature_ratio
    encode_time = raw_bits / PAPER_ENCODER_BPS
    decode_time = (semantic_bits * 3.0) / PAPER_DECODER_BPS
    return semantic_bits / rate_bps + encode_time + decode_time


def choose_adaptive_mode(sinr_db: float, quality_rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    if sinr_db < -15:
        target_miou = 0.80
    elif sinr_db < -10:
        target_miou = 0.88
    elif sinr_db < -5:
        target_miou = 0.925
    elif sinr_db < 0:
        target_miou = 0.95
    else:
        target_miou = 0.965

    for row in quality_rows:
        if float(row["mean_iou_mean"]) >= target_miou:
            return row
    return quality_rows[-1]


def simulate_policies(quality_rows: list[dict[str, float | str]], sinr: np.ndarray) -> dict[str, object]:
    raw_bits = RAW_RGB_BYTES * 8.0
    rate = shannon_rate_bps(sinr)
    raw_time = raw_bits / rate

    paper_like = next(row for row in quality_rows if row["mode"] == "paper_like")
    low = next(row for row in quality_rows if row["mode"] == "low")
    high = next(row for row in quality_rows if row["mode"] == "high")

    fixed_paper_time = delivery_time(raw_bits, float(paper_like["feature_ratio_mean"]), rate)
    fixed_low_time = delivery_time(raw_bits, float(low["feature_ratio_mean"]), rate)
    fixed_high_time = delivery_time(raw_bits, float(high["feature_ratio_mean"]), rate)

    chosen_modes = [choose_adaptive_mode(float(x), quality_rows) for x in sinr]
    adaptive_time = np.array(
        [delivery_time(raw_bits, float(row["feature_ratio_mean"]), np.array([r]))[0] for row, r in zip(chosen_modes, rate)],
        dtype=float,
    )
    adaptive_miou = np.array([float(row["mean_iou_mean"]) for row in chosen_modes], dtype=float)
    adaptive_ratio = np.array([float(row["feature_ratio_mean"]) for row in chosen_modes], dtype=float)

    def summary(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
        }

    mode_counts: dict[str, int] = {}
    for row in chosen_modes:
        mode_counts[str(row["mode"])] = mode_counts.get(str(row["mode"]), 0) + 1

    return {
        "sinr_count": int(sinr.size),
        "sinr_db": summary(sinr),
        "raw_time_s": summary(raw_time),
        "fixed_low_time_s": summary(fixed_low_time),
        "fixed_paper_time_s": summary(fixed_paper_time),
        "fixed_high_time_s": summary(fixed_high_time),
        "adaptive_time_s": summary(adaptive_time),
        "fixed_low_miou": float(low["mean_iou_mean"]),
        "fixed_paper_miou": float(paper_like["mean_iou_mean"]),
        "fixed_high_miou": float(high["mean_iou_mean"]),
        "adaptive_miou_mean": float(np.mean(adaptive_miou)),
        "adaptive_ratio_mean": float(np.mean(adaptive_ratio)),
        "adaptive_mode_counts": mode_counts,
        "adaptive_vs_fixed_paper_time_reduction_pct": float(
            (np.mean(fixed_paper_time) - np.mean(adaptive_time)) / np.mean(fixed_paper_time) * 100.0
        ),
        "fixed_paper_vs_raw_time_reduction_pct": float((np.mean(raw_time) - np.mean(fixed_paper_time)) / np.mean(raw_time) * 100.0),
        "adaptive_vs_raw_time_reduction_pct": float((np.mean(raw_time) - np.mean(adaptive_time)) / np.mean(raw_time) * 100.0),
    }


def write_quality_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_quality_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"mode", "description"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def _font(size: int = 14) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, x_label: str, y_label: str) -> None:
    left, top, right, bottom = box
    draw.rectangle([left, top, right, bottom], outline="#cbd5e1")
    draw.line([left, bottom, right, bottom], fill="#334155", width=2)
    draw.line([left, top, left, bottom], fill="#334155", width=2)
    draw.text((left, 18), title, fill="#0f172a", font=_font(18))
    draw.text(((left + right) // 2 - 120, bottom + 42), x_label, fill="#334155", font=_font(13))
    draw.text((16, top - 2), y_label, fill="#334155", font=_font(13))


def _scale_points(xs: list[float], ys: list[float], box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = max((y_max - y_min) * 0.08, 0.02)
    y_min -= y_pad
    y_max += y_pad
    points: list[tuple[int, int]] = []
    for x, y in zip(xs, ys):
        px = left + int((x - x_min) / max(x_max - x_min, 1e-9) * (right - left))
        py = bottom - int((y - y_min) / max(y_max - y_min, 1e-9) * (bottom - top))
        points.append((px, py))
    return points


def _save_quality_plot(rows: list[dict[str, float | str]], path: Path) -> None:
    image = Image.new("RGB", (980, 620), "white")
    draw = ImageDraw.Draw(image)
    box = (110, 80, 900, 500)
    _draw_axes(draw, box, "Compression-quality trade-off on Cityscapes labels", "payload ratio vs raw RGB", "quality")
    ratios = [float(row["feature_ratio_mean"]) for row in rows]
    ious = [float(row["mean_iou_mean"]) for row in rows]
    accs = [float(row["pixel_accuracy_mean"]) for row in rows]
    modes = [str(row["mode"]) for row in rows]
    for label, series, color in [("mean IoU", ious, "#2e74b5"), ("pixel accuracy", accs, "#1f9d55")]:
        pts = _scale_points(ratios, series, box)
        draw.line(pts, fill=color, width=3)
        for pt in pts:
            draw.ellipse([pt[0] - 5, pt[1] - 5, pt[0] + 5, pt[1] + 5], fill=color)
        draw.text((730, 105 + (0 if label == "mean IoU" else 24)), label, fill=color, font=_font(13))
    pts = _scale_points(ratios, ious, box)
    for mode, pt in zip(modes, pts):
        draw.text((pt[0] + 6, pt[1] - 14), mode, fill="#475569", font=_font(11))
    for value in [min(ratios), max(ratios)]:
        x = box[0] + int((value - min(ratios)) / max(max(ratios) - min(ratios), 1e-9) * (box[2] - box[0]))
        draw.text((x - 28, box[3] + 10), f"{value:.3f}", fill="#475569", font=_font(11))
    image.save(path)


def _save_bar_plot(title: str, labels: list[str], values: list[float], y_label: str, path: Path) -> None:
    image = Image.new("RGB", (980, 560), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = (90, 70, 920, 440)
    draw.text((left, 20), title, fill="#0f172a", font=_font(18))
    draw.line([left, bottom, right, bottom], fill="#334155", width=2)
    draw.line([left, top, left, bottom], fill="#334155", width=2)
    draw.text((16, top - 2), y_label, fill="#334155", font=_font(13))
    max_value = max(values) * 1.15 if values else 1.0
    width = (right - left) / max(len(values), 1)
    colors = ["#64748b", "#4f7cac", "#2e74b5", "#7aa6c2", "#1f9d55", "#7c3aed"]
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = int(left + idx * width + 18)
        x1 = int(left + (idx + 1) * width - 18)
        y0 = bottom - int(value / max_value * (bottom - top))
        draw.rectangle([x0, y0, x1, bottom], fill=colors[idx % len(colors)])
        draw.text((x0, y0 - 22), f"{value:.2f}", fill="#334155", font=_font(11))
        draw.text((x0, bottom + 12), label, fill="#334155", font=_font(11))
    image.save(path)


def make_figures(rows: list[dict[str, float | str]], policy: dict[str, object]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _save_quality_plot(rows, FIG_DIR / "quality_vs_payload.png")
    labels = ["raw", "fixed low", "fixed paper", "fixed high", "adaptive"]
    means = [
        policy["raw_time_s"]["mean"],
        policy["fixed_low_time_s"]["mean"],
        policy["fixed_paper_time_s"]["mean"],
        policy["fixed_high_time_s"]["mean"],
        policy["adaptive_time_s"]["mean"],
    ]
    _save_bar_plot(
        "Estimated delivery time under AirTalking SINR samples",
        labels,
        means,
        "mean delivery time (s)",
        FIG_DIR / "delivery_time_by_policy.png",
    )
    counts = policy["adaptive_mode_counts"]
    _save_bar_plot(
        "Adaptive mode usage by channel condition",
        list(counts.keys()),
        [float(v) for v in counts.values()],
        "selected links",
        FIG_DIR / "adaptive_mode_usage.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot experiment for adaptive semantic compression.")
    parser.add_argument("--gt-root", default=str(DEFAULT_GT_ROOT))
    parser.add_argument("--sinr", default=str(DEFAULT_SINR))
    parser.add_argument("--sample-limit", type=int, default=400)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--reuse-quality", action="store_true", help="Reuse an existing compression_quality.csv and only recompute policy figures.")
    args = parser.parse_args()

    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quality_csv = OUT_DIR / "compression_quality.csv"
    if args.reuse_quality and quality_csv.exists():
        quality_rows = read_quality_csv(quality_csv)
        sample_count: int | str = "reused"
    else:
        files = find_label_files(Path(args.gt_root), splits)
        subset = evenly_spaced_subset(files, args.sample_limit)
        if not subset:
            raise FileNotFoundError(f"No Cityscapes labelIds files found under {args.gt_root}")
        quality_rows = aggregate_quality(subset)
        write_quality_csv(quality_rows, quality_csv)
        sample_count = len(subset)
    sinr = load_sinr_samples(Path(args.sinr))
    policy = simulate_policies(quality_rows, sinr)
    result = {
        "dataset": "Cityscapes gtFine labelIds",
        "splits": list(splits),
        "sample_count": sample_count,
        "compression_modes": quality_rows,
        "policy_summary": policy,
    }
    (OUT_DIR / "policy_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(quality_rows, policy)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
