from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE_ROOT = ROOT / "dataset" / "leftImg8bit_trainvaltest" / "leftImg8bit"
DEFAULT_LABEL_ROOT = ROOT / "dataset" / "gtFine_trainvaltest" / "gtFine"
DEFAULT_OUT = ROOT / "studies" / "neural_encoder_decoder" / "results" / "smoke"
IGNORE_INDEX = 255
NUM_CLASSES = 19


# Cityscapes labelIds -> trainIds for the 19 semantic classes.
LABELID_TO_TRAINID = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for label_id, train_id in {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}.items():
    LABELID_TO_TRAINID[label_id] = train_id


@dataclass(frozen=True)
class SamplePair:
    image: Path
    label: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_pairs(image_root: Path, label_root: Path, split: str) -> list[SamplePair]:
    image_files = sorted((image_root / split).rglob("*_leftImg8bit.png"))
    pairs: list[SamplePair] = []
    for image_path in image_files:
        rel = image_path.relative_to(image_root / split)
        label_name = image_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        label_path = label_root / split / rel.parent / label_name
        if label_path.exists():
            pairs.append(SamplePair(image=image_path, label=label_path))
    return pairs


def evenly_spaced_subset(items: Sequence[SamplePair], limit: Optional[int]) -> list[SamplePair]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return list(items)
    indexes = np.linspace(0, len(items) - 1, limit, dtype=int)
    return [items[int(index)] for index in indexes]


class CityscapesSemanticDataset(Dataset):
    def __init__(
        self,
        image_root: Path,
        label_root: Path,
        split: str,
        image_size: tuple[int, int],
        sample_limit: Optional[int],
    ) -> None:
        self.pairs = evenly_spaced_subset(collect_pairs(image_root, label_root, split), sample_limit)
        if not self.pairs:
            raise FileNotFoundError(f"No Cityscapes pairs found for split={split!r}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        width, height = self.image_size
        image = Image.open(pair.image).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        label = Image.open(pair.label).resize((width, height), Image.Resampling.NEAREST)

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_arr = (image_arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        label_arr = LABELID_TO_TRAINID[np.asarray(label, dtype=np.uint8)]

        image_tensor = torch.from_numpy(image_arr.transpose(2, 0, 1)).float()
        label_tensor = torch.from_numpy(label_arr.astype(np.int64))
        return image_tensor, label_tensor


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SemanticEncoderDecoder(nn.Module):
    def __init__(self, latent_channels: int = 8, width: int = 24, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            ConvBNReLU(3, width, stride=2),
            ConvBNReLU(width, width * 2, stride=2),
            ConvBNReLU(width * 2, width * 4, stride=2),
            nn.Conv2d(width * 4, latent_channels, kernel_size=1),
        )
        self.decode_head = nn.Sequential(
            ConvBNReLU(latent_channels, width * 4),
            ConvBNReLU(width * 4, width * 2),
            ConvBNReLU(width * 2, width),
            nn.Conv2d(width, num_classes, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = nn.functional.interpolate(latent, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decode_head[0](x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decode_head[1](x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decode_head[2](x)
        x = self.decode_head[3](x)
        if x.shape[-2:] != output_size:
            x = nn.functional.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), x.shape[-2:])


class PaperLikeSemanticEncoderDecoder(nn.Module):
    def __init__(self, latent_channels: int = 20, width: int = 24, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.stem = nn.Sequential(ConvBNReLU(3, width), ResidualBlock(width))
        self.down1 = nn.Sequential(ConvBNReLU(width, width * 2, stride=2), ResidualBlock(width * 2))
        self.down2 = nn.Sequential(ConvBNReLU(width * 2, width * 4, stride=2), ResidualBlock(width * 4))
        self.down3 = nn.Sequential(ConvBNReLU(width * 4, width * 4, stride=2), ResidualBlock(width * 4))
        self.to_latent = nn.Conv2d(width * 4, latent_channels, kernel_size=1)

        self.from_latent = nn.Sequential(ConvBNReLU(latent_channels, width * 4), ResidualBlock(width * 4))
        self.up1 = nn.Sequential(ConvBNReLU(width * 4, width * 4), ResidualBlock(width * 4))
        self.up2 = nn.Sequential(ConvBNReLU(width * 4, width * 2), ResidualBlock(width * 2))
        self.up3 = nn.Sequential(ConvBNReLU(width * 2, width), ResidualBlock(width))
        self.classifier = nn.Conv2d(width, num_classes, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.to_latent(x)

    def decode(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = self.from_latent(latent)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)
        x = self.classifier(x)
        if x.shape[-2:] != output_size:
            x = nn.functional.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), x.shape[-2:])


def build_model(model_name: str, latent_channels: int, width: int) -> nn.Module:
    if model_name == "tiny":
        return SemanticEncoderDecoder(latent_channels=latent_channels, width=width)
    if model_name == "paperlite":
        return PaperLikeSemanticEncoderDecoder(latent_channels=latent_channels, width=width)
    raise ValueError(f"Unknown model {model_name!r}")


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    pred = pred[valid].view(-1)
    target = target[valid].view(-1)
    if target.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    index = target * num_classes + pred
    hist = torch.bincount(index, minlength=num_classes * num_classes)
    return hist.reshape(num_classes, num_classes).cpu()


def metrics_from_confusion(conf: torch.Tensor) -> dict[str, float]:
    conf = conf.float()
    diag = torch.diag(conf)
    denom = conf.sum(1) + conf.sum(0) - diag
    valid = denom > 0
    iou = torch.zeros_like(denom)
    iou[valid] = diag[valid] / denom[valid]
    pixel_acc = diag.sum() / conf.sum().clamp_min(1.0)
    return {
        "pixel_accuracy": float(pixel_acc.item()),
        "mean_iou": float(iou[valid].mean().item()) if bool(valid.any()) else 0.0,
        "valid_classes": int(valid.sum().item()),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def compute_class_weights(dataset: Dataset, num_classes: int, smoothing: float = 1.0) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, labels in dataset:
        valid = labels != IGNORE_INDEX
        if bool(valid.any()):
            counts += torch.bincount(labels[valid].view(-1), minlength=num_classes).double()
    freq = (counts + smoothing) / (counts.sum() + smoothing * num_classes)
    weights = 1.0 / torch.log(1.02 + freq)
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights.float()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        pred = logits.argmax(1).cpu()
        conf += confusion_matrix(pred, labels.cpu(), NUM_CLASSES)
        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    out = metrics_from_confusion(conf)
    out["loss"] = total_loss / max(total_items, 1)
    return out


@torch.no_grad()
def measure_timing(model: SemanticEncoderDecoder, sample: torch.Tensor, device: torch.device, runs: int) -> dict[str, float]:
    model.eval()
    sample = sample.to(device)
    for _ in range(3):
        latent = model.encode(sample)
        _ = model.decode(latent, sample.shape[-2:])

    encode_times: list[float] = []
    decode_times: list[float] = []
    full_times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        latent = model.encode(sample)
        encode_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        _ = model.decode(latent, sample.shape[-2:])
        decode_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        _ = model(sample)
        full_times.append(time.perf_counter() - start)

    return {
        "encode_ms_median": float(np.median(encode_times) * 1000.0),
        "decode_ms_median": float(np.median(decode_times) * 1000.0),
        "full_ms_median": float(np.median(full_times) * 1000.0),
    }


def payload_ratio(latent_channels: int, downsample_factor: int, quant_bits: int, image_size: tuple[int, int]) -> float:
    width, height = image_size
    latent_w = int(np.ceil(width / downsample_factor))
    latent_h = int(np.ceil(height / downsample_factor))
    latent_bits = latent_channels * latent_w * latent_h * quant_bits
    raw_rgb_bits = 3 * width * height * 8
    return latent_bits / raw_rgb_bits


def paper_comparison(ratio: float, timing: dict[str, float], image_size: tuple[int, int]) -> dict[str, float]:
    width, height = image_size
    raw_rgb_bits = 3 * width * height * 8
    latent_bits = raw_rgb_bits * ratio
    encode_s = timing["encode_ms_median"] / 1000.0
    decode_s = timing["decode_ms_median"] / 1000.0
    return {
        "paper_rho_c": 0.104,
        "paper_rho_r": 3.0,
        "rho_c_abs_error": abs(ratio - 0.104),
        "rho_c_rel_error_pct": abs(ratio - 0.104) / 0.104 * 100.0,
        "paper_encoding_bitrate_mbps": 91.30,
        "paper_decoding_bitrate_mbps": 23.23,
        "measured_encode_input_throughput_mbps": raw_rgb_bits / max(encode_s, 1e-12) / 1e6,
        "measured_decode_latent_throughput_mbps": latent_bits / max(decode_s, 1e-12) / 1e6,
        "measured_decode_restoration_throughput_mbps": (latent_bits * 3.0) / max(decode_s, 1e-12) / 1e6,
    }


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_airtalking_summary(path: Path, result: dict[str, object]) -> None:
    comparison = result["paper_comparison"]
    model = result["model"]
    final_metrics = result["final_metrics"]
    best_metrics = result["best_metrics"]
    ratio = float(model["payload_ratio"])
    payload = {
        "source": "trained_cityscapes_rgb_to_semantic_encoder_decoder",
        "rho_c_feature_uncompressed_mean": ratio,
        "rho_c_feature_png_mean": ratio,
        "rho_c_uncompressed_mean": ratio,
        "rho_c_png_mean": ratio,
        "rho_r_proxy": float(comparison["paper_rho_r"]),
        "encode_bitrate_mbps_median": float(comparison["measured_encode_input_throughput_mbps"]),
        "decode_bitrate_mbps_median": float(comparison["measured_decode_restoration_throughput_mbps"]),
        "feature_encode_bitrate_mbps_median": float(comparison["measured_encode_input_throughput_mbps"]),
        "feature_decode_bitrate_mbps_median": float(comparison["measured_decode_restoration_throughput_mbps"]),
        "semantic_quality_miou_final": float(final_metrics["val_mean_iou"]),
        "semantic_quality_miou_best": float(best_metrics["val_mean_iou"]),
        "pixel_accuracy_final": float(final_metrics["val_pixel_accuracy"]),
        "pixel_accuracy_best": float(best_metrics["val_pixel_accuracy"]),
        "num_samples": int(result["val_samples"]),
        "model": model,
        "timing": result["timing"],
        "paper_comparison": comparison,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight Cityscapes semantic encoder/decoder.")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--label-root", default=str(DEFAULT_LABEL_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--val-limit", type=int, default=32)
    parser.add_argument("--model", choices=["tiny", "paperlite"], default="tiny")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=260710)
    parser.add_argument("--timing-runs", type=int, default=10)
    parser.add_argument("--class-balanced-loss", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--eval-checkpoint", default=None, help="Load a saved checkpoint before training/evaluation.")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (args.image_width, args.image_height)
    train_ds = CityscapesSemanticDataset(Path(args.image_root), Path(args.label_root), "train", image_size, args.train_limit)
    val_ds = CityscapesSemanticDataset(Path(args.image_root), Path(args.label_root), "val", image_size, args.val_limit)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, args.latent_channels, args.width).to(device)
    checkpoint_payload: Optional[dict[str, object]] = None
    if args.eval_checkpoint:
        checkpoint_payload = torch.load(args.eval_checkpoint, map_location=device)
        model.load_state_dict(checkpoint_payload["model"])
    class_weights = compute_class_weights(train_ds, NUM_CLASSES).to(device) if args.class_balanced_loss else None
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
        if args.scheduler == "cosine"
        else None
    )

    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    best_row: Optional[dict[str, float | int]] = None
    if args.epochs <= 0:
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": 0,
            "train_loss": None,
            "val_loss": val_metrics["loss"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
            "valid_classes": val_metrics["valid_classes"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        best_row = dict(checkpoint_payload.get("best_metrics", row)) if checkpoint_payload else dict(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    else:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            if scheduler is not None:
                scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_pixel_accuracy": val_metrics["pixel_accuracy"],
                "val_mean_iou": val_metrics["mean_iou"],
                "valid_classes": val_metrics["valid_classes"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            if best_row is None or float(row["val_mean_iou"]) > float(best_row["val_mean_iou"]):
                best_row = dict(row)
                if args.save_checkpoint:
                    torch.save({"model": model.state_dict(), "args": vars(args), "best_metrics": best_row}, out_dir / "best_semantic_encoder_decoder.pt")
            print(json.dumps(row, ensure_ascii=False), flush=True)

    timing_batch = next(iter(val_loader))[0][:1]
    timing = measure_timing(model, timing_batch, device, args.timing_runs)
    elapsed = time.perf_counter() - started
    ratio = payload_ratio(args.latent_channels, 8, 8, image_size)
    comparison = paper_comparison(ratio, timing, image_size)
    result = {
        "experiment": "cityscapes_rgb_to_semantic_encoder_decoder",
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs": args.epochs,
        "image_size": {"width": args.image_width, "height": args.image_height},
        "model": {
            "name": args.model,
            "latent_channels": args.latent_channels,
            "width": args.width,
            "downsample_factor": 8,
            "quantization_bits_assumed": 8,
            "payload_ratio": ratio,
        },
        "training": {
            "class_balanced_loss": bool(args.class_balanced_loss),
            "scheduler": args.scheduler,
            "learning_rate": args.lr,
        },
        "final_metrics": history[-1],
        "best_metrics": best_row or history[-1],
        "timing": timing,
        "paper_comparison": comparison,
        "elapsed_seconds": elapsed,
        "notes": [
            "This is a real trainable neural encoder/decoder baseline, not the earlier label-resize proxy.",
            "Payload ratio assumes 8-bit quantized latent features. Entropy coding is not implemented in this baseline.",
            "CPU smoke runs are for pipeline validation; paper-grade metrics need longer training and preferably GPU.",
        ],
    }
    write_history(out_dir / "training_history.csv", history)
    (out_dir / "result_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_airtalking_summary(out_dir / "airtalking_semantic_summary.json", result)
    if args.save_checkpoint:
        torch.save({"model": model.state_dict(), "args": vars(args), "result": result}, out_dir / "semantic_encoder_decoder.pt")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
