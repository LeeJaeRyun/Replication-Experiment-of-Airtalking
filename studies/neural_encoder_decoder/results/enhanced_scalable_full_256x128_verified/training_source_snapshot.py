from __future__ import annotations

"""Train and evaluate a scalable, paper-inspired semantic image codec.

Scientific boundary
-------------------
The AirTalking paper discloses an RGB -> modified U-Net semantic
representation -> transmission -> modified Pix2PixHD reconstruction outline.
It does not disclose the tensors, weights, complete losses, or training recipe.
Consequently this module is an independently specified follow-up experiment; it
must not be described as the paper authors' exact neural implementation.

The receiver-side decoder accepts only the quantized latent tensor.  There are
deliberately no encoder-to-decoder skip connections across the transmission
boundary.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE_ROOT = ROOT / "dataset" / "leftImg8bit_trainvaltest" / "leftImg8bit"
DEFAULT_LABEL_ROOT = ROOT / "dataset" / "gtFine_trainvaltest" / "gtFine"
DEFAULT_OUT = ROOT / "studies" / "neural_encoder_decoder" / "results" / "enhanced_semantic_codec"
DEFAULT_DATASET_MANIFEST = (
    ROOT / "studies" / "neural_encoder_decoder" / "results" / "dataset_audit_20260711" / "dataset_manifest.json"
)
IGNORE_INDEX = 255
NUM_CLASSES = 19
DEFAULT_ACTIVE_CHANNELS = (20, 40, 60, 80, 120)
PAPER_LIKE_CHANNELS = 80
LATENT_STRIDE = 16
QUANTIZATION_BITS = 8

CLASS_NAMES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

# Official Cityscapes trainId colors, in trainId order.
CITYSCAPES_PALETTE = np.asarray(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)

LABELID_TO_TRAINID = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _label_id, _train_id in {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}.items():
    LABELID_TO_TRAINID[_label_id] = _train_id


@dataclass(frozen=True)
class SamplePair:
    image: Path
    label: Path


def parse_active_channels(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError("active channels must be comma-separated integers") from exc
    else:
        parsed = tuple(int(item) for item in value)
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("active channels must contain positive integers")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("active channels must not contain duplicates")
    return tuple(sorted(parsed))


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # Older supported PyTorch releases lack warn_only.
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def collect_pairs(image_root: Path, label_root: Path, split: str) -> list[SamplePair]:
    pairs: list[SamplePair] = []
    for image_path in sorted((image_root / split).rglob("*_leftImg8bit.png")):
        relative = image_path.relative_to(image_root / split)
        label_name = image_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        label_path = label_root / split / relative.parent / label_name
        if label_path.exists():
            pairs.append(SamplePair(image=image_path, label=label_path))
    return pairs


def evenly_spaced_subset(items: Sequence[SamplePair], limit: Optional[int]) -> list[SamplePair]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return list(items)
    indexes = np.linspace(0, len(items) - 1, num=limit, dtype=np.int64)
    return [items[int(index)] for index in indexes]


class CityscapesRGBLabelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Cityscapes RGB/label pairs with paired train augmentation.

    Augmentation randomness is a pure function of seed, epoch, and sample index.
    Validation never calls a random transform, which makes repeated evaluation
    deterministic even when DataLoader workers are enabled on Windows.
    """

    def __init__(
        self,
        image_root: Path,
        label_root: Path,
        split: str,
        image_size: tuple[int, int],
        sample_limit: Optional[int] = None,
        training: bool = False,
        seed: int = 0,
        scale_max: float = 1.25,
        horizontal_flip_probability: float = 0.5,
        color_jitter: float = 0.15,
    ) -> None:
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("image_size must contain positive dimensions")
        if scale_max < 1.0:
            raise ValueError("scale_max must be at least 1.0")
        available_pairs = collect_pairs(image_root, label_root, split)
        self.available_pairs = len(available_pairs)
        self.pairs = evenly_spaced_subset(available_pairs, sample_limit)
        if not self.pairs:
            raise FileNotFoundError(
                f"No paired Cityscapes files found for split={split!r}; "
                f"checked {image_root / split} and {label_root / split}"
            )
        self.split = split
        self.image_size = image_size
        self.training = bool(training)
        self.seed = int(seed)
        self.scale_max = float(scale_max)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.color_jitter = float(color_jitter)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _paired_crop(
        image: Image.Image,
        label: Image.Image,
        target_size: tuple[int, int],
        scale: float,
        rng: random.Random,
    ) -> tuple[Image.Image, Image.Image]:
        target_width, target_height = target_size
        scaled_width = max(target_width, int(math.ceil(target_width * scale)))
        scaled_height = max(target_height, int(math.ceil(target_height * scale)))
        image = image.resize((scaled_width, scaled_height), Image.Resampling.BILINEAR)
        label = label.resize((scaled_width, scaled_height), Image.Resampling.NEAREST)
        left = rng.randint(0, scaled_width - target_width)
        top = rng.randint(0, scaled_height - target_height)
        box = (left, top, left + target_width, top + target_height)
        return image.crop(box), label.crop(box)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        with Image.open(pair.image) as source_image:
            image = source_image.convert("RGB")
        with Image.open(pair.label) as source_label:
            label = source_label.copy()

        width, height = self.image_size
        if self.training:
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index * 9_973)
            scale = rng.uniform(1.0, self.scale_max)
            image, label = self._paired_crop(image, label, (width, height), scale, rng)
            if rng.random() < self.horizontal_flip_probability:
                image = ImageOps.mirror(image)
                label = ImageOps.mirror(label)
            if self.color_jitter > 0.0:
                magnitude = self.color_jitter
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(1.0 - magnitude, 1.0 + magnitude))
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(1.0 - magnitude, 1.0 + magnitude))
                image = ImageEnhance.Color(image).enhance(rng.uniform(1.0 - magnitude, 1.0 + magnitude))
        else:
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            label = label.resize((width, height), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        label_ids = np.asarray(label, dtype=np.uint8)
        label_array = LABELID_TO_TRAINID[label_ids].astype(np.int64, copy=True)
        return (
            torch.from_numpy(image_array.transpose(2, 0, 1)).float(),
            torch.from_numpy(label_array).long(),
        )


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            ConvNormAct(channels, channels),
            ConvNormAct(channels, channels, activation=False),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.residual(inputs))


class ResidualSemanticEncoder(nn.Module):
    """A robust modified-U-Net-inspired encoder without transmitted skips."""

    def __init__(self, max_latent_channels: int = 120, base_width: int = 32) -> None:
        super().__init__()
        width = int(base_width)
        self.network = nn.Sequential(
            ConvNormAct(3, width, kernel_size=5, stride=2),
            ResidualBlock(width),
            ConvNormAct(width, width * 2, stride=2),
            ResidualBlock(width * 2),
            ConvNormAct(width * 2, width * 4, stride=2),
            ResidualBlock(width * 4),
            ConvNormAct(width * 4, width * 8, stride=2),
            ResidualBlock(width * 8),
            nn.Conv2d(width * 8, max_latent_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ResNet18SemanticEncoder(nn.Module):
    """Optional stride-16 ImageNet ResNet-18 encoder.

    Pretrained weights are opt-in because acquiring them may require network
    access.  The residual encoder above is the offline-safe default.
    """

    def __init__(self, max_latent_channels: int = 120, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("The resnet18 encoder requires a compatible torchvision installation") from exc
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.projection = nn.Sequential(nn.Conv2d(256, max_latent_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(inputs))


class EightBitSTEQuantizer(nn.Module):
    """Uniform [0, 1] uint8 fake quantizer with a straight-through gradient."""

    levels = 255

    @staticmethod
    def to_uint8(inputs: torch.Tensor) -> torch.Tensor:
        return torch.round(inputs.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

    @staticmethod
    def from_uint8(codes: torch.Tensor) -> torch.Tensor:
        if codes.dtype != torch.uint8:
            raise TypeError(f"Expected torch.uint8 codes, received {codes.dtype}")
        return codes.to(torch.float32) / 255.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dequantized = self.from_uint8(self.to_uint8(inputs))
        return inputs + (dequantized - inputs).detach()


class UpscaleResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels, out_channels)
        self.residual = ResidualBlock(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        return self.residual(self.project(inputs))


class Pix2PixHDInspiredDecoder(nn.Module):
    """Residual/upscale receiver with RGB and 19-class semantic heads."""

    def __init__(
        self,
        max_latent_channels: int = 120,
        base_width: int = 32,
        residual_blocks: int = 3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        width = int(base_width)
        self.from_latent = ConvNormAct(max_latent_channels, width * 8)
        self.bottleneck = nn.Sequential(*[ResidualBlock(width * 8) for _ in range(residual_blocks)])
        self.upscale = nn.Sequential(
            UpscaleResidualBlock(width * 8, width * 8),
            UpscaleResidualBlock(width * 8, width * 4),
            UpscaleResidualBlock(width * 4, width * 2),
            UpscaleResidualBlock(width * 2, width),
        )
        self.shared = ResidualBlock(width)
        self.rgb_head = nn.Sequential(nn.Conv2d(width, 3, kernel_size=7, padding=3), nn.Sigmoid())
        self.segmentation_head = nn.Conv2d(width, num_classes, kernel_size=1)

    def forward(self, latent: torch.Tensor, output_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.from_latent(latent)
        features = self.bottleneck(features)
        features = self.shared(self.upscale(features))
        reconstructed_rgb = self.rgb_head(features)
        segmentation_logits = self.segmentation_head(features)
        if reconstructed_rgb.shape[-2:] != output_size:
            reconstructed_rgb = F.interpolate(
                reconstructed_rgb, size=output_size, mode="bilinear", align_corners=False
            )
            segmentation_logits = F.interpolate(
                segmentation_logits, size=output_size, mode="bilinear", align_corners=False
            )
        return reconstructed_rgb, segmentation_logits


class ScalableSemanticCodec(nn.Module):
    """A single 120-channel codec supporting prefix-channel operating rates."""

    def __init__(
        self,
        max_latent_channels: int = 120,
        active_channels: Sequence[int] = DEFAULT_ACTIVE_CHANNELS,
        encoder_name: str = "residual",
        base_width: int = 32,
        decoder_residual_blocks: int = 3,
        pretrained_resnet: bool = False,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        rates = parse_active_channels(active_channels)
        if rates[-1] > max_latent_channels:
            raise ValueError("An active-channel rate exceeds max_latent_channels")
        self.max_latent_channels = int(max_latent_channels)
        self.active_channels = rates
        if encoder_name == "residual":
            self.encoder = ResidualSemanticEncoder(max_latent_channels, base_width)
        elif encoder_name == "resnet18":
            self.encoder = ResNet18SemanticEncoder(max_latent_channels, pretrained=pretrained_resnet)
        else:
            raise ValueError(f"Unknown encoder_name {encoder_name!r}")
        self.decoder = Pix2PixHDInspiredDecoder(
            max_latent_channels,
            base_width,
            decoder_residual_blocks,
            num_classes,
        )
        self.quantizer = EightBitSTEQuantizer()
        self.register_buffer("input_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def _validate_rate(self, active_channels: int) -> int:
        active = int(active_channels)
        if active not in self.active_channels:
            raise ValueError(f"active_channels={active} is not one of {self.active_channels}")
        return active

    def encode_continuous(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - self.input_mean) / self.input_std
        latent = self.encoder(normalized)
        if latent.shape[1] != self.max_latent_channels:
            raise RuntimeError("Encoder returned an unexpected latent channel count")
        return latent

    def encode_quantized_full(self, images: torch.Tensor) -> torch.Tensor:
        return self.quantizer(self.encode_continuous(images))

    def mask_latent(self, latent: torch.Tensor, active_channels: int) -> torch.Tensor:
        active = self._validate_rate(active_channels)
        if latent.shape[1] != self.max_latent_channels:
            raise ValueError("mask_latent expects the full-width latent tensor")
        mask = torch.arange(self.max_latent_channels, device=latent.device).view(1, -1, 1, 1) < active
        return latent * mask.to(dtype=latent.dtype)

    def encode(self, images: torch.Tensor, active_channels: int = PAPER_LIKE_CHANNELS) -> torch.Tensor:
        return self.mask_latent(self.encode_quantized_full(images), active_channels)

    @torch.no_grad()
    def encode_uint8(self, images: torch.Tensor, active_channels: int = PAPER_LIKE_CHANNELS) -> torch.Tensor:
        active = self._validate_rate(active_channels)
        full_codes = self.quantizer.to_uint8(self.encode_continuous(images))
        return full_codes[:, :active].contiguous()

    def decode(
        self,
        latent: torch.Tensor,
        output_size: tuple[int, int],
        active_channels: int = PAPER_LIKE_CHANNELS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = self._validate_rate(active_channels)
        if latent.shape[1] == self.max_latent_channels:
            receiver_latent = self.mask_latent(latent, active)
        elif latent.shape[1] == active:
            padding = latent.new_zeros(
                (latent.shape[0], self.max_latent_channels - active, latent.shape[2], latent.shape[3])
            )
            receiver_latent = torch.cat((latent, padding), dim=1)
        else:
            raise ValueError("Decoder input must contain active channels or the full latent width")
        return self.decoder(receiver_latent, output_size)

    def decode_uint8(
        self,
        codes: torch.Tensor,
        output_size: tuple[int, int],
        active_channels: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = self._validate_rate(active_channels)
        if codes.shape[1] != active:
            raise ValueError("Serialized code shape does not match active_channels")
        return self.decode(self.quantizer.from_uint8(codes), output_size, active)

    def forward(
        self,
        images: torch.Tensor,
        active_channels: int = PAPER_LIKE_CHANNELS,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(images, active_channels)
        reconstructed_rgb, segmentation_logits = self.decode(latent, images.shape[-2:], active_channels)
        return {
            "latent": latent,
            "reconstructed_rgb": reconstructed_rgb,
            "segmentation_logits": segmentation_logits,
        }


def theoretical_payload_ratio(
    active_channels: int,
    stride: int = LATENT_STRIDE,
    quantization_bits: int = QUANTIZATION_BITS,
) -> float:
    """Latent bytes divided by raw 8-bit RGB bytes for divisible dimensions."""

    return float(active_channels * quantization_bits) / float(3 * stride * stride * 8)


def latent_uint8_to_bytes(codes: torch.Tensor) -> bytes:
    if codes.dtype != torch.uint8:
        raise TypeError("The transport payload must originate from a torch.uint8 tensor")
    if codes.ndim != 3:
        raise ValueError("Serialize one CxHxW latent sample at a time")
    return codes.detach().to(device="cpu").contiguous().numpy().tobytes(order="C")


def latent_bytes_to_uint8(payload: bytes, shape: Sequence[int], device: torch.device) -> torch.Tensor:
    expected = int(np.prod(tuple(int(item) for item in shape), dtype=np.int64))
    if len(payload) != expected:
        raise ValueError(f"Payload length {len(payload)} does not match expected uint8 elements {expected}")
    array = np.frombuffer(payload, dtype=np.uint8).reshape(tuple(int(item) for item in shape)).copy()
    return torch.from_numpy(array).to(device=device)


def transmit_latent_batch(
    codes: torch.Tensor,
    device: torch.device,
    zlib_level: int = 6,
) -> tuple[torch.Tensor, int, int]:
    """Actually serialize, zlib-compress, decompress, and deserialize a batch."""

    received: list[torch.Tensor] = []
    raw_bytes = 0
    compressed_bytes = 0
    for sample in codes:
        payload = latent_uint8_to_bytes(sample)
        compressed = zlib.compress(payload, level=zlib_level)
        restored_payload = zlib.decompress(compressed)
        if restored_payload != payload:
            raise RuntimeError("zlib latent payload failed a byte-exact round trip")
        received.append(latent_bytes_to_uint8(restored_payload, sample.shape, device))
        raw_bytes += len(payload)
        compressed_bytes += len(compressed)
    return torch.stack(received, dim=0), raw_bytes, compressed_bytes


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    valid = target != ignore_index
    valid_float = valid.unsqueeze(1).to(dtype=probabilities.dtype)
    safe_target = target.masked_fill(~valid, 0)
    expected = F.one_hot(safe_target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    expected = expected.to(dtype=probabilities.dtype) * valid_float
    predicted = probabilities * valid_float
    reduce_dims = (0, 2, 3)
    intersection = (predicted * expected).sum(dim=reduce_dims)
    denominator = predicted.sum(dim=reduce_dims) + expected.sum(dim=reduce_dims)
    # When the whole crop is ignore-only, every smoothed class Dice is one and
    # the returned loss is exactly zero without a device-to-host synchronization.
    class_dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - class_dice.mean()


def ssim_score(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("SSIM inputs must have identical shapes")
    smallest_side = min(int(prediction.shape[-2]), int(prediction.shape[-1]))
    window = min(int(window_size), smallest_side)
    if window % 2 == 0:
        window = max(1, window - 1)
    padding = window // 2
    mean_x = F.avg_pool2d(prediction, window, stride=1, padding=padding)
    mean_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    variance_x = F.avg_pool2d(prediction * prediction, window, 1, padding) - mean_x.square()
    variance_y = F.avg_pool2d(target * target, window, 1, padding) - mean_y.square()
    covariance = F.avg_pool2d(prediction * target, window, 1, padding) - mean_x * mean_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean().clamp(-1.0, 1.0)


class CompositeSemanticCodecLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        rgb_l1_weight: float = 1.0,
        rgb_ssim_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.detach().clone())
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.rgb_l1_weight = float(rgb_l1_weight)
        self.rgb_ssim_weight = float(rgb_ssim_weight)

    def forward(
        self,
        reconstructed_rgb: torch.Tensor,
        segmentation_logits: torch.Tensor,
        target_rgb: torch.Tensor,
        target_segmentation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ce_sum = F.cross_entropy(
            segmentation_logits,
            target_segmentation,
            weight=self.class_weights,
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        valid = target_segmentation != IGNORE_INDEX
        if self.class_weights is None:
            ce_denominator = valid.sum().to(dtype=segmentation_logits.dtype)
        else:
            safe_target = target_segmentation.masked_fill(~valid, 0)
            pixel_weights = self.class_weights[safe_target] * valid.to(dtype=self.class_weights.dtype)
            ce_denominator = pixel_weights.sum()
        ce = ce_sum / ce_denominator.clamp_min(1.0)
        dice = soft_dice_loss(segmentation_logits, target_segmentation)
        rgb_l1 = F.l1_loss(reconstructed_rgb, target_rgb)
        rgb_ssim = 1.0 - ssim_score(reconstructed_rgb, target_rgb)
        total = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.rgb_l1_weight * rgb_l1
            + self.rgb_ssim_weight * rgb_ssim
        )
        return total, {"ce": ce, "dice": dice, "rgb_l1": rgb_l1, "rgb_ssim": rgb_ssim}


def confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    prediction = prediction[valid].reshape(-1).to(torch.int64)
    target = target[valid].reshape(-1).to(torch.int64)
    if target.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    encoded = target * num_classes + prediction
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes).cpu()


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, Any]:
    values = matrix.to(torch.float64)
    true_positive = torch.diag(values)
    union = values.sum(dim=1) + values.sum(dim=0) - true_positive
    valid = union > 0
    iou = torch.zeros_like(union)
    iou[valid] = true_positive[valid] / union[valid]
    pixel_accuracy = true_positive.sum() / values.sum().clamp_min(1.0)
    per_class: list[Optional[float]] = [float(iou[index].item()) if bool(valid[index]) else None for index in range(len(iou))]
    return {
        "pixel_accuracy": float(pixel_accuracy.item()),
        "mean_iou": float(iou[valid].mean().item()) if bool(valid.any()) else 0.0,
        "valid_classes": int(valid.sum().item()),
        "per_class_iou": per_class,
    }


def compute_class_weights(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    num_classes: int = NUM_CLASSES,
    clip_min: float = 0.25,
    clip_max: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clip_min <= 0.0 or clip_max < clip_min:
        raise ValueError("Invalid class-weight clipping interval")
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, labels in dataset:
        valid = labels != IGNORE_INDEX
        if bool(valid.any()):
            counts += torch.bincount(labels[valid].reshape(-1), minlength=num_classes).to(torch.float64)
    frequency = (counts + 1.0) / (counts.sum() + num_classes)
    weights = 1.0 / torch.log(1.02 + frequency)
    weights = weights / weights.mean().clamp_min(1e-12)
    weights = weights.clamp(min=float(clip_min), max=float(clip_max))
    return weights.to(torch.float32), counts


def rgb_uint8_to_bytes(image: torch.Tensor) -> bytes:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Serialize one RGB tensor in CxHxW form")
    codes = torch.round(image.detach().clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    array = codes.to(device="cpu").permute(1, 2, 0).contiguous().numpy()
    return array.tobytes(order="C")


def _empty_quality_accumulator() -> dict[str, Any]:
    return {
        "loss": 0.0,
        "ce": 0.0,
        "dice": 0.0,
        "rgb_l1_component": 0.0,
        "rgb_ssim_loss": 0.0,
        "ssim": 0.0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
        "rgb_elements": 0,
        "items": 0,
        "confusion": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64),
        "latent_raw_bytes": 0,
        "latent_zlib_bytes": 0,
        "raw_rgb_bytes": 0,
    }


def _update_quality_accumulator(
    accumulator: dict[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    reconstructed: torch.Tensor,
    logits: torch.Tensor,
    loss_function: CompositeSemanticCodecLoss,
) -> None:
    loss, components = loss_function(reconstructed, logits, images, labels)
    batch_size = int(images.shape[0])
    difference = reconstructed - images
    accumulator["loss"] += float(loss.item()) * batch_size
    accumulator["ce"] += float(components["ce"].item()) * batch_size
    accumulator["dice"] += float(components["dice"].item()) * batch_size
    accumulator["rgb_l1_component"] += float(components["rgb_l1"].item()) * batch_size
    accumulator["rgb_ssim_loss"] += float(components["rgb_ssim"].item()) * batch_size
    accumulator["ssim"] += float(ssim_score(reconstructed, images).item()) * batch_size
    accumulator["squared_error"] += float(difference.square().sum().item())
    accumulator["absolute_error"] += float(difference.abs().sum().item())
    accumulator["rgb_elements"] += int(difference.numel())
    accumulator["items"] += batch_size
    prediction = logits.argmax(dim=1).detach().cpu()
    accumulator["confusion"] += confusion_matrix(prediction, labels.detach().cpu())


def _finish_quality_accumulator(accumulator: dict[str, Any], active_channels: int) -> dict[str, Any]:
    items = max(int(accumulator["items"]), 1)
    elements = max(int(accumulator["rgb_elements"]), 1)
    mse = float(accumulator["squared_error"]) / elements
    metrics = metrics_from_confusion(accumulator["confusion"])
    raw_rgb_bytes = int(accumulator["raw_rgb_bytes"])
    raw_payload = int(accumulator["latent_raw_bytes"])
    zlib_payload = int(accumulator["latent_zlib_bytes"])
    profile: dict[str, Any] = {
        "active_channels": int(active_channels),
        "operating_point": "paper_like" if active_channels == PAPER_LIKE_CHANNELS else f"rate_{active_channels}",
        "theoretical_rho_raw_rgb": theoretical_payload_ratio(active_channels),
        "loss": float(accumulator["loss"]) / items,
        "cross_entropy": float(accumulator["ce"]) / items,
        "dice_loss": float(accumulator["dice"]) / items,
        "rgb_l1_loss_component": float(accumulator["rgb_l1_component"]) / items,
        "rgb_ssim_loss": float(accumulator["rgb_ssim_loss"]) / items,
        "rgb_l1": float(accumulator["absolute_error"]) / elements,
        "rgb_mse": mse,
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-10))),
        "ssim": float(accumulator["ssim"]) / items,
        "pixel_accuracy": metrics["pixel_accuracy"],
        "mean_iou": metrics["mean_iou"],
        "valid_classes": metrics["valid_classes"],
        "per_class_iou": metrics["per_class_iou"],
        "evaluated_samples": int(accumulator["items"]),
        "raw_rgb_bytes": raw_rgb_bytes,
        "latent_uint8_bytes": raw_payload,
        "latent_zlib_bytes": zlib_payload,
        "measured_rho_uint8_over_raw_rgb": raw_payload / max(raw_rgb_bytes, 1),
        "measured_rho_zlib_over_raw_rgb": zlib_payload / max(raw_rgb_bytes, 1),
        "zlib_savings_fraction": 1.0 - zlib_payload / max(raw_payload, 1),
    }
    return profile


@torch.no_grad()
def evaluate_rate(
    model: ScalableSemanticCodec,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: CompositeSemanticCodecLoss,
    device: torch.device,
    active_channels: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    model.eval()
    accumulator = _empty_quality_accumulator()
    for images, labels in loader:
        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        output = model(images, active_channels)
        _update_quality_accumulator(
            accumulator,
            images,
            labels,
            output["reconstructed_rgb"],
            output["segmentation_logits"],
            loss_function,
        )
    return _finish_quality_accumulator(accumulator, active_channels), accumulator["confusion"].clone()


@torch.no_grad()
def evaluate_multi_rate_transport(
    model: ScalableSemanticCodec,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: CompositeSemanticCodecLoss,
    device: torch.device,
    active_channels: Sequence[int],
    zlib_level: int,
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor]]:
    """Evaluate quality after a real byte and zlib transmission round trip."""

    model.eval()
    rates = tuple(int(rate) for rate in active_channels)
    accumulators = {rate: _empty_quality_accumulator() for rate in rates}
    for images, labels in loader:
        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        full_codes = model.quantizer.to_uint8(model.encode_continuous(images))
        raw_rgb_payloads = [rgb_uint8_to_bytes(sample) for sample in images]
        raw_rgb_bytes = sum(len(payload) for payload in raw_rgb_payloads)
        for rate in rates:
            transmitted_codes, payload_bytes, compressed_bytes = transmit_latent_batch(
                full_codes[:, :rate].contiguous(), device=device, zlib_level=zlib_level
            )
            reconstructed, logits = model.decode_uint8(transmitted_codes, images.shape[-2:], rate)
            accumulator = accumulators[rate]
            accumulator["latent_raw_bytes"] += payload_bytes
            accumulator["latent_zlib_bytes"] += compressed_bytes
            accumulator["raw_rgb_bytes"] += raw_rgb_bytes
            _update_quality_accumulator(
                accumulator,
                images,
                labels,
                reconstructed,
                logits,
                loss_function,
            )
    profiles = [_finish_quality_accumulator(accumulators[rate], rate) for rate in rates]
    matrices = {rate: accumulators[rate]["confusion"].clone() for rate in rates}
    return profiles, matrices


def _make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: ScalableSemanticCodec,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loss_function: CompositeSemanticCodecLoss,
    device: torch.device,
    active_channels: Sequence[int],
    rate_rng: random.Random,
    accumulation_steps: int,
    gradient_clip_norm: float,
    amp_enabled: bool,
    global_step: int,
) -> tuple[dict[str, Any], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "rgb_l1": 0.0, "rgb_ssim": 0.0}
    sample_count = 0
    rate_counts = {int(rate): 0 for rate in active_channels}
    maximum_rate = max(int(rate) for rate in active_channels)
    minimum_rate = min(int(rate) for rate in active_channels)
    middle_rates = [int(rate) for rate in active_channels if int(rate) not in (minimum_rate, maximum_rate)]
    accumulation_steps = max(1, int(accumulation_steps))

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        random_rate = rate_rng.choice(middle_rates) if middle_rates else minimum_rate
        # Standard sandwich-style coverage: both extremes every batch, plus a
        # seeded intermediate operating point when one exists.
        sandwich_rates = tuple(dict.fromkeys((maximum_rate, minimum_rate, random_rate)))
        for rate in sandwich_rates:
            rate_counts[rate] += int(images.shape[0])

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            quantized_full = model.encode_quantized_full(images)
            pass_losses: list[torch.Tensor] = []
            pass_components: list[dict[str, torch.Tensor]] = []
            for rate in sandwich_rates:
                reconstructed, logits = model.decode(quantized_full, images.shape[-2:], rate)
                loss, components = loss_function(reconstructed, logits, images, labels)
                pass_losses.append(loss)
                pass_components.append(components)
            combined_loss = torch.stack(pass_losses).mean()

        if not bool(torch.isfinite(combined_loss).detach()):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}")
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        actual_group_size = min(accumulation_steps, len(loader) - group_start)
        scaler.scale(combined_loss / max(actual_group_size, 1)).backward()
        is_last_batch = batch_index + 1 == len(loader)
        should_step = (batch_index + 1) % accumulation_steps == 0 or is_last_batch
        if should_step:
            scaler.unscale_(optimizer)
            if gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        batch_size = int(images.shape[0])
        component_names = ("ce", "dice", "rgb_l1", "rgb_ssim")
        component_means = [
            torch.stack([components[name] for components in pass_components]).mean().detach()
            for name in component_names
        ]
        logged_values = torch.stack([combined_loss.detach(), *component_means]).float().cpu().tolist()
        totals["loss"] += float(logged_values[0]) * batch_size
        for name, value in zip(component_names, logged_values[1:]):
            totals[name] += float(value) * batch_size
        sample_count += batch_size

    denominator = max(sample_count, 1)
    result: dict[str, Any] = {name: value / denominator for name, value in totals.items()}
    result["samples"] = sample_count
    result["rate_sample_counts"] = {str(rate): count for rate, count in rate_counts.items()}
    return result, global_step


def capture_rng_state(rate_rng: random.Random, train_generator: torch.Generator) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "rate_sampler": rate_rng.getstate(),
        "train_loader_generator": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: Mapping[str, Any],
    rate_rng: random.Random,
    train_generator: torch.Generator,
) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "rate_sampler" in state:
        rate_rng.setstate(state["rate_sampler"])
    if "train_loader_generator" in state:
        train_generator.set_state(state["train_loader_generator"])


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def make_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Any,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_mean_iou: float,
    best_epoch: int,
    best_model_state: Mapping[str, torch.Tensor],
    history: Sequence[Mapping[str, Any]],
    rate_rng: random.Random,
    train_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "checkpoint_schema": "enhanced_semantic_codec_v1",
        # state_dict() tensors share live module storage. Clone them so later
        # load_state_dict calls cannot mutate an in-memory checkpoint candidate.
        "model": cpu_state_dict(model),
        "resume_supported": True,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "args": vars(args).copy(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_mean_iou": float(best_mean_iou),
        "best_epoch": int(best_epoch),
        "best_model": dict(best_model_state),
        "history": [dict(row) for row in history],
        "rng": capture_rng_state(rate_rng, train_generator),
    }


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid checkpoint: {path}")
    return payload


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write a checkpoint atomically so an interrupted save cannot corrupt resume state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


RESUME_CONFIG_KEYS = (
    "epochs",
    "image_root",
    "label_root",
    "dataset_manifest",
    "verify_dataset_content",
    "dataset_fingerprint",
    "training_source_sha256",
    "train_limit",
    "val_limit",
    "full_data",
    "image_width",
    "image_height",
    "batch_size",
    "encoder",
    "pretrained_resnet",
    "base_width",
    "decoder_residual_blocks",
    "max_latent_channels",
    "active_channels",
    "paper_like_channels",
    "learning_rate",
    "weight_decay",
    "minimum_lr_ratio",
    "gradient_accumulation",
    "gradient_clip_norm",
    "amp",
    "class_balanced_loss",
    "class_weight_clip_min",
    "class_weight_clip_max",
    "ce_weight",
    "dice_weight",
    "rgb_l1_weight",
    "rgb_ssim_weight",
    "train_scale_max",
    "horizontal_flip_probability",
    "color_jitter",
    "seed",
    "deterministic",
    "device",
)


def validate_resume_configuration(
    checkpoint_args: Mapping[str, Any],
    args: argparse.Namespace,
    allow_missing_keys: Sequence[str] = (),
) -> None:
    current = vars(args)
    allowed_missing = set(allow_missing_keys)
    mismatches: list[str] = []
    for key in RESUME_CONFIG_KEYS:
        if key not in checkpoint_args or key not in current:
            if key in allowed_missing:
                continue
            mismatches.append(f"{key}: missing from checkpoint/current arguments")
            continue
        saved_value = checkpoint_args[key]
        current_value = current[key]
        if isinstance(saved_value, (list, tuple)) or isinstance(current_value, (list, tuple)):
            saved_value = tuple(saved_value)
            current_value = tuple(current_value)
        if saved_value != current_value:
            mismatches.append(f"{key}: checkpoint={saved_value!r}, current={current_value!r}")
    if mismatches:
        detail = "\n  - ".join(mismatches)
        raise ValueError(f"Resume configuration mismatch:\n  - {detail}")


def verify_dataset_audit_manifest(
    manifest_path: Path,
    image_root: Path,
    label_root: Path,
    verify_content: bool,
) -> dict[str, Any]:
    """Verify manifest roots/inventory and optionally all recorded content hashes."""
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_status = manifest.get("status")
    fingerprint = manifest.get("fingerprint")
    dataset_info = manifest.get("dataset")
    if not isinstance(audit_status, Mapping) or audit_status.get("strict_pass") is not True:
        raise ValueError(f"Dataset audit manifest did not pass strict validation: {manifest_path}")
    if not isinstance(fingerprint, Mapping) or not fingerprint.get("digest"):
        raise ValueError(f"Dataset audit manifest has no fingerprint digest: {manifest_path}")
    if not isinstance(dataset_info, Mapping):
        raise ValueError(f"Dataset audit manifest has no dataset roots: {manifest_path}")
    dataset_root = Path(str(dataset_info.get("dataset_root", ""))).resolve()
    manifest_image_root = Path(str(dataset_info.get("left_img8bit_root", ""))).resolve()
    manifest_label_root = Path(str(dataset_info.get("gt_fine_root", ""))).resolve()
    if manifest_image_root != image_root.resolve() or manifest_label_root != label_root.resolve():
        raise ValueError(
            "Dataset manifest roots do not match the training roots: "
            f"manifest image/label={manifest_image_root}, {manifest_label_root}; "
            f"training image/label={image_root.resolve()}, {label_root.resolve()}"
        )
    inventory = fingerprint.get("inventory")
    if not isinstance(inventory, list):
        raise ValueError(f"Dataset fingerprint inventory is missing: {manifest_path}")
    current_files = {
        path.relative_to(dataset_root).as_posix(): path
        for path in dataset_root.rglob("*")
        if path.is_file()
    }
    expected_paths = {str(record.get("relative_path")) for record in inventory if isinstance(record, Mapping)}
    if set(current_files) != expected_paths:
        missing = sorted(expected_paths - set(current_files))[:5]
        extra = sorted(set(current_files) - expected_paths)[:5]
        raise ValueError(f"Dataset inventory changed since audit; missing={missing}, extra={extra}")

    canonical = hashlib.sha256()
    canonical.update(b"cityscapes-audit-fingerprint-v1\0")
    content_checked = 0
    for record in sorted(inventory, key=lambda item: str(item["relative_path"])):
        relative_path = str(record["relative_path"])
        path = current_files[relative_path]
        expected_size = int(record["size_bytes"])
        if path.stat().st_size != expected_size:
            raise ValueError(f"Dataset file size changed since audit: {relative_path}")
        expected_hash = record.get("content_sha256")
        actual_hash: Optional[str] = None
        if expected_hash is not None:
            if verify_content:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                actual_hash = digest.hexdigest()
                if actual_hash != str(expected_hash):
                    raise ValueError(f"Dataset content hash changed since audit: {relative_path}")
                content_checked += 1
            else:
                actual_hash = str(expected_hash)
        canonical_record = json.dumps(
            [relative_path, expected_size, actual_hash],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        canonical.update(canonical_record)
        canonical.update(b"\n")
    calculated_digest = canonical.hexdigest()
    expected_digest = str(fingerprint["digest"])
    if verify_content and calculated_digest != expected_digest:
        raise ValueError(
            f"Dataset fingerprint mismatch: manifest={expected_digest}, current={calculated_digest}"
        )
    return {
        "fingerprint": expected_digest,
        "fingerprint_algorithm": str(fingerprint.get("algorithm", "unknown")),
        "fingerprint_content_hash_policy": fingerprint.get("content_hash_policy"),
        "fingerprint_inventory_file_count": len(inventory),
        "fingerprint_content_hashed_file_count": fingerprint.get("content_hashed_file_count"),
        "fingerprint_current_inventory_verified": True,
        "fingerprint_current_content_verified": bool(verify_content),
        "fingerprint_content_files_rehashed": content_checked,
        "fingerprint_verification_seconds": time.perf_counter() - started,
        "audit_manifest": str(manifest_path),
        "audit_strict_pass": True,
    }


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_timing(
    model: ScalableSemanticCodec,
    sample: torch.Tensor,
    device: torch.device,
    active_channels: int,
    runs: int,
) -> dict[str, Any]:
    model.eval()
    sample = sample.to(device)
    runs = max(1, int(runs))
    for _ in range(2):
        latent = model.encode(sample, active_channels)
        model.decode(latent, sample.shape[-2:], active_channels)
    synchronize_if_cuda(device)

    encode_times: list[float] = []
    decode_times: list[float] = []
    full_times: list[float] = []
    for _ in range(runs):
        synchronize_if_cuda(device)
        started = time.perf_counter()
        latent = model.encode(sample, active_channels)
        synchronize_if_cuda(device)
        encode_times.append(time.perf_counter() - started)

        synchronize_if_cuda(device)
        started = time.perf_counter()
        model.decode(latent, sample.shape[-2:], active_channels)
        synchronize_if_cuda(device)
        decode_times.append(time.perf_counter() - started)

        synchronize_if_cuda(device)
        started = time.perf_counter()
        model(sample, active_channels)
        synchronize_if_cuda(device)
        full_times.append(time.perf_counter() - started)

    def distribution(values: Sequence[float]) -> dict[str, float]:
        milliseconds = np.asarray(values, dtype=np.float64) * 1000.0
        return {
            "median_ms": float(np.median(milliseconds)),
            "mean_ms": float(np.mean(milliseconds)),
            "min_ms": float(np.min(milliseconds)),
            "max_ms": float(np.max(milliseconds)),
        }

    return {
        "active_channels": int(active_channels),
        "runs": runs,
        "batch_size": int(sample.shape[0]),
        "cuda_synchronized": device.type == "cuda",
        "encode_including_8bit_fake_quantization": distribution(encode_times),
        "decode_from_latent_only": distribution(decode_times),
        "full_forward": distribution(full_times),
        "excludes_cpu_zlib_transport": True,
    }


def colorize_segmentation(label: np.ndarray) -> np.ndarray:
    output = np.zeros((*label.shape, 3), dtype=np.uint8)
    valid = (label >= 0) & (label < NUM_CLASSES)
    output[valid] = CITYSCAPES_PALETTE[label[valid].astype(np.int64)]
    return output


@torch.no_grad()
def save_qualitative_panel(
    path: Path,
    model: ScalableSemanticCodec,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    active_channels: int,
    sample_count: int,
    zlib_level: int,
) -> int:
    model.eval()
    panels: list[tuple[Image.Image, Image.Image, Image.Image, Image.Image]] = []
    for images, labels in loader:
        images_device = images.to(device)
        codes = model.encode_uint8(images_device, active_channels)
        received, _, _ = transmit_latent_batch(codes, device, zlib_level)
        reconstructed, logits = model.decode_uint8(received, images.shape[-2:], active_channels)
        predictions = logits.argmax(dim=1).cpu().numpy()
        reconstructed_array = (
            reconstructed.detach().cpu().clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy() * 255.0
        ).round().astype(np.uint8)
        original_array = (images.clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
        label_array = labels.numpy()
        for index in range(images.shape[0]):
            panels.append(
                (
                    Image.fromarray(original_array[index], mode="RGB"),
                    Image.fromarray(reconstructed_array[index], mode="RGB"),
                    Image.fromarray(colorize_segmentation(label_array[index]), mode="RGB"),
                    Image.fromarray(colorize_segmentation(predictions[index]), mode="RGB"),
                )
            )
            if len(panels) >= sample_count:
                break
        if len(panels) >= sample_count:
            break
    if not panels:
        return 0

    width, height = panels[0][0].size
    header_height = 22
    canvas = Image.new("RGB", (width * 4, (height + header_height) * len(panels)), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    titles = ("input RGB", "reconstructed RGB", "ground truth", "prediction")
    for row, images_row in enumerate(panels):
        top = row * (height + header_height)
        for column, panel in enumerate(images_row):
            left = column * width
            draw.text((left + 4, top + 4), titles[column], fill=(0, 0, 0))
            canvas.paste(panel, (left, top + header_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return len(panels)


def assert_strict_json(value: Any, location: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float at {location}: {value!r}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {location} is not a string: {key!r}")
            assert_strict_json(child, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_strict_json(child, f"{location}[{index}]")
        return
    raise TypeError(f"Unsupported strict JSON value at {location}: {type(value).__name__}")


def write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_strict_json(payload)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    # Parsing our own output catches accidental encoder extensions or truncation.
    json.loads(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + "\n", encoding="utf-8")


def write_dict_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "epoch",
        "train_loss",
        "train_ce",
        "train_dice",
        "train_rgb_l1",
        "train_rgb_ssim",
        "val_loss",
        "val_mean_iou",
        "val_pixel_accuracy",
        "val_psnr_db",
        "val_ssim",
        "learning_rate",
        "global_step",
        "sampled_rate_counts_json",
    )
    write_dict_rows(path, history, fields)


def write_rate_quality(path: Path, profiles: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "active_channels",
        "operating_point",
        "theoretical_rho_raw_rgb",
        "measured_rho_uint8_over_raw_rgb",
        "measured_rho_zlib_over_raw_rgb",
        "latent_uint8_bytes",
        "latent_zlib_bytes",
        "raw_rgb_bytes",
        "mean_iou",
        "pixel_accuracy",
        "psnr_db",
        "ssim",
        "rgb_l1",
        "loss",
        "evaluated_samples",
    )
    write_dict_rows(path, profiles, fields)


def write_per_class_iou(path: Path, profile: Mapping[str, Any]) -> None:
    values = profile["per_class_iou"]
    rows = [
        {"train_id": index, "class_name": CLASS_NAMES[index], "iou": values[index]}
        for index in range(NUM_CLASSES)
    ]
    write_dict_rows(path, rows, ("train_id", "class_name", "iou"))


def write_confusion_matrix(path: Path, matrix: torch.Tensor) -> None:
    fields = ("ground_truth",) + tuple(f"pred_{name.replace(' ', '_')}" for name in CLASS_NAMES)
    rows: list[dict[str, Any]] = []
    values = matrix.to(torch.int64).tolist()
    for row_index, class_name in enumerate(CLASS_NAMES):
        row: dict[str, Any] = {"ground_truth": class_name}
        for column_index, field in enumerate(fields[1:]):
            row[field] = int(values[row_index][column_index])
        rows.append(row)
    write_dict_rows(path, rows, fields)


def collect_environment(device: torch.device) -> dict[str, Any]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            torchvision_version: Optional[str] = version("torchvision")
        except PackageNotFoundError:
            torchvision_version = None
    except ImportError:
        torchvision_version = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version,
        "numpy_version": str(np.__version__),
        "pillow_version": str(Image.__version__),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "git_commit": git_commit,
        "working_directory": str(Path.cwd()),
        "argv": list(sys.argv),
        "command_windows": subprocess.list2cmdline(sys.argv),
    }


def write_airtalking_summary(path: Path, result: Mapping[str, Any]) -> None:
    profiles = list(result["multi_rate_profiles"])
    paper_profile = next(profile for profile in profiles if profile["active_channels"] == PAPER_LIKE_CHANNELS)
    timing = result["timing"]
    encode_ms = float(timing["encode_including_8bit_fake_quantization"]["median_ms"])
    decode_ms = float(timing["decode_from_latent_only"]["median_ms"])
    image_size = result["dataset"]["image_size"]
    raw_bits = int(image_size["width"]) * int(image_size["height"]) * 3 * 8
    rho_uint8 = float(paper_profile["measured_rho_uint8_over_raw_rgb"])
    rho_zlib = float(paper_profile["measured_rho_zlib_over_raw_rgb"])
    rho_r_proxy = 3.0
    feature_encode_mbps = raw_bits / max(encode_ms / 1000.0, 1e-12) / 1e6
    feature_decode_mbps = (
        raw_bits * rho_uint8 * rho_r_proxy / max(decode_ms / 1000.0, 1e-12) / 1e6
    )
    payload = {
        "schema_version": 2,
        "source": "enhanced_scalable_cityscapes_codec_paper_inspired_follow_up",
        "scientific_scope": result["scientific_scope"],
        "paper_like_active_channels": PAPER_LIKE_CHANNELS,
        "rho_c_feature_uncompressed_mean": rho_uint8,
        "rho_c_feature_zlib_mean": rho_zlib,
        "rho_c_uncompressed_mean": rho_zlib,
        "rho_c_zlib_mean": rho_zlib,
        "png_denominator_fields_omitted": True,
        "png_denominator_fields_note": "Input PNG byte sizes were not measured; PNG-ratio aliases are intentionally absent.",
        "transport_encoding": "contiguous_uint8_latent; optional zlib level recorded in result_summary",
        "rho_r_proxy": rho_r_proxy,
        "semantic_quality_miou_best": paper_profile["mean_iou"],
        "semantic_quality_miou_final": paper_profile["mean_iou"],
        "pixel_accuracy_best": paper_profile["pixel_accuracy"],
        "pixel_accuracy_final": paper_profile["pixel_accuracy"],
        "rgb_reconstruction_psnr_db": paper_profile["psnr_db"],
        "rgb_reconstruction_ssim": paper_profile["ssim"],
        "feature_encode_bitrate_mbps_median": feature_encode_mbps,
        "feature_decode_bitrate_mbps_median": feature_decode_mbps,
        "feature_bitrate_contract": "encode=raw_RGB_bits/encoder_time; decode=latent_uint8_bits*rho_r_proxy/decoder_time",
        "zlib_bitrate_fields_omitted": True,
        "zlib_bitrate_fields_note": "GPU timing excludes D2H/H2D, serialization, zlib compression, and decompression.",
        "num_samples": result["dataset"]["val_samples"],
        "multi_rate_profiles": [
            {
                "active_channels": profile["active_channels"],
                "operating_point": profile["operating_point"],
                "rho_uint8": profile["measured_rho_uint8_over_raw_rgb"],
                "rho_zlib": profile["measured_rho_zlib_over_raw_rgb"],
                "mean_iou": profile["mean_iou"],
                "pixel_accuracy": profile["pixel_accuracy"],
                "psnr_db": profile["psnr_db"],
                "ssim": profile["ssim"],
            }
            for profile in profiles
        ],
        "timing": timing,
    }
    write_strict_json(path, payload)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a scalable Cityscapes RGB semantic codec. This is a paper-inspired follow-up, "
            "not an exact reconstruction of undisclosed AirTalking neural-network details."
        )
    )
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--label-root", default=str(DEFAULT_LABEL_ROOT))
    parser.add_argument(
        "--dataset-manifest",
        default=str(DEFAULT_DATASET_MANIFEST),
        help="Strict Cityscapes audit manifest whose fingerprint is attached to result provenance.",
    )
    parser.add_argument(
        "--verify-dataset-content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-hash every RGB and labelIds file recorded by the manifest before training/evaluation.",
    )
    parser.add_argument("--out", default=None, help="Output directory (a smoke-specific default is used with --smoke).")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means every paired training image.")
    parser.add_argument("--val-limit", type=int, default=0, help="0 means every paired validation image.")
    parser.add_argument("--full-data", action="store_true", help="Force both sample limits to zero.")
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--encoder", choices=("residual", "resnet18"), default="residual")
    parser.add_argument("--pretrained-resnet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--decoder-residual-blocks", type=int, default=3)
    parser.add_argument("--max-latent-channels", type=int, default=120)
    parser.add_argument(
        "--active-channels",
        type=parse_active_channels,
        default=DEFAULT_ACTIVE_CHANNELS,
        help="Comma-separated scalable prefix widths; default: 20,40,60,80,120.",
    )
    parser.add_argument("--paper-like-channels", type=int, default=PAPER_LIKE_CHANNELS)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.02)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-balanced-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weight-clip-min", type=float, default=0.25)
    parser.add_argument("--class-weight-clip-max", type=float, default=4.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--rgb-l1-weight", type=float, default=1.0)
    parser.add_argument("--rgb-ssim-weight", type=float, default=0.25)

    parser.add_argument("--train-scale-max", type=float, default=1.25)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--color-jitter", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=260711)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default=None, help="Resume all train and RNG state from this checkpoint.")
    parser.add_argument(
        "--allow-legacy-resume-provenance",
        action="store_true",
        help="Explicitly allow a pre-v2 checkpoint missing dataset fingerprint arguments; recorded in metadata.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--timing-runs", type=int, default=20)
    parser.add_argument("--zlib-level", type=int, choices=range(0, 10), default=6)
    parser.add_argument("--qualitative-samples", type=int, default=4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny one-epoch 64x32 CPU/GPU pipeline validation while retaining all five rates.",
    )
    parser.set_defaults(dataset_fingerprint=None, training_source_sha256=None)
    return parser


def apply_smoke_configuration(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.epochs = 1
    args.train_limit = 2
    args.val_limit = 1
    args.image_width = 64
    args.image_height = 32
    args.batch_size = 1
    args.num_workers = 0
    args.base_width = min(int(args.base_width), 8)
    args.decoder_residual_blocks = 1
    args.gradient_accumulation = 1
    args.timing_runs = 1
    args.qualitative_samples = 1
    args.encoder = "residual"
    args.pretrained_resnet = False
    args.verify_dataset_content = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.full_data:
        args.train_limit = 0
        args.val_limit = 0
    if args.epochs < 0:
        parser.error("--epochs cannot be negative")
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch size must be positive and workers cannot be negative")
    if args.image_width % LATENT_STRIDE or args.image_height % LATENT_STRIDE:
        parser.error(f"image width and height must be divisible by latent stride {LATENT_STRIDE}")
    if args.image_width <= 0 or args.image_height <= 0:
        parser.error("image dimensions must be positive")
    if args.max_latent_channels < max(args.active_channels):
        parser.error("max latent channels is smaller than an active rate")
    if args.paper_like_channels not in args.active_channels:
        parser.error("paper-like channels must be one of the active channel rates")
    if args.paper_like_channels != PAPER_LIKE_CHANNELS:
        parser.error("this experiment defines the paper_like operating point at exactly 80 channels")
    if args.max_latent_channels != max(args.active_channels):
        parser.error("max latent channels must equal the largest active rate for an exact scalable payload")
    if args.gradient_accumulation <= 0:
        parser.error("gradient accumulation must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        parser.error("invalid optimizer hyperparameters")
    if not 0.0 <= args.minimum_lr_ratio <= 1.0:
        parser.error("minimum LR ratio must be in [0, 1]")
    if not 0.0 <= args.horizontal_flip_probability <= 1.0:
        parser.error("horizontal flip probability must be in [0, 1]")
    if args.train_scale_max < 1.0 or not 0.0 <= args.color_jitter < 1.0:
        parser.error("invalid augmentation parameters")
    for name in ("ce_weight", "dice_weight", "rgb_l1_weight", "rgb_ssim_weight"):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")


def main(argv: Optional[Sequence[str]] = None) -> int:
    launch_time_utc = datetime.now(timezone.utc).isoformat()
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    apply_smoke_configuration(args)
    validate_arguments(args, parser)
    if args.out is None:
        default_output = DEFAULT_OUT.parent / "enhanced_smoke" if args.smoke else DEFAULT_OUT
        args.out = str(default_output)

    set_global_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = (args.image_width, args.image_height)
    image_root = Path(args.image_root).resolve()
    label_root = Path(args.label_root).resolve()
    dataset_manifest_path = Path(args.dataset_manifest).resolve() if args.dataset_manifest else None
    dataset_manifest_provenance: dict[str, Any] = {}
    if dataset_manifest_path is not None:
        if not dataset_manifest_path.exists():
            raise FileNotFoundError(f"Dataset audit manifest not found: {dataset_manifest_path}")
        dataset_manifest_provenance = verify_dataset_audit_manifest(
            dataset_manifest_path,
            image_root,
            label_root,
            verify_content=bool(args.verify_dataset_content),
        )
    args.dataset_fingerprint = dataset_manifest_provenance.get("fingerprint")
    source_path = Path(__file__).resolve()
    source_bytes = source_path.read_bytes()
    args.training_source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_snapshot_path = output_dir / "training_source_snapshot.py"
    source_snapshot_path.write_bytes(source_bytes)
    launch_manifest_path = output_dir / "launch_manifest.json"
    launch_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "started",
        "started_utc": launch_time_utc,
        "argv": list(sys.argv if argv is None else [str(source_path), *argv]),
        "command_windows": subprocess.list2cmdline(sys.argv if argv is None else [str(source_path), *argv]),
        "working_directory": str(Path.cwd()),
        "training_source": str(source_path),
        "training_source_snapshot": str(source_snapshot_path),
        "training_source_sha256": args.training_source_sha256,
        "dataset_manifest": str(dataset_manifest_path) if dataset_manifest_path is not None else None,
        "dataset_fingerprint": args.dataset_fingerprint,
        "dataset_content_reverified": dataset_manifest_provenance.get("fingerprint_current_content_verified"),
        "effective_configuration": vars(args).copy(),
        "environment": collect_environment(device),
    }
    write_strict_json(launch_manifest_path, launch_manifest)
    started = time.perf_counter()

    train_dataset = CityscapesRGBLabelDataset(
        image_root,
        label_root,
        "train",
        image_size,
        sample_limit=args.train_limit,
        training=True,
        seed=args.seed,
        scale_max=args.train_scale_max,
        horizontal_flip_probability=args.horizontal_flip_probability,
        color_jitter=args.color_jitter,
    )
    validation_dataset = CityscapesRGBLabelDataset(
        image_root,
        label_root,
        "val",
        image_size,
        sample_limit=args.val_limit,
        training=False,
        seed=args.seed,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + 17)
    loader_options: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    model = ScalableSemanticCodec(
        max_latent_channels=args.max_latent_channels,
        active_channels=args.active_channels,
        encoder_name=args.encoder,
        base_width=args.base_width,
        decoder_residual_blocks=args.decoder_residual_blocks,
        pretrained_resnet=args.pretrained_resnet,
    ).to(device)

    if args.class_balanced_loss:
        class_weights_cpu, class_pixel_counts = compute_class_weights(
            train_dataset,
            clip_min=args.class_weight_clip_min,
            clip_max=args.class_weight_clip_max,
        )
    else:
        class_weights_cpu = torch.ones(NUM_CLASSES, dtype=torch.float32)
        class_pixel_counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    loss_function = CompositeSemanticCodecLoss(
        class_weights=class_weights_cpu.to(device) if args.class_balanced_loss else None,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
        rgb_l1_weight=args.rgb_l1_weight,
        rgb_ssim_weight=args.rgb_ssim_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate * args.minimum_lr_ratio,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = _make_grad_scaler(amp_enabled)
    rate_rng = random.Random(args.seed + 29)

    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 1
    best_mean_iou = -1.0
    best_epoch = 0
    best_model_state = cpu_state_dict(model)
    resume_legacy_missing_keys: list[str] = []
    if args.resume:
        resume_path = Path(args.resume).resolve()
        checkpoint = load_torch_checkpoint(resume_path, device)
        if checkpoint.get("resume_supported", True) is not True:
            raise ValueError(
                f"Checkpoint is evaluation-only and cannot resume optimizer state: {resume_path}. "
                "Use last_checkpoint.pt or final_checkpoint.pt instead."
            )
        checkpoint_args = checkpoint.get("args")
        if not isinstance(checkpoint_args, Mapping):
            raise ValueError(f"Resume checkpoint has no argument provenance: {resume_path}")
        checkpoint_epoch = int(checkpoint.get("epoch", 0))
        newly_required_keys = ("dataset_manifest", "verify_dataset_content", "dataset_fingerprint")
        resume_legacy_missing_keys = [key for key in newly_required_keys if key not in checkpoint_args]
        evaluation_only_resume = checkpoint_epoch >= int(args.epochs)
        if resume_legacy_missing_keys and not (evaluation_only_resume or args.allow_legacy_resume_provenance):
            raise ValueError(
                "Resume checkpoint predates dataset fingerprint provenance and would continue training without "
                f"proof of identical data: missing {resume_legacy_missing_keys}. Re-run from scratch or pass "
                "--allow-legacy-resume-provenance explicitly."
            )
        validate_resume_configuration(
            checkpoint_args,
            args,
            allow_missing_keys=resume_legacy_missing_keys if (evaluation_only_resume or args.allow_legacy_resume_provenance) else (),
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_mean_iou = float(checkpoint.get("best_mean_iou", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = [dict(row) for row in checkpoint.get("history", [])]
        if checkpoint.get("best_model") is not None:
            best_model_state = {name: value.detach().cpu() for name, value in checkpoint["best_model"].items()}
        else:
            best_model_state = cpu_state_dict(model)
        if checkpoint.get("rng") is not None:
            restore_rng_state(checkpoint["rng"], rate_rng, train_generator)

    last_completed_epoch = start_epoch - 1
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    last_checkpoint_path = output_dir / "last_checkpoint.pt"
    final_checkpoint_path = output_dir / "final_checkpoint.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            loss_function,
            device,
            args.active_channels,
            rate_rng,
            args.gradient_accumulation,
            args.gradient_clip_norm,
            amp_enabled,
            global_step,
        )
        validation_profile, _ = evaluate_rate(
            model, validation_loader, loss_function, device, args.paper_like_channels
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce": train_metrics["ce"],
            "train_dice": train_metrics["dice"],
            "train_rgb_l1": train_metrics["rgb_l1"],
            "train_rgb_ssim": train_metrics["rgb_ssim"],
            "val_loss": validation_profile["loss"],
            "val_mean_iou": validation_profile["mean_iou"],
            "val_pixel_accuracy": validation_profile["pixel_accuracy"],
            "val_psnr_db": validation_profile["psnr_db"],
            "val_ssim": validation_profile["ssim"],
            "learning_rate": learning_rate,
            "global_step": global_step,
            "sampled_rate_counts_json": json.dumps(train_metrics["rate_sample_counts"], sort_keys=True),
        }
        history.append(row)
        improved = float(validation_profile["mean_iou"]) > best_mean_iou
        if improved:
            best_mean_iou = float(validation_profile["mean_iou"])
            best_epoch = epoch
            best_model_state = cpu_state_dict(model)
        checkpoint_payload = make_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            args,
            epoch,
            global_step,
            best_mean_iou,
            best_epoch,
            best_model_state,
            history,
            rate_rng,
            train_generator,
        )
        atomic_torch_save(checkpoint_payload, last_checkpoint_path)
        if improved:
            atomic_torch_save(checkpoint_payload, best_checkpoint_path)
        last_completed_epoch = epoch
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": row["train_loss"],
                    "val_mean_iou_80ch": row["val_mean_iou"],
                    "val_psnr_db_80ch": row["val_psnr_db"],
                    "best_epoch": best_epoch,
                },
                allow_nan=False,
            ),
            flush=True,
        )

    if not history:
        validation_profile, _ = evaluate_rate(
            model, validation_loader, loss_function, device, args.paper_like_channels
        )
        best_mean_iou = float(validation_profile["mean_iou"])
        best_epoch = last_completed_epoch
        best_model_state = cpu_state_dict(model)
        history.append(
            {
                "epoch": last_completed_epoch,
                "train_loss": None,
                "train_ce": None,
                "train_dice": None,
                "train_rgb_l1": None,
                "train_rgb_ssim": None,
                "val_loss": validation_profile["loss"],
                "val_mean_iou": validation_profile["mean_iou"],
                "val_pixel_accuracy": validation_profile["pixel_accuracy"],
                "val_psnr_db": validation_profile["psnr_db"],
                "val_ssim": validation_profile["ssim"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "global_step": global_step,
                "sampled_rate_counts_json": "{}",
            }
        )

    final_train_checkpoint = make_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        args,
        last_completed_epoch,
        global_step,
        best_mean_iou,
        best_epoch,
        best_model_state,
        history,
        rate_rng,
        train_generator,
    )
    atomic_torch_save(final_train_checkpoint, final_checkpoint_path)
    if not last_checkpoint_path.exists():
        atomic_torch_save(final_train_checkpoint, last_checkpoint_path)
    # Always overwrite the run-local best file from in-memory provenance, but
    # mark it evaluation-only: its weights come from the best epoch whereas the
    # final optimizer/RNG state belongs to the last epoch. Resume must use last
    # or final checkpoint, whose state is internally consistent.
    best_checkpoint_payload = dict(final_train_checkpoint)
    best_checkpoint_payload["model"] = dict(best_model_state)
    best_checkpoint_payload["resume_supported"] = False
    best_checkpoint_payload["optimizer"] = None
    best_checkpoint_payload["scheduler"] = None
    best_checkpoint_payload["scaler"] = None
    best_checkpoint_payload["rng"] = None
    best_checkpoint_payload["checkpoint_role"] = "evaluation_only_best_80ch"
    atomic_torch_save(best_checkpoint_payload, best_checkpoint_path)

    # Compare the 80-channel validation-best checkpoint with the last epoch over
    # all five rates. The adaptive follow-up consumes every rate, so selecting
    # solely on 80-channel mIoU could silently harm the rate-quality frontier.
    best_checkpoint_for_evaluation = load_torch_checkpoint(best_checkpoint_path, device)
    candidate_states = {
        "best_80ch_checkpoint": best_checkpoint_for_evaluation["model"],
        "last_epoch_checkpoint": final_train_checkpoint["model"],
    }
    evaluated_candidates: dict[str, dict[str, Any]] = {}
    for candidate_name, candidate_state in candidate_states.items():
        model.load_state_dict(candidate_state)
        candidate_profiles, candidate_confusions = evaluate_multi_rate_transport(
            model,
            validation_loader,
            loss_function,
            device,
            args.active_channels,
            args.zlib_level,
        )
        rate_mious = [float(profile["mean_iou"]) for profile in candidate_profiles]
        paper_candidate = next(
            profile for profile in candidate_profiles if profile["active_channels"] == args.paper_like_channels
        )
        evaluated_candidates[candidate_name] = {
            "profiles": candidate_profiles,
            "confusions": candidate_confusions,
            "mean_rate_miou": float(sum(rate_mious) / max(len(rate_mious), 1)),
            "minimum_rate_miou": float(min(rate_mious)) if rate_mious else 0.0,
            "paper_like_miou": float(paper_candidate["mean_iou"]),
        }
    selected_candidate_name = max(
        evaluated_candidates,
        key=lambda name: (
            evaluated_candidates[name]["mean_rate_miou"],
            evaluated_candidates[name]["minimum_rate_miou"],
            evaluated_candidates[name]["paper_like_miou"],
        ),
    )
    selected_candidate = evaluated_candidates[selected_candidate_name]
    model.load_state_dict(candidate_states[selected_candidate_name])
    multi_rate_profiles = selected_candidate["profiles"]
    rate_confusions = selected_candidate["confusions"]
    paper_profile = next(
        profile for profile in multi_rate_profiles if profile["active_channels"] == args.paper_like_channels
    )
    timing_sample = next(iter(validation_loader))[0][:1]
    timing = measure_timing(
        model,
        timing_sample,
        device,
        args.paper_like_channels,
        args.timing_runs,
    )
    qualitative_path = output_dir / "qualitative_panel_paper_like.png"
    saved_panel_count = save_qualitative_panel(
        qualitative_path,
        model,
        validation_loader,
        device,
        args.paper_like_channels,
        args.qualitative_samples,
        args.zlib_level,
    )

    history_path = output_dir / "training_history.csv"
    rate_quality_path = output_dir / "rate_quality.csv"
    rate_quality_best_path = output_dir / "rate_quality_best_80ch_checkpoint.csv"
    rate_quality_last_path = output_dir / "rate_quality_last_epoch_checkpoint.csv"
    per_class_path = output_dir / "per_class_iou_paper_like.csv"
    confusion_path = output_dir / "confusion_matrix_paper_like.csv"
    summary_path = output_dir / "result_summary.json"
    airtalking_path = output_dir / "airtalking_semantic_summary.json"
    write_history(history_path, history)
    write_rate_quality(rate_quality_path, multi_rate_profiles)
    write_rate_quality(rate_quality_best_path, evaluated_candidates["best_80ch_checkpoint"]["profiles"])
    write_rate_quality(rate_quality_last_path, evaluated_candidates["last_epoch_checkpoint"]["profiles"])
    write_per_class_iou(per_class_path, paper_profile)
    write_confusion_matrix(confusion_path, rate_confusions[args.paper_like_channels])

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    scientific_scope = {
        "classification": "paper-inspired independently specified follow-up experiment",
        "paper_disclosed_outline": (
            "RGB to modified U-Net semantic representation, compressed/transmitted representation, "
            "modified Pix2PixHD visual reconstruction"
        ),
        "undisclosed_by_paper": (
            "exact tensor layout, neural weights, losses, optimizer schedule, augmentations, and training recipe"
        ),
        "claim_not_made": "This is not the paper authors' exact encoder/decoder reproduction.",
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "enhanced_scalable_cityscapes_semantic_codec",
        "scientific_scope": scientific_scope,
        "status": "completed",
        "best_weights_reloaded_for_final_evaluation": selected_candidate_name == "best_80ch_checkpoint",
        "final_weight_selection": {
            "criterion": "highest mean mIoU across all five rates; minimum-rate and 80ch mIoU break ties",
            "selected": selected_candidate_name,
            "candidates": {
                name: {
                    "mean_rate_miou": values["mean_rate_miou"],
                    "minimum_rate_miou": values["minimum_rate_miou"],
                    "paper_like_miou": values["paper_like_miou"],
                }
                for name, values in evaluated_candidates.items()
            },
        },
        "seed": int(args.seed),
        "determinism": {
            "requested": bool(args.deterministic),
            "cudnn_deterministic": bool(args.deterministic),
            "algorithm_enforcement": "warn_only when supported",
            "guarantee": False,
            "known_limitation": (
                "PyTorch CUDA nll_loss2d may warn that no deterministic implementation is available; "
                "seeded repeatability is requested but bitwise identity is not claimed."
            ),
        },
        "dataset": {
            "name": "Cityscapes fine annotations, 19 trainId classes",
            "image_root": str(image_root),
            "label_root": str(label_root),
            "available_train_pairs": train_dataset.available_pairs,
            "available_val_pairs": validation_dataset.available_pairs,
            "train_samples": len(train_dataset),
            "val_samples": len(validation_dataset),
            "train_limit": int(args.train_limit),
            "val_limit": int(args.val_limit),
            "full_data": bool(args.train_limit <= 0 and args.val_limit <= 0),
            "image_size": {"width": args.image_width, "height": args.image_height},
            "train_augmentation": {
                "paired_random_scale_crop_max": args.train_scale_max,
                "horizontal_flip_probability": args.horizontal_flip_probability,
                "brightness_contrast_color_jitter": args.color_jitter,
                "rng_key": "seed + epoch*1000003 + index*9973",
            },
            "validation_transform": "deterministic bilinear RGB resize and nearest-neighbor label resize",
            **dataset_manifest_provenance,
        },
        "model": {
            "encoder": args.encoder,
            "pretrained_resnet": bool(args.pretrained_resnet),
            "decoder": "Pix2PixHD-inspired residual/upscale dual RGB+segmentation decoder",
            "receiver_input_contract": "quantized latent only; no encoder skip tensors cross the link",
            "latent_stride": LATENT_STRIDE,
            "max_latent_channels": args.max_latent_channels,
            "active_channels": list(args.active_channels),
            "paper_like_active_channels": args.paper_like_channels,
            "latent_bound": "[0, 1] via sigmoid",
            "quantization": "8-bit uniform STE in every train/inference forward",
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
        },
        "training": {
            "epochs_requested": args.epochs,
            "last_completed_epoch": last_completed_epoch,
            "global_optimizer_steps": global_step,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "CosineAnnealingLR",
            "minimum_lr_ratio": args.minimum_lr_ratio,
            "amp_requested": bool(args.amp),
            "amp_effective": amp_enabled,
            "gradient_accumulation": args.gradient_accumulation,
            "gradient_clip_norm": args.gradient_clip_norm,
            "rate_training": "maximum and minimum rates plus one seeded random intermediate decoder pass per batch",
            "loss": {
                "cross_entropy_weight": args.ce_weight,
                "dice_weight": args.dice_weight,
                "rgb_l1_weight": args.rgb_l1_weight,
                "rgb_ssim_weight": args.rgb_ssim_weight,
                "class_balancing": bool(args.class_balanced_loss),
                "class_weight_clip": [args.class_weight_clip_min, args.class_weight_clip_max],
                "class_weights": [float(value) for value in class_weights_cpu.tolist()],
                "class_pixel_counts_for_weight_estimate": [int(value) for value in class_pixel_counts.tolist()],
            },
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_mean_iou_during_training": best_mean_iou,
            "resumed_from": str(Path(args.resume).resolve()) if args.resume else None,
            "legacy_resume_provenance_missing_keys": resume_legacy_missing_keys,
            "legacy_resume_override_explicit": bool(args.allow_legacy_resume_provenance),
        },
        "transport_measurement": {
            "latent_serialization": "actual contiguous CxHxW uint8 bytes, one sample per payload",
            "zlib_level": args.zlib_level,
            "round_trip_verified": True,
            "ratio_denominator": "actual serialized resized RGB uint8 bytes",
        },
        "multi_rate_profiles": multi_rate_profiles,
        "quality_metric_definitions": {
            "mean_iou": "19-class confusion-matrix IoU over all non-ignore validation pixels",
            "pixel_accuracy": "correct non-ignore validation pixels divided by all non-ignore pixels",
            "psnr_db": "10*log10(1/global validation MSE), with a 100 dB cap only for exact-zero MSE",
            "ssim": (
                "project-local differentiable SSIM proxy using a uniform 7x7 window, zero padding, "
                "C1=0.01^2 and C2=0.03^2; it is not Gaussian-window torchmetrics/skimage SSIM"
            ),
        },
        "paper_like_profile": paper_profile,
        "paper_like_per_class_iou": [
            {"train_id": index, "class_name": CLASS_NAMES[index], "iou": paper_profile["per_class_iou"][index]}
            for index in range(NUM_CLASSES)
        ],
        "paper_like_confusion_matrix": rate_confusions[args.paper_like_channels].tolist(),
        "timing": timing,
        "resource_usage": {
            "cuda_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "cuda_peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            ),
            "scope": "current process; includes training and final evaluation",
        },
        "qualitative_samples_saved": saved_panel_count,
        "artifacts": {
            "result_summary_json": str(summary_path),
            "airtalking_semantic_summary_json": str(airtalking_path),
            "training_history_csv": str(history_path),
            "rate_quality_csv": str(rate_quality_path),
            "rate_quality_best_80ch_checkpoint_csv": str(rate_quality_best_path),
            "rate_quality_last_epoch_checkpoint_csv": str(rate_quality_last_path),
            "per_class_iou_csv": str(per_class_path),
            "confusion_matrix_csv": str(confusion_path),
            "qualitative_panel_png": str(qualitative_path),
            "best_checkpoint": str(best_checkpoint_path),
            "last_resume_checkpoint": str(last_checkpoint_path),
            "final_training_checkpoint": str(final_checkpoint_path),
        },
        "environment": collect_environment(device),
        "provenance": {
            "launch_manifest": str(launch_manifest_path),
            "training_source_snapshot": str(source_snapshot_path),
            "training_source_sha256": args.training_source_sha256,
            "started_utc": launch_time_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
        "effective_configuration": vars(args).copy(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "interpretation_notes": [
            "A smoke run validates mechanics only; it is not evidence of converged semantic quality.",
            "The raw rho values are exact for the observed uint8 tensor byte counts; zlib rho is content dependent.",
            "The 80-channel stride-16 8-bit operating point has theoretical rho 0.1041667 relative to raw RGB.",
            "Quality must be compared at the same dataset split, resize, checkpoint selection, and transport path.",
        ],
    }
    write_strict_json(summary_path, result)
    write_airtalking_summary(airtalking_path, result)
    launch_manifest.update(
        {
            "status": "completed",
            "finished_utc": result["provenance"]["finished_utc"],
            "result_summary": str(summary_path),
            "selected_final_weights": selected_candidate_name,
        }
    )
    write_strict_json(launch_manifest_path, launch_manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "output_directory": str(output_dir),
                "best_epoch": best_epoch,
                "selected_final_weights": selected_candidate_name,
                "paper_like_mean_iou": paper_profile["mean_iou"],
                "paper_like_psnr_db": paper_profile["psnr_db"],
                "paper_like_ssim": paper_profile["ssim"],
                "paper_like_rho_uint8": paper_profile["measured_rho_uint8_over_raw_rgb"],
                "paper_like_rho_zlib": paper_profile["measured_rho_zlib_over_raw_rgb"],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def cuda_oom_guidance() -> str:
    return (
        "CUDA out of memory while running the enhanced semantic codec. No checkpoint is silently marked "
        "successful. Free GPU memory or retry with, for example: --image-width 256 --image-height 128 "
        "--base-width 16 --batch-size 1 --gradient-accumulation 16. The --smoke flag is the smallest "
        "pipeline check."
    )


def cli_entry() -> int:
    try:
        return main()
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"ERROR: {cuda_oom_guidance()}", file=sys.stderr, flush=True)
        return 2
    except RuntimeError as exc:
        # Some older CUDA/PyTorch combinations raise the base RuntimeError type.
        if "out of memory" not in str(exc).lower() or "cuda" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"ERROR: {cuda_oom_guidance()}\nCUDA detail: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli_entry())
