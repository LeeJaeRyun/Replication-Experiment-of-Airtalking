from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import platform
import shutil
import statistics
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import validate_full_adaptive_results as result_validator


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parents[1]
AIR_TALKING_CODE = WORKSPACE_ROOT / "studies" / "airtalking_reproduction" / "code"
if str(AIR_TALKING_CODE) not in sys.path:
    sys.path.insert(0, str(AIR_TALKING_CODE))

from airtalking_reproduction import (  # noqa: E402
    AREAS,
    POLICIES,
    AssumedParams,
    PaperParams,
    SemanticCompressionMode,
    SemanticProfile,
    SimulationResult,
    aggregate,
    run_single,
    write_timeseries_npz,
)


DEFAULT_METADATA = (
    WORKSPACE_ROOT
    / "studies"
    / "airtalking_reproduction"
    / "results"
    / "airtalking_cityscapes_calibrated_final_p012"
    / "run_metadata.json"
)
DEFAULT_QUALITY = STUDY_ROOT / "results" / "probe_outputs" / "compression_quality.csv"
DEFAULT_OUT = STUDY_ROOT / "results" / "full_adaptive_results"
DEFAULT_NEURAL_SUMMARY = (
    WORKSPACE_ROOT
    / "studies"
    / "neural_encoder_decoder"
    / "results"
    / "paperlike_timed_latent20"
    / "airtalking_semantic_summary.json"
)
ADAPTIVE_MODE_ORDER = ("emergency", "low", "medium", "paper_like", "high")
DEFAULT_SINR_BOUNDS = (-15.0, -10.0, -5.0, 0.0, math.inf)
LEGACY_PROXY_QUALITY_THRESHOLDS = (0.80, 0.88, 0.925, 0.95, 0.965)

MULTI_RATE_LIST_KEYS = (
    "multi_rate_profiles",
    "rate_profiles",
    "multi_rate_results",
    "profiles_by_rate",
    "operating_points",
    "profiles",
)
ACTIVE_CHANNEL_KEYS = ("active_channels", "num_active_channels", "latent_channels", "channels", "channel_count")
RAW_RATIO_KEYS = (
    "rho_uint8",
    "measured_rho_uint8_over_raw_rgb",
    "payload_ratio_raw",
    "raw_payload_ratio",
    "rho_raw",
    "rho_uncompressed",
    "theoretical_rho_raw_rgb",
)
ZLIB_RATIO_KEYS = (
    "rho_zlib",
    "measured_rho_zlib_over_raw_rgb",
    "payload_ratio_zlib",
    "zlib_payload_ratio",
    "compressed_payload_ratio",
    "zlib_ratio",
)
GENERIC_RATIO_KEYS = ("payload_ratio", "rho_c", "compression_ratio", "ratio")
MIOU_KEYS = ("mean_iou", "miou", "m_iou", "semantic_quality", "semantic_quality_miou", "val_mean_iou")
PSNR_KEYS = ("psnr_db", "psnr", "rgb_psnr_db", "rgb_reconstruction_psnr_db")
SSIM_KEYS = ("ssim", "ssim_score", "rgb_ssim", "rgb_reconstruction_ssim")
ENCODE_BITRATE_KEYS = (
    "feature_encode_bitrate_mbps_median",
    "encode_bitrate_mbps_median",
    "measured_encode_bitrate_mbps",
    "measured_encode_input_throughput_mbps",
    "encoder_bitrate_mbps",
    "encode_throughput_mbps",
)
DECODE_BITRATE_KEYS = (
    "feature_decode_bitrate_mbps_median",
    "decode_bitrate_mbps_median",
    "measured_decode_bitrate_mbps",
    "measured_decode_restoration_throughput_mbps",
    "decoder_bitrate_mbps",
    "decode_throughput_mbps",
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def dataclass_from_dict(cls, values: dict) -> object:
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def load_base_params(metadata_path: Path) -> Tuple[PaperParams, AssumedParams, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    paper = dataclass_from_dict(PaperParams, metadata["paper_params"])
    assumed = dataclass_from_dict(AssumedParams, metadata["assumed_params"])
    return paper, assumed, metadata


def load_quality_rows(path: Path) -> List[dict[str, float | str]]:
    rows: List[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"mode", "description"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No compression quality rows found in {path}")
    return sorted(rows, key=lambda item: float(item["feature_ratio_mean"]))


def _find_alias(mapping: Mapping[str, object], aliases: Sequence[str]) -> Tuple[Optional[object], Optional[str]]:
    lowered = {str(key).lower(): (str(key), value) for key, value in mapping.items()}
    for alias in aliases:
        if alias in mapping:
            return mapping[alias], alias
        match = lowered.get(alias.lower())
        if match is not None:
            return match[1], match[0]
    return None, None


def _find_nested_alias(
    mapping: Mapping[str, object],
    aliases: Sequence[str],
    nested_keys: Sequence[str] = (),
) -> Tuple[Optional[object], Optional[str]]:
    value, key = _find_alias(mapping, aliases)
    if key is not None:
        return value, key
    for nested_key in nested_keys:
        nested, actual_nested_key = _find_alias(mapping, (nested_key,))
        if actual_nested_key is None or not isinstance(nested, Mapping):
            continue
        value, key = _find_alias(nested, aliases)
        if key is not None:
            return value, f"{actual_nested_key}.{key}"
    return None, None


def _numeric_alias(
    mapping: Mapping[str, object],
    aliases: Sequence[str],
    context: str,
    *,
    nested_keys: Sequence[str] = (),
    required: bool = True,
) -> Tuple[Optional[float], Optional[str]]:
    raw_value, source_key = _find_nested_alias(mapping, aliases, nested_keys)
    if source_key is None:
        if required:
            raise ValueError(f"{context} is missing; accepted keys: {', '.join(aliases)}")
        return None, None
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} from {source_key!r} must be numeric, got {raw_value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} from {source_key!r} must be finite, got {raw_value!r}")
    return value, source_key


def _normalize_neural_anchor(anchor: dict[str, object]) -> dict[str, object]:
    """Add legacy timing aliases needed by existing metadata/report consumers."""
    normalized = dict(anchor)
    raw_timing = anchor.get("timing")
    if not isinstance(raw_timing, Mapping):
        return normalized
    timing = dict(raw_timing)
    encode_group = timing.get("encode_including_8bit_fake_quantization")
    decode_group = timing.get("decode_from_latent_only")
    if "encode_ms_median" not in timing and isinstance(encode_group, Mapping) and "median_ms" in encode_group:
        timing["encode_ms_median"] = encode_group["median_ms"]
    if "decode_ms_median" not in timing and isinstance(decode_group, Mapping) and "median_ms" in decode_group:
        timing["decode_ms_median"] = decode_group["median_ms"]
    normalized["timing"] = timing
    return normalized


def load_neural_anchor(path: Path) -> Optional[dict[str, object]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Neural summary must contain a JSON object: {path}")
    return _normalize_neural_anchor(payload)


def _extract_multi_rate_profiles(neural_anchor: Mapping[str, object]) -> Optional[list[Mapping[str, object]]]:
    containers: list[Tuple[str, Mapping[str, object]]] = [("root", neural_anchor)]
    for parent_key in ("multi_rate", "evaluation", "metrics", "results", "codec", "summary"):
        nested, actual_key = _find_alias(neural_anchor, (parent_key,))
        if actual_key is not None and isinstance(nested, Mapping):
            containers.append((actual_key, nested))
    for container_name, container in containers:
        value, key = _find_alias(container, MULTI_RATE_LIST_KEYS)
        if key is None:
            continue
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{container_name}.{key} must be an array of neural rate profiles")
        profiles: list[Mapping[str, object]] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValueError(f"{container_name}.{key}[{index}] must be an object")
            profiles.append(item)
        if not profiles:
            raise ValueError(f"{container_name}.{key} is present but empty")
        return profiles
    return None


def _parse_multi_rate_profile(profile: Mapping[str, object], index: int) -> dict[str, object]:
    context = f"multi_rate_profiles[{index}]"
    active_value, active_key = _numeric_alias(profile, ACTIVE_CHANNEL_KEYS, f"{context}.active_channels")
    assert active_value is not None and active_key is not None
    if active_value <= 0.0 or not active_value.is_integer():
        raise ValueError(f"{context}.{active_key} must be a positive integer, got {active_value!r}")

    ratio_nested = ("payload_ratio", "payload_ratios", "ratios", "compression")
    raw_ratio, raw_key = _numeric_alias(
        profile,
        RAW_RATIO_KEYS + ("raw", "uint8", "uncompressed"),
        f"{context}.raw_payload_ratio",
        nested_keys=ratio_nested,
        required=False,
    )
    zlib_ratio, zlib_key = _numeric_alias(
        profile,
        ZLIB_RATIO_KEYS + ("zlib", "compressed"),
        f"{context}.zlib_payload_ratio",
        nested_keys=ratio_nested,
        required=False,
    )
    generic_ratio, generic_key = _numeric_alias(
        profile,
        GENERIC_RATIO_KEYS,
        f"{context}.payload_ratio",
        required=False,
    )
    if raw_ratio is not None:
        selected_ratio, ratio_basis, ratio_key = raw_ratio, "raw", raw_key
    elif zlib_ratio is not None:
        selected_ratio, ratio_basis, ratio_key = zlib_ratio, "zlib", zlib_key
    elif generic_ratio is not None:
        selected_ratio, ratio_basis, ratio_key = generic_ratio, "unspecified", generic_key
    else:
        accepted = ", ".join((*RAW_RATIO_KEYS, *ZLIB_RATIO_KEYS, *GENERIC_RATIO_KEYS))
        raise ValueError(f"{context} has no payload ratio; accepted keys: {accepted}")
    if selected_ratio <= 0.0:
        raise ValueError(f"{context}.{ratio_key} must be greater than zero")

    metric_nested = ("metrics", "quality", "quality_metrics", "reconstruction")
    mean_iou, mean_iou_key = _numeric_alias(
        profile,
        MIOU_KEYS,
        f"{context}.mean_iou",
        nested_keys=metric_nested,
    )
    assert mean_iou is not None and mean_iou_key is not None
    if not 0.0 <= mean_iou <= 1.0:
        raise ValueError(f"{context}.{mean_iou_key} must be between 0 and 1")
    psnr, psnr_key = _numeric_alias(
        profile,
        PSNR_KEYS,
        f"{context}.psnr",
        nested_keys=metric_nested,
        required=False,
    )
    ssim, ssim_key = _numeric_alias(
        profile,
        SSIM_KEYS,
        f"{context}.ssim",
        nested_keys=metric_nested,
        required=False,
    )
    if psnr is None or ssim is None:
        missing = ", ".join(name for name, value in (("PSNR", psnr), ("SSIM", ssim)) if value is None)
        warnings.warn(f"{context} is missing optional {missing}; selection still uses payload ratio and mIoU", RuntimeWarning)
    return {
        "active_channels": int(active_value),
        "payload_ratio": selected_ratio,
        "payload_ratio_basis": ratio_basis,
        "payload_ratio_source_key": ratio_key,
        "rho_raw": raw_ratio,
        "rho_zlib": zlib_ratio,
        "mean_iou": mean_iou,
        "mean_iou_source_key": mean_iou_key,
        "psnr_db": psnr,
        "psnr_source_key": psnr_key,
        "ssim": ssim,
        "ssim_source_key": ssim_key,
    }


def apply_neural_anchor(
    rows: Sequence[dict[str, float | str]],
    neural_anchor: Optional[dict[str, object]],
    quality_mode: str,
) -> List[dict[str, float | str]]:
    if neural_anchor is None:
        return [dict(row) for row in rows]
    if quality_mode not in {"record_only", "selection"}:
        raise ValueError(f"Unknown neural quality mode: {quality_mode!r}")

    multi_rate_profiles = _extract_multi_rate_profiles(neural_anchor)
    if multi_rate_profiles is not None:
        target_rows = {str(row.get("mode")): dict(row) for row in rows if str(row.get("mode")) in ADAPTIVE_MODE_ORDER}
        missing_modes = [mode for mode in ADAPTIVE_MODE_ORDER if mode not in target_rows]
        if missing_modes:
            raise ValueError(f"Quality table is missing adaptive modes required for neural mapping: {missing_modes}")
        if len(multi_rate_profiles) != len(ADAPTIVE_MODE_ORDER):
            raise ValueError(
                f"multi_rate_profiles must contain exactly {len(ADAPTIVE_MODE_ORDER)} rates to replace "
                f"{ADAPTIVE_MODE_ORDER}; got {len(multi_rate_profiles)}"
            )
        parsed_profiles = sorted(
            (_parse_multi_rate_profile(profile, index) for index, profile in enumerate(multi_rate_profiles)),
            key=lambda profile: float(profile["payload_ratio"]),
        )
        ratios = [float(profile["payload_ratio"]) for profile in parsed_profiles]
        channels = [int(profile["active_channels"]) for profile in parsed_profiles]
        if len(set(ratios)) != len(ratios):
            raise ValueError(f"multi_rate_profiles payload ratios must be unique, got {ratios}")
        if len(set(channels)) != len(channels):
            raise ValueError(f"multi_rate_profiles active channel counts must be unique, got {channels}")

        source = str(neural_anchor.get("source", "trained_multi_rate_neural_semantic_codec"))
        updates: dict[str, dict[str, object]] = {}
        for mode, profile in zip(ADAPTIVE_MODE_ORDER, parsed_profiles):
            ratio = float(profile["payload_ratio"])
            quality = float(profile["mean_iou"])
            updates[mode] = {
                "feature_ratio_mean": ratio,
                "feature_ratio_median": ratio,
                "neural_encoder_decoder_miou": quality,
                "neural_active_channels": int(profile["active_channels"]),
                "neural_payload_ratio": ratio,
                "neural_payload_ratio_basis": str(profile["payload_ratio_basis"]),
                "neural_payload_ratio_source_key": str(profile["payload_ratio_source_key"]),
                "neural_encoder_decoder_source": source,
                "description": f"neural multi-rate profile ({int(profile['active_channels'])} active channels)",
            }
            if quality_mode == "selection":
                updates[mode]["mean_iou_mean"] = quality
                updates[mode]["mean_iou_median"] = quality
            if profile["rho_raw"] is not None:
                updates[mode]["neural_rho_raw"] = float(profile["rho_raw"])
            if profile["rho_zlib"] is not None:
                updates[mode]["neural_rho_zlib"] = float(profile["rho_zlib"])
                updates[mode]["zlib_ratio_mean"] = float(profile["rho_zlib"])
                updates[mode]["zlib_ratio_median"] = float(profile["rho_zlib"])
            if profile["psnr_db"] is not None:
                updates[mode]["neural_psnr_db"] = float(profile["psnr_db"])
            if profile["ssim"] is not None:
                updates[mode]["neural_ssim"] = float(profile["ssim"])

        out = []
        for row in rows:
            updated = dict(row)
            mode = str(updated.get("mode"))
            if mode in updates:
                updated.update(updates[mode])
            out.append(updated)
        return sorted(out, key=lambda item: float(item["feature_ratio_mean"]))

    rho_c, _ = _numeric_alias(
        neural_anchor,
        (
            "rho_c_feature_uncompressed_mean",
            "rho_c_uncompressed_mean",
            "rho_uint8",
            "payload_ratio_raw",
            "payload_ratio",
            "rho_c",
        ),
        "single-rate neural payload ratio",
    )
    neural_quality, _ = _numeric_alias(
        neural_anchor,
        ("semantic_quality_miou_best", "semantic_quality_miou_final", "val_mean_iou", *MIOU_KEYS),
        "single-rate neural mIoU",
    )
    assert rho_c is not None and neural_quality is not None
    out: List[dict[str, float | str]] = []
    for row in rows:
        updated = dict(row)
        if str(updated["mode"]) == "paper_like":
            updated["feature_ratio_mean"] = rho_c
            updated["feature_ratio_median"] = rho_c
            updated["neural_encoder_decoder_miou"] = neural_quality
            updated["neural_encoder_decoder_source"] = str(neural_anchor.get("source", "trained_cityscapes_rgb_to_semantic_encoder_decoder"))
            updated["description"] = "paper-like payload anchored to trained neural encoder/decoder"
            if quality_mode == "selection":
                updated["mean_iou_mean"] = neural_quality
                updated["mean_iou_median"] = neural_quality
        out.append(updated)
    return out


def apply_neural_bitrates(
    paper: PaperParams,
    neural_anchor: Optional[Mapping[str, object]],
) -> Tuple[PaperParams, dict[str, object]]:
    if neural_anchor is None:
        return paper, {"applied": False, "reason": "no_neural_summary"}
    nested = ("throughput", "timing", "metrics", "paper_comparison", "benchmark")
    encode_mbps, encode_key = _numeric_alias(
        neural_anchor,
        ENCODE_BITRATE_KEYS,
        "neural encode bitrate",
        nested_keys=nested,
        required=False,
    )
    decode_mbps, decode_key = _numeric_alias(
        neural_anchor,
        DECODE_BITRATE_KEYS,
        "neural decode bitrate",
        nested_keys=nested,
        required=False,
    )
    if encode_mbps is None and decode_mbps is None:
        warnings.warn(
            "Neural summary has no measured encode/decode bitrate; calibrated base PaperParams are retained",
            RuntimeWarning,
        )
        return paper, {"applied": False, "reason": "measured_bitrates_missing"}
    if encode_mbps is None or decode_mbps is None:
        missing = "encode" if encode_mbps is None else "decode"
        raise ValueError(f"Neural summary has an incomplete bitrate pair: missing measured {missing} bitrate")
    if encode_mbps <= 0.0 or decode_mbps <= 0.0:
        raise ValueError("Measured neural encode/decode bitrates must both be greater than zero")
    updated = PaperParams(
        **{
            **paper.__dict__,
            "enc_bitrate": encode_mbps * 1e6,
            "dec_bitrate": decode_mbps * 1e6,
        }
    )
    return updated, {
        "applied": True,
        "encode_bitrate_mbps": encode_mbps,
        "decode_bitrate_mbps": decode_mbps,
        "encode_source_key": encode_key,
        "decode_source_key": decode_key,
    }


def parse_quality_thresholds(value: str | None) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if len(pieces) != len(DEFAULT_SINR_BOUNDS):
        raise ValueError(
            f"Explicit quality thresholds require {len(DEFAULT_SINR_BOUNDS)} comma-separated values; "
            f"got {len(pieces)}"
        )
    try:
        thresholds = tuple(float(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError("Explicit quality thresholds must all be numeric") from exc
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Explicit quality thresholds must be finite values between 0 and 1")
    if any(right < left for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("Explicit quality thresholds must be in nondecreasing SINR-bin order")
    return thresholds


def _has_multi_rate_anchor(neural_anchor: Optional[Mapping[str, object]]) -> bool:
    return neural_anchor is not None and _extract_multi_rate_profiles(neural_anchor) is not None


def resolve_adaptive_thresholds(
    rows: Sequence[dict[str, float | str]],
    neural_anchor: Optional[Mapping[str, object]],
    neural_quality_mode: str,
    requested_rule: str = "auto",
    explicit_quality_thresholds: Optional[Sequence[float]] = None,
) -> tuple[Tuple[Tuple[float, float], ...], dict[str, object]]:
    """Resolve five pre-declared SINR bins to the qualities used by selection."""
    if requested_rule not in {
        "auto",
        "measured_ordered",
        "ordered_mode_quality",
        "legacy_proxy",
        "explicit",
    }:
        raise ValueError(f"Unknown adaptive threshold rule: {requested_rule!r}")
    multi_rate = _has_multi_rate_anchor(neural_anchor)
    resolved_rule = requested_rule
    if requested_rule == "auto":
        if multi_rate and neural_quality_mode == "selection":
            resolved_rule = "measured_ordered"
        elif multi_rate:
            resolved_rule = "ordered_mode_quality"
        else:
            resolved_rule = "legacy_proxy"

    if resolved_rule == "explicit":
        if explicit_quality_thresholds is None:
            raise ValueError("--adaptive-threshold-rule explicit requires --adaptive-quality-thresholds")
        qualities = tuple(float(value) for value in explicit_quality_thresholds)
        quality_source = "cli_explicit"
    elif explicit_quality_thresholds is not None:
        raise ValueError("--adaptive-quality-thresholds may only be used with --adaptive-threshold-rule explicit")
    elif resolved_rule == "legacy_proxy":
        qualities = LEGACY_PROXY_QUALITY_THRESHOLDS
        quality_source = "legacy_label_proxy_constants"
    elif resolved_rule == "measured_ordered":
        if not multi_rate:
            raise ValueError("measured_ordered requires a neural summary with five multi-rate profiles")
        if neural_quality_mode != "selection":
            raise ValueError(
                "measured_ordered requires --neural-quality-mode selection; record_only preserves the "
                "quality CSV values used by selection"
            )
        measured = [
            float(row["neural_encoder_decoder_miou"])
            for row in rows
            if str(row.get("mode")) in ADAPTIVE_MODE_ORDER
            and "neural_encoder_decoder_miou" in row
        ]
        if len(measured) != len(DEFAULT_SINR_BOUNDS):
            raise ValueError("measured_ordered requires one measured neural mIoU for every adaptive rate")
        qualities = tuple(sorted(measured))
        quality_source = "measured_multi_rate_neural_miou_ascending"
    else:
        ordered_rows = sorted(
            (row for row in rows if str(row.get("mode")) in ADAPTIVE_MODE_ORDER),
            key=lambda row: float(row["feature_ratio_mean"]),
        )
        if len(ordered_rows) != len(DEFAULT_SINR_BOUNDS):
            raise ValueError("ordered_mode_quality requires exactly five adaptive modes")
        qualities = tuple(float(row["mean_iou_mean"]) for row in ordered_rows)
        if any(right < left for left, right in zip(qualities, qualities[1:])):
            raise ValueError("Adaptive mode quality must be nondecreasing with payload rate")
        quality_source = (
            "quality_csv_selection_values_record_only"
            if multi_rate and neural_quality_mode == "record_only"
            else "active_mode_quality_ascending_rate"
        )

    if len(qualities) != len(DEFAULT_SINR_BOUNDS):
        raise ValueError(f"Threshold rule must produce exactly {len(DEFAULT_SINR_BOUNDS)} qualities")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in qualities):
        raise ValueError("Resolved adaptive quality thresholds must be finite values between 0 and 1")
    thresholds = tuple(zip(DEFAULT_SINR_BOUNDS, qualities))
    return thresholds, {
        "requested_rule": requested_rule,
        "resolved_rule": resolved_rule,
        "quality_source": quality_source,
        "sinr_bounds_db": [None if math.isinf(value) else value for value in DEFAULT_SINR_BOUNDS],
        "quality_thresholds": list(qualities),
        "actual_thresholds": [
            [None if math.isinf(bound) else bound, quality]
            for bound, quality in thresholds
        ],
        "null_means_unbounded": True,
    }


def build_profiles(
    rows: Sequence[dict[str, float | str]],
    target_thresholds: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[SemanticProfile, SemanticProfile]:
    modes = tuple(
        SemanticCompressionMode(
            name=str(row["mode"]),
            rho_c=float(row["feature_ratio_mean"]),
            quality=float(row["mean_iou_mean"]),
        )
        for row in rows
    )
    paper_like = next((mode for mode in modes if mode.name == "paper_like"), modes[len(modes) // 2])
    fixed = SemanticProfile(name="fixed_paper_like", strategy="fixed", modes=(paper_like,))
    adaptive = SemanticProfile(
        name="adaptive_semantic",
        strategy="adaptive",
        modes=modes,
        target_thresholds=tuple(target_thresholds) if target_thresholds is not None else SemanticProfile.target_thresholds,
    )
    return fixed, adaptive


def reachable_adaptive_modes(profile: SemanticProfile) -> list[str]:
    """Return modes selected by one representative SINR from every configured bin."""
    lower = -math.inf
    selected: list[str] = []
    modes = sorted(profile.modes, key=lambda mode: mode.rho_c)
    for upper, required_quality in profile.target_thresholds:
        if math.isinf(lower):
            representative = upper - 5.0
        elif math.isinf(upper):
            representative = lower + 5.0
        else:
            representative = (lower + upper) / 2.0
        target = profile.target_thresholds[-1][1]
        for bin_upper, bin_quality in profile.target_thresholds:
            if representative < bin_upper:
                target = bin_quality
                break
        chosen = next((mode for mode in modes if mode.quality >= target), max(modes, key=lambda mode: mode.quality))
        if chosen.name not in selected:
            selected.append(chosen.name)
        lower = upper
    return selected


def profile_to_metadata(profile: SemanticProfile) -> dict[str, object]:
    """Return strict-JSON metadata without changing the in-memory profile.

    ``None`` is the JSON representation of the final, unbounded SINR interval.
    The explicit flag keeps that meaning available to metadata consumers while
    avoiding Python's non-standard ``Infinity`` JSON token.
    """
    thresholds: list[list[float | None]] = []
    for upper_sinr, required_quality in profile.target_thresholds:
        if math.isinf(upper_sinr) and upper_sinr > 0.0:
            json_upper_sinr: float | None = None
        else:
            json_upper_sinr = upper_sinr
        thresholds.append([json_upper_sinr, required_quality])
    return {
        "name": profile.name,
        "strategy": profile.strategy,
        "modes": [mode.__dict__ for mode in profile.modes],
        "target_thresholds": thresholds,
        "null_means_unbounded": True,
        "target_thresholds_semantics": (
            "Each pair is [exclusive upper SINR bound in dB, required quality]; "
            "a null upper bound means the final interval is unbounded."
        ),
    }


def profile_from_metadata(metadata: dict[str, object]) -> SemanticProfile:
    """Load profiles written by :func:`profile_to_metadata`.

    Numeric legacy thresholds are also accepted, so consumers can read both
    pre-fix Python-generated metadata and the strict-JSON representation.
    """
    raw_thresholds = metadata.get("target_thresholds", ())
    null_means_unbounded = bool(metadata.get("null_means_unbounded", False))
    thresholds: list[tuple[float, float]] = []
    for raw_threshold in raw_thresholds:  # type: ignore[union-attr]
        upper_sinr, required_quality = raw_threshold
        if upper_sinr is None:
            if not null_means_unbounded:
                raise ValueError("null SINR threshold requires null_means_unbounded=true")
            upper_sinr = math.inf
        thresholds.append((float(upper_sinr), float(required_quality)))
    raw_modes = metadata.get("modes", ())
    modes = tuple(
        SemanticCompressionMode(
            name=str(raw_mode["name"]),
            rho_c=float(raw_mode["rho_c"]),
            quality=float(raw_mode["quality"]),
        )
        for raw_mode in raw_modes  # type: ignore[union-attr]
    )
    return SemanticProfile(
        name=str(metadata["name"]),
        strategy=str(metadata["strategy"]),
        modes=modes,
        target_thresholds=tuple(thresholds),
    )


def write_metadata_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path, *, required: bool = True) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.exists():
        if required:
            raise FileNotFoundError(resolved)
        return {"path": str(resolved), "exists": False, "sha256": None, "size_bytes": None}
    return {
        "path": str(resolved),
        "exists": True,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def artifact_provenance(path: Path) -> dict[str, object]:
    record = file_provenance(path)
    record.pop("exists", None)
    return record


def require_integrity_validation(result: Mapping[str, object]) -> None:
    if result.get("passed") is not True:
        raise RuntimeError("Adaptive result integrity validation failed")


def sanitize_neural_anchor_metadata(
    neural_anchor: Optional[Mapping[str, object]],
) -> Optional[dict[str, object]]:
    """Keep measured evidence while excluding arbitrary nested/stale path metadata."""
    if neural_anchor is None:
        return None
    safe: dict[str, object] = {}
    for key in (
        "schema_version",
        "source",
        "paper_like_active_channels",
        "rho_c_feature_uncompressed_mean",
        "semantic_quality_miou_best",
        "semantic_quality_miou_final",
        "pixel_accuracy_best",
        "feature_encode_bitrate_mbps_median",
        "feature_decode_bitrate_mbps_median",
    ):
        value = neural_anchor.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
    timing = neural_anchor.get("timing")
    if isinstance(timing, Mapping):
        safe["timing"] = {
            key: value
            for key, value in timing.items()
            if key in {"encode_ms_median", "decode_ms_median", "full_ms_median"}
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        }
    profiles = _extract_multi_rate_profiles(neural_anchor)
    if profiles is not None:
        safe["multi_rate_profiles"] = [
            {
                "active_channels": parsed["active_channels"],
                "payload_ratio": parsed["payload_ratio"],
                "payload_ratio_basis": parsed["payload_ratio_basis"],
                "mean_iou": parsed["mean_iou"],
                "psnr_db": parsed["psnr_db"],
                "ssim": parsed["ssim"],
            }
            for parsed in (
                _parse_multi_rate_profile(profile, index)
                for index, profile in enumerate(profiles)
            )
        ]
    safe["sanitized_for_run_metadata"] = True
    safe["nested_source_metadata_embedded"] = False
    return safe


def command_environment(argv: Sequence[str]) -> dict[str, object]:
    full_argv = [sys.executable, *argv]
    return {
        "argv": full_argv,
        "command_windows": subprocess.list2cmdline(full_argv),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


def _run_repeat_task(
    task: Tuple[int, str, int, bool, PaperParams, AssumedParams, Optional[SemanticProfile]],
) -> SimulationResult:
    """Top-level worker entry point so Windows ``spawn`` can pickle it."""
    area, policy, repeat, semantic_enabled, paper, assumed, profile = task
    return run_single(
        area,
        policy,
        repeat,
        semantic_enabled,
        paper,
        assumed,
        semantic_profile=profile,
    )


def run_mode(
    mode_name: str,
    semantic_enabled: bool,
    profile: Optional[SemanticProfile],
    paper: PaperParams,
    assumed: AssumedParams,
    repeats: int,
    areas: Sequence[int],
    policies: Sequence[str],
    repeat_metrics: Optional[List[dict[str, object]]] = None,
    workers: int = 1,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[int, Dict[str, SimulationResult]]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    owned_executor: Optional[ProcessPoolExecutor] = None
    if executor is None and workers > 1:
        owned_executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        executor = owned_executor
    out: Dict[int, Dict[str, SimulationResult]] = {}
    try:
        for area in areas:
            out[area] = {}
            for policy in policies:
                print(
                    f"[{mode_name}] area={area} policy={policy} repeats={repeats} workers={workers}",
                    flush=True,
                )
                tasks = [
                    (area, policy, repeat, semantic_enabled, paper, assumed, profile)
                    for repeat in range(repeats)
                ]
                if executor is None:
                    reps = [_run_repeat_task(task) for task in tasks]
                else:
                    # executor.map preserves repeat order, matching workers=1 output ordering.
                    reps = list(executor.map(_run_repeat_task, tasks, chunksize=1))
                if repeat_metrics is not None:
                    for repeat, result in enumerate(reps):
                        repeat_metrics.append(
                            {
                                "mode": mode_name,
                                "area": area,
                                "policy": policy,
                                "repeat": repeat,
                                **result.summary,
                            }
                        )
                out[area][policy] = aggregate(reps)
    finally:
        if owned_executor is not None:
            owned_executor.shutdown(wait=True, cancel_futures=False)
    return out


REPEAT_METRIC_ORDER = (
    "finished",
    "flight_energy_per_req",
    "nonflight_energy_per_req",
    "avg_time",
    "avg_travel",
    "encodes",
    "decodes",
    "semantic_quality",
    "semantic_payload_ratio",
    "sinr_median_db",
)


def _repeat_metric_keys(rows: Sequence[Mapping[str, object]]) -> list[str]:
    excluded = {"mode", "area", "policy", "repeat"}
    available = {key for row in rows for key in row if key not in excluded}
    ordered = [key for key in REPEAT_METRIC_ORDER if key in available]
    return ordered + sorted(available - set(ordered))


def write_repeat_metrics_csv(rows: Sequence[Mapping[str, object]], out_dir: Path) -> Path:
    if not rows:
        raise ValueError("No repeat metrics were collected")
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = _repeat_metric_keys(rows)
    fields_out = ["mode", "area", "policy", "repeat", *metric_keys]
    path = out_dir / "repeat_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields_out)
        writer.writeheader()
        for source_row in rows:
            row = {field: "" for field in fields_out}
            row.update({field: source_row[field] for field in fields_out if field in source_row})
            writer.writerow(row)
    return path


_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _student_t_critical_95(sample_count: int) -> float:
    degrees_freedom = sample_count - 1
    if degrees_freedom <= 0:
        return 0.0
    if degrees_freedom <= 30:
        return _T_CRITICAL_975[degrees_freedom]
    if degrees_freedom <= 40:
        return 2.021
    if degrees_freedom <= 60:
        return 2.000
    if degrees_freedom <= 120:
        return 1.980
    return 1.960


def write_statistical_summary_csv(rows: Sequence[Mapping[str, object]], out_dir: Path) -> Path:
    if not rows:
        raise ValueError("No repeat metrics were collected")
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[Tuple[str, int, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["mode"]), int(row["area"]), str(row["policy"]))
        groups.setdefault(key, []).append(row)
    metric_keys = _repeat_metric_keys(rows)
    statistic_suffixes = ("n", "mean", "std", "ci95_margin", "ci95_low", "ci95_high")
    fields_out = ["mode", "area", "policy", "repeat_count", "ci95_method"]
    fields_out.extend(f"{metric}_{suffix}" for metric in metric_keys for suffix in statistic_suffixes)

    path = out_dir / "statistical_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields_out)
        writer.writeheader()
        for (mode, area, policy), group_rows in sorted(groups.items()):
            output_row: dict[str, object] = {field: "" for field in fields_out}
            output_row.update(
                {
                    "mode": mode,
                    "area": area,
                    "policy": policy,
                    "repeat_count": len(group_rows),
                    "ci95_method": "two-sided Student t; n=1 uses zero-width descriptive interval",
                }
            )
            for metric in metric_keys:
                values = [
                    float(row[metric])
                    for row in group_rows
                    if metric in row
                    and isinstance(row[metric], (int, float))
                    and not isinstance(row[metric], bool)
                    and math.isfinite(float(row[metric]))
                ]
                output_row[f"{metric}_n"] = len(values)
                if not values:
                    continue
                mean = statistics.fmean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                margin = _student_t_critical_95(len(values)) * std / math.sqrt(len(values))
                output_row[f"{metric}_mean"] = mean
                output_row[f"{metric}_std"] = std
                output_row[f"{metric}_ci95_margin"] = margin
                output_row[f"{metric}_ci95_low"] = mean - margin
                output_row[f"{metric}_ci95_high"] = mean + margin
            writer.writerow(output_row)
    return path


def write_summary_csv(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    core = [
        "mode",
        "area",
        "policy",
        "finished",
        "flight_energy_per_req",
        "nonflight_energy_per_req",
        "avg_time",
        "avg_travel",
        "encodes",
        "decodes",
        "semantic_quality",
        "semantic_payload_ratio",
        "sinr_median_db",
    ]
    extras = sorted(
        {
            key
            for mode_results in results.values()
            for area_results in mode_results.values()
            for result in area_results.values()
            for key in result.summary
            if key not in core
        }
    )
    fields_out = core + extras
    path = out_dir / "summary_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields_out)
        writer.writeheader()
        for mode_name, mode_results in results.items():
            for area, area_results in mode_results.items():
                for policy, result in area_results.items():
                    row = {field: "" for field in fields_out}
                    row.update({"mode": mode_name, "area": area, "policy": policy})
                    row.update(result.summary)
                    writer.writerow(row)
    return path


def write_mode_usage_csv(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path) -> Path:
    keys = sorted(
        {
            key
            for mode_results in results.values()
            for area_results in mode_results.values()
            for result in area_results.values()
            for key in result.summary
            if key.startswith("mode_") and key.endswith("_count")
        }
    )
    path = out_dir / "compression_mode_usage.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["mode", "area", "policy", *keys])
        writer.writeheader()
        for mode_name, mode_results in results.items():
            for area, area_results in mode_results.items():
                for policy, result in area_results.items():
                    row = {key: result.summary.get(key, 0.0) for key in keys}
                    row.update({"mode": mode_name, "area": area, "policy": policy})
                    writer.writerow(row)
    return path


def _draw_axes(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], title: str, x_label: str, y_label: str) -> None:
    left, top, right, bottom = box
    draw.text((left, 24), title, font=_font(20, True), fill="#0f172a")
    draw.line((left, bottom, right, bottom), fill="#334155", width=2)
    draw.line((left, top, left, bottom), fill="#334155", width=2)
    draw.text(((left + right) // 2 - 90, bottom + 30), x_label, font=_font(13), fill="#334155")
    draw.text((left, top - 26), y_label, font=_font(13), fill="#334155")
    for frac in (0.25, 0.5, 0.75):
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill="#e2e8f0", width=1)


def save_finished_by_area(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path) -> Path:
    path = out_dir / "figures" / "finished_by_area_greedy_mcts.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1180, 620), "white")
    draw = ImageDraw.Draw(image)
    box = (90, 85, 1090, 500)
    title = "Finished requests by area: fixed semantic vs adaptive semantic"
    _draw_axes(draw, box, title, "area side length (m)", "finished requests")
    modes = ["fixed_paper_like", "adaptive_semantic", "nonsemantic"]
    areas = sorted({area for mode_results in results.values() for area in mode_results})
    policies = list(
        dict.fromkeys(
            policy
            for mode_results in results.values()
            for area_results in mode_results.values()
            for policy in area_results
        )
    )
    colors = {
        "fixed_paper_like": "#2563eb",
        "adaptive_semantic": "#16a34a",
        "nonsemantic": "#64748b",
    }
    values = [
        results[mode][area][policy].summary["finished"]
        for area in areas
        for policy in policies
        for mode in modes
        if mode in results and area in results[mode] and policy in results[mode][area]
    ]
    max_value = max(values) * 1.12 if values else 1.0
    group_width = (box[2] - box[0]) / max(len(areas), 1)
    bar_width = group_width / (len(policies) * len(modes) + 2)
    for area_idx, area in enumerate(areas):
        base_x = box[0] + area_idx * group_width + bar_width
        for policy_idx, policy in enumerate(policies):
            for mode_idx, mode in enumerate(modes):
                if mode not in results or area not in results[mode] or policy not in results[mode][area]:
                    continue
                value = results[mode][area][policy].summary["finished"]
                offset = (policy_idx * len(modes) + mode_idx) * bar_width
                x0 = int(base_x + offset)
                x1 = int(x0 + bar_width * 0.78)
                y0 = int(box[3] - value / max_value * (box[3] - box[1]))
                draw.rectangle((x0, y0, x1, box[3]), fill=colors[mode])
        draw.text((int(box[0] + area_idx * group_width + group_width * 0.35), box[3] + 12), str(area), font=_font(12), fill="#334155")
    y_font = _font(11)
    for frac in (0.0, 0.5, 1.0):
        value = max_value * frac
        y = int(box[3] - frac * (box[3] - box[1]))
        draw.text((30, y - 8), f"{value:.0f}", font=y_font, fill="#475569")
    x = 110
    for mode in modes:
        draw.rectangle((x, 565, x + 20, 585), fill=colors[mode])
        draw.text((x + 28, 563), mode, font=_font(13), fill="#334155")
        x += 230
    draw.text((780, 563), f"policy order: {', '.join(policies)}", font=_font(11), fill="#64748b")
    image.save(path)
    return path


def quality_axis_limits(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    span = max(max(finite) - min(finite), 0.02)
    padding = max(0.02, span * 0.08)
    lower, upper = max(0.0, min(finite) - padding), min(1.0, max(finite) + padding)
    if upper - lower < 1e-9:
        upper = min(1.0, lower + 0.05)
        lower = max(0.0, upper - 0.05)
    return lower, upper


def save_tradeoff(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path, area: int = 300) -> Path:
    path = out_dir / "figures" / f"latency_quality_tradeoff_{area}m.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (980, 620), "white")
    draw = ImageDraw.Draw(image)
    box = (105, 85, 880, 500)
    _draw_axes(draw, box, f"Latency-quality trade-off at {area} x {area} m", "average time cost (s)", "semantic quality")
    modes = ["nonsemantic", "fixed_paper_like", "adaptive_semantic"]
    colors = {"nonsemantic": "#64748b", "fixed_paper_like": "#2563eb", "adaptive_semantic": "#16a34a"}
    points: list[tuple[str, str, float, float]] = []
    for mode in modes:
        if mode not in results or area not in results[mode]:
            continue
        for policy in results[mode][area]:
            if policy not in results[mode][area]:
                continue
            summary = results[mode][area][policy].summary
            points.append((mode, policy, summary["avg_time"], summary["semantic_quality"]))
    xs = [point[2] for point in points] or [1.0]
    ys = [point[3] for point in points] or [1.0]
    x_min, x_max = min(xs) * 0.88, max(xs) * 1.08
    y_min, y_max = quality_axis_limits(ys)
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 0.05

    def xy(x_value: float, y_value: float) -> tuple[int, int]:
        x = box[0] + int((x_value - x_min) / (x_max - x_min) * (box[2] - box[0]))
        y = box[3] - int((y_value - y_min) / (y_max - y_min) * (box[3] - box[1]))
        return x, y

    for mode, policy, avg_time, quality in points:
        x, y = xy(avg_time, quality)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colors[mode])
        draw.text((x + 8, y - 8), policy, font=_font(10), fill="#334155")
    for frac in (0.0, 0.5, 1.0):
        x_value = x_min + (x_max - x_min) * frac
        x = box[0] + int(frac * (box[2] - box[0]))
        draw.text((x - 16, box[3] + 10), f"{x_value:.1f}", font=_font(11), fill="#475569")
        y_value = y_min + (y_max - y_min) * frac
        y = box[3] - int(frac * (box[3] - box[1]))
        draw.text((35, y - 7), f"{y_value:.2f}", font=_font(11), fill="#475569")
    x = 120
    for mode in modes:
        draw.ellipse((x, 565, x + 13, 578), fill=colors[mode])
        draw.text((x + 22, 561), mode, font=_font(13), fill="#334155")
        x += 230
    image.save(path)
    return path


def save_adaptive_mode_usage(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path, area: int = 300) -> Path:
    path = out_dir / "figures" / f"adaptive_mode_usage_{area}m.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1080, 600), "white")
    draw = ImageDraw.Draw(image)
    box = (90, 80, 1000, 465)
    _draw_axes(draw, box, f"Adaptive compression mode usage at {area} x {area} m", "policy", "selected requests")
    mode_keys = [
        "mode_emergency_count",
        "mode_low_count",
        "mode_medium_count",
        "mode_paper_like_count",
        "mode_high_count",
    ]
    labels = ["emergency", "low", "medium", "paper_like", "high"]
    colors = ["#0f766e", "#22c55e", "#84cc16", "#eab308", "#f97316"]
    rows = results["adaptive_semantic"][area]
    policies = list(rows)
    max_value = max((sum(result.summary.get(key, 0.0) for key in mode_keys) for result in rows.values()), default=1.0) * 1.1
    group_width = (box[2] - box[0]) / max(len(policies), 1)
    for idx, policy in enumerate(policies):
        x0 = int(box[0] + idx * group_width + group_width * 0.22)
        x1 = int(box[0] + (idx + 1) * group_width - group_width * 0.22)
        y_cursor = box[3]
        for key, color in zip(mode_keys, colors):
            value = rows[policy].summary.get(key, 0.0)
            height = int(value / max(max_value, 1e-9) * (box[3] - box[1]))
            y0 = y_cursor - height
            draw.rectangle((x0, y0, x1, y_cursor), fill=color)
            y_cursor = y0
        draw.text((x0 - 5, box[3] + 12), policy, font=_font(11), fill="#334155")
    for frac in (0.0, 0.5, 1.0):
        value = max_value * frac
        y = int(box[3] - frac * (box[3] - box[1]))
        draw.text((28, y - 8), f"{value:.0f}", font=_font(11), fill="#475569")
    x = 100
    for label, color in zip(labels, colors):
        draw.rectangle((x, 520, x + 18, 538), fill=color)
        draw.text((x + 25, 517), label, font=_font(12), fill="#334155")
        x += 170
    image.save(path)
    return path


def select_representative_area(results: Mapping[str, Mapping[int, object]]) -> int:
    areas = sorted({area for mode_results in results.values() for area in mode_results})
    if not areas:
        raise ValueError("Cannot select a representative area from empty results")
    return 300 if 300 in areas else areas[len(areas) // 2]


def save_figures(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path) -> Dict[str, str]:
    area = select_representative_area(results)
    paths = [
        save_finished_by_area(results, out_dir),
        save_tradeoff(results, out_dir, area),
        save_adaptive_mode_usage(results, out_dir, area),
    ]
    return {path.name: str(path) for path in paths}


def pct_change(new: float, old: float) -> float:
    return (new - old) / old * 100.0 if old else float("nan")


def format_row(summary: dict[str, float]) -> str:
    return (
        f"{summary['finished']:.1f} | {summary['avg_time']:.2f} | "
        f"{summary['flight_energy_per_req']:.1f} | {summary['semantic_quality']:.3f} | "
        f"{summary['semantic_payload_ratio']:.3f}"
    )


def write_analysis(
    results: Dict[str, Dict[int, Dict[str, SimulationResult]]],
    quality_rows: Sequence[dict[str, float | str]],
    neural_anchor: Optional[dict[str, object]],
    neural_quality_mode: str,
    out_dir: Path,
    elapsed: float,
    repeats: int,
    t_slots: int,
    metadata_path: Path,
    threshold_configuration: Optional[Mapping[str, object]] = None,
) -> Path:
    area = select_representative_area(results)
    has_neural_multi_rate = _has_multi_rate_anchor(neural_anchor)
    if has_neural_multi_rate and neural_quality_mode == "selection":
        neural_reflection = (
            "- Neural encoder/decoder reflection: all five adaptive levels use measured payload and mIoU "
            "from one trained scalable neural codec for mode selection."
        )
    elif has_neural_multi_rate:
        neural_reflection = (
            "- Neural encoder/decoder reflection: all five measured neural payloads and mIoUs are recorded, "
            "while `record_only` deliberately keeps the quality CSV values for selection."
        )
    else:
        neural_reflection = (
            "- Neural encoder/decoder reflection: only the paper-like payload is anchored to a trained "
            "codec; the remaining levels retain Cityscapes label-proxy quality."
        )
    scope_note = (
        "- The result is a trade-off study, not a claim that the original AirTalking neural network was "
        "reproduced. Its weights and training recipe are not public; this run uses the explicitly specified "
        "paper-inspired scalable codec."
        if has_neural_multi_rate
        else "- The result is a trade-off study, not a claim that the original AirTalking paper is fully "
        "reproduced. The authors' encoder/decoder is not public and some adaptive qualities use a label proxy."
    )
    lines: List[str] = [
        "# Adaptive semantic compression follow-up research",
        "",
        "## What changed",
        "",
        "This experiment moves beyond the earlier SINR-only probe. The adaptive compression policy is now inside the UAV scheduler, so each candidate UAV relay action is evaluated with the payload ratio selected for that candidate link condition.",
        "",
        "## Data and settings",
        "",
        "- Public semantic data: Cityscapes gtFine train/val label maps.",
        f"- Base simulator parameters: `{metadata_path.resolve()}`.",
        f"- Repeats: {repeats}; simulation slots per repeat: {t_slots}.",
        "- Compared modes: nonsemantic raw payload, fixed Cityscapes paper-like semantic payload, and channel-aware adaptive semantic payload.",
        neural_reflection,
        "",
        "## Compression table",
        "",
        "| mode | payload ratio | mean IoU |",
        "|---|---:|---:|",
    ]
    for row in quality_rows:
        lines.append(f"| {row['mode']} | {float(row['feature_ratio_mean']):.6f} | {float(row['mean_iou_mean']):.3f} |")

    if neural_anchor is not None:
        lines.extend(["", "## Neural encoder/decoder anchor", ""])
        optional_fields = (
            ("rho_c", ("rho_c_feature_uncompressed_mean", "rho_uint8", "payload_ratio"), 6),
            ("pixel accuracy", ("pixel_accuracy_best", "pixel_accuracy_final"), 4),
            ("mIoU", ("semantic_quality_miou_best", "semantic_quality_miou_final", *MIOU_KEYS), 4),
        )
        for label, aliases, decimals in optional_fields:
            value, _ = _numeric_alias(neural_anchor, aliases, label, required=False)
            if value is not None:
                lines.append(f"- {label}: {value:.{decimals}f}")
        timing = neural_anchor.get("timing")
        if isinstance(timing, Mapping):
            encode_ms, _ = _numeric_alias(timing, ("encode_ms_median",), "encode time", required=False)
            decode_ms, _ = _numeric_alias(timing, ("decode_ms_median",), "decode time", required=False)
            if encode_ms is not None and decode_ms is not None:
                lines.append(f"- encode/decode median time: {encode_ms:.2f} ms / {decode_ms:.2f} ms")
        if neural_quality_mode == "record_only":
            quality_meaning = (
                "neural payloads and neural mIoU are recorded for every rate, but selection quality "
                "remains the input quality CSV"
            )
        else:
            quality_meaning = "measured neural mIoU is used both for recording and mode selection"
        lines.append(f"- adaptive quality mode: {neural_quality_mode}; {quality_meaning}.")
        if threshold_configuration is not None:
            lines.append(
                "- SINR-to-quality threshold rule: "
                f"{threshold_configuration.get('resolved_rule')} "
                f"({threshold_configuration.get('quality_source')}); actual thresholds are preserved in run metadata."
            )

    lines.extend(
        [
            "",
            f"## {area} x {area} m result table",
            "",
            "| policy | mode | finished | avg time (s) | flight J/req | semantic quality | payload ratio |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    available_policies = [
        policy
        for policy in results.get("adaptive_semantic", {}).get(area, {})
        if all(mode in results and area in results[mode] and policy in results[mode][area] for mode in ["nonsemantic", "fixed_paper_like", "adaptive_semantic"])
    ]
    for policy in available_policies:
        for mode in ["nonsemantic", "fixed_paper_like", "adaptive_semantic"]:
            summary = results[mode][area][policy].summary
            lines.append(f"| {policy} | {mode} | {format_row(summary)} |")

    lines.extend(["", f"## Adaptive vs fixed semantic at {area} x {area} m", ""])
    lines.append("| policy | finished change | avg time change | quality change | payload-ratio change |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy in available_policies:
        fixed = results["fixed_paper_like"][area][policy].summary
        adaptive = results["adaptive_semantic"][area][policy].summary
        lines.append(
            "| "
            f"{policy} | {pct_change(adaptive['finished'], fixed['finished']):.1f}% | "
            f"{pct_change(adaptive['avg_time'], fixed['avg_time']):.1f}% | "
            f"{adaptive['semantic_quality'] - fixed['semantic_quality']:+.3f} | "
            f"{pct_change(adaptive['semantic_payload_ratio'], fixed['semantic_payload_ratio']):.1f}% |"
        )

    anchor_policy = "Greedy" if "Greedy" in available_policies else available_policies[0]
    second_policy = "MCTS" if "MCTS" in available_policies else available_policies[-1]
    fixed_greedy = results["fixed_paper_like"][area][anchor_policy].summary
    adaptive_greedy = results["adaptive_semantic"][area][anchor_policy].summary
    fixed_mcts = results["fixed_paper_like"][area][second_policy].summary
    adaptive_mcts = results["adaptive_semantic"][area][second_policy].summary
    lines.extend(
        [
            "",
            "## Main interpretation",
            "",
            f"- {anchor_policy} at {area} m: adaptive finished {adaptive_greedy['finished']:.1f} requests vs fixed {fixed_greedy['finished']:.1f}; average time changed from {fixed_greedy['avg_time']:.2f}s to {adaptive_greedy['avg_time']:.2f}s.",
            f"- {second_policy} at {area} m: adaptive finished {adaptive_mcts['finished']:.1f} requests vs fixed {fixed_mcts['finished']:.1f}; average time changed from {fixed_mcts['avg_time']:.2f}s to {adaptive_mcts['avg_time']:.2f}s.",
            scope_note,
            "- Adaptive compression improves scheduling latency by changing the payload-quality choice per candidate link. In weak links it can choose a smaller payload than the fixed paper-like mode; in stronger links it may spend more payload to preserve semantic quality.",
            "- Therefore the key claim is not that adaptive always lowers the average payload ratio. The measured gain is a latency/completion improvement with a controlled semantic-quality drop.",
            "",
            "## Generated artifacts",
            "",
            "- `summary_metrics.csv`: final numeric comparison for all modes, areas, and policies.",
            "- `compression_mode_usage.csv`: selected adaptive mode counts.",
            "- `timeseries_and_sinr_samples.npz`: time series and SINR samples for later plotting.",
            "- `figures/`: result figures for area scaling, latency-quality trade-off, and adaptive mode usage.",
            "",
            f"Elapsed wall time: {elapsed:.1f} seconds.",
        ]
    )
    path = out_dir / "adaptive_followup_research_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full adaptive semantic compression follow-up experiments.")
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--quality", default=str(DEFAULT_QUALITY))
    parser.add_argument("--neural-summary", default=str(DEFAULT_NEURAL_SUMMARY))
    parser.add_argument("--neural-quality-mode", choices=["record_only", "selection"], default="selection")
    parser.add_argument(
        "--adaptive-threshold-rule",
        "--threshold-rule",
        dest="adaptive_threshold_rule",
        choices=["auto", "measured_ordered", "ordered_mode_quality", "legacy_proxy", "explicit"],
        default="auto",
        help=(
            "SINR-bin quality rule. auto uses measured_ordered for a selected multi-rate neural "
            "frontier and legacy_proxy otherwise."
        ),
    )
    parser.add_argument(
        "--adaptive-quality-thresholds",
        "--quality-thresholds",
        dest="adaptive_quality_thresholds",
        default=None,
        help="Five comma-separated qualities for --adaptive-threshold-rule explicit.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--t-slots", type=int, default=None)
    parser.add_argument("--areas", default=",".join(str(area) for area in AREAS))
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="ProcessPool workers for independent repeats. Default 1 preserves sequential execution.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    try:
        areas = tuple(int(value.strip()) for value in args.areas.split(",") if value.strip())
    except ValueError:
        parser.error("--areas must contain comma-separated integers")
    policies = tuple(value.strip() for value in args.policies.split(",") if value.strip())
    if not areas:
        parser.error("--areas must contain at least one area")
    if any(area <= 0 for area in areas):
        parser.error("--areas values must all be positive")
    if len(set(areas)) != len(areas):
        parser.error("--areas must not contain duplicates")
    if not policies:
        parser.error("--policies must contain at least one policy")
    if len(set(policies)) != len(policies):
        parser.error("--policies must not contain duplicates")
    unknown_policies = [policy for policy in policies if policy not in POLICIES]
    if unknown_policies:
        parser.error(f"Unknown policies {unknown_policies}; supported policies are {list(POLICIES)}")
    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.t_slots is not None and args.t_slots < 1:
        parser.error("--t-slots must be at least 1")
    try:
        explicit_thresholds = parse_quality_thresholds(args.adaptive_quality_thresholds)
    except ValueError as exc:
        parser.error(str(exc))

    metadata_path = Path(args.metadata)
    quality_path = Path(args.quality)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    neural_path = Path(args.neural_summary)
    metadata_out = out_dir / "run_metadata.json"
    runner_source = Path(__file__).resolve()
    runner_snapshot = out_dir / "runner_source_snapshot.py"
    shutil.copy2(runner_source, runner_snapshot)
    validator_source = Path(result_validator.__file__).resolve()
    validator_snapshot = out_dir / "validator_source_snapshot.py"
    shutil.copy2(validator_source, validator_snapshot)
    started_at = utc_now()
    started = time.perf_counter()
    environment = command_environment(sys.argv)
    metadata_state: dict[str, object] = {
        "schema_version": 2,
        "status": "running",
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "argv": environment["argv"],
        "command_windows": environment["command_windows"],
        "environment": environment,
        "runner_source": {
            "path": str(runner_source),
            "sha256": sha256_file(runner_source),
            "snapshot_path": str(runner_snapshot.resolve()),
            "snapshot_sha256": sha256_file(runner_snapshot),
        },
        "validator_source": {
            "path": str(validator_source),
            "sha256": sha256_file(validator_source),
            "snapshot_path": str(validator_snapshot.resolve()),
            "snapshot_sha256": sha256_file(validator_snapshot),
        },
        "metadata_path": str(metadata_out.resolve()),
    }
    write_metadata_json(metadata_out, metadata_state)

    try:
        input_files = {
            "metadata": file_provenance(metadata_path),
            "quality": file_provenance(quality_path),
            "neural_summary": file_provenance(neural_path, required=False),
        }
        paper, assumed, _source_metadata = load_base_params(metadata_path)
        repeats = args.repeats if args.repeats is not None else paper.repeats
        t_slots = args.t_slots if args.t_slots is not None else paper.t_slots
        if repeats < 1 or t_slots < 1:
            raise ValueError("Resolved repeats and t_slots must both be at least 1")
        paper = PaperParams(**{**paper.__dict__, "repeats": repeats, "t_slots": t_slots})
        neural_anchor = load_neural_anchor(neural_path)
        paper, neural_bitrate_application = apply_neural_bitrates(paper, neural_anchor)
        quality_rows = apply_neural_anchor(
            load_quality_rows(quality_path), neural_anchor, args.neural_quality_mode
        )
        target_thresholds, threshold_configuration = resolve_adaptive_thresholds(
            quality_rows,
            neural_anchor,
            args.neural_quality_mode,
            args.adaptive_threshold_rule,
            explicit_thresholds,
        )
        fixed_profile, adaptive_profile = build_profiles(quality_rows, target_thresholds)
        reachable_modes = reachable_adaptive_modes(adaptive_profile)
        threshold_configuration["reachable_modes_across_configured_sinr_bins"] = reachable_modes
        threshold_configuration["reachable_mode_count"] = len(reachable_modes)
        if threshold_configuration["resolved_rule"] in {"measured_ordered", "ordered_mode_quality"} and len(reachable_modes) < 2:
            raise ValueError(
                "Adaptive threshold configuration collapses to fewer than two reachable modes across the SINR range"
            )

        repeat_metric_rows: List[dict[str, object]] = []
        executor_context = (
            nullcontext(None)
            if args.workers == 1
            else ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        )
        with executor_context as executor:
            results: Dict[str, Dict[int, Dict[str, SimulationResult]]] = {
                "nonsemantic": run_mode(
                    "nonsemantic", False, None, paper, assumed, repeats, areas, policies,
                    repeat_metrics=repeat_metric_rows, workers=args.workers, executor=executor,
                ),
                "fixed_paper_like": run_mode(
                    "fixed_paper_like", True, fixed_profile, paper, assumed, repeats, areas, policies,
                    repeat_metrics=repeat_metric_rows, workers=args.workers, executor=executor,
                ),
                "adaptive_semantic": run_mode(
                    "adaptive_semantic", True, adaptive_profile, paper, assumed, repeats, areas, policies,
                    repeat_metrics=repeat_metric_rows, workers=args.workers, executor=executor,
                ),
            }

        summary_csv = write_summary_csv(results, out_dir)
        repeat_metrics_csv = write_repeat_metrics_csv(repeat_metric_rows, out_dir)
        statistical_summary_csv = write_statistical_summary_csv(repeat_metric_rows, out_dir)
        usage_csv = write_mode_usage_csv(results, out_dir)
        npz_path = write_timeseries_npz(results, out_dir)
        figure_paths = save_figures(results, out_dir)
        elapsed = time.perf_counter() - started
        report_path = write_analysis(
            results,
            quality_rows,
            neural_anchor,
            args.neural_quality_mode,
            out_dir,
            elapsed,
            repeats,
            t_slots,
            metadata_path,
            threshold_configuration,
        )
        validation_path = out_dir / "result_validation.json"
        validation_rows, validation_columns, validation_load_errors = result_validator.load_rows_with_schema(summary_csv)
        validation_result = result_validator.validate(
            validation_rows,
            available_columns=validation_columns,
            pre_errors=validation_load_errors,
            source_summary_path=summary_csv,
            expected_areas=areas,
            expected_policies=policies,
        )
        result_validator.write_result(validation_path, validation_result)
        metadata_state["result_validation"] = {
            "path": str(validation_path.resolve()),
            "passed": validation_result["passed"],
            "schema_version": validation_result["schema_version"],
        }
        require_integrity_validation(validation_result)

        output_paths: dict[str, Path] = {
            "summary_metrics_csv": summary_csv,
            "repeat_metrics_csv": repeat_metrics_csv,
            "statistical_summary_csv": statistical_summary_csv,
            "compression_mode_usage_csv": usage_csv,
            "timeseries_and_sinr_samples_npz": npz_path,
            "adaptive_followup_research_report_md": report_path,
            "result_validation_json": validation_path,
            "runner_source_snapshot": runner_snapshot,
            "validator_source_snapshot": validator_snapshot,
        }
        output_paths.update({f"figure_{name}": Path(path) for name, path in figure_paths.items()})
        artifacts = {name: artifact_provenance(path) for name, path in output_paths.items()}
        completed_at = utc_now()
        metadata_state.update(
            {
                "status": "completed",
                "completed_at_utc": completed_at,
                "elapsed_seconds": elapsed,
                "input_files": input_files,
                "source_metadata": str(metadata_path.resolve()),
                "source_quality": str(quality_path.resolve()),
                "source_neural_encoder_decoder": str(neural_path.resolve()) if neural_anchor is not None else None,
                "neural_encoder_decoder_anchor": sanitize_neural_anchor_metadata(neural_anchor),
                "neural_quality_mode": args.neural_quality_mode,
                "neural_quality_mode_semantics": (
                    "record_only preserves quality CSV values for mode selection and records measured neural mIoU separately"
                    if args.neural_quality_mode == "record_only"
                    else "selection uses measured neural mIoU as the mode selection quality"
                ),
                "neural_bitrate_application": neural_bitrate_application,
                "adaptive_threshold_configuration": threshold_configuration,
                "base_paper_params": paper.__dict__,
                "base_assumed_params": assumed.__dict__,
                "source_reproduction_metadata_embedded": False,
                "source_reproduction_metadata_note": (
                    "Only parsed parameter snapshots are embedded; nested source artifact paths are excluded."
                ),
                "profiles": {
                    "fixed_paper_like": profile_to_metadata(fixed_profile),
                    "adaptive_semantic": profile_to_metadata(adaptive_profile),
                },
                "areas": list(areas),
                "policies": list(policies),
                "representative_area": select_representative_area(results),
                "workers": args.workers,
                "repeat_metrics_csv": str(repeat_metrics_csv.resolve()),
                "statistical_summary_csv": str(statistical_summary_csv.resolve()),
                "ci95_method": "two-sided Student t; n=1 uses zero-width descriptive interval",
                "artifacts": artifacts,
            }
        )
        write_metadata_json(metadata_out, metadata_state)
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir.resolve()),
                    "status": metadata_state["status"],
                    "validation_passed": validation_result["passed"],
                    "metadata": str(metadata_out.resolve()),
                    "artifacts": artifacts,
                    "workers": args.workers,
                    "elapsed_seconds": elapsed,
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except BaseException as exc:
        metadata_state.update(
            {
                "status": "failed",
                "completed_at_utc": utc_now(),
                "elapsed_seconds": time.perf_counter() - started,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_metadata_json(metadata_out, metadata_state)
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
