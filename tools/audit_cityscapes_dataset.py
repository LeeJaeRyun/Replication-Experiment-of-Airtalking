"""Audit a local Cityscapes tree and write reproducible provenance artifacts.

The tool is intentionally single-process.  A full audit already reads every RGB
and labelIds file for hashing and every labelIds image for a pixel histogram;
additional workers tend to compete with training jobs for disk bandwidth.

Only the Python standard library, Pillow, and NumPy are required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import PIL
from PIL import Image


SCHEMA_VERSION = "cityscapes-dataset-audit/v1"
SPLITS = ("train", "val", "test")
ARTIFACT_SUFFIXES = {
    "rgb": "_leftImg8bit.png",
    "gt_color": "_gtFine_color.png",
    "gt_instanceIds": "_gtFine_instanceIds.png",
    "gt_labelIds": "_gtFine_labelIds.png",
    "gt_polygons": "_gtFine_polygons.json",
}
PNG_KINDS = ("rgb", "gt_color", "gt_instanceIds", "gt_labelIds")
GT_KINDS = ("gt_color", "gt_instanceIds", "gt_labelIds", "gt_polygons")

# Official Cityscapes raw labelId -> 19-class trainId mapping.
CITYSCAPES_CLASSES = (
    (0, 7, "road"),
    (1, 8, "sidewalk"),
    (2, 11, "building"),
    (3, 12, "wall"),
    (4, 13, "fence"),
    (5, 17, "pole"),
    (6, 19, "traffic light"),
    (7, 20, "traffic sign"),
    (8, 21, "vegetation"),
    (9, 22, "terrain"),
    (10, 23, "sky"),
    (11, 24, "person"),
    (12, 25, "rider"),
    (13, 26, "car"),
    (14, 27, "truck"),
    (15, 28, "bus"),
    (16, 31, "train"),
    (17, 32, "motorcycle"),
    (18, 33, "bicycle"),
)
RAW_TO_TRAIN = {raw_id: train_id for train_id, raw_id, _ in CITYSCAPES_CLASSES}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Progress:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.started = time.perf_counter()

    def log(self, message: str) -> None:
        if not self.quiet:
            elapsed = time.perf_counter() - self.started
            print(f"[cityscapes-audit +{elapsed:,.1f}s] {message}", file=sys.stderr, flush=True)


def find_cityscapes_roots(dataset_root: Path) -> tuple[Path, Path]:
    candidates = (
        (dataset_root / "leftImg8bit", dataset_root / "gtFine"),
        (
            dataset_root / "leftImg8bit_trainvaltest" / "leftImg8bit",
            dataset_root / "gtFine_trainvaltest" / "gtFine",
        ),
    )
    for left_root, gt_root in candidates:
        if left_root.is_dir() and gt_root.is_dir():
            return left_root, gt_root
    raise FileNotFoundError(
        "Could not find leftImg8bit and gtFine roots below "
        f"{dataset_root}. Supported layouts are <root>/leftImg8bit + <root>/gtFine "
        "and the official *_trainvaltest wrapper directories."
    )


def _key_for(path: Path, split_root: Path, suffix: str) -> tuple[str, str]:
    relative = path.relative_to(split_root)
    if len(relative.parts) != 2:
        raise ValueError(
            f"expected <city>/<file> below {split_root}, got {relative.as_posix()}"
        )
    city = relative.parts[0]
    if not path.name.endswith(suffix):
        raise ValueError(f"file does not end with {suffix}: {path}")
    stem = path.name[: -len(suffix)]
    return city, stem


def collect_artifacts(
    left_root: Path, gt_root: Path
) -> tuple[
    dict[str, dict[str, dict[tuple[str, str], Path]]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    artifacts: dict[str, dict[str, dict[tuple[str, str], Path]]] = {
        split: {kind: {} for kind in ARTIFACT_SUFFIXES} for split in SPLITS
    }
    duplicates: list[dict[str, str]] = []
    layout_errors: list[dict[str, str]] = []

    for split in SPLITS:
        for kind, suffix in ARTIFACT_SUFFIXES.items():
            type_root = left_root if kind == "rgb" else gt_root
            split_root = type_root / split
            if not split_root.is_dir():
                layout_errors.append(
                    {"split": split, "kind": kind, "error": f"missing directory: {split_root}"}
                )
                continue
            for path in sorted(split_root.rglob(f"*{suffix}")):
                if not path.is_file():
                    continue
                try:
                    key = _key_for(path, split_root, suffix)
                except ValueError as exc:
                    layout_errors.append(
                        {"split": split, "kind": kind, "path": str(path), "error": str(exc)}
                    )
                    continue
                if key in artifacts[split][kind]:
                    duplicates.append(
                        {
                            "split": split,
                            "kind": kind,
                            "sample": f"{key[0]}/{key[1]}",
                            "first": str(artifacts[split][kind][key]),
                            "duplicate": str(path),
                        }
                    )
                else:
                    artifacts[split][kind][key] = path
    return artifacts, duplicates, layout_errors


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def inventory_files(dataset_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    root_resolved = dataset_root.resolve()
    output_resolved = output_dir.resolve()
    if _is_relative_to(output_resolved, root_resolved):
        raise ValueError(
            "output directory must be outside dataset root so generated files cannot "
            "change the dataset fingerprint"
        )
    records: list[dict[str, Any]] = []
    for path in sorted(dataset_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file():
            records.append(
                {
                    "relative_path": path.relative_to(dataset_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "content_sha256": None,
                }
            )
    return records


def sha256_stream(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda item: str(item))}


def _png_stat_template() -> dict[str, Any]:
    return {
        "files_discovered": 0,
        "files_inspected": 0,
        "dimensions": Counter(),
        "modes": Counter(),
        "read_errors": [],
    }


def _inspect_png_header(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as image:
        return tuple(int(value) for value in image.size), str(image.mode)


def _inspect_label_and_count(path: Path) -> tuple[tuple[int, int], str, Counter[int], str]:
    # labelIds PNGs are small enough to read once into memory.  The same encoded
    # bytes are used for content SHA-256 and Pillow decoding, avoiding two disk reads.
    encoded = path.read_bytes()
    content_sha256 = hashlib.sha256(encoded).hexdigest()
    with Image.open(io.BytesIO(encoded)) as image:
        size = tuple(int(value) for value in image.size)
        mode = str(image.mode)
        array = np.asarray(image)
        if array.ndim != 2:
            raise ValueError(f"labelIds must be single-channel, got array shape {array.shape}")
        values, counts = np.unique(array, return_counts=True)
    histogram = Counter(
        {int(value): int(count) for value, count in zip(values.tolist(), counts.tolist())}
    )
    return size, mode, histogram, content_sha256


def _parse_polygon(path: Path) -> tuple[tuple[int, int] | None, int, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    schema_errors: list[str] = []
    if not isinstance(document, dict):
        return None, 0, ["top-level JSON value is not an object"]
    width = document.get("imgWidth")
    height = document.get("imgHeight")
    if not isinstance(width, int) or not isinstance(height, int):
        dimensions = None
        schema_errors.append("imgWidth/imgHeight are not integers")
    else:
        dimensions = (width, height)
    objects = document.get("objects")
    if not isinstance(objects, list):
        return dimensions, 0, schema_errors + ["objects is not a list"]
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            schema_errors.append(f"objects[{index}] is not an object")
            continue
        if not isinstance(obj.get("label"), str):
            schema_errors.append(f"objects[{index}].label is not a string")
        polygon = obj.get("polygon")
        if not isinstance(polygon, list):
            schema_errors.append(f"objects[{index}].polygon is not a list")
    return dimensions, len(objects), schema_errors


def _sample_id(key: tuple[str, str]) -> str:
    return f"{key[0]}/{key[1]}"


def _all_intersections(values: dict[str, set[str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for first_index, first in enumerate(SPLITS):
        for second in SPLITS[first_index + 1 :]:
            result[f"{first}__{second}"] = sorted(values[first] & values[second])
    return result


def _histogram_summary(raw_histogram: Counter[int]) -> dict[str, Any]:
    total = int(sum(raw_histogram.values()))
    class_counts = {
        str(train_id): int(raw_histogram.get(raw_id, 0))
        for train_id, raw_id, _ in CITYSCAPES_CLASSES
    }
    valid = int(sum(class_counts.values()))
    ignored = total - valid
    return {
        "total_pixels": total,
        "valid_19_class_pixels": valid,
        "ignored_pixels": ignored,
        "ignore_ratio": (ignored / total) if total else None,
        "valid_19_class_ratio": (valid / total) if total else None,
        "raw_label_id_histogram": {
            str(raw_id): int(raw_histogram[raw_id]) for raw_id in sorted(raw_histogram)
        },
        "train_id_histogram": class_counts,
    }


def _fingerprint(inventory: list[dict[str, Any]]) -> str:
    """Hash an explicitly documented canonical stream of inventory records."""
    digest = hashlib.sha256()
    digest.update(b"cityscapes-audit-fingerprint-v1\0")
    for record in inventory:
        canonical = json.dumps(
            [record["relative_path"], record["size_bytes"], record["content_sha256"]],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _append_error(errors: list[dict[str, Any]], code: str, detail: Any) -> None:
    errors.append({"code": code, "detail": detail})


def audit_dataset(
    dataset_root: Path,
    output_dir: Path,
    *,
    hash_policy: str = "rgb-labelids",
    progress_every: int = 250,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the audit and return a strict-JSON-compatible manifest dictionary."""
    if hash_policy not in {"rgb-labelids", "all"}:
        raise ValueError("hash_policy must be 'rgb-labelids' or 'all'")
    if progress_every < 1:
        raise ValueError("progress_every must be >= 1")

    progress = Progress(quiet=quiet)
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    wall_started = utc_now()
    left_root, gt_root = find_cityscapes_roots(dataset_root)
    progress.log(f"found leftImg8bit={left_root} and gtFine={gt_root}")

    artifacts, duplicates, layout_errors = collect_artifacts(left_root, gt_root)
    inventory = inventory_files(dataset_root, output_dir)
    inventory_by_path = {record["relative_path"]: record for record in inventory}
    progress.log(
        f"discovered {len(inventory):,} total files and "
        f"{sum(len(artifacts[s]['rgb']) for s in SPLITS):,} RGB samples"
    )

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if duplicates:
        _append_error(errors, "duplicate_artifact_keys", duplicates)
    if layout_errors:
        _append_error(errors, "layout_errors", layout_errors)

    split_results: dict[str, Any] = {}
    png_dimensions: dict[str, dict[str, dict[tuple[str, str], tuple[int, int]]]] = {
        split: {kind: {} for kind in PNG_KINDS} for split in SPLITS
    }
    polygon_dimensions: dict[str, dict[tuple[str, str], tuple[int, int]]] = {
        split: {} for split in SPLITS
    }
    raw_histograms = {split: Counter() for split in SPLITS}
    content_hash_errors: list[dict[str, str]] = []
    hashed_file_count = 0
    hashed_bytes = 0
    processed_pngs = 0
    total_pngs = sum(
        len(artifacts[split][kind]) for split in SPLITS for kind in PNG_KINDS
    )

    for split in SPLITS:
        kind_keys = {kind: set(artifacts[split][kind]) for kind in ARTIFACT_SUFFIXES}
        rgb_keys = kind_keys["rgb"]
        missing_by_type = {
            kind: sorted(_sample_id(key) for key in rgb_keys - kind_keys[kind])
            for kind in GT_KINDS
        }
        orphan_by_type = {
            kind: sorted(_sample_id(key) for key in kind_keys[kind] - rgb_keys)
            for kind in GT_KINDS
        }
        all_sets_equal = all(kind_keys[kind] == rgb_keys for kind in GT_KINDS)
        complete_keys = set.intersection(*(kind_keys[kind] for kind in ARTIFACT_SUFFIXES))

        city_lists = {
            kind: sorted({city for city, _ in keys}) for kind, keys in kind_keys.items()
        }
        split_result: dict[str, Any] = {
            "counts": {kind: len(artifacts[split][kind]) for kind in ARTIFACT_SUFFIXES},
            "one_to_one": {
                "all_rgb_have_exactly_one_of_each_gt_type": all_sets_equal,
                "complete_sample_count": len(complete_keys),
                "missing_gt_for_rgb": missing_by_type,
                "orphan_gt_without_rgb": orphan_by_type,
            },
            "cities": {
                "rgb_city_count": len(city_lists["rgb"]),
                "rgb_city_list": city_lists["rgb"],
                "by_artifact_type": city_lists,
                "all_artifact_types_match_rgb_cities": all(
                    city_lists[kind] == city_lists["rgb"] for kind in GT_KINDS
                ),
            },
            "png_checks": {kind: _png_stat_template() for kind in PNG_KINDS},
            "polygon_json": {
                "files_discovered": len(artifacts[split]["gt_polygons"]),
                "files_parsed": 0,
                "object_count": 0,
                "parse_errors": [],
                "schema_errors": [],
            },
        }

        if not all_sets_equal:
            _append_error(
                errors,
                "one_to_one_correspondence_failed",
                {"split": split, "missing": missing_by_type, "orphan": orphan_by_type},
            )

        for kind in PNG_KINDS:
            png_stats = split_result["png_checks"][kind]
            png_stats["files_discovered"] = len(artifacts[split][kind])
            for key, path in artifacts[split][kind].items():
                try:
                    relative = path.relative_to(dataset_root).as_posix()
                    record = inventory_by_path[relative]
                    if kind == "gt_labelIds":
                        size, mode, histogram, content_sha256 = _inspect_label_and_count(path)
                        raw_histograms[split].update(histogram)
                        record["content_sha256"] = content_sha256
                        hashed_file_count += 1
                        hashed_bytes += int(record["size_bytes"])
                    else:
                        if kind == "rgb" or hash_policy == "all":
                            record["content_sha256"] = sha256_stream(path)
                            hashed_file_count += 1
                            hashed_bytes += int(record["size_bytes"])
                        size, mode = _inspect_png_header(path)
                    png_dimensions[split][kind][key] = size
                    png_stats["files_inspected"] += 1
                    png_stats["dimensions"][f"{size[0]}x{size[1]}"] += 1
                    png_stats["modes"][mode] += 1
                except Exception as exc:  # retain every path/error in the provenance artifact
                    detail = {
                        "path": path.relative_to(dataset_root).as_posix(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    png_stats["read_errors"].append(detail)
                    if kind in {"rgb", "gt_labelIds"} or hash_policy == "all":
                        content_hash_errors.append(detail)
                processed_pngs += 1
                if processed_pngs % progress_every == 0 or processed_pngs == total_pngs:
                    progress.log(f"inspected PNG {processed_pngs:,}/{total_pngs:,}")

        polygons = split_result["polygon_json"]
        for index, (key, path) in enumerate(artifacts[split]["gt_polygons"].items(), start=1):
            try:
                dimensions, object_count, schema_issues = _parse_polygon(path)
                polygons["files_parsed"] += 1
                polygons["object_count"] += object_count
                if dimensions is not None:
                    polygon_dimensions[split][key] = dimensions
                if schema_issues:
                    polygons["schema_errors"].append(
                        {
                            "path": path.relative_to(dataset_root).as_posix(),
                            "errors": schema_issues,
                        }
                    )
            except Exception as exc:
                polygons["parse_errors"].append(
                    {
                        "path": path.relative_to(dataset_root).as_posix(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if index % progress_every == 0 or index == len(artifacts[split]["gt_polygons"]):
                progress.log(
                    f"parsed {split} polygon JSON {index:,}/"
                    f"{len(artifacts[split]['gt_polygons']):,}"
                )

        dimension_mismatches: list[dict[str, Any]] = []
        for key in sorted(complete_keys):
            observed: dict[str, str] = {}
            for kind in PNG_KINDS:
                size = png_dimensions[split][kind].get(key)
                if size is not None:
                    observed[kind] = f"{size[0]}x{size[1]}"
            polygon_size = polygon_dimensions[split].get(key)
            if polygon_size is not None:
                observed["gt_polygons"] = f"{polygon_size[0]}x{polygon_size[1]}"
            if len(set(observed.values())) > 1:
                dimension_mismatches.append({"sample": _sample_id(key), "observed": observed})
        split_result["cross_artifact_dimensions"] = {
            "samples_compared": len(complete_keys),
            "all_dimensions_match": not dimension_mismatches,
            "mismatches": dimension_mismatches,
        }

        for kind in PNG_KINDS:
            png_stats = split_result["png_checks"][kind]
            png_stats["dimensions"] = _counter_dict(png_stats["dimensions"])
            png_stats["modes"] = _counter_dict(png_stats["modes"])
            if png_stats["read_errors"]:
                _append_error(
                    errors,
                    "png_read_errors",
                    {"split": split, "kind": kind, "errors": png_stats["read_errors"]},
                )
        if polygons["parse_errors"]:
            _append_error(
                errors, "polygon_json_parse_errors", {"split": split, **polygons}
            )
        if polygons["schema_errors"]:
            _append_error(
                errors,
                "polygon_json_schema_errors",
                {"split": split, "errors": polygons["schema_errors"]},
            )
        if dimension_mismatches:
            _append_error(
                errors,
                "cross_artifact_dimension_mismatch",
                {"split": split, "mismatches": dimension_mismatches},
            )
        split_result["semantic_label_pixels"] = _histogram_summary(raw_histograms[split])
        split_results[split] = split_result

    # Hash any selected file not reached through the canonical layout.  Such a
    # misplaced file is already a strict layout error, but it still belongs in the
    # promised content-hash scope and therefore must influence the fingerprint.
    for index, record in enumerate(inventory, start=1):
        relative = str(record["relative_path"])
        selected = hash_policy == "all" or relative.endswith(
            (ARTIFACT_SUFFIXES["rgb"], ARTIFACT_SUFFIXES["gt_labelIds"])
        )
        if not selected or record["content_sha256"] is not None:
            continue
        path = dataset_root / relative
        try:
            record["content_sha256"] = sha256_stream(path)
            hashed_file_count += 1
            hashed_bytes += int(record["size_bytes"])
        except Exception as exc:
            content_hash_errors.append(
                {
                    "path": relative,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if index % progress_every == 0:
            progress.log(f"content-hashed inventory record {index:,}/{len(inventory):,}")

    if content_hash_errors:
        _append_error(errors, "content_hash_errors", content_hash_errors)

    stems_by_split = {
        split: {stem for _, stem in artifacts[split]["rgb"]} for split in SPLITS
    }
    cities_by_split = {
        split: {city for city, _ in artifacts[split]["rgb"]} for split in SPLITS
    }
    stem_intersections = _all_intersections(stems_by_split)
    city_intersections = _all_intersections(cities_by_split)
    leakage = {
        "stem_intersections": stem_intersections,
        "city_intersections": city_intersections,
        "has_stem_leakage": any(stem_intersections.values()),
        "has_city_leakage": any(city_intersections.values()),
    }
    if leakage["has_stem_leakage"]:
        _append_error(errors, "split_stem_leakage", stem_intersections)
    if leakage["has_city_leakage"]:
        _append_error(errors, "split_city_leakage", city_intersections)

    trainval_raw = raw_histograms["train"] + raw_histograms["val"]
    trainval_semantics = _histogram_summary(trainval_raw)
    test_semantics = split_results["test"]["semantic_label_pixels"]
    test_has_eval_pixels = bool(test_semantics["valid_19_class_pixels"])
    if test_has_eval_pixels:
        test_detection = "contains_19_class_evaluation_pixels"
        test_explanation = (
            "This local test labelIds set contains pixels from the 19 evaluation classes. "
            "It is not the usual public Cityscapes test placeholder distribution; verify its "
            "origin and authorization before treating it as test ground truth."
        )
        warnings.append({"code": "test_contains_semantic_labels", "detail": test_explanation})
    else:
        test_detection = "official_public_test_placeholder_not_semantic_ground_truth"
        test_explanation = (
            "No test pixel maps to any of the 19 evaluation trainIds. The packaged test "
            "gtFine files therefore contain only ignored/placeholder regions and are not "
            "semantic ground truth for local accuracy or mIoU evaluation. Official test "
            "predictions must be evaluated through the Cityscapes evaluation server."
        )
        warnings.append({"code": "test_gt_is_not_semantic_ground_truth", "detail": test_explanation})

    fingerprint = _fingerprint(inventory)
    progress.log(
        f"computed dataset fingerprint {fingerprint}; content-hashed "
        f"{hashed_file_count:,} files ({hashed_bytes / (1024 ** 3):,.2f} GiB)"
    )

    elapsed = time.perf_counter() - progress.started
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "name": "Cityscapes",
            "dataset_root": str(dataset_root),
            "left_img8bit_root": str(left_root),
            "gt_fine_root": str(gt_root),
        },
        "run": {
            "started_at_utc": wall_started,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "workers": 1,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pillow_version": PIL.__version__,
        },
        "audit_scope": {
            "splits": list(SPLITS),
            "gt_types": list(GT_KINDS),
            "png_dimensions_and_modes": (
                "all discovered RGB, color, instanceIds, and labelIds PNG files; "
                "labelIds pixel payloads were fully decoded, other PNGs were header-inspected"
            ),
            "polygon_json": "all discovered gtFine polygons JSON files parsed and schema-checked",
            "pixel_histogram": (
                "all labelIds pixels in train and val; test labelIds pixels were also fully "
                "counted solely to detect whether public test GT is semantic"
            ),
            "content_hash_policy": hash_policy,
        },
        "status": {
            "strict_pass": not errors,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        },
        "splits": split_results,
        "leakage": leakage,
        "train_val_19_class_histogram": trainval_semantics,
        "test_ground_truth_semantics": {
            "detection": test_detection,
            "has_19_class_evaluation_pixels": test_has_eval_pixels,
            "explanation": test_explanation,
            "total_pixels": test_semantics["total_pixels"],
            "valid_19_class_pixels": test_semantics["valid_19_class_pixels"],
            "ignored_pixels": test_semantics["ignored_pixels"],
            "ignore_ratio": test_semantics["ignore_ratio"],
            "raw_label_id_histogram": test_semantics["raw_label_id_histogram"],
        },
        "fingerprint": {
            "algorithm": "SHA-256",
            "digest": fingerprint,
            "canonicalization": (
                "Start with UTF-8 bytes 'cityscapes-audit-fingerprint-v1\\0'. For every "
                "inventory record sorted by POSIX relative_path, append compact UTF-8 JSON "
                "[relative_path,size_bytes,content_sha256_or_null] followed by LF. SHA-256 "
                "the resulting byte stream."
            ),
            "inventory_scope": "every regular file recursively below dataset_root",
            "inventory_file_count": len(inventory),
            "inventory_total_size_bytes": int(sum(r["size_bytes"] for r in inventory)),
            "content_hash_policy": (
                "all files" if hash_policy == "all" else "all RGB leftImg8bit and gtFine labelIds files"
            ),
            "content_hashed_file_count": hashed_file_count,
            "content_hashed_size_bytes": hashed_bytes,
            "inventory": inventory,
        },
    }
    # Prove JSON-standard compatibility here rather than only in the writer.
    json.dumps(manifest, ensure_ascii=False, allow_nan=False)
    return manifest


def histogram_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = {
        "train": manifest["splits"]["train"]["semantic_label_pixels"],
        "val": manifest["splits"]["val"]["semantic_label_pixels"],
        "trainval": manifest["train_val_19_class_histogram"],
    }
    for scope, summary in sources.items():
        total = int(summary["total_pixels"])
        valid = int(summary["valid_19_class_pixels"])
        for train_id, raw_id, class_name in CITYSCAPES_CLASSES:
            count = int(summary["train_id_histogram"][str(train_id)])
            rows.append(
                {
                    "scope": scope,
                    "train_id": train_id,
                    "raw_label_id": raw_id,
                    "class_name": class_name,
                    "pixel_count": count,
                    "ratio_of_all_pixels": count / total if total else "",
                    "ratio_within_valid_19_classes": count / valid if valid else "",
                    "is_ignore": False,
                }
            )
        ignored = int(summary["ignored_pixels"])
        rows.append(
            {
                "scope": scope,
                "train_id": 255,
                "raw_label_id": "all non-mapped labelIds",
                "class_name": "ignore",
                "pixel_count": ignored,
                "ratio_of_all_pixels": ignored / total if total else "",
                "ratio_within_valid_19_classes": "",
                "is_ignore": True,
            }
        )
    return rows


def render_summary(manifest: dict[str, Any]) -> str:
    status = manifest["status"]
    lines = [
        "# Cityscapes 데이터셋 감사 요약",
        "",
        f"- 엄격 감사 결과: **{'통과' if status['strict_pass'] else '실패'}**",
        f"- 데이터 루트: `{manifest['dataset']['dataset_root']}`",
        f"- 실행 시간: {manifest['run']['elapsed_seconds']:.3f}초 (worker 1개, 순차 I/O)",
        f"- 오류/경고: {status['error_count']} / {status['warning_count']}",
        "",
        "## 표본 수와 1:1 대응",
        "",
        "| split | RGB | color GT | instanceIds GT | labelIds GT | polygons GT | 완전 대응 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for split in SPLITS:
        result = manifest["splits"][split]
        counts = result["counts"]
        paired = result["one_to_one"]["all_rgb_have_exactly_one_of_each_gt_type"]
        lines.append(
            f"| {split} | {counts['rgb']:,} | {counts['gt_color']:,} | "
            f"{counts['gt_instanceIds']:,} | {counts['gt_labelIds']:,} | "
            f"{counts['gt_polygons']:,} | {'예' if paired else '아니오'} |"
        )
    lines.extend(["", "## 도시와 split 누수", ""])
    for split in SPLITS:
        cities = manifest["splits"][split]["cities"]
        lines.append(
            f"- **{split}**: {cities['rgb_city_count']}개 — "
            + ", ".join(cities["rgb_city_list"])
        )
    leakage = manifest["leakage"]
    lines.extend(
        [
            "",
            f"- stem 누수: **{'있음' if leakage['has_stem_leakage'] else '없음'}**",
            f"- city 누수: **{'있음' if leakage['has_city_leakage'] else '없음'}**",
            "",
            "## PNG 및 polygon JSON 검사",
            "",
        ]
    )
    for split in SPLITS:
        result = manifest["splits"][split]
        png_text = []
        for kind in PNG_KINDS:
            check = result["png_checks"][kind]
            dims = ", ".join(f"{key}: {value:,}" for key, value in check["dimensions"].items())
            modes = ", ".join(f"{key}: {value:,}" for key, value in check["modes"].items())
            png_text.append(
                f"  - {kind}: {check['files_inspected']:,}/{check['files_discovered']:,}, "
                f"dimensions [{dims}], modes [{modes}]"
            )
        lines.append(f"- **{split}**")
        lines.extend(png_text)
        polygons = result["polygon_json"]
        lines.append(
            f"  - polygons JSON: {polygons['files_parsed']:,}/{polygons['files_discovered']:,} "
            f"파싱, 객체 {polygons['object_count']:,}개, 파싱 오류 "
            f"{len(polygons['parse_errors']):,}개, 스키마 오류 {len(polygons['schema_errors']):,}개"
        )
    lines.extend(
        [
            "",
            "검사 범위: 인식된 RGB·GT PNG 전부의 크기와 모드를 검사했습니다. "
            "labelIds는 모든 픽셀을 완전히 디코딩했고, 나머지 PNG는 헤더를 검사했습니다. "
            "모든 polygons JSON은 파싱 및 기본 스키마 검사를 했습니다.",
            "",
            "## train/val 19-class 픽셀 분포",
            "",
            "Cityscapes의 원본 `labelId`를 학습용 `trainId` 0~18로 매핑했습니다. "
            "19개에 속하지 않는 픽셀은 `ignore`로 합쳤습니다.",
            "",
            "| trainId | labelId | 클래스 | train 픽셀 | val 픽셀 | train+val 픽셀 |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    train_hist = manifest["splits"]["train"]["semantic_label_pixels"]["train_id_histogram"]
    val_hist = manifest["splits"]["val"]["semantic_label_pixels"]["train_id_histogram"]
    total_hist = manifest["train_val_19_class_histogram"]["train_id_histogram"]
    for train_id, raw_id, name in CITYSCAPES_CLASSES:
        lines.append(
            f"| {train_id} | {raw_id} | {name} | {train_hist[str(train_id)]:,} | "
            f"{val_hist[str(train_id)]:,} | {total_hist[str(train_id)]:,} |"
        )
    for label, summary in (
        ("train", manifest["splits"]["train"]["semantic_label_pixels"]),
        ("val", manifest["splits"]["val"]["semantic_label_pixels"]),
        ("train+val", manifest["train_val_19_class_histogram"]),
    ):
        lines.append(
            f"- {label} ignore 비율: **{summary['ignore_ratio']:.6%}** "
            f"({summary['ignored_pixels']:,}/{summary['total_pixels']:,} 픽셀)"
        )
    test = manifest["test_ground_truth_semantics"]
    if test["has_19_class_evaluation_pixels"]:
        korean_test_explanation = (
            "이 로컬 test labelIds에는 19개 평가 클래스 픽셀이 있습니다. 일반 공개 "
            "Cityscapes test placeholder 분포와 다르므로, test 정답으로 사용하기 전에 "
            "출처와 사용 권한을 확인해야 합니다."
        )
    else:
        korean_test_explanation = (
            "test 픽셀 중 19개 평가 trainId에 매핑되는 픽셀이 하나도 없습니다. 따라서 "
            "동봉된 test gtFine은 무시 영역/placeholder일 뿐, 로컬 정확도나 mIoU를 계산할 "
            "수 있는 의미 분할 정답이 아닙니다. test 예측은 Cityscapes 공식 평가 서버로 "
            "평가해야 합니다."
        )
    lines.extend(
        [
            "",
            "## test GT 해석 주의",
            "",
            f"**{korean_test_explanation}**",
            "",
            f"전수 검사 결과 test의 19-class 유효 픽셀은 "
            f"{test['valid_19_class_pixels']:,}/{test['total_pixels']:,}개이고, "
            f"ignore 비율은 {test['ignore_ratio']:.6%}입니다.",
            "",
            "## 재현 가능한 데이터셋 지문",
            "",
            f"- SHA-256: `{manifest['fingerprint']['digest']}`",
            f"- inventory: {manifest['fingerprint']['inventory_file_count']:,}개 파일, "
            f"{manifest['fingerprint']['inventory_total_size_bytes']:,} bytes",
            f"- content SHA-256: {manifest['fingerprint']['content_hashed_file_count']:,}개 파일, "
            f"{manifest['fingerprint']['content_hashed_size_bytes']:,} bytes",
            f"- 정책: {manifest['fingerprint']['content_hash_policy']}",
            "",
            "지문은 모든 파일의 상대 경로와 크기를 사용하며, 기본 정책에서는 모든 RGB와 "
            "labelIds 파일의 실제 바이트 SHA-256도 사용합니다. 정확한 정규화 방법과 전체 "
            "파일 inventory는 `dataset_manifest.json`에 기록되어 있습니다.",
            "",
            "## 용어",
            "",
            "- **stem**: 한 장면을 식별하는 공통 파일명 부분입니다. RGB와 네 GT 파일이 같은 stem을 가져야 합니다.",
            "- **labelId**: Cityscapes 원본 의미 클래스 번호입니다.",
            "- **trainId**: 학습·평가에 쓰도록 19개 클래스를 0~18로 다시 번호 붙인 값입니다.",
            "- **ignore 비율**: 19개 평가 클래스에 속하지 않아 손실·mIoU 계산에서 제외해야 하는 픽셀 비율입니다.",
            "- **fingerprint**: 파일 경로·크기·내용 해시를 하나의 SHA-256으로 합친 데이터셋 식별값입니다.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(manifest: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.json"
    summary_path = output_dir / "dataset_summary.md"
    histogram_path = output_dir / "class_histogram.csv"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            manifest,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    summary_path.write_text(render_summary(manifest), encoding="utf-8", newline="\n")
    rows = histogram_rows(manifest)
    with histogram_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"manifest": manifest_path, "summary": summary_path, "histogram": histogram_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Cityscapes counts, pairing, leakage, image/JSON integrity, labels, and provenance."
    )
    parser.add_argument("--dataset-root", "--root", default="dataset", type=Path)
    parser.add_argument(
        "--output-dir",
        "--out",
        default=Path("studies/neural_encoder_decoder/results/dataset_audit_20260711"),
        type=Path,
    )
    parser.add_argument(
        "--hash-policy",
        choices=("rgb-labelids", "all"),
        default="rgb-labelids",
        help="Default hashes all RGB and labelIds content; 'all' hashes every inventory file.",
    )
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-strict-exit",
        action="store_true",
        help="Write results but exit 0 even when integrity errors are found.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = audit_dataset(
        args.dataset_root,
        args.output_dir,
        hash_policy=args.hash_policy,
        progress_every=args.progress_every,
        quiet=args.quiet,
    )
    paths = write_outputs(manifest, args.output_dir)
    print(
        json.dumps(
            {
                "strict_pass": manifest["status"]["strict_pass"],
                "elapsed_seconds": manifest["run"]["elapsed_seconds"],
                "fingerprint": manifest["fingerprint"]["digest"],
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    if not manifest["status"]["strict_pass"] and not args.no_strict_exit:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
