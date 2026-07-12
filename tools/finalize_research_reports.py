from __future__ import annotations

"""Finalize the three Korean research reports from machine-readable evidence.

The Markdown files in ``reports/`` are immutable templates.  This program always
reads those templates and writes ``*_final.md`` plus ``*_final.docx`` to a
separate output directory, so rerunning it cannot progressively rewrite an
already-rendered report.

No experimental value is guessed.  In strict mode (the default), missing core
artifacts, required columns/fields, ambiguous files, and stale provenance paths
are errors.  ``--allow-incomplete`` is intended for pipeline smoke checks: a
missing artifact becomes an explicit Korean ``미실행/증거 없음`` statement.
Malformed JSON/CSV remains fatal in both modes because corrupt evidence must
never be presented as an experiment that merely has not run yet.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

AREAS = (100, 200, 300, 400, 500)
POLICIES = ("Stochastic", "LinUCB", "SA", "Greedy", "MCTS")
OPT_POLICIES = ("LinUCB", "SA", "Greedy", "MCTS")
ADAPTIVE_MODES = ("nonsemantic", "fixed_paper_like", "adaptive_semantic")
REPRODUCTION_MODES = ("semantic", "nonsemantic")
CI95_METHOD = "two-sided Student t; n=1 uses zero-width descriptive interval"
CORE_REPEAT_METRICS = (
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

_T_CRITICAL_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

REPORT_TEMPLATES = {
    "neural": "01_인코더_디코더_딥러닝_과정.md",
    "reproduction": "02_AirTalking_논문_실험_재현.md",
    "adaptive": "03_후속연구_적응형_의미압축.md",
}

EXPECTED_AUTO_KEYS = {
    "enhanced_dataset_fingerprint",
    "enhanced_environment",
    "enhanced_command_result",
    "enhanced_best_epoch",
    "enhanced_sample_counts",
    "enhanced_training_elapsed",
    "enhanced_run_status",
    "enhanced_rate_quality_table",
    "enhanced_paperlike_metrics",
    "enhanced_timing_and_resources",
    "enhanced_training_curve_summary",
    "enhanced_qualitative_audit",
    "enhanced_ablation_results",
    "enhanced_semantic_profile_summary",
    "enhanced_reproduction_provenance",
    "enhanced_airtalking_command",
    "enhanced_paperlike_codec_for_simulator",
    "enhanced_airtalking_full_results",
    "enhanced_vs_legacy_system_delta",
    "enhanced_paper_verification_counts",
    "strengthened_reproduction_sensitivity_results",
    "enhanced_neural_rate_quality_table",
    "preregistered_quality_guardrails",
    "enhanced_adaptive_ablation_results",
    "enhanced_adaptive_statistics_command",
    "enhanced_adaptive_main_results",
    "enhanced_adaptive_paired_deltas",
    "enhanced_adaptive_confidence_intervals",
    "enhanced_adaptive_quality_guardrail_results",
    "enhanced_adaptive_mode_usage",
    "enhanced_adaptive_generalization",
    "enhanced_adaptive_airtalking_verification",
    "enhanced_adaptive_claim_status",
}

AUTO_RE = re.compile(
    r"<!--\s*AUTO:\s*([A-Za-z0-9_]+)\s*-->(?:[ \t]*미생성)?"
)


class ReportFinalizationError(RuntimeError):
    """Raised when evidence cannot safely be turned into a final report."""


@dataclass
class Diagnostics:
    allow_incomplete: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def missing(self, message: str) -> None:
        if self.allow_incomplete:
            self.warnings.append(message)
        else:
            self.errors.append(message)

    def stale(self, message: str) -> None:
        if self.allow_incomplete:
            self.warnings.append(message)
        else:
            self.errors.append(message)

    def fail_if_errors(self) -> None:
        if self.errors:
            details = "\n".join(f"  - {item}" for item in self.errors)
            raise ReportFinalizationError(
                "보고서 최종화를 중단했습니다. 다음 증거 오류를 해결하세요:\n" + details
            )


@dataclass(frozen=True)
class InputPaths:
    enhanced_dir: Path
    reproduction_dir: Path
    adaptive_dir: Path
    reports_dir: Path
    output_dir: Path
    baseline_neural_dir: Path | None = None
    baseline_reproduction_dir: Path | None = None
    baseline_adaptive_dir: Path | None = None


@dataclass
class Evidence:
    enhanced_summary: dict[str, Any] | None = None
    enhanced_rates: list[dict[str, str]] | None = None
    enhanced_history: list[dict[str, str]] | None = None
    enhanced_confusion: list[dict[str, str]] | None = None
    enhanced_airtalking: dict[str, Any] | None = None
    reproduction_summary: list[dict[str, str]] | None = None
    reproduction_repeats: list[dict[str, str]] | None = None
    reproduction_statistics: list[dict[str, str]] | None = None
    reproduction_verification: list[dict[str, str]] | None = None
    reproduction_verification_manifest: dict[str, Any] | None = None
    reproduction_metadata: dict[str, Any] | None = None
    adaptive_summary: list[dict[str, str]] | None = None
    adaptive_repeats: list[dict[str, str]] | None = None
    adaptive_statistics: list[dict[str, str]] | None = None
    adaptive_usage: list[dict[str, str]] | None = None
    adaptive_validation: dict[str, Any] | None = None
    adaptive_metadata: dict[str, Any] | None = None
    baseline_neural_summary: dict[str, Any] | None = None
    baseline_neural_airtalking: dict[str, Any] | None = None
    baseline_neural_dir: Path | None = None
    baseline_reproduction_summary: list[dict[str, str]] | None = None
    baseline_reproduction_verification: list[dict[str, str]] | None = None
    baseline_adaptive_summary: list[dict[str, str]] | None = None


def _ensure_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ReportFinalizationError(
            f"{label} 경로가 존재하지 않습니다(오래되었거나 오타인 경로): {resolved}"
        )
    if not resolved.is_dir():
        raise ReportFinalizationError(f"{label}은 디렉터리가 아닙니다: {resolved}")
    return resolved


def _artifact_path(
    directory: Path, filename: str, diagnostics: Diagnostics, label: str
) -> Path | None:
    path = directory / filename
    if not path.is_file():
        diagnostics.missing(f"{label} 파일이 없습니다: {path}")
        return None
    return path


def _load_json(
    path: Path | None, diagnostics: Diagnostics, label: str
) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ReportFinalizationError(f"{label} JSON을 읽을 수 없습니다: {path}: {exc}") from exc
    def reject_nonstandard_constant(constant: str) -> None:
        raise ValueError(f"비표준 비유한 상수 {constant!r}")

    try:
        value = json.loads(text, parse_constant=reject_nonstandard_constant)
    except json.JSONDecodeError as exc:
        raise ReportFinalizationError(
            f"{label} JSON 문법 오류: {path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        raise ReportFinalizationError(
            f"{label} JSON에 NaN/Infinity 같은 비표준 비유한값이 있습니다: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReportFinalizationError(
            f"{label} JSON 최상위 값은 object여야 합니다: {path}"
        )
    return value


def _load_csv(
    path: Path | None,
    diagnostics: Diagnostics,
    label: str,
    required_columns: Iterable[str] = (),
    *,
    require_rows: bool = True,
) -> list[dict[str, str]] | None:
    if path is None:
        return None
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None:
                raise ReportFinalizationError(f"{label} CSV header가 없습니다: {path}")
            headers = [header.strip() if header is not None else "" for header in reader.fieldnames]
            if not all(headers):
                raise ReportFinalizationError(f"{label} CSV에 빈 header가 있습니다: {path}")
            if len(headers) != len(set(headers)):
                raise ReportFinalizationError(f"{label} CSV에 중복 header가 있습니다: {path}")
            reader.fieldnames = headers
            rows: list[dict[str, str]] = []
            for line_number, raw in enumerate(reader, start=2):
                if None in raw:
                    raise ReportFinalizationError(
                        f"{label} CSV {line_number}행의 열 수가 header보다 많습니다: {path}"
                    )
                row = {key: (value if value is not None else "") for key, value in raw.items()}
                rows.append(row)
    except csv.Error as exc:
        raise ReportFinalizationError(f"{label} CSV 문법 오류: {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ReportFinalizationError(f"{label} CSV UTF-8 해석 오류: {path}: {exc}") from exc
    except OSError as exc:
        raise ReportFinalizationError(f"{label} CSV를 읽을 수 없습니다: {path}: {exc}") from exc

    missing_columns = sorted(set(required_columns) - set(headers))
    if missing_columns:
        diagnostics.missing(
            f"{label} CSV 필수 열 누락 {missing_columns}: {path}; 실제 열={headers}"
        )
    if require_rows and not rows:
        diagnostics.missing(f"{label} CSV에 데이터 행이 없습니다: {path}")
    return rows


def _find_one_csv(
    directory: Path, pattern: str, diagnostics: Diagnostics, label: str
) -> Path | None:
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not matches:
        diagnostics.missing(f"{label} 파일이 없습니다: {directory / pattern}")
        return None
    if len(matches) > 1:
        message = f"{label} 파일이 여러 개라 선택이 모호합니다: {', '.join(str(p) for p in matches)}"
        if diagnostics.allow_incomplete:
            diagnostics.warnings.append(message + f"; 사전순 첫 파일 {matches[0].name} 사용")
        else:
            diagnostics.errors.append(message)
    return matches[0]


def _nested_get(value: Mapping[str, Any] | None, path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _require_json_fields(
    value: Mapping[str, Any] | None,
    paths: Iterable[str],
    diagnostics: Diagnostics,
    label: str,
) -> None:
    if value is None:
        return
    missing = [path for path in paths if _nested_get(value, path) is None]
    if missing:
        diagnostics.missing(f"{label} JSON 필수 필드 누락: {missing}")


def _validate_json_numbers(
    value: Mapping[str, Any] | None,
    paths: Iterable[str],
    diagnostics: Diagnostics,
    label: str,
) -> None:
    if value is None:
        return
    invalid = [path for path in paths if _float(_nested_get(value, path)) is None]
    if invalid:
        diagnostics.missing(
            f"{label} JSON 필수 수치가 비어 있거나 비수치/비유한값입니다: {invalid}"
        )


def _validate_csv_cells(
    rows: Sequence[Mapping[str, str]] | None,
    *,
    numeric_columns: Iterable[str] = (),
    text_columns: Iterable[str] = (),
    diagnostics: Diagnostics,
    label: str,
) -> None:
    if rows is None:
        return
    numeric = tuple(numeric_columns)
    text = tuple(text_columns)
    invalid: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        for column in numeric:
            if column in row and _float(row.get(column)) is None:
                invalid.append(f"{row_number}행 {column}={row.get(column)!r}")
        for column in text:
            if column in row and not str(row.get(column, "")).strip():
                invalid.append(f"{row_number}행 {column}=빈값")
        if len(invalid) >= 20:
            break
    if invalid:
        diagnostics.missing(
            f"{label} CSV 필수 셀이 비었거나 비수치/비유한값입니다(최대 20개 표시): {invalid}"
        )


def _validate_unique_csv_keys(
    rows: Sequence[Mapping[str, str]] | None,
    columns: Sequence[str],
    diagnostics: Diagnostics,
    label: str,
) -> None:
    if rows is None:
        return
    seen: set[tuple[str, ...]] = set()
    duplicates: list[tuple[str, ...]] = []
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in columns)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        diagnostics.missing(
            f"{label} CSV에 중복 key {list(columns)}가 있습니다: {duplicates[:20]}"
        )


def _looks_absolute_windows_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")


def _resolve_reference(raw: str, result_dir: Path, *, require_file: bool = True) -> Path | None:
    """Resolve an artifact reference with one deterministic search order.

    Relative paths are interpreted against the producing result directory,
    then the repository root, then the current working directory.  A required
    provenance reference denotes a concrete file, never merely an existing
    directory.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate_text = os.path.expandvars(os.path.expanduser(raw.strip()))
    candidates: list[Path] = []
    if _looks_absolute_windows_path(candidate_text):
        candidates.append(Path(str(PureWindowsPath(candidate_text))))
    else:
        normalized = candidate_text.replace("\\", os.sep)
        path = Path(normalized)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend((result_dir / path, ROOT / path, Path.cwd() / path))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() if require_file else resolved.exists():
            return resolved
    return None


def _reference_exists(raw: str, result_dir: Path) -> bool:
    return _resolve_reference(raw, result_dir, require_file=True) is not None


def _check_path_references(
    mapping: Mapping[str, Any] | None,
    keys: Iterable[str],
    result_dir: Path,
    diagnostics: Diagnostics,
    label: str,
) -> None:
    if mapping is None:
        return
    for key in keys:
        raw = _nested_get(mapping, key)
        if raw is None:
            diagnostics.missing(f"{label}.{key} 필수 경로 필드가 누락되었습니다")
            continue
        if not isinstance(raw, str):
            diagnostics.missing(f"{label}.{key} 경로 필드가 문자열이 아닙니다: {raw!r}")
            continue
        if not raw.strip():
            diagnostics.missing(f"{label}.{key} 필수 경로 필드가 빈 문자열입니다")
            continue
        if _resolve_reference(raw, result_dir, require_file=True) is None:
            diagnostics.stale(
                f"{label}.{key}가 존재하는 파일이 아닌 오래된 경로/디렉터리를 가리킵니다: {raw}"
            )


def _check_artifact_map(
    summary: Mapping[str, Any] | None,
    result_dir: Path,
    diagnostics: Diagnostics,
) -> None:
    artifacts = _nested_get(summary, "artifacts")
    if not isinstance(artifacts, Mapping):
        return
    for name, raw in artifacts.items():
        if not isinstance(raw, str):
            diagnostics.missing(
                f"강화 codec result_summary.artifacts.{name} 경로 필드가 문자열이 아닙니다: {raw!r}"
            )
        elif not raw.strip():
            diagnostics.missing(
                f"강화 codec result_summary.artifacts.{name} 필수 경로가 비어 있습니다"
            )
        elif _resolve_reference(raw, result_dir, require_file=True) is None:
            diagnostics.stale(
                f"강화 codec result_summary.artifacts.{name}가 존재하는 파일이 아닌 오래된 경로/디렉터리를 가리킵니다: {raw}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_cell_key(row: Mapping[str, Any]) -> tuple[str, int, str] | None:
    mode = str(row.get("mode", "")).strip()
    policy = str(row.get("policy", "")).strip()
    area_number = _float(row.get("area"))
    if (
        not mode
        or not policy
        or area_number is None
        or not area_number.is_integer()
    ):
        return None
    return mode, int(area_number), policy


def _canonical_repeat_id(row: Mapping[str, Any]) -> int | None:
    value = _float(row.get("repeat"))
    if value is None or not value.is_integer() or value < 0:
        return None
    return int(value)


def _index_canonical_rows(
    rows: Sequence[Mapping[str, Any]] | None,
    diagnostics: Diagnostics,
    label: str,
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    invalid: list[int] = []
    duplicates: list[tuple[str, int, str]] = []
    for row_number, row in enumerate(rows or [], start=2):
        key = _canonical_cell_key(row)
        if key is None:
            invalid.append(row_number)
            continue
        if key in indexed:
            duplicates.append(key)
        else:
            indexed[key] = row
    if invalid:
        diagnostics.missing(f"{label}에 정규화할 수 없는 mode/area/policy key 행이 있습니다: {invalid[:20]}")
    if duplicates:
        diagnostics.missing(f"{label}에 정규화 후 중복 key가 있습니다: {duplicates[:20]}")
    return indexed


def _expected_reproduction_keys() -> set[tuple[str, int, str]]:
    semantic = {("semantic", area, policy) for area in AREAS for policy in POLICIES}
    nonsemantic = {("nonsemantic", 300, policy) for policy in OPT_POLICIES}
    return semantic | nonsemantic


def _expected_adaptive_keys() -> set[tuple[str, int, str]]:
    return {
        (mode, area, policy)
        for mode in ADAPTIVE_MODES
        for area in AREAS
        for policy in POLICIES
    }


def _validate_exact_coverage(
    actual: set[tuple[str, int, str]],
    expected: set[tuple[str, int, str]],
    diagnostics: Diagnostics,
    label: str,
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        diagnostics.missing(
            f"{label} mode/area/policy 조합 coverage 불일치: "
            f"누락={missing[:20]}(총 {len(missing)}), 초과={extra[:20]}(총 {len(extra)})"
        )


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


def _numbers_match(actual: Any, expected: float) -> bool:
    parsed = _float(actual)
    return parsed is not None and math.isclose(parsed, expected, rel_tol=1e-9, abs_tol=1e-9)


def _positive_integer(value: Any) -> int | None:
    parsed = _float(value)
    if parsed is None or not parsed.is_integer() or parsed < 1:
        return None
    return int(parsed)


def _nonnegative_integer(value: Any) -> int | None:
    parsed = _float(value)
    if parsed is None or not parsed.is_integer() or parsed < 0:
        return None
    return int(parsed)


def _validate_repeat_statistics(
    *,
    summary_rows: Sequence[Mapping[str, str]] | None,
    repeat_rows: Sequence[Mapping[str, str]] | None,
    statistics_rows: Sequence[Mapping[str, str]] | None,
    expected_keys: set[tuple[str, int, str]],
    repeats_expected: int | None,
    diagnostics: Diagnostics,
    label: str,
) -> None:
    """Independently recompute every required mean/std/Student-t interval."""
    summary = _index_canonical_rows(summary_rows, diagnostics, f"{label} summary")
    statistics_index = _index_canonical_rows(
        statistics_rows, diagnostics, f"{label} statistical_summary"
    )
    _validate_exact_coverage(set(summary), expected_keys, diagnostics, f"{label} summary")
    _validate_exact_coverage(
        set(statistics_index), expected_keys, diagnostics, f"{label} statistical_summary"
    )

    groups: dict[tuple[str, int, str], dict[int, Mapping[str, str]]] = {}
    invalid_repeats: list[int] = []
    duplicate_repeats: list[tuple[str, int, str, int]] = []
    for row_number, row in enumerate(repeat_rows or [], start=2):
        key = _canonical_cell_key(row)
        repeat_id = _canonical_repeat_id(row)
        if key is None or repeat_id is None:
            invalid_repeats.append(row_number)
            continue
        group = groups.setdefault(key, {})
        if repeat_id in group:
            duplicate_repeats.append((*key, repeat_id))
        else:
            group[repeat_id] = row
    if invalid_repeats:
        diagnostics.missing(
            f"{label} repeat_metrics에 유효하지 않은 key/repeat 행이 있습니다: {invalid_repeats[:20]}"
        )
    if duplicate_repeats:
        diagnostics.missing(
            f"{label} repeat_metrics에 정규화 후 중복 key가 있습니다: {duplicate_repeats[:20]}"
        )
    _validate_exact_coverage(set(groups), expected_keys, diagnostics, f"{label} repeat_metrics")

    if repeats_expected is None:
        diagnostics.missing(f"{label} metadata repeat 수가 양의 정수가 아닙니다")
        return

    mismatches: list[str] = []
    expected_repeat_ids = set(range(repeats_expected))
    for key in sorted(expected_keys):
        repeat_group = groups.get(key)
        summary_row = summary.get(key)
        statistics_row = statistics_index.get(key)
        if repeat_group is None or summary_row is None or statistics_row is None:
            continue
        actual_repeat_ids = set(repeat_group)
        if actual_repeat_ids != expected_repeat_ids:
            mismatches.append(
                f"{key}: repeat 번호 actual={sorted(actual_repeat_ids)}, expected={sorted(expected_repeat_ids)}"
            )
            continue
        if not _numbers_match(statistics_row.get("repeat_count"), float(repeats_expected)):
            mismatches.append(
                f"{key}: repeat_count={statistics_row.get('repeat_count')!r}, expected={repeats_expected}"
            )
        if str(statistics_row.get("ci95_method", "")).strip() != CI95_METHOD:
            mismatches.append(f"{key}: ci95_method 계약 불일치")

        ordered_rows = [repeat_group[index] for index in range(repeats_expected)]
        for metric in CORE_REPEAT_METRICS:
            values = [_float(row.get(metric)) for row in ordered_rows]
            if any(value is None for value in values):
                mismatches.append(f"{key}: {metric} repeat 값이 비었거나 비수치/비유한값")
                continue
            finite_values = [float(value) for value in values if value is not None]
            mean = statistics.fmean(finite_values)
            std = statistics.stdev(finite_values) if len(finite_values) > 1 else 0.0
            margin = _student_t_critical_95(len(finite_values)) * std / math.sqrt(len(finite_values))
            expected_statistics = {
                f"{metric}_n": float(len(finite_values)),
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci95_margin": margin,
                f"{metric}_ci95_low": mean - margin,
                f"{metric}_ci95_high": mean + margin,
            }
            if not _numbers_match(summary_row.get(metric), mean):
                mismatches.append(
                    f"{key}: summary {metric}={summary_row.get(metric)!r}, repeat 평균={mean!r}"
                )
            for column, expected_value in expected_statistics.items():
                if not _numbers_match(statistics_row.get(column), expected_value):
                    mismatches.append(
                        f"{key}: {column}={statistics_row.get(column)!r}, 재계산={expected_value!r}"
                    )
            if len(mismatches) >= 40:
                break
        if len(mismatches) >= 40:
            break
    if mismatches:
        diagnostics.missing(
            f"{label} summary/repeat/statistics 독립 재계산 불일치(최대 40개): {mismatches[:40]}"
        )


def _validate_source_summary_snapshot(
    manifest: Mapping[str, Any] | None,
    summary_path: Path | None,
    result_dir: Path,
    diagnostics: Diagnostics,
    label: str,
) -> None:
    if manifest is None or summary_path is None:
        return
    source = manifest.get("source_summary")
    if not isinstance(source, Mapping):
        diagnostics.missing(f"{label}.source_summary 필수 object가 누락되었습니다")
        return
    raw_path = source.get("path")
    expected_hash = source.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        diagnostics.missing(f"{label}.source_summary.path는 nonblank 문자열이어야 합니다")
        resolved = None
    else:
        resolved = _resolve_reference(raw_path, result_dir, require_file=True)
        if resolved is None:
            diagnostics.stale(
                f"{label}.source_summary.path가 존재하는 파일을 가리키지 않습니다: {raw_path}"
            )
        elif resolved != summary_path.resolve():
            diagnostics.stale(
                f"{label}.source_summary.path가 현재 summary_metrics.csv와 다릅니다: {resolved}"
            )
    if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is None:
        diagnostics.missing(f"{label}.source_summary.sha256은 64자리 SHA256이어야 합니다")
    else:
        actual_hash = _sha256(summary_path)
        if expected_hash.lower() != actual_hash:
            diagnostics.stale(
                f"{label}.source_summary.sha256 불일치: 기록={expected_hash.lower()}, 실제={actual_hash}"
            )


def _validate_adaptive_validation_contract(
    validation: Mapping[str, Any] | None,
    summary_path: Path | None,
    summary_rows: Sequence[Mapping[str, str]] | None,
    adaptive_dir: Path,
    diagnostics: Diagnostics,
) -> None:
    label = "강화 적응형 result_validation"
    if validation is None:
        return
    schema_version = _positive_integer(validation.get("schema_version"))
    if schema_version is None or schema_version < 2:
        diagnostics.missing(f"{label}.schema_version은 2 이상이어야 합니다")
    if validation.get("passed") is not True:
        diagnostics.missing(f"{label}.passed는 반드시 true여야 합니다")
    actual_count = len(summary_rows or [])
    expected_count = len(_expected_adaptive_keys())
    if not _numbers_match(validation.get("row_count"), float(actual_count)):
        diagnostics.missing(
            f"{label}.row_count 불일치: 기록={validation.get('row_count')!r}, 실제={actual_count}"
        )
    if not _numbers_match(validation.get("expected_row_count"), float(expected_count)):
        diagnostics.missing(
            f"{label}.expected_row_count 불일치: 기록={validation.get('expected_row_count')!r}, 계약={expected_count}"
        )
    contract = validation.get("expected_combinations")
    if not isinstance(contract, Mapping):
        diagnostics.missing(f"{label}.expected_combinations 필수 object가 누락되었습니다")
    else:
        modes = contract.get("modes")
        areas = contract.get("areas")
        policies = contract.get("policies")
        actual_areas = {_positive_integer(value) for value in areas} if isinstance(areas, list) else set()
        if not isinstance(modes, list) or set(modes) != set(ADAPTIVE_MODES):
            diagnostics.missing(f"{label}.expected_combinations.modes 계약 불일치: {modes!r}")
        if actual_areas != set(AREAS):
            diagnostics.missing(f"{label}.expected_combinations.areas 계약 불일치: {areas!r}")
        if not isinstance(policies, list) or set(policies) != set(POLICIES):
            diagnostics.missing(f"{label}.expected_combinations.policies 계약 불일치: {policies!r}")
        if not _numbers_match(contract.get("count"), float(expected_count)):
            diagnostics.missing(f"{label}.expected_combinations.count 계약 불일치: {contract.get('count')!r}")
    for field_name in (
        "errors",
        "missing_combinations",
        "extra_combinations",
        "duplicates",
        "duplicate_combinations",
        "missing_values",
        "invalid_canonical_keys",
        "invalid_numeric_values",
        "nonfinite_values",
        "zero_denominators",
    ):
        value = validation.get(field_name)
        if not isinstance(value, list) or value:
            diagnostics.missing(f"{label}.{field_name}는 빈 list여야 합니다: {value!r}")
    adaptive_vs_fixed = validation.get("adaptive_vs_fixed")
    if not isinstance(adaptive_vs_fixed, Mapping) or adaptive_vs_fixed.get("evaluated") is not True:
        diagnostics.missing(f"{label}.adaptive_vs_fixed.evaluated는 true여야 합니다")
    else:
        summary_index = _index_canonical_rows(summary_rows, diagnostics, f"{label} source summary")
        comparisons = adaptive_vs_fixed.get("comparisons")
        comparison_index: dict[tuple[int, str], Mapping[str, Any]] = {}
        malformed: list[int] = []
        duplicates: list[tuple[int, str]] = []
        if isinstance(comparisons, list):
            for row_number, item in enumerate(comparisons, start=1):
                if not isinstance(item, Mapping):
                    malformed.append(row_number)
                    continue
                area = _positive_integer(item.get("area"))
                policy = str(item.get("policy", "")).strip()
                if area is None or not policy:
                    malformed.append(row_number)
                    continue
                key = (area, policy)
                if key in comparison_index:
                    duplicates.append(key)
                else:
                    comparison_index[key] = item
        else:
            malformed.append(0)
        expected_comparisons = {(area, policy) for area in AREAS for policy in POLICIES}
        if malformed or duplicates or set(comparison_index) != expected_comparisons:
            diagnostics.missing(
                f"{label}.adaptive_vs_fixed.comparisons coverage/unique 계약 불일치: "
                f"malformed={malformed[:20]}, duplicates={duplicates[:20]}, "
                f"missing={sorted(expected_comparisons - set(comparison_index))[:20]}, "
                f"extra={sorted(set(comparison_index) - expected_comparisons)[:20]}"
            )
        comparison_mismatches: list[str] = []
        quality_deltas: list[float] = []
        for area, policy in sorted(expected_comparisons):
            item = comparison_index.get((area, policy))
            fixed = summary_index.get(("fixed_paper_like", area, policy))
            adaptive = summary_index.get(("adaptive_semantic", area, policy))
            if item is None or fixed is None or adaptive is None:
                continue
            fixed_finished = _float(fixed.get("finished"))
            fixed_time = _float(fixed.get("avg_time"))
            adaptive_finished = _float(adaptive.get("finished"))
            adaptive_time = _float(adaptive.get("avg_time"))
            fixed_quality = _float(fixed.get("semantic_quality"))
            adaptive_quality = _float(adaptive.get("semantic_quality"))
            if None in (
                fixed_finished,
                fixed_time,
                adaptive_finished,
                adaptive_time,
                fixed_quality,
                adaptive_quality,
            ) or fixed_finished == 0.0 or fixed_time == 0.0:
                comparison_mismatches.append(f"{area}/{policy}: source 값이 비교 불가능")
                continue
            finished_delta = (float(adaptive_finished) - float(fixed_finished)) / float(fixed_finished) * 100.0
            time_delta = (float(adaptive_time) - float(fixed_time)) / float(fixed_time) * 100.0
            quality_delta = float(adaptive_quality) - float(fixed_quality)
            quality_deltas.append(quality_delta)
            expected_pass = finished_delta >= 0.0 and time_delta <= 0.0 and quality_delta >= -0.10
            for field_name, expected_value in (
                ("finished_delta_pct", finished_delta),
                ("avg_time_delta_pct", time_delta),
                ("quality_delta", quality_delta),
            ):
                if not _numbers_match(item.get(field_name), expected_value):
                    comparison_mismatches.append(
                        f"{area}/{policy}:{field_name}={item.get(field_name)!r}, 재계산={expected_value!r}"
                    )
            if item.get("pass") is not expected_pass:
                comparison_mismatches.append(
                    f"{area}/{policy}:pass={item.get('pass')!r}, 재계산={expected_pass!r}"
                )
        if quality_deltas and not _numbers_match(
            adaptive_vs_fixed.get("max_quality_drop"), min(quality_deltas)
        ):
            comparison_mismatches.append(
                "adaptive_vs_fixed.max_quality_drop가 source summary 재계산과 다름"
            )
        if comparison_mismatches:
            diagnostics.missing(
                f"{label}.adaptive_vs_fixed 재계산 불일치(최대 40개): {comparison_mismatches[:40]}"
            )
    _validate_source_summary_snapshot(
        validation, summary_path, adaptive_dir, diagnostics, label
    )


def _validate_reproduction_verifier_contract(
    manifest: Mapping[str, Any] | None,
    summary_path: Path | None,
    verification_rows: Sequence[Mapping[str, str]] | None,
    reproduction_dir: Path,
    diagnostics: Diagnostics,
) -> None:
    label = "강화 논문 재현 verifier companion"
    if manifest is None:
        return
    schema_version = _positive_integer(manifest.get("schema_version"))
    if schema_version is None or schema_version < 2:
        diagnostics.missing(f"{label}.schema_version은 2 이상이어야 합니다")
    if manifest.get("status") != "completed":
        diagnostics.missing(f"{label}.status는 'completed'여야 합니다")
    actual_count = len(verification_rows or [])
    if not _numbers_match(manifest.get("row_count"), float(actual_count)):
        diagnostics.missing(
            f"{label}.row_count 불일치: 기록={manifest.get('row_count')!r}, 실제={actual_count}"
        )
    recorded_counts = manifest.get("verdict_counts")
    actual_observed = _verdict_counts(verification_rows)
    actual_counts = {verdict: actual_observed.get(verdict, 0) for verdict in ("match", "partial", "mismatch")}
    parsed_counts = (
        {str(key): _nonnegative_integer(value) for key, value in recorded_counts.items()}
        if isinstance(recorded_counts, Mapping)
        else None
    )
    if parsed_counts != actual_counts:
        diagnostics.missing(
            f"{label}.verdict_counts 불일치: 기록={recorded_counts!r}, 실제={actual_counts!r}"
        )
    qualitative_count = _nonnegative_integer(manifest.get("qualitative_row_count"))
    qualitative_counts = manifest.get("qualitative_verdict_counts")
    parsed_qualitative = (
        {str(key): _nonnegative_integer(value) for key, value in qualitative_counts.items()}
        if isinstance(qualitative_counts, Mapping)
        else None
    )
    if (
        qualitative_count is None
        or parsed_qualitative is None
        or set(parsed_qualitative) != {"match", "partial", "mismatch"}
        or any(value is None for value in parsed_qualitative.values())
        or sum(int(value) for value in parsed_qualitative.values() if value is not None)
        != qualitative_count
    ):
        diagnostics.missing(
            f"{label} qualitative row/count 계약 불일치: "
            f"row_count={manifest.get('qualitative_row_count')!r}, counts={qualitative_counts!r}"
        )
    _validate_source_summary_snapshot(
        manifest, summary_path, reproduction_dir, diagnostics, label
    )


def _validate_usage_contract(
    usage_rows: Sequence[Mapping[str, str]] | None,
    expected_keys: set[tuple[str, int, str]],
    diagnostics: Diagnostics,
) -> None:
    label = "강화 적응형 mode 사용량"
    usage = _index_canonical_rows(usage_rows, diagnostics, label)
    _validate_exact_coverage(set(usage), expected_keys, diagnostics, label)
    columns = sorted(
        {
            column
            for row in usage.values()
            for column in row
            if column.startswith("mode_") and column.endswith("_count")
        }
    )
    if not columns:
        diagnostics.missing(f"{label}에 mode_*_count 수치 열이 없습니다")
        return
    invalid: list[str] = []
    for key, row in usage.items():
        for column in columns:
            value = _float(row.get(column))
            if value is None or value < 0:
                invalid.append(f"{key}:{column}={row.get(column)!r}")
                if len(invalid) >= 40:
                    break
        if len(invalid) >= 40:
            break
    if invalid:
        diagnostics.missing(
            f"{label} count가 비었거나 비수치/비유한/음수입니다(최대 40개): {invalid}"
        )


def load_evidence(paths: InputPaths, diagnostics: Diagnostics) -> Evidence:
    enhanced_dir = _ensure_directory(paths.enhanced_dir, "강화 codec 결과")
    reproduction_dir = _ensure_directory(paths.reproduction_dir, "논문 재현 결과")
    adaptive_dir = _ensure_directory(paths.adaptive_dir, "적응형 후속 결과")
    _ensure_directory(paths.reports_dir, "보고서 템플릿")

    evidence = Evidence()

    evidence.enhanced_summary = _load_json(
        _artifact_path(enhanced_dir, "result_summary.json", diagnostics, "강화 codec 요약"),
        diagnostics,
        "강화 codec 요약",
    )
    evidence.enhanced_rates = _load_csv(
        _artifact_path(enhanced_dir, "rate_quality.csv", diagnostics, "강화 codec rate-quality"),
        diagnostics,
        "강화 codec rate-quality",
        {
            "active_channels",
            "measured_rho_uint8_over_raw_rgb",
            "measured_rho_zlib_over_raw_rgb",
            "mean_iou",
            "pixel_accuracy",
            "psnr_db",
            "ssim",
        },
    )
    evidence.enhanced_history = _load_csv(
        _artifact_path(enhanced_dir, "training_history.csv", diagnostics, "강화 codec 학습 이력"),
        diagnostics,
        "강화 codec 학습 이력",
        {"epoch", "train_loss", "val_loss", "val_mean_iou", "val_pixel_accuracy"},
    )
    evidence.enhanced_confusion = _load_csv(
        _artifact_path(
            enhanced_dir,
            "confusion_matrix_paper_like.csv",
            diagnostics,
            "강화 codec confusion matrix",
        ),
        diagnostics,
        "강화 codec confusion matrix",
    )
    evidence.enhanced_airtalking = _load_json(
        _artifact_path(
            enhanced_dir,
            "airtalking_semantic_summary.json",
            diagnostics,
            "강화 codec AirTalking profile",
        ),
        diagnostics,
        "강화 codec AirTalking profile",
    )
    _require_json_fields(
        evidence.enhanced_summary,
        {
            "status",
            "dataset.name",
            "dataset.train_samples",
            "dataset.val_samples",
            "dataset.image_size.width",
            "dataset.image_size.height",
            "model.encoder",
            "model.decoder",
            "model.paper_like_active_channels",
            "training.epochs_requested",
            "training.best_epoch",
            "paper_like_profile.mean_iou",
            "paper_like_profile.pixel_accuracy",
            "paper_like_profile.psnr_db",
            "paper_like_profile.ssim",
            "timing.encode_including_8bit_fake_quantization.median_ms",
            "timing.decode_from_latent_only.median_ms",
            "environment.device",
            "environment.command_windows",
            "elapsed_seconds",
            "artifacts.best_checkpoint",
            "artifacts.last_resume_checkpoint",
            "artifacts.final_training_checkpoint",
            "provenance.launch_manifest",
            "provenance.training_source_snapshot",
        },
        diagnostics,
        "강화 codec 요약",
    )
    _validate_json_numbers(
        evidence.enhanced_summary,
        {
            "dataset.train_samples",
            "dataset.val_samples",
            "dataset.image_size.width",
            "dataset.image_size.height",
            "model.paper_like_active_channels",
            "training.epochs_requested",
            "training.best_epoch",
            "paper_like_profile.mean_iou",
            "paper_like_profile.pixel_accuracy",
            "paper_like_profile.psnr_db",
            "paper_like_profile.ssim",
            "timing.encode_including_8bit_fake_quantization.median_ms",
            "timing.decode_from_latent_only.median_ms",
            "elapsed_seconds",
        },
        diagnostics,
        "강화 codec 요약",
    )
    if evidence.enhanced_summary is not None and evidence.enhanced_summary.get("status") != "completed":
        diagnostics.missing(
            "강화 codec result_summary.status가 'completed'가 아닙니다: "
            f"{evidence.enhanced_summary.get('status')!r}"
        )
    _require_json_fields(
        evidence.enhanced_airtalking,
        {
            "schema_version",
            "paper_like_active_channels",
            "rho_c_feature_uncompressed_mean",
            "semantic_quality_miou_final",
            "pixel_accuracy_final",
            "multi_rate_profiles",
        },
        diagnostics,
        "강화 codec AirTalking profile",
    )
    _validate_csv_cells(
        evidence.enhanced_rates,
        numeric_columns=(
            "active_channels",
            "measured_rho_uint8_over_raw_rgb",
            "measured_rho_zlib_over_raw_rgb",
            "mean_iou",
            "pixel_accuracy",
            "psnr_db",
            "ssim",
        ),
        diagnostics=diagnostics,
        label="강화 codec rate-quality",
    )
    actual_rates = {
        int(value)
        for row in evidence.enhanced_rates or []
        if (value := _float(row.get("active_channels"))) is not None
    }
    if evidence.enhanced_rates is not None and actual_rates != {20, 40, 60, 80, 120}:
        diagnostics.missing(
            "강화 codec rate-quality 활성 채널 계약 불일치: "
            f"기대=[20, 40, 60, 80, 120], 실제={sorted(actual_rates)}"
        )
    _validate_unique_csv_keys(
        evidence.enhanced_rates,
        ("active_channels",),
        diagnostics,
        "강화 codec rate-quality",
    )
    _validate_csv_cells(
        evidence.enhanced_history,
        numeric_columns=(
            "epoch",
            "train_loss",
            "val_loss",
            "val_mean_iou",
            "val_pixel_accuracy",
        ),
        diagnostics=diagnostics,
        label="강화 codec 학습 이력",
    )
    _validate_unique_csv_keys(
        evidence.enhanced_history,
        ("epoch",),
        diagnostics,
        "강화 codec 학습 이력",
    )
    if evidence.enhanced_confusion is not None:
        if len(evidence.enhanced_confusion) != 19 or any(len(row) != 20 for row in evidence.enhanced_confusion):
            diagnostics.missing(
                "강화 codec confusion matrix는 ground-truth label 열 + 19×19 수치여야 합니다: "
                f"행={len(evidence.enhanced_confusion)}, 열={len(evidence.enhanced_confusion[0]) if evidence.enhanced_confusion else 0}"
            )
        elif evidence.enhanced_confusion:
            headers = list(evidence.enhanced_confusion[0])
            _validate_csv_cells(
                evidence.enhanced_confusion,
                numeric_columns=headers[1:],
                text_columns=(headers[0],),
                diagnostics=diagnostics,
                label="강화 codec confusion matrix",
            )
    _check_artifact_map(evidence.enhanced_summary, enhanced_dir, diagnostics)
    _check_path_references(
        evidence.enhanced_summary,
        ("provenance.launch_manifest", "provenance.training_source_snapshot"),
        enhanced_dir,
        diagnostics,
        "강화 codec 요약",
    )

    reproduction_summary_path = _artifact_path(
        reproduction_dir, "summary_metrics.csv", diagnostics, "강화 논문 재현 요약"
    )
    reproduction_repeat_path = _artifact_path(
        reproduction_dir, "repeat_metrics.csv", diagnostics, "강화 논문 재현 repeat 원시값"
    )
    reproduction_statistics_path = _artifact_path(
        reproduction_dir, "statistical_summary.csv", diagnostics, "강화 논문 재현 통계 요약"
    )
    reproduction_verification_path = _find_one_csv(
        reproduction_dir,
        "verification_against_paper*.csv",
        diagnostics,
        "강화 논문 재현 verifier",
    )
    reproduction_manifest_path = (
        _artifact_path(
            reproduction_dir,
            reproduction_verification_path.with_suffix(".json").name,
            diagnostics,
            "강화 논문 재현 verifier companion",
        )
        if reproduction_verification_path is not None
        else None
    )
    evidence.reproduction_summary = _load_csv(
        reproduction_summary_path,
        diagnostics,
        "강화 논문 재현 요약",
        {
            "mode",
            "area",
            "policy",
            "finished",
            "flight_energy_per_req",
            "avg_time",
            "avg_travel",
            "encodes",
            "decodes",
            "nonflight_energy_per_req",
            "semantic_quality",
            "semantic_payload_ratio",
            "sinr_median_db",
        },
    )
    evidence.reproduction_repeats = _load_csv(
        reproduction_repeat_path,
        diagnostics,
        "강화 논문 재현 repeat 원시값",
        {"mode", "area", "policy", "repeat", *CORE_REPEAT_METRICS},
    )
    evidence.reproduction_statistics = _load_csv(
        reproduction_statistics_path,
        diagnostics,
        "강화 논문 재현 통계 요약",
        {
            "mode",
            "area",
            "policy",
            "repeat_count",
            "ci95_method",
            *{
                f"{metric}_{suffix}"
                for metric in CORE_REPEAT_METRICS
                for suffix in ("n", "mean", "std", "ci95_margin", "ci95_low", "ci95_high")
            },
        },
    )
    evidence.reproduction_verification = _load_csv(
        reproduction_verification_path,
        diagnostics,
        "강화 논문 재현 verifier",
        {"check", "metric", "paper_visual_estimate", "reproduction", "verdict"},
    )
    evidence.reproduction_verification_manifest = _load_json(
        reproduction_manifest_path,
        diagnostics,
        "강화 논문 재현 verifier companion",
    )
    evidence.reproduction_metadata = _load_json(
        _artifact_path(
            reproduction_dir, "run_metadata.json", diagnostics, "강화 논문 재현 metadata"
        ),
        diagnostics,
        "강화 논문 재현 metadata",
    )
    _validate_csv_cells(
        evidence.reproduction_summary,
        numeric_columns=("area", *CORE_REPEAT_METRICS),
        text_columns=("mode", "policy"),
        diagnostics=diagnostics,
        label="강화 논문 재현 요약",
    )
    _validate_unique_csv_keys(
        evidence.reproduction_summary,
        ("mode", "area", "policy"),
        diagnostics,
        "강화 논문 재현 요약",
    )
    _validate_csv_cells(
        evidence.reproduction_repeats,
        numeric_columns=("area", "repeat", *CORE_REPEAT_METRICS),
        text_columns=("mode", "policy"),
        diagnostics=diagnostics,
        label="강화 논문 재현 repeat 원시값",
    )
    _validate_unique_csv_keys(
        evidence.reproduction_repeats,
        ("mode", "area", "policy", "repeat"),
        diagnostics,
        "강화 논문 재현 repeat 원시값",
    )
    _validate_csv_cells(
        evidence.reproduction_statistics,
        numeric_columns=(
            "area",
            "repeat_count",
            *tuple(
                f"{metric}_{suffix}"
                for metric in CORE_REPEAT_METRICS
                for suffix in ("n", "mean", "std", "ci95_margin", "ci95_low", "ci95_high")
            ),
        ),
        text_columns=("mode", "policy", "ci95_method"),
        diagnostics=diagnostics,
        label="강화 논문 재현 통계 요약",
    )
    _validate_unique_csv_keys(
        evidence.reproduction_statistics,
        ("mode", "area", "policy"),
        diagnostics,
        "강화 논문 재현 통계 요약",
    )
    _validate_csv_cells(
        evidence.reproduction_verification,
        numeric_columns=("paper_visual_estimate", "reproduction"),
        text_columns=("check", "metric", "verdict"),
        diagnostics=diagnostics,
        label="강화 논문 재현 verifier",
    )
    _require_json_fields(
        evidence.reproduction_metadata,
        {"paper_params.repeats", "paper_params.t_slots", "semantic_profile"},
        diagnostics,
        "강화 논문 재현 metadata",
    )
    _check_path_references(
        evidence.reproduction_metadata,
        (
            "semantic_profile.source",
            "summary_metrics_csv",
            "repeat_metrics_csv",
            "statistical_summary_csv",
        ),
        reproduction_dir,
        diagnostics,
        "강화 논문 재현 metadata",
    )
    _validate_repeat_statistics(
        summary_rows=evidence.reproduction_summary,
        repeat_rows=evidence.reproduction_repeats,
        statistics_rows=evidence.reproduction_statistics,
        expected_keys=_expected_reproduction_keys(),
        repeats_expected=_positive_integer(
            _nested_get(evidence.reproduction_metadata, "paper_params.repeats")
        ),
        diagnostics=diagnostics,
        label="강화 논문 재현",
    )
    _validate_reproduction_verifier_contract(
        evidence.reproduction_verification_manifest,
        reproduction_summary_path,
        evidence.reproduction_verification,
        reproduction_dir,
        diagnostics,
    )

    adaptive_summary_path = _artifact_path(
        adaptive_dir, "summary_metrics.csv", diagnostics, "강화 적응형 요약"
    )
    evidence.adaptive_summary = _load_csv(
        adaptive_summary_path,
        diagnostics,
        "강화 적응형 요약",
        {
            "mode",
            "area",
            "policy",
            "finished",
            "flight_energy_per_req",
            "avg_time",
            "semantic_quality",
            "semantic_payload_ratio",
            "nonflight_energy_per_req",
            "avg_travel",
            "encodes",
            "decodes",
            "sinr_median_db",
        },
    )
    evidence.adaptive_repeats = _load_csv(
        _artifact_path(adaptive_dir, "repeat_metrics.csv", diagnostics, "강화 적응형 repeat 원시값"),
        diagnostics,
        "강화 적응형 repeat 원시값",
        {"mode", "area", "policy", "repeat", *CORE_REPEAT_METRICS},
    )
    evidence.adaptive_statistics = _load_csv(
        _artifact_path(
            adaptive_dir, "statistical_summary.csv", diagnostics, "강화 적응형 통계 요약"
        ),
        diagnostics,
        "강화 적응형 통계 요약",
        {
            "mode",
            "area",
            "policy",
            "repeat_count",
            "ci95_method",
            "finished_mean",
            "finished_ci95_low",
            "finished_ci95_high",
            "avg_time_mean",
            "avg_time_ci95_low",
            "avg_time_ci95_high",
            "semantic_quality_mean",
            "semantic_quality_ci95_low",
            "semantic_quality_ci95_high",
            *{
                f"{metric}_{suffix}"
                for metric in CORE_REPEAT_METRICS
                for suffix in ("n", "mean", "std", "ci95_margin", "ci95_low", "ci95_high")
            },
        },
    )
    evidence.adaptive_usage = _load_csv(
        _artifact_path(
            adaptive_dir, "compression_mode_usage.csv", diagnostics, "강화 적응형 mode 사용량"
        ),
        diagnostics,
        "강화 적응형 mode 사용량",
        {"mode", "area", "policy"},
    )
    evidence.adaptive_validation = _load_json(
        _artifact_path(
            adaptive_dir, "result_validation.json", diagnostics, "강화 적응형 검증"
        ),
        diagnostics,
        "강화 적응형 검증",
    )
    evidence.adaptive_metadata = _load_json(
        _artifact_path(adaptive_dir, "run_metadata.json", diagnostics, "강화 적응형 metadata"),
        diagnostics,
        "강화 적응형 metadata",
    )
    _validate_csv_cells(
        evidence.adaptive_summary,
        numeric_columns=("area", *CORE_REPEAT_METRICS),
        text_columns=("mode", "policy"),
        diagnostics=diagnostics,
        label="강화 적응형 요약",
    )
    _validate_unique_csv_keys(
        evidence.adaptive_summary,
        ("mode", "area", "policy"),
        diagnostics,
        "강화 적응형 요약",
    )
    _validate_csv_cells(
        evidence.adaptive_repeats,
        numeric_columns=(
            "area",
            "repeat",
            *CORE_REPEAT_METRICS,
        ),
        text_columns=("mode", "policy"),
        diagnostics=diagnostics,
        label="강화 적응형 repeat 원시값",
    )
    _validate_unique_csv_keys(
        evidence.adaptive_repeats,
        ("mode", "area", "policy", "repeat"),
        diagnostics,
        "강화 적응형 repeat 원시값",
    )
    _validate_csv_cells(
        evidence.adaptive_statistics,
        numeric_columns=(
            "area",
            "repeat_count",
            *tuple(
                f"{metric}_{suffix}"
                for metric in CORE_REPEAT_METRICS
                for suffix in ("n", "mean", "std", "ci95_margin", "ci95_low", "ci95_high")
            ),
        ),
        text_columns=("mode", "policy", "ci95_method"),
        diagnostics=diagnostics,
        label="강화 적응형 통계 요약",
    )
    _validate_unique_csv_keys(
        evidence.adaptive_statistics,
        ("mode", "area", "policy"),
        diagnostics,
        "강화 적응형 통계 요약",
    )
    _require_json_fields(
        evidence.adaptive_validation,
        {
            "schema_version",
            "source_summary.path",
            "source_summary.sha256",
            "passed",
            "row_count",
            "expected_row_count",
            "expected_combinations.modes",
            "expected_combinations.areas",
            "expected_combinations.policies",
            "expected_combinations.count",
            "errors",
            "missing_combinations",
            "duplicate_combinations",
            "nonfinite_values",
            "adaptive_vs_fixed",
        },
        diagnostics,
        "강화 적응형 검증",
    )
    _require_json_fields(
        evidence.adaptive_metadata,
        {"base_paper_params.repeats", "base_paper_params.t_slots", "profiles", "elapsed_seconds"},
        diagnostics,
        "강화 적응형 metadata",
    )
    _check_path_references(
        evidence.adaptive_metadata,
        (
            "source_metadata",
            "source_quality",
            "source_neural_encoder_decoder",
            "repeat_metrics_csv",
            "statistical_summary_csv",
            "artifacts.summary_metrics_csv.path",
            "artifacts.repeat_metrics_csv.path",
            "artifacts.statistical_summary_csv.path",
            "artifacts.compression_mode_usage_csv.path",
        ),
        adaptive_dir,
        diagnostics,
        "강화 적응형 metadata",
    )
    adaptive_expected_keys = _expected_adaptive_keys()
    _validate_repeat_statistics(
        summary_rows=evidence.adaptive_summary,
        repeat_rows=evidence.adaptive_repeats,
        statistics_rows=evidence.adaptive_statistics,
        expected_keys=adaptive_expected_keys,
        repeats_expected=_positive_integer(
            _nested_get(evidence.adaptive_metadata, "base_paper_params.repeats")
        ),
        diagnostics=diagnostics,
        label="강화 적응형",
    )
    _validate_usage_contract(evidence.adaptive_usage, adaptive_expected_keys, diagnostics)
    _validate_adaptive_validation_contract(
        evidence.adaptive_validation,
        adaptive_summary_path,
        evidence.adaptive_summary,
        adaptive_dir,
        diagnostics,
    )

    if paths.baseline_neural_dir is not None:
        baseline_neural = _ensure_directory(paths.baseline_neural_dir, "기존 neural 기준선")
        evidence.baseline_neural_dir = baseline_neural
        evidence.baseline_neural_summary = _load_json(
            _artifact_path(
                baseline_neural, "result_summary.json", diagnostics, "기존 neural 기준선 요약"
            ),
            diagnostics,
            "기존 neural 기준선 요약",
        )
        evidence.baseline_neural_airtalking = _load_json(
            _artifact_path(
                baseline_neural,
                "airtalking_semantic_summary.json",
                diagnostics,
                "기존 neural 기준선 AirTalking profile",
            ),
            diagnostics,
            "기존 neural 기준선 AirTalking profile",
        )
    else:
        diagnostics.missing("기존 neural 기준선 디렉터리가 지정되지 않았습니다")

    if paths.baseline_reproduction_dir is not None:
        baseline_repro = _ensure_directory(
            paths.baseline_reproduction_dir, "기존 논문 재현 기준선"
        )
        evidence.baseline_reproduction_summary = _load_csv(
            _artifact_path(
                baseline_repro,
                "summary_metrics.csv",
                diagnostics,
                "기존 논문 재현 기준선 요약",
            ),
            diagnostics,
            "기존 논문 재현 기준선 요약",
            {"mode", "area", "policy", "finished", "avg_time", "flight_energy_per_req"},
        )
        evidence.baseline_reproduction_verification = _load_csv(
            _find_one_csv(
                baseline_repro,
                "verification_against_paper*.csv",
                diagnostics,
                "기존 논문 재현 verifier",
            ),
            diagnostics,
            "기존 논문 재현 verifier",
            {"check", "verdict"},
        )
    else:
        diagnostics.missing("기존 논문 재현 기준선 디렉터리가 지정되지 않았습니다")

    if paths.baseline_adaptive_dir is not None:
        baseline_adaptive = _ensure_directory(paths.baseline_adaptive_dir, "기존 적응형 기준선")
        evidence.baseline_adaptive_summary = _load_csv(
            _artifact_path(
                baseline_adaptive,
                "summary_metrics.csv",
                diagnostics,
                "기존 적응형 기준선 요약",
            ),
            diagnostics,
            "기존 적응형 기준선 요약",
            {"mode", "area", "policy", "finished", "avg_time"},
        )

    diagnostics.fail_if_errors()
    return evidence


def _missing(label: str, detail: str | None = None) -> str:
    suffix = f" ({detail})" if detail else ""
    return f"**상태: 미실행/증거 없음.** 다음 항목을 뒷받침하는 산출물이 없습니다: {label}{suffix}."


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(value: Any, digits: int = 3, *, comma: bool = False) -> str:
    number = _float(value)
    if number is None:
        return "미실행/증거 없음"
    spec = f",.{digits}f" if comma else f".{digits}f"
    return format(number, spec)


def _fmt_int(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "미실행/증거 없음"
    return f"{int(round(number)):,}"


def _pct(value: Any, digits: int = 1, *, signed: bool = False) -> str:
    number = _float(value)
    if number is None:
        return "미실행/증거 없음"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}%"


def _ratio_delta(new: Any, old: Any) -> float | None:
    new_value, old_value = _float(new), _float(old)
    if new_value is None or old_value in (None, 0.0):
        return None
    return (new_value - old_value) / old_value * 100.0


def _md_cell(value: Any) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    materialized = [list(row) for row in rows]
    if not materialized:
        return _missing("표")
    lines = [
        "| " + " | ".join(_md_cell(item) for item in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in materialized:
        if len(row) != len(headers):
            raise ReportFinalizationError(
                f"내부 보고서 표 열 수 불일치: header={len(headers)}, row={len(row)}"
            )
        lines.append("| " + " | ".join(_md_cell(item) for item in row) + " |")
    return "\n".join(lines)


def _rows_for_mode(
    rows: Sequence[Mapping[str, str]] | None, mode: str
) -> list[Mapping[str, str]]:
    if rows is None:
        return []
    return [row for row in rows if row.get("mode") == mode]


def _sort_area_policy(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    policy_order = {name: index for index, name in enumerate(("Stochastic", "LinUCB", "SA", "Greedy", "MCTS"))}
    return sorted(
        rows,
        key=lambda row: (
            _float(row.get("area")) if _float(row.get("area")) is not None else math.inf,
            policy_order.get(row.get("policy", ""), 999),
        ),
    )


def _index_rows(
    rows: Sequence[Mapping[str, str]] | None,
) -> dict[tuple[str, int, str], Mapping[str, str]]:
    index: dict[tuple[str, int, str], Mapping[str, str]] = {}
    for row in rows or []:
        area = _float(row.get("area"))
        if area is None:
            continue
        index[(row.get("mode", ""), int(area), row.get("policy", ""))] = row
    return index


def _mean(rows: Iterable[Mapping[str, str]], key: str) -> float | None:
    values = [_float(row.get(key)) for row in rows]
    valid = [value for value in values if value is not None]
    return statistics.fmean(valid) if valid else None


def _rate_quality_table(evidence: Evidence) -> str:
    if not evidence.enhanced_rates:
        return _missing("강화 codec의 rate-quality 결과", "rate_quality.csv 없음")
    rows = sorted(
        evidence.enhanced_rates,
        key=lambda row: _float(row.get("active_channels")) or math.inf,
    )
    return md_table(
        ("활성 채널", "지점", "ρ uint8", "ρ zlib", "mIoU", "pixel acc.", "PSNR(dB)", "SSIM", "평가 표본"),
        (
            (
                _fmt_int(row.get("active_channels")),
                row.get("operating_point") or "기록 없음",
                _fmt(row.get("measured_rho_uint8_over_raw_rgb"), 6),
                _fmt(row.get("measured_rho_zlib_over_raw_rgb"), 6),
                _fmt(row.get("mean_iou"), 6),
                _fmt(row.get("pixel_accuracy"), 6),
                _fmt(row.get("psnr_db"), 3),
                _fmt(row.get("ssim"), 6),
                _fmt_int(row.get("evaluated_samples")),
            )
            for row in rows
        ),
    )


def _render_dataset(evidence: Evidence) -> str:
    summary = evidence.enhanced_summary
    dataset = _nested_get(summary, "dataset")
    if not isinstance(dataset, Mapping):
        return _missing("강화 학습 데이터셋 provenance", "result_summary.json의 dataset 없음")
    size = dataset.get("image_size") if isinstance(dataset.get("image_size"), Mapping) else {}
    augmentation = dataset.get("train_augmentation")
    fingerprint = dataset.get("fingerprint") or dataset.get("dataset_fingerprint")
    table = md_table(
        ("항목", "기록값"),
        (
            ("데이터셋", dataset.get("name", "기록 없음")),
            ("RGB root", dataset.get("image_root", "기록 없음")),
            ("label root", dataset.get("label_root", "기록 없음")),
            ("발견한 train/val pair", f"{_fmt_int(dataset.get('available_train_pairs'))} / {_fmt_int(dataset.get('available_val_pairs'))}"),
            ("실제 사용 train/val", f"{_fmt_int(dataset.get('train_samples'))} / {_fmt_int(dataset.get('val_samples'))}"),
            ("입력 크기", f"{_fmt_int(size.get('width'))}×{_fmt_int(size.get('height'))}"),
            ("전체 데이터 사용", str(dataset.get("full_data", "기록 없음"))),
            ("증강 설정", json.dumps(augmentation, ensure_ascii=False, sort_keys=True) if augmentation is not None else "기록 없음"),
            ("검증 변환", dataset.get("validation_transform", "기록 없음")),
            ("암호학적 fingerprint", fingerprint or "기록 없음"),
        ),
    )
    note = (
        "`result_summary.json`에 기록된 경로·pair 수·사용 표본·변환 설정을 그대로 옮겼다. "
        + ("fingerprint 값도 함께 기록되어 파일 집합 동일성을 확인할 수 있다." if fingerprint else "파일 목록 해시 또는 fingerprint 필드는 기록되지 않았으므로, 이 표만으로 파일 byte 단위 동일성을 증명할 수는 없다.")
    )
    return table + "\n\n" + note


def _render_environment(evidence: Evidence) -> str:
    env = _nested_get(evidence.enhanced_summary, "environment")
    if not isinstance(env, Mapping):
        return _missing("강화 학습 실행 환경")
    return md_table(
        ("항목", "실행 기록"),
        (
            ("UTC 시각", env.get("timestamp_utc", "기록 없음")),
            ("OS", env.get("platform", "기록 없음")),
            ("Python", env.get("python_version", "기록 없음")),
            ("PyTorch / torchvision", f"{env.get('torch_version', '기록 없음')} / {env.get('torchvision_version', '기록 없음')}"),
            ("장치 / GPU", f"{env.get('device', '기록 없음')} / {env.get('gpu_name', '기록 없음')}"),
            ("CUDA / cuDNN", f"{env.get('cuda_version', '기록 없음')} / {env.get('cudnn_version', '기록 없음')}"),
            ("NumPy / Pillow", f"{env.get('numpy_version', '기록 없음')} / {env.get('pillow_version', '기록 없음')}"),
            ("git commit", env.get("git_commit") or "기록 없음"),
        ),
    )


def _render_command_inline(evidence: Evidence) -> str:
    command = _nested_get(evidence.enhanced_summary, "environment.command_windows")
    if not isinstance(command, str) or not command.strip():
        return "미실행/증거 없음(실제 명령 미기록)"
    return "`" + command.replace("`", "\\`").replace("|", "\\|") + "`"


def _render_best_epoch(evidence: Evidence) -> str:
    value = _nested_get(evidence.enhanced_summary, "training.best_epoch")
    return _fmt_int(value) if _float(value) is not None else "미실행/증거 없음"


def _render_sample_counts_inline(evidence: Evidence) -> str:
    train = _nested_get(evidence.enhanced_summary, "dataset.train_samples")
    val = _nested_get(evidence.enhanced_summary, "dataset.val_samples")
    if _float(train) is None or _float(val) is None:
        return "미실행/증거 없음"
    return f"{_fmt_int(train)} / {_fmt_int(val)}"


def _render_elapsed_inline(evidence: Evidence) -> str:
    value = _nested_get(evidence.enhanced_summary, "elapsed_seconds")
    if _float(value) is None:
        return "미실행/증거 없음"
    return f"{_fmt(value, 1, comma=True)}초 ({_fmt(float(value) / 3600.0, 2)}시간)"


def _render_status_inline(evidence: Evidence) -> str:
    status = _nested_get(evidence.enhanced_summary, "status")
    if status is None:
        return "미실행/증거 없음"
    last_epoch = _nested_get(evidence.enhanced_summary, "training.last_completed_epoch")
    requested = _nested_get(evidence.enhanced_summary, "training.epochs_requested")
    detail = ""
    if _float(last_epoch) is not None and _float(requested) is not None:
        detail = f"; epoch {_fmt_int(last_epoch)}/{_fmt_int(requested)}"
    return f"{status}{detail} (result_summary.json 기록)"


def _render_paperlike(evidence: Evidence) -> str:
    profile = _nested_get(evidence.enhanced_summary, "paper_like_profile")
    if not isinstance(profile, Mapping):
        return _missing("80채널 paper-like 평가")
    baseline = evidence.baseline_neural_airtalking or {}
    table = md_table(
        ("항목", "강화 80채널", "재학습·CPU 재평가 기준선"),
        (
            ("ρ uint8 / raw RGB", _fmt(profile.get("measured_rho_uint8_over_raw_rgb"), 7), _fmt(baseline.get("rho_c_feature_uncompressed_mean"), 7)),
            ("ρ zlib / raw RGB", _fmt(profile.get("measured_rho_zlib_over_raw_rgb"), 7), "미측정"),
            ("mIoU", _fmt(profile.get("mean_iou"), 6), _fmt(baseline.get("semantic_quality_miou_final"), 6)),
            ("pixel accuracy", _fmt(profile.get("pixel_accuracy"), 6), _fmt(baseline.get("pixel_accuracy_final"), 6)),
            ("RGB PSNR", _fmt(profile.get("psnr_db"), 3), "미측정"),
            ("RGB SSIM", _fmt(profile.get("ssim"), 6), "미측정"),
            ("평가 표본", _fmt_int(profile.get("evaluated_samples")), _fmt_int(baseline.get("num_samples"))),
        ),
    )
    baseline_source = (
        evidence.baseline_neural_dir.name
        if isinstance(evidence.baseline_neural_dir, Path)
        else "기록 없음"
    )
    return table + (
        f"\n\n이 표의 CPU 기준선 입력 디렉터리는 `{baseline_source}`다. "
        "작업 전 원래 저장 결과를 사용하는 04 비교 보고서와는 기준선 run이 다를 수 있다."
        "\n\nρ 값은 실제 직렬화된 uint8 latent byte를 raw uint8 RGB byte로 나눈 값이다. "
        "기존 기준선의 ρ는 8비트 가정값이므로 측정 의미가 동일하지 않다. "
        "또한 기존 기준선은 128×64, train/val 512/256 부분집합, segmentation-only 설정이고 "
        "강화 모델은 256×128, train/val 2,975/500 전체 분할, RGB+분할 이중 decoder이다. "
        "따라서 mIoU 차이는 전체 시스템 강화의 결과이지, 모델 구조 하나만의 통제된 ablation 효과가 아니다. "
        "논문 latent 규약이 비공개여서 0.104에 가깝다는 사실만으로 저자 codec 재현이라고 할 수 없다."
    )


def _render_timing(evidence: Evidence) -> str:
    summary = evidence.enhanced_summary
    timing = _nested_get(summary, "timing")
    env = _nested_get(summary, "environment")
    model = _nested_get(summary, "model")
    dataset = _nested_get(summary, "dataset")
    training = _nested_get(summary, "training")
    if not all(isinstance(item, Mapping) for item in (timing, env, model, dataset, training)):
        return _missing("강화 codec 속도·자원 기록")
    size = dataset.get("image_size") if isinstance(dataset.get("image_size"), Mapping) else {}
    return md_table(
        ("항목", "기록값"),
        (
            ("장치", f"{env.get('device', '기록 없음')} / {env.get('gpu_name', '기록 없음')}"),
            ("입력 / timing batch", f"{_fmt_int(size.get('width'))}×{_fmt_int(size.get('height'))} / {_fmt_int(timing.get('batch_size'))}"),
            ("timing 반복 / CUDA 동기화", f"{_fmt_int(timing.get('runs'))} / {timing.get('cuda_synchronized', '기록 없음')}"),
            ("encode median", _fmt(_nested_get(timing, "encode_including_8bit_fake_quantization.median_ms"), 3) + " ms"),
            ("decode median", _fmt(_nested_get(timing, "decode_from_latent_only.median_ms"), 3) + " ms"),
            ("full forward median", _fmt(_nested_get(timing, "full_forward.median_ms"), 3) + " ms"),
            ("학습 가능 / 전체 parameter", f"{_fmt_int(model.get('trainable_parameters'))} / {_fmt_int(model.get('total_parameters'))}"),
            ("AMP 요청 / 실제", f"{training.get('amp_requested', '기록 없음')} / {training.get('amp_effective', '기록 없음')}"),
            ("zlib 시간 포함", "아니오" if timing.get("excludes_cpu_zlib_transport") is True else "기록 확인 필요"),
        ),
    )


def _render_training_curve(evidence: Evidence) -> str:
    history = evidence.enhanced_history
    if not history:
        return _missing("강화 codec 학습 곡선", "training_history.csv 없음")
    best_epoch = _nested_get(evidence.enhanced_summary, "training.best_epoch")
    selected = [history[0]]
    if len(history) > 1:
        selected.append(history[-1])
    for row in history:
        if _float(row.get("epoch")) == _float(best_epoch) and row not in selected:
            selected.append(row)
    selected.sort(key=lambda row: _float(row.get("epoch")) or math.inf)
    table = md_table(
        ("epoch", "train loss", "val loss", "val mIoU", "val pixel acc.", "val PSNR", "val SSIM", "LR"),
        (
            (
                _fmt_int(row.get("epoch")),
                _fmt(row.get("train_loss"), 6),
                _fmt(row.get("val_loss"), 6),
                _fmt(row.get("val_mean_iou"), 6),
                _fmt(row.get("val_pixel_accuracy"), 6),
                _fmt(row.get("val_psnr_db"), 3),
                _fmt(row.get("val_ssim"), 6),
                _fmt(row.get("learning_rate"), 8),
            )
            for row in selected
        ),
    )
    return f"학습 이력에는 총 {len(history):,}개 epoch 행이 있다. 아래 표는 첫 epoch, 마지막 epoch, best epoch(중복 제외)를 보여 준다.\n\n{table}"


def _confusion_accuracy(rows: Sequence[Mapping[str, str]] | None) -> tuple[float | None, int]:
    if not rows:
        return None, 0
    headers = list(rows[0])
    row_label_candidates = [header for header in headers if any(_float(row.get(header)) is None for row in rows)]
    row_label = row_label_candidates[0] if row_label_candidates else headers[0]
    numeric_headers = [header for header in headers if header != row_label]
    total = 0.0
    diagonal = 0.0
    for index, row in enumerate(rows):
        for header in numeric_headers:
            value = _float(row.get(header))
            if value is not None:
                total += value
        label = str(row.get(row_label, ""))
        if label in numeric_headers:
            diagonal_value = _float(row.get(label))
        elif index < len(numeric_headers):
            diagonal_value = _float(row.get(numeric_headers[index]))
        else:
            diagonal_value = None
        if diagonal_value is not None:
            diagonal += diagonal_value
    return (diagonal / total if total > 0 else None), int(total)


def _path_from_artifact(
    summary: Mapping[str, Any] | None, key: str, result_dir: Path
) -> Path | None:
    raw = _nested_get(summary, f"artifacts.{key}")
    if not isinstance(raw, str):
        return None
    return _resolve_reference(raw, result_dir, require_file=True)


def _relative_markdown_path(path: Path, output_dir: Path) -> str:
    try:
        relative = os.path.relpath(path, output_dir)
    except ValueError:
        return path.as_posix()
    return Path(relative).as_posix()


def _markdown_destination(path: str) -> str:
    """Use angle brackets when a local Markdown destination contains spaces."""
    return f"<{path}>" if any(character.isspace() for character in path) else path


def _render_qualitative(evidence: Evidence, output_dir: Path, enhanced_dir: Path) -> str:
    count = _nested_get(evidence.enhanced_summary, "qualitative_samples_saved")
    accuracy, pixels = _confusion_accuracy(evidence.enhanced_confusion)
    panel = _path_from_artifact(
        evidence.enhanced_summary, "qualitative_panel_png", enhanced_dir
    )
    parts = [
        md_table(
            ("감사 항목", "기록"),
            (
                ("정성 panel 표본 수", _fmt_int(count)),
                ("confusion matrix 총 유효 pixel", _fmt_int(pixels)),
                ("confusion matrix 대각 비율", _fmt(accuracy, 6)),
                ("panel 파일", str(panel) if panel is not None else "기록 없음"),
            ),
        )
    ]
    if panel is not None and panel.is_file():
        parts.append(
            "아래 panel은 입력 RGB, 복원 RGB, 정답/예측 segmentation의 정성 점검용이다. 작은 표본이므로 정량 지표를 대체하지 않는다.\n\n"
            f"![강화 codec paper-like 정성 panel]({_markdown_destination(_relative_markdown_path(panel, output_dir))})"
        )
    else:
        parts.append(_missing("정성 panel 그림", "파일 경로가 없거나 현재 위치에 없음"))
    return "\n\n".join(parts)


def _append_existing_figures(
    body: str,
    output_dir: Path,
    figures: Sequence[tuple[str, str, Path]],
) -> str:
    """Append only verified local figures; synthetic fixtures need not provide them."""
    blocks = [body]
    for alt_text, explanation, path in figures:
        if not path.is_file():
            continue
        destination = _markdown_destination(
            _relative_markdown_path(path.resolve(), output_dir)
        )
        blocks.append(f"{explanation}\n\n![{alt_text}]({destination})")
    return "\n\n".join(blocks)


def _render_no_neural_ablation(_: Evidence) -> str:
    return _missing(
        "강화 codec 인과적 ablation 결과",
        "요청된 optimizer/loss/pretraining/prefix-vs-rate별 모델을 한 요인씩 바꾼 별도 결과 파일이 없음; rate-quality 5점은 ablation이 아니라 한 scalable 모델의 operating point 평가임",
    )


def _render_semantic_profile(evidence: Evidence) -> str:
    profile = evidence.enhanced_airtalking
    if not isinstance(profile, Mapping):
        return _missing("강화 AirTalking semantic profile")
    encode_bitrate = profile.get(
        "feature_encode_bitrate_mbps_median",
        profile.get("encode_bitrate_mbps_median"),
    )
    decode_bitrate = profile.get(
        "feature_decode_bitrate_mbps_median",
        profile.get("decode_bitrate_mbps_median"),
    )
    return md_table(
        ("필드", "profile 기록값"),
        (
            ("schema version", profile.get("schema_version", "기록 없음")),
            ("출처 분류", _nested_get(profile, "scientific_scope.classification") or profile.get("source", "기록 없음")),
            ("paper-like 채널", _fmt_int(profile.get("paper_like_active_channels"))),
            ("ρ uint8", _fmt(profile.get("rho_c_feature_uncompressed_mean"), 7)),
            ("ρ zlib", _fmt(profile.get("rho_c_feature_zlib_mean"), 7)),
            ("mIoU / pixel acc.", f"{_fmt(profile.get('semantic_quality_miou_final'), 6)} / {_fmt(profile.get('pixel_accuracy_final'), 6)}"),
            ("RGB PSNR / SSIM", f"{_fmt(profile.get('rgb_reconstruction_psnr_db'), 3)} / {_fmt(profile.get('rgb_reconstruction_ssim'), 6)}"),
            ("encode / decode 처리율", f"{_fmt(encode_bitrate, 3)} / {_fmt(decode_bitrate, 3)} Mbps"),
            ("평가 표본", _fmt_int(profile.get("num_samples"))),
            ("multi-rate 지점 수", _fmt_int(len(profile.get("multi_rate_profiles", [])) if isinstance(profile.get("multi_rate_profiles"), list) else None)),
        ),
    )


def _render_reproduction_provenance(evidence: Evidence, diagnostics: Diagnostics) -> str:
    metadata = evidence.reproduction_metadata
    if not isinstance(metadata, Mapping):
        return _missing("강화 논문 재현 provenance")
    paper = metadata.get("paper_params") if isinstance(metadata.get("paper_params"), Mapping) else {}
    assumed = metadata.get("assumed_params") if isinstance(metadata.get("assumed_params"), Mapping) else {}
    semantic = metadata.get("semantic_profile") if isinstance(metadata.get("semantic_profile"), Mapping) else {}
    stale_warnings = [warning for warning in diagnostics.warnings if warning.startswith("강화 논문 재현 metadata")]
    table = md_table(
        ("항목", "metadata 기록"),
        (
            ("repeat × slot", f"{_fmt_int(paper.get('repeats'))} × {_fmt_int(paper.get('t_slots'))}"),
            ("UAV / device", f"{_fmt_int(paper.get('n_uav'))} / {_fmt_int(paper.get('n_device'))}"),
            ("seed", _fmt_int(assumed.get("seed"))),
            ("request probability", _fmt(assumed.get("request_probability"), 6)),
            ("semantic profile 적용", semantic.get("applied", "기록 없음")),
            ("profile source", semantic.get("source", "기록 없음")),
            ("profile kind / raw basis", f"{semantic.get('profile_kind', '기록 없음')} / {semantic.get('raw_basis', '기록 없음')}"),
            ("ρc / ρr", f"{_fmt(semantic.get('rho_c'), 7)} / {_fmt(semantic.get('rho_r'), 3)}"),
            ("전체 실행 시간", _fmt(metadata.get("elapsed_seconds"), 1, comma=True) + "초"),
        ),
    )
    if stale_warnings:
        table += "\n\n**경로 감사 경고:** " + " ".join(stale_warnings)
    return table


def _recorded_command(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    for path in (
        "run_provenance.command_windows",
        "environment.command_windows",
        "command_windows",
        "command",
    ):
        value = _nested_get(metadata, path)
        if isinstance(value, str) and value.strip():
            return value
    for path in ("run_provenance.argv", "argv"):
        argv = _nested_get(metadata, path)
        if isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv):
            return " ".join(argv)
    return None


def _render_reproduction_command(evidence: Evidence) -> str:
    command = _recorded_command(evidence.reproduction_metadata)
    if command is None:
        return _missing(
            "강화 논문 재현의 실제 CLI 원문",
            "run_metadata.json에 command/argv가 기록되지 않음; 결과 파일 존재만으로 실행 명령을 역추정하지 않음",
        )
    return "```powershell\n" + command + "\n```"


def _render_simulator_profile(evidence: Evidence) -> str:
    metadata = evidence.reproduction_metadata
    profile = metadata.get("semantic_profile") if isinstance(metadata, Mapping) and isinstance(metadata.get("semantic_profile"), Mapping) else None
    if not isinstance(profile, Mapping):
        return _missing("simulator가 소비한 강화 codec profile")
    return md_table(
        ("simulator 필드", "적용값"),
        (
            ("적용 여부", profile.get("applied", "기록 없음")),
            ("source", profile.get("source", "기록 없음")),
            ("profile kind", profile.get("profile_kind", "기록 없음")),
            ("raw basis", profile.get("raw_basis", "기록 없음")),
            ("encoder / decoder mode", f"{profile.get('encoder_mode', '기록 없음')} / {profile.get('decoder_mode', '기록 없음')}"),
            ("ρc / ρr", f"{_fmt(profile.get('rho_c'), 7)} / {_fmt(profile.get('rho_r'), 3)}"),
            ("encoder / decoder bit/s", f"{_fmt(profile.get('enc_bitrate'), 1, comma=True)} / {_fmt(profile.get('dec_bitrate'), 1, comma=True)}"),
            ("profile 표본", _fmt_int(profile.get("num_samples"))),
        ),
    )


def _render_reproduction_full(evidence: Evidence) -> str:
    rows = _sort_area_policy(_rows_for_mode(evidence.reproduction_summary, "semantic"))
    if not rows:
        return _missing("강화 semantic simulator 전체 결과")
    summary_table = md_table(
        ("면적(m)", "정책", "완료", "평균 시간(s)", "비행 J/request", "평균 이동", "encode", "decode"),
        (
            (
                _fmt_int(row.get("area")),
                row.get("policy", "기록 없음"),
                _fmt(row.get("finished"), 1),
                _fmt(row.get("avg_time"), 2),
                _fmt(row.get("flight_energy_per_req"), 1, comma=True),
                _fmt(row.get("avg_travel"), 2),
                _fmt(row.get("encodes"), 1),
                _fmt(row.get("decodes"), 1),
            )
            for row in rows
        ),
    )
    statistics_rows = _sort_area_policy(
        _rows_for_mode(evidence.reproduction_statistics, "semantic")
    )
    if not statistics_rows:
        return summary_table + "\n\n" + _missing("강화 semantic simulator repeat 통계")
    statistics_table = md_table(
        (
            "면적(m)",
            "정책",
            "n",
            "완료 mean ± sample std [95% CI]",
            "시간 mean ± sample std [95% CI]",
            "품질 mean ± sample std [95% CI]",
        ),
        (
            (
                _fmt_int(row.get("area")),
                row.get("policy", "기록 없음"),
                _fmt_int(row.get("repeat_count")),
                f"{_fmt(row.get('finished_mean'), 2)} ± {_fmt(row.get('finished_std'), 2)} "
                f"[{_fmt(row.get('finished_ci95_low'), 2)}, {_fmt(row.get('finished_ci95_high'), 2)}]",
                f"{_fmt(row.get('avg_time_mean'), 2)} ± {_fmt(row.get('avg_time_std'), 2)} "
                f"[{_fmt(row.get('avg_time_ci95_low'), 2)}, {_fmt(row.get('avg_time_ci95_high'), 2)}]",
                f"{_fmt(row.get('semantic_quality_mean'), 6)} ± {_fmt(row.get('semantic_quality_std'), 6)} "
                f"[{_fmt(row.get('semantic_quality_ci95_low'), 6)}, {_fmt(row.get('semantic_quality_ci95_high'), 6)}]",
            )
            for row in statistics_rows
        ),
    )
    return (
        summary_table
        + "\n\n**repeat 원시값에서 독립 검증한 재현 통계**\n\n"
        + statistics_table
        + f"\n\n표준편차는 sample std(ddof=1), 신뢰구간은 `{CI95_METHOD}` 계약을 사용한다."
    )


def _render_reproduction_delta(evidence: Evidence) -> str:
    current = _index_rows(evidence.reproduction_summary)
    baseline = _index_rows(evidence.baseline_reproduction_summary)
    pairs: list[tuple[Mapping[str, str], Mapping[str, str]]] = []
    for key, row in current.items():
        if key[0] == "semantic" and key in baseline:
            pairs.append((row, baseline[key]))
    if not pairs:
        return _missing("강화 profile과 기존 simulator 기준선의 대응 조합 비교")
    policies = ("Stochastic", "LinUCB", "SA", "Greedy", "MCTS")
    output = []
    for policy in policies:
        subset = [(new, old) for new, old in pairs if new.get("policy") == policy]
        if not subset:
            continue
        new_finished = statistics.fmean(float(new["finished"]) for new, _ in subset)
        old_finished = statistics.fmean(float(old["finished"]) for _, old in subset)
        new_time = statistics.fmean(float(new["avg_time"]) for new, _ in subset)
        old_time = statistics.fmean(float(old["avg_time"]) for _, old in subset)
        new_energy = statistics.fmean(float(new["flight_energy_per_req"]) for new, _ in subset)
        old_energy = statistics.fmean(float(old["flight_energy_per_req"]) for _, old in subset)
        output.append(
            (
                policy,
                len(subset),
                _fmt(new_finished, 2),
                _pct(_ratio_delta(new_finished, old_finished), signed=True),
                _fmt(new_time, 2),
                _pct(_ratio_delta(new_time, old_time), signed=True),
                _pct(_ratio_delta(new_energy, old_energy), signed=True),
            )
        )
    table = md_table(
        ("정책", "대응 면적", "강화 완료 평균", "완료 변화", "강화 시간 평균", "시간 변화", "비행에너지 변화"),
        output,
    )
    return "같은 mode·면적·정책 key끼리 대응시킨 뒤 정책별로 면적 평균을 냈다. 양수 변화는 증가, 음수 변화는 감소다.\n\n" + table


def _verdict_counts(rows: Sequence[Mapping[str, str]] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows or []:
        verdict = (row.get("verdict") or "기록 없음").strip().lower()
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def _render_verification_counts(evidence: Evidence) -> str:
    rows = evidence.reproduction_verification
    if not rows:
        return _missing("강화 논문 재현 verifier 결과")
    groups: dict[str, dict[str, int]] = {}
    for row in rows:
        check = row.get("check") or "기록 없음"
        verdict = (row.get("verdict") or "기록 없음").lower()
        groups.setdefault(check, {})[verdict] = groups.setdefault(check, {}).get(verdict, 0) + 1
    output = []
    for check in sorted(groups):
        counts = groups[check]
        output.append((check, sum(counts.values()), counts.get("match", 0), counts.get("partial", 0), counts.get("mismatch", 0)))
    total = _verdict_counts(rows)
    output.append(("전체", len(rows), total.get("match", 0), total.get("partial", 0), total.get("mismatch", 0)))
    return md_table(("verifier check", "행", "match", "partial", "mismatch"), output) + "\n\n`paper_visual_estimate`는 그림 판독 근삿값이므로 이 count는 원자료의 통계 검정이 아니라 독립 구현의 근사 일치도 감사다."


def _render_sensitivity(evidence: Evidence) -> str:
    current = _verdict_counts(evidence.reproduction_verification)
    baseline = _verdict_counts(evidence.baseline_reproduction_verification)
    if not current or not baseline:
        return _missing("강화/기존 verifier 민감도 비교")
    verdicts = sorted(set(current) | set(baseline), key=lambda item: ("match", "partial", "mismatch").index(item) if item in {"match", "partial", "mismatch"} else 99)
    table = md_table(
        ("판정", "기존 profile", "강화 profile", "행 수 변화"),
        ((verdict, baseline.get(verdict, 0), current.get(verdict, 0), current.get(verdict, 0) - baseline.get(verdict, 0)) for verdict in verdicts),
    )
    return table + "\n\n이 표는 codec/profile 교체에 따른 verifier 판정 민감도만 보여 준다. density penalty, workload, power, request probability를 한 요인씩 바꾼 별도 sweep 산출물은 없어 그 민감도 수치는 **미실행/증거 없음**이다."


def _render_guardrails(evidence: Evidence) -> str:
    metadata = evidence.adaptive_metadata
    profiles = metadata.get("profiles") if isinstance(metadata, Mapping) and isinstance(metadata.get("profiles"), Mapping) else None
    adaptive = profiles.get("adaptive_semantic") if isinstance(profiles, Mapping) and isinstance(profiles.get("adaptive_semantic"), Mapping) else None
    thresholds = adaptive.get("target_thresholds") if isinstance(adaptive, Mapping) else None
    if not isinstance(thresholds, list) or not thresholds:
        return _missing("품질 guardrail threshold metadata")
    rows = []
    lower: float | None = None
    for item in thresholds:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        upper, quality = item
        if upper is None:
            interval = f"SINR ≥ {_fmt(lower, 1)} dB" if lower is not None else "전체"
        elif lower is None:
            interval = f"SINR < {_fmt(upper, 1)} dB"
        else:
            interval = f"{_fmt(lower, 1)} ≤ SINR < {_fmt(upper, 1)} dB"
        rows.append((interval, _fmt(quality, 6)))
        lower = _float(upper)
    return md_table(("SINR 구간", "metadata 요구 품질"), rows) + "\n\n이 threshold는 `run_metadata.json`에 기록된 실행 설정이다. 별도 사전등록 문서·시각·해시는 산출물에 없으므로 **외부 사전등록을 완료했다는 증거는 없음**으로 해석한다."


def _adaptive_mode_comparison(evidence: Evidence) -> str:
    rows = evidence.adaptive_summary or []
    if not rows:
        return _missing("적응형 system-mode 비교")
    output = []
    labels = (("nonsemantic", "비의미 전송"), ("fixed_paper_like", "고정 paper-like"), ("adaptive_semantic", "적응형"))
    for mode, label in labels:
        subset = _rows_for_mode(rows, mode)
        if subset:
            output.append((label, len(subset), _fmt(_mean(subset, "finished"), 2), _fmt(_mean(subset, "avg_time"), 2), _fmt(_mean(subset, "flight_energy_per_req"), 1, comma=True), _fmt(_mean(subset, "semantic_quality"), 6), _fmt(_mean(subset, "semantic_payload_ratio"), 6)))
    if not output:
        return _missing("적응형 system-mode 비교")
    return md_table(("system mode", "조합", "완료 평균", "시간 평균", "비행 J/request", "품질 평균", "payload 평균"), output) + "\n\n이는 저장된 세 system mode 비교이며, neural loss·threshold·zlib·latency를 한 요인씩 제거한 인과적 ablation은 아니다. 해당 세부 ablation은 **미실행/증거 없음**이다."


def _render_adaptive_statistics_command(evidence: Evidence, paths: InputPaths) -> str:
    command = _recorded_command(evidence.adaptive_metadata)
    if command is not None:
        return "실제 실행 명령은 metadata에 다음과 같이 기록됐다.\n\n```powershell\n" + command + "\n```"
    summary = paths.adaptive_dir / "summary_metrics.csv"
    validation = paths.adaptive_dir / "result_validation.json"
    return (
        "**실제 full-run CLI 원문은 metadata에 기록되지 않아 증거 없음.** 아래 명령은 저장된 요약의 구조·방향성 invariant를 다시 검사하는 재검증 명령이며, 원 실험 실행 명령으로 주장하지 않는다.\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe studies\\adaptive_semantic_compression\\code\\validate_full_adaptive_results.py `\n"
        f"  --summary \"{summary}\" `\n"
        f"  --out \"{validation}\"\n"
        "```"
    )


def _render_adaptive_main(evidence: Evidence) -> str:
    rows = _sort_area_policy(_rows_for_mode(evidence.adaptive_summary, "adaptive_semantic"))
    if not rows:
        return _missing("강화 adaptive_semantic 전체 결과")
    return md_table(
        ("면적(m)", "정책", "완료", "평균 시간(s)", "비행 J/request", "품질", "payload ratio", "encode", "decode"),
        (
            (
                _fmt_int(row.get("area")),
                row.get("policy", "기록 없음"),
                _fmt(row.get("finished"), 1),
                _fmt(row.get("avg_time"), 2),
                _fmt(row.get("flight_energy_per_req"), 1, comma=True),
                _fmt(row.get("semantic_quality"), 6),
                _fmt(row.get("semantic_payload_ratio"), 6),
                _fmt(row.get("encodes"), 1),
                _fmt(row.get("decodes"), 1),
            )
            for row in rows
        ),
    )


def _paired_repeat_differences(
    rows: Sequence[Mapping[str, str]] | None,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str, int], Mapping[str, str]] = {}
    for row in rows or []:
        area, repeat = _float(row.get("area")), _float(row.get("repeat"))
        if area is None or repeat is None:
            continue
        by_key[(row.get("mode", ""), int(area), row.get("policy", ""), int(repeat))] = row
    groups: dict[tuple[int, str], dict[str, list[float]]] = {}
    for (mode, area, policy, repeat), adaptive in by_key.items():
        if mode != "adaptive_semantic":
            continue
        fixed = by_key.get(("fixed_paper_like", area, policy, repeat))
        if fixed is None:
            continue
        group = groups.setdefault((area, policy), {"finished": [], "avg_time": [], "semantic_quality": []})
        for metric in tuple(group):
            new, old = _float(adaptive.get(metric)), _float(fixed.get(metric))
            if new is not None and old is not None:
                group[metric].append(new - old)
    output: list[dict[str, Any]] = []
    for (area, policy), metrics in groups.items():
        n = min((len(values) for values in metrics.values()), default=0)
        if n == 0:
            continue
        row: dict[str, Any] = {"area": area, "policy": policy, "n": n}
        for metric, values in metrics.items():
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_delta"] = mean
            row[f"{metric}_std"] = std
        output.append(row)
    return sorted(output, key=lambda row: (row["area"], row["policy"]))


def _render_paired_deltas(evidence: Evidence) -> str:
    rows = _paired_repeat_differences(evidence.adaptive_repeats)
    if not rows:
        return _missing("fixed_paper_like 대비 repeat-matched paired 변화", "repeat_metrics.csv에서 동일 area·policy·repeat 쌍을 찾지 못함")
    table = md_table(
        ("면적", "정책", "paired n", "완료 Δ", "시간 Δ(s)", "품질 Δ"),
        ((row["area"], row["policy"], row["n"], _fmt(row["finished_delta"], 3), _fmt(row["avg_time_delta"], 3), _fmt(row["semantic_quality_delta"], 6)) for row in rows),
    )
    return "동일한 area·policy·repeat 번호의 `adaptive_semantic - fixed_paper_like`를 먼저 계산한 뒤 평균했다. 음의 시간 Δ는 단축, 음의 품질 Δ는 품질 하락이다.\n\n" + table


def _render_confidence_intervals(evidence: Evidence) -> str:
    rows = _sort_area_policy(_rows_for_mode(evidence.adaptive_statistics, "adaptive_semantic"))
    if not rows:
        return _missing("adaptive_semantic 95% 신뢰구간", "statistical_summary.csv 없음")
    table = md_table(
        ("면적", "정책", "n", "완료 평균 [95% CI]", "시간 평균 [95% CI]", "품질 평균 [95% CI]", "방법"),
        (
            (
                _fmt_int(row.get("area")),
                row.get("policy", "기록 없음"),
                _fmt_int(row.get("repeat_count")),
                f"{_fmt(row.get('finished_mean'), 2)} [{_fmt(row.get('finished_ci95_low'), 2)}, {_fmt(row.get('finished_ci95_high'), 2)}]",
                f"{_fmt(row.get('avg_time_mean'), 2)} [{_fmt(row.get('avg_time_ci95_low'), 2)}, {_fmt(row.get('avg_time_ci95_high'), 2)}]",
                f"{_fmt(row.get('semantic_quality_mean'), 6)} [{_fmt(row.get('semantic_quality_ci95_low'), 6)}, {_fmt(row.get('semantic_quality_ci95_high'), 6)}]",
                row.get("ci95_method", "기록 없음"),
            )
            for row in rows
        ),
    )
    return table + "\n\n이 파일은 각 mode의 평균 CI를 기록한다. paired difference 자체의 CI, p-value, Holm 다중비교 보정은 별도 열이 없으므로 **증거 없음**이며 통계적 유의성을 주장하지 않는다."


def _render_guardrail_results(evidence: Evidence) -> str:
    validation = evidence.adaptive_validation
    comparisons = _nested_get(validation, "adaptive_vs_fixed.comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        return _missing("adaptive-vs-fixed 품질 guardrail 검증 결과")
    table = md_table(
        ("면적", "정책", "완료 변화", "시간 변화", "품질 Δ", "조합 가설 pass"),
        (
            (
                _fmt_int(item.get("area")),
                item.get("policy", "기록 없음"),
                _pct(item.get("finished_delta_pct"), signed=True),
                _pct(item.get("avg_time_delta_pct"), signed=True),
                _fmt(item.get("quality_delta"), 6),
                item.get("pass", "기록 없음"),
            )
            for item in comparisons
            if isinstance(item, Mapping)
        ),
    )
    passed = validation.get("passed") if isinstance(validation, Mapping) else None
    all_comparisons_pass = _nested_get(
        validation, "adaptive_vs_fixed.all_comparisons_pass"
    )
    minimum = _nested_get(validation, "adaptive_vs_fixed.max_quality_drop")
    return (
        f"구조 무결성 `passed`={passed}, 방향 가설 "
        f"`all_comparisons_pass`={all_comparisons_pass}, 기록된 최저 품질 "
        f"Δ={_fmt(minimum, 6)}이다.\n\n{table}\n\n"
        "상위 `passed`는 coverage·고유 키·finite 값·비교 분모 무결성만 뜻한다. "
        "조합별 `pass`와 `all_comparisons_pass`는 저장된 방향·품질 guardrail이며, "
        "논문 일치나 통계적 유의성을 뜻하지 않는다."
    )


def _usage_columns(rows: Sequence[Mapping[str, str]]) -> list[str]:
    keys = {key for row in rows for key in row}
    return sorted(key for key in keys if key.startswith("mode_") and key.endswith("_count") and key not in {"mode_adaptive_semantic_count", "mode_fixed_count"})


def _render_mode_usage(evidence: Evidence) -> str:
    rows = _rows_for_mode(evidence.adaptive_usage, "adaptive_semantic")
    if not rows:
        return _missing("adaptive_semantic mode 사용량")
    columns = _usage_columns(rows)
    if not columns:
        return _missing("adaptive_semantic 세부 mode count 열")
    output = []
    policies = ("Stochastic", "LinUCB", "SA", "Greedy", "MCTS")
    for policy in policies:
        subset = [row for row in rows if row.get("policy") == policy]
        totals = {column: sum(_float(row.get(column)) or 0.0 for row in subset) for column in columns}
        denominator = sum(totals.values())
        output.append(
            [
                policy,
                *(
                    f"{_fmt(value, 1)} ({_fmt(value / denominator * 100.0 if denominator else None, 1)}%)"
                    for value in totals.values()
                ),
                _fmt(denominator, 1),
            ]
        )
    labels = [column.removeprefix("mode_").removesuffix("_count") for column in columns]
    return "면적 5개에 저장된 평균 count를 정책별로 합산했다. 괄호는 해당 정책 내 기록 count 비율이다.\n\n" + md_table(("정책", *labels, "합계"), output)


def _aggregate_adaptive_delta(
    evidence: Evidence, group_key: str
) -> list[tuple[Any, int, str, str, str]]:
    index = _index_rows(evidence.adaptive_summary)
    pairs: list[tuple[Mapping[str, str], Mapping[str, str]]] = []
    for (mode, area, policy), adaptive in index.items():
        fixed = index.get(("fixed_paper_like", area, policy))
        if mode == "adaptive_semantic" and fixed is not None:
            pairs.append((adaptive, fixed))
    groups: dict[Any, list[tuple[Mapping[str, str], Mapping[str, str]]]] = {}
    for adaptive, fixed in pairs:
        key: Any = int(float(adaptive["area"])) if group_key == "area" else adaptive.get("policy")
        groups.setdefault(key, []).append((adaptive, fixed))
    output = []
    for key in sorted(groups, key=str):
        values = groups[key]
        finished = [_ratio_delta(a.get("finished"), f.get("finished")) for a, f in values]
        times = [_ratio_delta(a.get("avg_time"), f.get("avg_time")) for a, f in values]
        qualities = [(_float(a.get("semantic_quality")) or 0.0) - (_float(f.get("semantic_quality")) or 0.0) for a, f in values if _float(a.get("semantic_quality")) is not None and _float(f.get("semantic_quality")) is not None]
        output.append((key, len(values), _pct(statistics.fmean([v for v in finished if v is not None]), signed=True) if any(v is not None for v in finished) else "기록 없음", _pct(statistics.fmean([v for v in times if v is not None]), signed=True) if any(v is not None for v in times) else "기록 없음", _fmt(statistics.fmean(qualities), 6) if qualities else "기록 없음"))
    return output


def _render_generalization(evidence: Evidence) -> str:
    by_area = _aggregate_adaptive_delta(evidence, "area")
    by_policy = _aggregate_adaptive_delta(evidence, "policy")
    if not by_area or not by_policy:
        return _missing("면적·정책 일반화 비교")
    return (
        "각 조합의 fixed 대비 상대 변화를 먼저 구한 다음 그룹 안에서 평균했다.\n\n"
        "**면적별(정책 평균)**\n\n"
        + md_table(("면적", "정책 수", "완료 변화", "시간 변화", "품질 Δ"), by_area)
        + "\n\n**정책별(면적 평균)**\n\n"
        + md_table(("정책", "면적 수", "완료 변화", "시간 변화", "품질 Δ"), by_policy)
    )


def _render_adaptive_airtalking(evidence: Evidence) -> str:
    adaptive_index = _index_rows(evidence.adaptive_summary)
    reproduction_index = _index_rows(evidence.reproduction_summary)
    differences: list[float] = []
    compared = 0
    for (mode, area, policy), fixed in adaptive_index.items():
        if mode != "fixed_paper_like":
            continue
        reproduction = reproduction_index.get(("semantic", area, policy))
        if reproduction is None:
            continue
        compared += 1
        for metric in ("finished", "avg_time", "flight_energy_per_req", "avg_travel", "encodes", "decodes"):
            left, right = _float(fixed.get(metric)), _float(reproduction.get(metric))
            if left is not None and right is not None:
                differences.append(abs(left - right))
    if compared == 0:
        return _missing("적응형 실행의 fixed profile과 강화 재현 simulator 연결 일관성")
    neural_mode = _nested_get(evidence.adaptive_metadata, "neural_quality_mode")
    source_neural = _nested_get(evidence.adaptive_metadata, "source_neural_encoder_decoder")
    return md_table(
        ("감사 항목", "결과"),
        (
            ("fixed-vs-reproduction 대응 조합", compared),
            ("공통 지표 최대 절대차", _fmt(max(differences) if differences else None, 9)),
            ("neural quality mode", neural_mode or "기록 없음"),
            ("neural summary source", source_neural or "기록 없음"),
        ),
    ) + "\n\n이 비교는 simulator 입력 연결의 일관성 감사다. adaptive 결과 자체를 Fig. 3~6 `verify_against_paper.py`로 다시 평가한 verifier CSV는 필수 adaptive 산출물에 없으므로 논문 그림 일치도는 **미실행/증거 없음**이다."


def _render_adaptive_claim(evidence: Evidence) -> str:
    validation_passed = _nested_get(evidence.adaptive_validation, "passed") is True
    neural_mode = _nested_get(evidence.adaptive_metadata, "neural_quality_mode")
    anchor = _nested_get(evidence.adaptive_metadata, "neural_encoder_decoder_anchor")
    multi_rate = isinstance(anchor, Mapping) and isinstance(anchor.get("multi_rate_profiles"), list) and len(anchor.get("multi_rate_profiles", [])) >= 5
    repeats = evidence.adaptive_repeats or []
    repeat_counts: dict[tuple[str, int, str], set[int]] = {}
    for row in repeats:
        key = _canonical_cell_key(row)
        repeat_id = _canonical_repeat_id(row)
        if key is not None and repeat_id is not None:
            repeat_counts.setdefault(key, set()).add(repeat_id)
    minimum_repeats = min((len(ids) for ids in repeat_counts.values()), default=0)
    if validation_passed and neural_mode == "selection" and multi_rate:
        status = (
            "강화 5-rate codec 연결과 로컬 validator 통과 확인; "
            f"mode·area·policy 조합별 최소 repeat {minimum_repeats}개 기록"
        )
    elif evidence.adaptive_summary:
        status = "결과는 존재하지만 강화 5-rate codec selection 연결 또는 validator 통과 증거가 불충분"
    else:
        status = "미실행/증거 없음"
    return status + "; 논문 정확 재현·인과 효과·통계적 유의성은 별도 증거 없이는 주장하지 않음"


def build_replacements(
    evidence: Evidence, paths: InputPaths, diagnostics: Diagnostics
) -> dict[str, str]:
    replacements = {
        "enhanced_dataset_fingerprint": _render_dataset(evidence),
        "enhanced_environment": _render_environment(evidence),
        "enhanced_command_result": _render_command_inline(evidence),
        "enhanced_best_epoch": _render_best_epoch(evidence),
        "enhanced_sample_counts": _render_sample_counts_inline(evidence),
        "enhanced_training_elapsed": _render_elapsed_inline(evidence),
        "enhanced_run_status": _render_status_inline(evidence),
        "enhanced_rate_quality_table": _rate_quality_table(evidence),
        "enhanced_paperlike_metrics": _render_paperlike(evidence),
        "enhanced_timing_and_resources": _render_timing(evidence),
        "enhanced_training_curve_summary": _render_training_curve(evidence),
        "enhanced_qualitative_audit": _render_qualitative(
            evidence, paths.output_dir, paths.enhanced_dir
        ),
        "enhanced_ablation_results": _render_no_neural_ablation(evidence),
        "enhanced_semantic_profile_summary": _render_semantic_profile(evidence),
        "enhanced_reproduction_provenance": _render_reproduction_provenance(evidence, diagnostics),
        "enhanced_airtalking_command": _render_reproduction_command(evidence),
        "enhanced_paperlike_codec_for_simulator": _render_simulator_profile(evidence),
        "enhanced_airtalking_full_results": _append_existing_figures(
            _render_reproduction_full(evidence),
            paths.output_dir,
            (
                (
                    "AirTalking 정책별 완료 request 수",
                    "아래 그림은 면적과 정책에 따른 완료 request 수의 10회 반복 평균을 보여 준다. 오차막대가 없으므로 정확한 불확실성은 반복 통계 표를 함께 봐야 한다.",
                    paths.reproduction_dir / "figures" / "finished_requests.png",
                ),
                (
                    "300m semantic과 nonsemantic 비교",
                    "300×300 m에서 semantic과 nonsemantic 경로의 방향 차이를 시각적으로 확인한다. 이 그림도 논문 일치 증명이 아니라 독립 구현 내부 비교다.",
                    paths.reproduction_dir
                    / "figures"
                    / "semantic_vs_nonsemantic_300m.png",
                ),
            ),
        ),
        "enhanced_vs_legacy_system_delta": _render_reproduction_delta(evidence),
        "enhanced_paper_verification_counts": _render_verification_counts(evidence),
        "strengthened_reproduction_sensitivity_results": _render_sensitivity(evidence),
        "enhanced_neural_rate_quality_table": _rate_quality_table(evidence),
        "preregistered_quality_guardrails": _render_guardrails(evidence),
        "enhanced_adaptive_ablation_results": _adaptive_mode_comparison(evidence),
        "enhanced_adaptive_statistics_command": _render_adaptive_statistics_command(evidence, paths),
        "enhanced_adaptive_main_results": _append_existing_figures(
            _render_adaptive_main(evidence),
            paths.output_dir,
            (
                (
                    "고정과 적응형의 면적별 완료 request",
                    "Greedy·MCTS를 포함한 정책별 fixed/adaptive 완료 수를 면적에 따라 비교한다. 막대는 10회 반복 평균이며, 신뢰구간은 뒤의 통계 표에서 확인한다.",
                    paths.adaptive_dir
                    / "figures"
                    / "finished_by_area_greedy_mcts.png",
                ),
                (
                    "300m 적응형 압축 mode 사용량",
                    "300×300 m에서 실제로 선택된 mode를 보여 준다. 실측 mIoU가 비단조여서 emergency·low·high만 도달 가능했고 medium·paper_like는 선택되지 않았다.",
                    paths.adaptive_dir
                    / "figures"
                    / "adaptive_mode_usage_300m.png",
                ),
                (
                    "300m 지연-품질 균형",
                    "300×300 m의 평균 시간과 기록된 semantic quality를 함께 본다. 별도 점은 system mode·정책별 평균이며 paired 유의성 검정을 대체하지 않는다.",
                    paths.adaptive_dir
                    / "figures"
                    / "latency_quality_tradeoff_300m.png",
                ),
            ),
        ),
        "enhanced_adaptive_paired_deltas": _render_paired_deltas(evidence),
        "enhanced_adaptive_confidence_intervals": _render_confidence_intervals(evidence),
        "enhanced_adaptive_quality_guardrail_results": _render_guardrail_results(evidence),
        "enhanced_adaptive_mode_usage": _render_mode_usage(evidence),
        "enhanced_adaptive_generalization": _render_generalization(evidence),
        "enhanced_adaptive_airtalking_verification": _render_adaptive_airtalking(evidence),
        "enhanced_adaptive_claim_status": _render_adaptive_claim(evidence),
    }
    if set(replacements) != EXPECTED_AUTO_KEYS:
        missing = sorted(EXPECTED_AUTO_KEYS - set(replacements))
        extra = sorted(set(replacements) - EXPECTED_AUTO_KEYS)
        raise ReportFinalizationError(
            f"내부 AUTO 치환 map 불일치: 누락={missing}, 초과={extra}"
        )
    return replacements


def _read_templates(reports_dir: Path) -> dict[str, tuple[Path, str]]:
    templates: dict[str, tuple[Path, str]] = {}
    all_keys: list[str] = []
    for report_kind, filename in REPORT_TEMPLATES.items():
        path = reports_dir / filename
        if not path.is_file():
            raise ReportFinalizationError(f"보고서 AUTO 템플릿이 없습니다: {path}")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise ReportFinalizationError(f"보고서 템플릿을 읽을 수 없습니다: {path}: {exc}") from exc
        keys = AUTO_RE.findall(text)
        if len(keys) != len(set(keys)):
            duplicates = sorted({key for key in keys if keys.count(key) > 1})
            raise ReportFinalizationError(f"템플릿 AUTO 키가 중복되었습니다: {path}: {duplicates}")
        templates[report_kind] = (path, text)
        all_keys.extend(keys)
    if len(all_keys) != 33 or set(all_keys) != EXPECTED_AUTO_KEYS:
        raise ReportFinalizationError(
            "보고서 템플릿 AUTO 계약 불일치: "
            f"개수={len(all_keys)}(기대 33), 누락={sorted(EXPECTED_AUTO_KEYS - set(all_keys))}, "
            f"알 수 없는 키={sorted(set(all_keys) - EXPECTED_AUTO_KEYS)}"
        )
    return templates


def render_markdown_reports(
    templates: Mapping[str, tuple[Path, str]],
    replacements: Mapping[str, str],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for report_kind, (source_path, text) in templates.items():
        source_keys = set(AUTO_RE.findall(text))

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in replacements:
                raise ReportFinalizationError(f"치환값이 없는 AUTO 키: {key}")
            return replacements[key]

        rendered = AUTO_RE.sub(replace, text)
        if AUTO_RE.search(rendered) or "<!-- AUTO:" in rendered:
            raise ReportFinalizationError(f"최종 Markdown에 AUTO 표식이 남았습니다: {source_path}")
        if not source_keys:
            raise ReportFinalizationError(f"AUTO 키가 없는 파일을 템플릿으로 읽었습니다: {source_path}")
        output_path = output_dir / f"{source_path.stem}_final.md"
        output_path.write_text(rendered, encoding="utf-8", newline="\n")
        outputs.append(output_path)
    return outputs


def _split_md_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    cells: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(stripped):
        char = stripped[index]
        if char == "\\" and index + 1 < len(stripped) and stripped[index + 1] == "|":
            # Only consume an actual table escape.  In particular, the
            # backslashes in C:\\경로\\파일 이름 and \\\\server shares are
            # ordinary path data; only \| affects table cell boundaries.
            current.append(stripped[index + 1])
            index += 2
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    cells.append("".join(current).strip())
    return cells


def _is_table_separator(line: str) -> bool:
    cells = _split_md_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def markdown_to_docx(
    markdown_path: Path,
    docx_path: Path,
    warnings: list[str],
    *,
    generated_at: datetime | None = None,
) -> None:
    try:
        try:
            from tools.docx_report_design import build_docx
        except ModuleNotFoundError:
            from docx_report_design import build_docx
    except ImportError as exc:
        raise ReportFinalizationError(
            "DOCX 생성을 위해 python-docx가 필요합니다. requirements.txt를 설치하세요."
        ) from exc

    try:
        build_docx(
            markdown_path,
            docx_path,
            warnings,
            split_md_row=_split_md_row,
            is_table_separator=_is_table_separator,
            generated_at=generated_at,
        )
    except ImportError as exc:
        raise ReportFinalizationError(
            "DOCX 생성을 위해 python-docx가 필요합니다. requirements.txt를 설치하세요."
        ) from exc


def finalize_reports(paths: InputPaths, *, allow_incomplete: bool = False) -> dict[str, Any]:
    diagnostics = Diagnostics(allow_incomplete=allow_incomplete)
    normalized = InputPaths(
        enhanced_dir=_ensure_directory(paths.enhanced_dir, "강화 codec 결과"),
        reproduction_dir=_ensure_directory(paths.reproduction_dir, "논문 재현 결과"),
        adaptive_dir=_ensure_directory(paths.adaptive_dir, "적응형 후속 결과"),
        reports_dir=_ensure_directory(paths.reports_dir, "보고서 템플릿"),
        output_dir=paths.output_dir.expanduser().resolve(),
        baseline_neural_dir=paths.baseline_neural_dir,
        baseline_reproduction_dir=paths.baseline_reproduction_dir,
        baseline_adaptive_dir=paths.baseline_adaptive_dir,
    )
    templates = _read_templates(normalized.reports_dir)
    evidence = load_evidence(normalized, diagnostics)
    replacements = build_replacements(evidence, normalized, diagnostics)
    markdown_paths = render_markdown_reports(templates, replacements, normalized.output_dir)
    for path in markdown_paths:
        if AUTO_RE.search(path.read_text(encoding="utf-8")):
            raise ReportFinalizationError(f"최종 AUTO 0개 검사 실패: {path}")

    docx_paths: list[Path] = []
    docx_generated_at = datetime.now(timezone.utc).replace(microsecond=0)
    for markdown_path in markdown_paths:
        docx_path = markdown_path.with_suffix(".docx")
        markdown_to_docx(
            markdown_path,
            docx_path,
            diagnostics.warnings,
            generated_at=docx_generated_at,
        )
        docx_paths.append(docx_path)

    finalizer_source = Path(__file__).resolve()
    renderer_source = ROOT / "tools" / "docx_report_design.py"
    artifact_records = {
        "markdown": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in markdown_paths
        ],
        "docx": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in docx_paths
        ],
    }
    manifest = {
        "schema_version": 2,
        "mode": "allow-incomplete" if allow_incomplete else "strict",
        "auto_template_key_count": 33,
        "auto_remaining_in_final_markdown": 0,
        "markdown": [str(path) for path in markdown_paths],
        "docx": [str(path) for path in docx_paths],
        "artifacts": artifact_records,
        "finalizer_source": {
            "path": str(finalizer_source),
            "sha256": _sha256(finalizer_source),
        },
        "docx_converter_source": {
            "path": str(finalizer_source),
            "sha256": _sha256(finalizer_source),
            "function": "markdown_to_docx",
        },
        "docx_renderer_source": {
            "path": str(renderer_source),
            "sha256": _sha256(renderer_source),
        },
        "docx_generated_at_utc": docx_generated_at.isoformat(),
        "input_directories": {
            "enhanced": str(normalized.enhanced_dir),
            "reproduction": str(normalized.reproduction_dir),
            "adaptive": str(normalized.adaptive_dir),
            "baseline_neural": (
                str(normalized.baseline_neural_dir)
                if normalized.baseline_neural_dir is not None
                else None
            ),
            "baseline_reproduction": (
                str(normalized.baseline_reproduction_dir)
                if normalized.baseline_reproduction_dir is not None
                else None
            ),
            "baseline_adaptive": (
                str(normalized.baseline_adaptive_dir)
                if normalized.baseline_adaptive_dir is not None
                else None
            ),
        },
        "warnings": diagnostics.warnings,
        "templates_preserved": [str(path) for path, _ in templates.values()],
    }
    manifest_path = normalized.output_dir / "finalization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def _parse_baseline_overrides(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    allowed = {"neural", "reproduction", "adaptive"}
    for value in values:
        if "=" not in value:
            raise ReportFinalizationError(
                f"--baseline-dir 형식은 KIND=PATH입니다: {value!r}"
            )
        kind, raw_path = value.split("=", 1)
        kind = kind.strip().lower()
        if kind not in allowed:
            raise ReportFinalizationError(
                f"--baseline-dir KIND는 {sorted(allowed)} 중 하나여야 합니다: {kind!r}"
            )
        if kind in result:
            raise ReportFinalizationError(f"--baseline-dir가 중복되었습니다: {kind}")
        result[kind] = Path(raw_path.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="33개 AUTO 표식을 실험 산출물로 치환하고 최종 Markdown/DOCX 3개를 생성합니다."
    )
    parser.add_argument(
        "--enhanced-result-dir",
        "--enhanced-dir",
        dest="enhanced_dir",
        type=Path,
        required=True,
        help="강화 encoder/decoder 결과 디렉터리",
    )
    parser.add_argument(
        "--reproduction-result-dir",
        "--reproduction-dir",
        dest="reproduction_dir",
        type=Path,
        required=True,
        help="강화 AirTalking 재현 결과 디렉터리",
    )
    parser.add_argument(
        "--adaptive-result-dir",
        "--adaptive-dir",
        dest="adaptive_dir",
        type=Path,
        required=True,
        help="강화 적응형 후속 연구 결과 디렉터리",
    )
    parser.add_argument(
        "--baseline-neural-dir",
        type=Path,
        default=ROOT / "studies" / "neural_encoder_decoder" / "results" / "paperlike_timed_latent20",
        help="기존 neural encoder/decoder 기준선 디렉터리",
    )
    parser.add_argument(
        "--baseline-reproduction-dir",
        type=Path,
        default=ROOT / "studies" / "airtalking_reproduction" / "results" / "airtalking_cityscapes_calibrated_final_p012",
        help="기존 AirTalking 재현 기준선 디렉터리",
    )
    parser.add_argument(
        "--baseline-adaptive-dir",
        type=Path,
        default=None,
        help="선택적인 기존 적응형 기준선 디렉터리",
    )
    parser.add_argument(
        "--baseline-dir",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help="기준선 경로 override(neural, reproduction, adaptive); 여러 번 사용 가능",
    )
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="최종본 디렉터리(기본: REPORTS_DIR/final)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="누락 산출물을 명시적 '미실행/증거 없음'으로 기록합니다. 손상 JSON/CSV는 여전히 실패합니다.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        overrides = _parse_baseline_overrides(args.baseline_dir)
        reports_dir = args.reports_dir
        paths = InputPaths(
            enhanced_dir=args.enhanced_dir,
            reproduction_dir=args.reproduction_dir,
            adaptive_dir=args.adaptive_dir,
            reports_dir=reports_dir,
            output_dir=args.output_dir or reports_dir / "final",
            baseline_neural_dir=overrides.get("neural", args.baseline_neural_dir),
            baseline_reproduction_dir=overrides.get(
                "reproduction", args.baseline_reproduction_dir
            ),
            baseline_adaptive_dir=overrides.get("adaptive", args.baseline_adaptive_dir),
        )
        result = finalize_reports(paths, allow_incomplete=args.allow_incomplete)
    except ReportFinalizationError as exc:
        parser.exit(2, f"오류: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
