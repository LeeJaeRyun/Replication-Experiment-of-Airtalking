from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = STUDY_ROOT / "results" / "full_adaptive_results" / "summary_metrics.csv"
DEFAULT_OUT = STUDY_ROOT / "results" / "full_adaptive_results" / "result_validation.json"
MODES = ("nonsemantic", "fixed_paper_like", "adaptive_semantic")
AREAS = (100, 200, 300, 400, 500)
POLICIES = ("Stochastic", "LinUCB", "SA", "Greedy", "MCTS")
KEY_COLUMNS = ("mode", "area", "policy")
METRIC_COLUMNS = ("finished", "avg_time", "semantic_quality")
REQUIRED_COLUMNS = (*KEY_COLUMNS, *METRIC_COLUMNS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def load_rows_with_schema(path: Path) -> tuple[list[dict[str, object]], tuple[str, ...], list[dict[str, object]]]:
    """Read raw cells so malformed input becomes validation output, not an exception."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            columns = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], (), [_error("summary_read_error", f"Could not read summary CSV: {exc}", exception_type=type(exc).__name__)]
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    errors = (
        [_error("missing_required_columns", f"Summary CSV is missing required columns: {', '.join(missing)}.", columns=missing)]
        if missing
        else []
    )
    return rows, columns, errors


def load_rows(path: Path) -> list[dict[str, object]]:
    rows, _, _ = load_rows_with_schema(path)
    return rows


def _text(value: object) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def _number(value: object) -> tuple[float | None, str | None]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "not_numeric"
    return (number, None) if math.isfinite(number) else (None, "nonfinite")


def _area(value: object) -> tuple[int | None, str | None]:
    number, problem = _number(value)
    if problem or number is None:
        return None, problem
    if not number.is_integer():
        return None, "not_integer"
    return (int(number), None) if number > 0 else (None, "not_positive")


def _source(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"path": None, "sha256": None}
    resolved = path.resolve()
    try:
        digest: str | None = sha256_file(resolved)
    except OSError:
        digest = None
    return {"path": str(resolved), "sha256": digest}


def validate(
    rows: list[dict[str, object]],
    *,
    available_columns: Sequence[str] | None = None,
    pre_errors: Sequence[Mapping[str, object]] = (),
    source_summary_path: Path | None = None,
    expected_areas: Sequence[int] | None = None,
    expected_policies: Sequence[str] | None = None,
) -> dict[str, object]:
    errors = [dict(error) for error in pre_errors]
    columns = set(available_columns or {key for row in rows for key in row})
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns and not any(error.get("code") == "missing_required_columns" for error in errors):
        errors.append(_error("missing_required_columns", f"Summary is missing required columns: {', '.join(missing_columns)}.", columns=missing_columns))

    normalized: list[dict[str, object]] = []
    missing_values: list[dict[str, object]] = []
    invalid_keys: list[dict[str, object]] = []
    invalid_numeric: list[dict[str, object]] = []
    nonfinite: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        mode, policy = _text(row.get("mode")), _text(row.get("policy"))
        area, area_problem = _area(row.get("area"))
        invalid = False
        for column, value in (("mode", mode), ("policy", policy)):
            if value is None:
                missing_values.append({"row": index, "column": column})
                invalid = True
        if area_problem:
            item = {"row": index, "column": "area", "value": str(row.get("area"))}
            if area_problem == "missing":
                missing_values.append(item)
            elif area_problem == "nonfinite":
                nonfinite.append(item)
            else:
                invalid_keys.append({**item, "reason": area_problem})
            invalid = True
        metrics: dict[str, float] = {}
        for column in METRIC_COLUMNS:
            number, problem = _number(row.get(column))
            if problem is None and number is not None:
                metrics[column] = number
            else:
                item = {"row": index, "column": column, "value": str(row.get(column))}
                if problem == "missing":
                    missing_values.append(item)
                elif problem == "nonfinite":
                    nonfinite.append(item)
                else:
                    invalid_numeric.append(item)
                invalid = True
        for column, value in row.items():
            if column in REQUIRED_COLUMNS or value is None or (isinstance(value, str) and not value.strip()):
                continue
            if isinstance(value, (int, float)):
                optional_number = float(value)
                optional_is_nonfinite = not math.isfinite(optional_number)
            elif isinstance(value, str) and value.strip().lower() in {
                "nan",
                "+nan",
                "-nan",
                "inf",
                "+inf",
                "-inf",
                "infinity",
                "+infinity",
                "-infinity",
            }:
                optional_is_nonfinite = True
            else:
                optional_is_nonfinite = False
            if optional_is_nonfinite:
                nonfinite.append({"row": index, "column": column, "value": str(value)})
                invalid = True
        if mode is not None and policy is not None and area is not None:
            normalized.append({"row": index, "mode": mode, "area": area, "policy": policy, "metrics": metrics, "valid": not invalid})

    diagnostic_groups = (
        (missing_values, "missing_required_values", "missing required value"),
        (invalid_keys, "invalid_canonical_keys", "invalid canonical key value"),
        (invalid_numeric, "invalid_numeric_values", "non-numeric metric value"),
        (nonfinite, "nonfinite_values", "non-finite required numeric value"),
    )
    for values, code, label in diagnostic_groups:
        if values:
            errors.append(_error(code, f"Found {len(values)} {label}(s).", count=len(values)))

    keyed: dict[tuple[str, int, str], list[dict[str, object]]] = {}
    for row in normalized:
        key = (str(row["mode"]), int(row["area"]), str(row["policy"]))
        keyed.setdefault(key, []).append(row)
    duplicates = [
        {"combination": key, "rows": [int(row["row"]) for row in values]}
        for key, values in sorted(keyed.items())
        if len(values) > 1
    ]
    if duplicates:
        errors.append(_error("duplicate_combinations", f"Found {len(duplicates)} duplicate canonical combination(s).", count=len(duplicates)))

    areas = (
        tuple(int(value) for value in expected_areas)
        if expected_areas is not None
        else tuple(dict.fromkeys(int(row["area"]) for row in normalized))
    )
    policies = (
        tuple(str(value).strip() for value in expected_policies)
        if expected_policies is not None
        else tuple(dict.fromkeys(str(row["policy"]) for row in normalized))
    )
    if not areas:
        errors.append(_error("empty_areas", "No valid area values were found."))
    if not policies:
        errors.append(_error("empty_policies", "No valid policy values were found."))
    expected = {(mode, area, policy) for mode in MODES for area in areas for policy in policies}
    missing, extra = sorted(expected - set(keyed)), sorted(set(keyed) - expected)
    if missing:
        errors.append(_error("missing_combinations", f"{len(missing)} required mode/area/policy combination(s) are missing.", count=len(missing)))
    if extra:
        errors.append(_error("extra_combinations", f"Found {len(extra)} unexpected mode/area/policy combination(s).", count=len(extra)))

    structural = bool(errors or duplicates or missing or extra)
    zero_denominators: list[dict[str, object]] = []
    if not structural:
        for area in areas:
            for policy in policies:
                metrics = keyed[("fixed_paper_like", area, policy)][0]["metrics"]
                assert isinstance(metrics, Mapping)
                for metric in ("finished", "avg_time"):
                    if float(metrics[metric]) == 0.0:
                        zero_denominators.append({"combination": ("fixed_paper_like", area, policy), "metric": metric, "value": 0.0})
    if zero_denominators:
        errors.append(_error("zero_denominator", f"Found {len(zero_denominators)} zero comparison denominator(s).", count=len(zero_denominators)))
        structural = True

    comparisons: list[dict[str, object]] = []
    if not structural:
        for area in areas:
            for policy in policies:
                fixed = keyed[("fixed_paper_like", area, policy)][0]["metrics"]
                adaptive = keyed[("adaptive_semantic", area, policy)][0]["metrics"]
                assert isinstance(fixed, Mapping) and isinstance(adaptive, Mapping)
                finished_delta = (float(adaptive["finished"]) - float(fixed["finished"])) / float(fixed["finished"]) * 100.0
                time_delta = (float(adaptive["avg_time"]) - float(fixed["avg_time"])) / float(fixed["avg_time"]) * 100.0
                quality_delta = float(adaptive["semantic_quality"]) - float(fixed["semantic_quality"])
                comparisons.append({"area": area, "policy": policy, "finished_delta_pct": finished_delta, "avg_time_delta_pct": time_delta, "quality_delta": quality_delta, "pass": finished_delta >= 0.0 and time_delta <= 0.0 and quality_delta >= -0.10})

    evaluated = not structural
    reason = None
    if not evaluated:
        reason = (
            f"Cannot evaluate adaptive-vs-fixed comparisons: {len(missing)} required mode/area/policy combination(s) are missing."
            if missing
            else "Cannot evaluate adaptive-vs-fixed comparisons because structural validation failed."
        )
    schema = {"required_columns": list(REQUIRED_COLUMNS), "key_columns": list(KEY_COLUMNS), "numeric_columns": ["area", *METRIC_COLUMNS], "canonical_key": ["mode", "integer(area)", "policy"]}
    result: dict[str, object] = {
        "schema_version": 2,
        "source_summary": _source(source_summary_path),
        "passed": not errors and evaluated,
        "row_count": len(rows),
        "expected_row_count": len(expected),
        "expected_combinations": {"modes": list(MODES), "areas": list(areas), "policies": list(policies), "count": len(expected)},
        "schema": schema,
        "expected_schema": schema,
        "missing_combinations": missing,
        "missing": {"combinations": missing, "values": missing_values, "required_columns": missing_columns},
        "extra_combinations": extra,
        "duplicates": duplicates,
        "duplicate_combinations": duplicates,
        "missing_values": missing_values,
        "invalid_canonical_keys": invalid_keys,
        "invalid_numeric_values": invalid_numeric,
        "nonfinite_values": nonfinite,
        "nonfinite": nonfinite,
        "zero_denominators": zero_denominators,
        "zero_denominator": zero_denominators,
        "errors": errors,
        "adaptive_vs_fixed": {
            "evaluated": evaluated,
            "reason": reason,
            "all_comparisons_pass": evaluated and all(bool(item["pass"]) for item in comparisons),
            "all_finished_ge_fixed": evaluated and all(float(item["finished_delta_pct"]) >= 0.0 for item in comparisons),
            "all_avg_time_le_fixed": evaluated and all(float(item["avg_time_delta_pct"]) <= 0.0 for item in comparisons),
            "max_quality_drop": min((float(item["quality_delta"]) for item in comparisons), default=None),
            "comparisons": comparisons,
        },
    }
    json.dumps(result, allow_nan=False)
    return result


def write_result(path: Path, result: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full adaptive semantic compression results.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    summary = Path(args.summary)
    rows, columns, load_errors = load_rows_with_schema(summary)
    result = validate(rows, available_columns=columns, pre_errors=load_errors, source_summary_path=summary)
    write_result(Path(args.out), result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
