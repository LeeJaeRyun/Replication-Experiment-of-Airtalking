from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = STUDY_ROOT / "results" / "full_adaptive_results" / "summary_metrics.csv"
DEFAULT_OUT = STUDY_ROOT / "results" / "full_adaptive_results" / "result_validation.json"
MODES = ("nonsemantic", "fixed_paper_like", "adaptive_semantic")
AREAS = (100, 200, 300, 400, 500)
POLICIES = ("Stochastic", "LinUCB", "SA", "Greedy", "MCTS")


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict[str, object] = {"mode": row["mode"], "policy": row["policy"], "area": int(float(row["area"]))}
            for key, value in row.items():
                if key in {"mode", "area", "policy"} or value == "":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def validate(rows: list[dict[str, object]]) -> dict[str, object]:
    lookup = {(str(row["mode"]), int(row["area"]), str(row["policy"])): row for row in rows}
    expected = {(mode, area, policy) for mode in MODES for area in AREAS for policy in POLICIES}
    missing = sorted(expected - set(lookup))
    extra = sorted(set(lookup) - expected)

    nonfinite: list[dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite.append({"row": idx, "key": key, "value": str(value)})

    comparisons: list[dict[str, object]] = []
    for area in AREAS:
        for policy in POLICIES:
            fixed = lookup[("fixed_paper_like", area, policy)]
            adaptive = lookup[("adaptive_semantic", area, policy)]
            finished_delta_pct = (float(adaptive["finished"]) - float(fixed["finished"])) / float(fixed["finished"]) * 100.0
            avg_time_delta_pct = (float(adaptive["avg_time"]) - float(fixed["avg_time"])) / float(fixed["avg_time"]) * 100.0
            quality_delta = float(adaptive["semantic_quality"]) - float(fixed["semantic_quality"])
            comparisons.append(
                {
                    "area": area,
                    "policy": policy,
                    "finished_delta_pct": finished_delta_pct,
                    "avg_time_delta_pct": avg_time_delta_pct,
                    "quality_delta": quality_delta,
                    "pass": finished_delta_pct >= 0.0 and avg_time_delta_pct <= 0.0 and quality_delta >= -0.10,
                }
            )

    passed = not missing and not extra and not nonfinite and all(bool(item["pass"]) for item in comparisons)
    return {
        "passed": passed,
        "row_count": len(rows),
        "expected_row_count": len(expected),
        "missing_combinations": missing,
        "extra_combinations": extra,
        "nonfinite_values": nonfinite,
        "adaptive_vs_fixed": {
            "all_finished_ge_fixed": all(float(item["finished_delta_pct"]) >= 0.0 for item in comparisons),
            "all_avg_time_le_fixed": all(float(item["avg_time_delta_pct"]) <= 0.0 for item in comparisons),
            "max_quality_drop": min(float(item["quality_delta"]) for item in comparisons),
            "comparisons": comparisons,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full adaptive semantic compression results.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    rows = load_rows(Path(args.summary))
    result = validate(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
