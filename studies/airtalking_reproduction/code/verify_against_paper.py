from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# Approximate final values read from the paper's rendered Figure 3, Figure 4,
# and Figure 6. The paper does not publish raw plotting data, so these are
# visual estimates used only for validation, not calibration.
PAPER_FINISHED = {
    100: {"Stochastic": 10, "LinUCB": 25, "SA": 31, "Greedy": 45, "MCTS": 47},
    200: {"Stochastic": 20, "LinUCB": 65, "SA": 65, "Greedy": 145, "MCTS": 135},
    300: {"Stochastic": 15, "LinUCB": 65, "SA": 50, "Greedy": 205, "MCTS": 195},
    400: {"Stochastic": 18, "LinUCB": 100, "SA": 85, "Greedy": 200, "MCTS": 225},
    500: {"Stochastic": 18, "LinUCB": 95, "SA": 35, "Greedy": 250, "MCTS": 210},
}

PAPER_AVG_TIME = {
    100: {"LinUCB": 32, "SA": 30, "Greedy": 27, "MCTS": 22},
    200: {"LinUCB": 9, "SA": 12, "Greedy": 7, "MCTS": 7},
    300: {"LinUCB": 10, "SA": 12, "Greedy": 5, "MCTS": 5},
    400: {"LinUCB": 8, "SA": 7, "Greedy": 4, "MCTS": 5},
    500: {"LinUCB": 10, "SA": 15, "Greedy": 4, "MCTS": 5},
}

PAPER_FLIGHT_ENERGY = {
    100: {"LinUCB": 36000, "SA": 32000, "Greedy": 30000, "MCTS": 24000},
    200: {"LinUCB": 10000, "SA": 14000, "Greedy": 8000, "MCTS": 8500},
    300: {"LinUCB": 10000, "SA": 13000, "Greedy": 5500, "MCTS": 6000},
    400: {"LinUCB": 8000, "SA": 10000, "Greedy": 5000, "MCTS": 5500},
    500: {"LinUCB": 10000, "SA": 16000, "Greedy": 4500, "MCTS": 5500},
}

PAPER_FIG6_FINISHED = {
    "semantic": {"LinUCB": 65, "SA": 50, "Greedy": 205, "MCTS": 180},
    "nonsemantic": {"LinUCB": 60, "SA": 45, "Greedy": 100, "MCTS": 65},
}


def load_actual(summary_path: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    out: dict[tuple[str, int, str], dict[str, float]] = {}
    with summary_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"mode", "area", "policy", "finished", "avg_time", "flight_energy_per_req"}
        missing_columns = sorted(required - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"Summary is missing required columns: {missing_columns}")
        for line_number, row in enumerate(reader, start=2):
            key = (str(row["mode"]).strip(), int(float(row["area"])), str(row["policy"]).strip())
            if key in out:
                raise ValueError(f"Duplicate summary key at line {line_number}: {key}")
            values = {
                key: float(value)
                for key, value in row.items()
                if key not in {"mode", "area", "policy"} and value != ""
            }
            nonfinite = [name for name, value in values.items() if not math.isfinite(value)]
            if nonfinite:
                raise ValueError(f"Non-finite summary values at line {line_number}: {nonfinite}")
            out[key] = values
    return out


def verdict(rel_error: float) -> str:
    if rel_error <= 0.25:
        return "match"
    if rel_error <= 0.50:
        return "partial"
    return "mismatch"


def compare_rows(actual: dict[tuple[str, int, str], dict[str, float]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    checks = [
        ("Figure 3 finished", "semantic", "finished", PAPER_FINISHED),
        ("Figure 4 avg_time", "semantic", "avg_time", PAPER_AVG_TIME),
        ("Figure 3 flight_energy", "semantic", "flight_energy_per_req", PAPER_FLIGHT_ENERGY),
    ]
    missing_requirements: list[str] = []
    for _, mode, metric, expected_by_area in checks:
        for area, expected_by_policy in expected_by_area.items():
            for policy in expected_by_policy:
                key = (mode, area, policy)
                if key not in actual or metric not in actual[key]:
                    missing_requirements.append(f"{key}:{metric}")
    for mode, expected_by_policy in PAPER_FIG6_FINISHED.items():
        for policy in expected_by_policy:
            key = (mode, 300, policy)
            if key not in actual or "finished" not in actual[key]:
                missing_requirements.append(f"{key}:finished")
    if missing_requirements:
        raise ValueError(f"Summary cannot support paper verification; missing {missing_requirements[:10]}")
    for check_name, mode, metric, expected_by_area in checks:
        for area, expected_by_policy in expected_by_area.items():
            for policy, expected in expected_by_policy.items():
                got = actual[(mode, area, policy)][metric]
                abs_error = got - expected
                rel_error = abs(abs_error) / expected if expected else 0.0
                rows.append(
                    {
                        "check": check_name,
                        "mode": mode,
                        "area": str(area),
                        "policy": policy,
                        "metric": metric,
                        "paper_visual_estimate": f"{expected:.3f}",
                        "reproduction": f"{got:.3f}",
                        "absolute_error": f"{abs_error:.3f}",
                        "relative_error": f"{rel_error:.3f}",
                        "verdict": verdict(rel_error),
                    }
                )
    for mode, expected_by_policy in PAPER_FIG6_FINISHED.items():
        for policy, expected in expected_by_policy.items():
            got = actual[(mode, 300, policy)]["finished"]
            abs_error = got - expected
            rel_error = abs(abs_error) / expected if expected else 0.0
            rows.append(
                {
                    "check": "Figure 6 finished",
                    "mode": mode,
                    "area": "300",
                    "policy": policy,
                    "metric": "finished",
                    "paper_visual_estimate": f"{expected:.3f}",
                    "reproduction": f"{got:.3f}",
                    "absolute_error": f"{abs_error:.3f}",
                    "relative_error": f"{rel_error:.3f}",
                    "verdict": verdict(rel_error),
                }
            )
    return rows


def qualitative_checks(actual: dict[tuple[str, int, str], dict[str, float]]) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    areas = [100, 200, 300, 400, 500]
    for policy in ["Stochastic", "LinUCB", "SA", "Greedy", "MCTS"]:
        values = [actual[("semantic", area, policy)]["finished"] for area in areas]
        monotonic = all(a <= b for a, b in zip(values, values[1:]))
        checks.append(
            (
                f"Finished requests increase with area for {policy}",
                "match" if monotonic else "partial",
                ", ".join(f"{area}:{value:.1f}" for area, value in zip(areas, values)),
            )
        )
    linucb_better_large = (
        actual[("semantic", 400, "LinUCB")]["finished"] > actual[("semantic", 400, "SA")]["finished"]
        and actual[("semantic", 500, "LinUCB")]["finished"] > actual[("semantic", 500, "SA")]["finished"]
    )
    checks.append(
        (
            "Paper statement: LinUCB outperforms SA as area enlarges",
            "match" if linucb_better_large else "mismatch",
            f"400m LinUCB={actual[('semantic', 400, 'LinUCB')]['finished']:.1f}, SA={actual[('semantic', 400, 'SA')]['finished']:.1f}; "
            f"500m LinUCB={actual[('semantic', 500, 'LinUCB')]['finished']:.1f}, SA={actual[('semantic', 500, 'SA')]['finished']:.1f}",
        )
    )
    sem_beats_ns = all(
        actual[("semantic", 300, policy)]["finished"] > actual[("nonsemantic", 300, policy)]["finished"]
        for policy in ["LinUCB", "SA", "Greedy", "MCTS"]
    )
    checks.append(
        (
            "Figure 6 direction: semantic beats non-semantic on finished requests",
            "match" if sem_beats_ns else "mismatch",
            "; ".join(
                f"{policy}: sem={actual[('semantic', 300, policy)]['finished']:.1f}, ns={actual[('nonsemantic', 300, policy)]['finished']:.1f}"
                for policy in ["LinUCB", "SA", "Greedy", "MCTS"]
            ),
        )
    )
    return checks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    rows: list[dict[str, str]],
    qualitative: list[tuple[str, str, str]],
    out_dir: Path,
    label: str,
    summary_path: Path,
) -> None:
    if not rows:
        raise ValueError("No quantitative verification rows were produced")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if not label else (label if label.startswith("_") else f"_{label}")
    csv_out = out_dir / f"verification_against_paper{suffix}.csv"
    md_out = out_dir / f"verification_against_paper{suffix}.md"
    json_out = out_dir / f"verification_against_paper{suffix}.json"
    with csv_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {"match": 0, "partial": 0, "mismatch": 0}
    for row in rows:
        counts[row["verdict"]] += 1
    qual_counts: dict[str, int] = {"match": 0, "partial": 0, "mismatch": 0}
    for _, status, _ in qualitative:
        qual_counts[status] += 1

    lines = [
        f"# AirTalking reproduction verification against paper figures"
        f"{f' ({label})' if label else ''}",
        "",
        "## Conclusion",
        "",
        "The current reproduction does not quantitatively match the paper figures. It preserves a few qualitative directions, especially that semantic processing outperforms the non-semantic baseline at 300 x 300 m2, but several key policy rankings and magnitudes differ.",
        "",
        "## Quantitative check summary",
        "",
        f"- Match: {counts['match']}",
        f"- Partial: {counts['partial']}",
        f"- Mismatch: {counts['mismatch']}",
        "",
        "A `match` means the reproduction is within 25% of the paper visual estimate; `partial` is within 50%; `mismatch` is outside 50%. Because the paper provides plots but not raw data, the paper-side numbers are visual estimates from rendered Figure 3, Figure 4, and Figure 6.",
        "",
        "## Qualitative checks",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for check, status, evidence in qualitative:
        lines.append(f"| {check} | {status} | {evidence} |")
    top_mismatches = sorted(rows, key=lambda row: float(row["relative_error"]), reverse=True)[:8]
    lines.extend(
        [
            "",
            "## Largest numeric deviations",
            "",
            "| Check | Area | Policy | Metric | Paper estimate | Reproduction | Relative error |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in top_mismatches:
        lines.append(
            f"| {row['check']} | {row['area']} | {row['policy']} | {row['metric']} | "
            f"{row['paper_visual_estimate']} | {row['reproduction']} | {row['relative_error']} |"
        )
    lines.extend(
        [
            "",
            "## Main mismatch types",
            "",
            "- Magnitude mismatch: several reproduced metrics are outside the 50% tolerance against visually estimated paper values.",
            "- Hidden-parameter sensitivity: request probability, workload distribution, propulsion/hover power, and detailed interference scheduling are not numerically disclosed in the paper.",
            "- Dataset/profile sensitivity: substitute semantic payload profiles can improve some metrics but shift completed-request counts and energy/latency trade-offs.",
            "",
            "## Likely causes",
            "",
            "- The paper does not publish raw source code, request probability, workload distribution, propulsion/hover power, codec power, or full interference scheduling details.",
            "- The reproduction uses assumed values for those hidden parameters, and those assumptions materially change latency, energy, and policy ranking.",
            "- The density interference correction added to mimic small-area interference improves one trend but inflates latency and flight energy relative to the paper.",
            "",
            "## Detailed CSV",
            "",
            f"See `{csv_out}` for row-level expected vs. reproduced values.",
            "",
        ]
    )
    md_out.write_text("\n".join(lines), encoding="utf-8")
    source_path = summary_path.resolve()
    verifier_source = Path(__file__).resolve()
    payload = {
        "schema_version": 2,
        "status": "completed",
        "source_summary": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "row_count": len(rows),
        "verdict_counts": counts,
        "qualitative_row_count": len(qualitative),
        "qualitative_verdict_counts": qual_counts,
        "thresholds": {"match_max_relative_error": 0.25, "partial_max_relative_error": 0.50},
        "artifacts": {
            "verification_csv": {"path": str(csv_out.resolve()), "sha256": sha256_file(csv_out)},
            "verification_markdown": {"path": str(md_out.resolve()), "sha256": sha256_file(md_out)},
        },
        "provenance": {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "argv": list(sys.argv),
            "command_windows": subprocess.list2cmdline(sys.argv),
            "python": platform.python_version(),
            "verifier_source": str(verifier_source),
            "verifier_source_sha256": sha256_file(verifier_source),
        },
    }
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"csv": str(csv_out), "markdown": str(md_out), "json": str(json_out)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="studies/airtalking_reproduction/results/airtalking_reproduction/summary_metrics.csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    actual = load_actual(summary_path)
    rows = compare_rows(actual)
    qualitative = qualitative_checks(actual)
    write_outputs(rows, qualitative, out_dir, args.label, summary_path)


if __name__ == "__main__":
    main()
