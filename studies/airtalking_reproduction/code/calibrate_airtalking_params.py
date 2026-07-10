from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from airtalking_reproduction import (
    AssumedParams,
    PaperParams,
    apply_semantic_summary,
    run_experiments,
    write_summary_csv,
)
from verify_against_paper import compare_rows, load_actual


BASE_OVERRIDES = {
    # Smaller image/task payloads are consistent with the paper's 4-15 s
    # average request latency; the earlier 420 Mb assumption made latency
    # one order of magnitude too large.
    "workload_mean_bits": 45e6,
    "workload_std_bits": 12e6,
    "workload_min_bits": 18e6,
    "workload_max_bits": 90e6,
    # Multirotor propulsion can be hundreds of watts, but the paper's plotted
    # per-request flight energy is much lower than the prior 680/610 W setting.
    # These values behave like an effective task-level movement/hover power.
    "p_move": 85.0,
    "p_hover": 75.0,
    "p_encode": 0.9,
    "p_decode": 0.9,
    "p_d2d_radio": 0.2,
}


def summarize_score(rows: list[dict[str, str]]) -> dict[str, float]:
    metric_weights = {
        "finished": 1.0,
        "avg_time": 1.2,
        "flight_energy_per_req": 1.2,
    }
    total = 0.0
    count = 0
    matches = partials = mismatches = 0
    for row in rows:
        rel = float(row["relative_error"])
        weight = metric_weights.get(row["metric"], 1.0)
        total += weight * min(rel, 3.0)
        count += 1
        if row["verdict"] == "match":
            matches += 1
        elif row["verdict"] == "partial":
            partials += 1
        else:
            mismatches += 1
    return {
        "score": total / max(count, 1),
        "match": matches,
        "partial": partials,
        "mismatch": mismatches,
    }


def run_candidate(
    candidate: dict[str, float | int],
    paper: PaperParams,
    semantic_summary: Path,
    repeats: int,
    t_slots: int,
) -> dict[str, object]:
    assumed_values = {**AssumedParams().__dict__, **BASE_OVERRIDES, **candidate}
    assumed = AssumedParams(**assumed_values)
    tuned_paper = replace(paper, repeats=repeats, t_slots=t_slots)
    tuned_paper, _ = apply_semantic_summary(
        tuned_paper,
        semantic_summary,
        "uncompressed",
        "paper",
        "paper",
        "feature",
    )
    results = run_experiments(tuned_paper, assumed)
    with TemporaryDirectory() as tmp:
        summary_path = write_summary_csv(results, Path(tmp))
        actual = load_actual(summary_path)
        rows = compare_rows(actual)
    out = {**candidate, **summarize_score(rows)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick hidden-parameter calibration for the AirTalking reproduction.")
    parser.add_argument(
        "--semantic-summary",
        default="studies/airtalking_reproduction/results/cityscapes_semantic_measurement/cityscapes_semantic_summary.json",
    )
    parser.add_argument("--out", default="studies/airtalking_reproduction/results/calibration/candidates.csv")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--t-slots", type=int, default=350)
    args = parser.parse_args()

    paper = PaperParams()
    semantic_summary = Path(args.semantic_summary)
    candidates: list[dict[str, float | int]] = []
    for request_probability in [0.010, 0.014, 0.018, 0.022, 0.026]:
        for density_interference_scale in [0.0, 1.0, 3.0, 6.0, 10.0]:
            for energy_weight in [1 / 5000, 1 / 9000, 1 / 15000, 1 / 25000]:
                for sa_iterations in [4, 8, 16]:
                    candidates.append(
                        {
                            "request_probability": request_probability,
                            "density_interference_scale": density_interference_scale,
                            "energy_weight": energy_weight,
                            "sa_iterations": sa_iterations,
                        }
                    )

    rows: list[dict[str, object]] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, 1):
        result = run_candidate(candidate, paper, semantic_summary, args.repeats, args.t_slots)
        rows.append(result)
        if index % 10 == 0 or index == total:
            best = min(rows, key=lambda item: float(item["score"]))
            print(json.dumps({"done": index, "total": total, "best": best}, indent=2))

    rows.sort(key=lambda item: float(item["score"]))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out": str(out_path), "best": rows[0]}, indent=2))


if __name__ == "__main__":
    main()
