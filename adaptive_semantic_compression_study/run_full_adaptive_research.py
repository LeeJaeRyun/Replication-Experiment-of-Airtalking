from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


DEFAULT_METADATA = ROOT / "outputs" / "airtalking_cityscapes_calibrated_final_p012" / "run_metadata.json"
DEFAULT_QUALITY = ROOT / "adaptive_semantic_compression_study" / "outputs" / "compression_quality.csv"
DEFAULT_OUT = ROOT / "adaptive_semantic_compression_study" / "full_adaptive_results"


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


def build_profiles(rows: Sequence[dict[str, float | str]]) -> Tuple[SemanticProfile, SemanticProfile]:
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
    adaptive = SemanticProfile(name="adaptive_semantic", strategy="adaptive", modes=modes)
    return fixed, adaptive


def run_mode(
    mode_name: str,
    semantic_enabled: bool,
    profile: Optional[SemanticProfile],
    paper: PaperParams,
    assumed: AssumedParams,
    repeats: int,
    areas: Sequence[int],
    policies: Sequence[str],
) -> Dict[int, Dict[str, SimulationResult]]:
    out: Dict[int, Dict[str, SimulationResult]] = {}
    for area in areas:
        out[area] = {}
        for policy in policies:
            print(f"[{mode_name}] area={area} policy={policy} repeats={repeats}", flush=True)
            reps = [
                run_single(
                    area,
                    policy,
                    repeat,
                    semantic_enabled,
                    paper,
                    assumed,
                    semantic_profile=profile,
                )
                for repeat in range(repeats)
            ]
            out[area][policy] = aggregate(reps)
    return out


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
    policies = ["Greedy", "MCTS"]
    areas = sorted({area for mode_results in results.values() for area in mode_results})
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
    draw.text((835, 563), "bars repeat as Greedy then MCTS in each area", font=_font(12), fill="#64748b")
    image.save(path)
    return path


def save_tradeoff(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path, area: int = 300) -> Path:
    path = out_dir / "figures" / "latency_quality_tradeoff_300m.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (980, 620), "white")
    draw = ImageDraw.Draw(image)
    box = (105, 85, 880, 500)
    _draw_axes(draw, box, "Latency-quality trade-off at 300 x 300 m", "average time cost (s)", "semantic quality")
    modes = ["nonsemantic", "fixed_paper_like", "adaptive_semantic"]
    colors = {"nonsemantic": "#64748b", "fixed_paper_like": "#2563eb", "adaptive_semantic": "#16a34a"}
    points: list[tuple[str, str, float, float]] = []
    for mode in modes:
        if mode not in results or area not in results[mode]:
            continue
        for policy in POLICIES:
            if policy not in results[mode][area]:
                continue
            summary = results[mode][area][policy].summary
            points.append((mode, policy, summary["avg_time"], summary["semantic_quality"]))
    xs = [point[2] for point in points] or [1.0]
    ys = [point[3] for point in points] or [1.0]
    x_min, x_max = min(xs) * 0.88, max(xs) * 1.08
    y_min, y_max = max(0.75, min(ys) - 0.02), min(1.01, max(ys) + 0.02)
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
        if policy in {"Greedy", "MCTS"}:
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
    path = out_dir / "figures" / "adaptive_mode_usage_300m.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1080, 600), "white")
    draw = ImageDraw.Draw(image)
    box = (90, 80, 1000, 465)
    _draw_axes(draw, box, "Adaptive compression mode usage at 300 x 300 m", "policy", "selected requests")
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
    policies = [policy for policy in POLICIES if policy in rows]
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


def save_figures(results: Dict[str, Dict[int, Dict[str, SimulationResult]]], out_dir: Path) -> Dict[str, str]:
    paths = [
        save_finished_by_area(results, out_dir),
        save_tradeoff(results, out_dir),
        save_adaptive_mode_usage(results, out_dir),
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
    out_dir: Path,
    elapsed: float,
    repeats: int,
    t_slots: int,
) -> Path:
    area = 300
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
        "- Base simulator: the calibrated AirTalking reproduction parameters from `outputs/airtalking_cityscapes_calibrated_final_p012/run_metadata.json`.",
        f"- Repeats: {repeats}; simulation slots per repeat: {t_slots}.",
        "- Compared modes: nonsemantic raw payload, fixed Cityscapes paper-like semantic payload, and channel-aware adaptive semantic payload.",
        "",
        "## Compression table",
        "",
        "| mode | payload ratio | mean IoU |",
        "|---|---:|---:|",
    ]
    for row in quality_rows:
        lines.append(f"| {row['mode']} | {float(row['feature_ratio_mean']):.6f} | {float(row['mean_iou_mean']):.3f} |")

    lines.extend(
        [
            "",
            "## 300 x 300 m result table",
            "",
            "| policy | mode | finished | avg time (s) | flight J/req | semantic quality | payload ratio |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    available_policies = [
        policy
        for policy in POLICIES
        if all(mode in results and area in results[mode] and policy in results[mode][area] for mode in ["nonsemantic", "fixed_paper_like", "adaptive_semantic"])
    ]
    for policy in available_policies:
        for mode in ["nonsemantic", "fixed_paper_like", "adaptive_semantic"]:
            summary = results[mode][area][policy].summary
            lines.append(f"| {policy} | {mode} | {format_row(summary)} |")

    lines.extend(["", "## Adaptive vs fixed semantic at 300 x 300 m", ""])
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
            f"- {anchor_policy} at 300 m: adaptive finished {adaptive_greedy['finished']:.1f} requests vs fixed {fixed_greedy['finished']:.1f}; average time changed from {fixed_greedy['avg_time']:.2f}s to {adaptive_greedy['avg_time']:.2f}s.",
            f"- {second_policy} at 300 m: adaptive finished {adaptive_mcts['finished']:.1f} requests vs fixed {fixed_mcts['finished']:.1f}; average time changed from {fixed_mcts['avg_time']:.2f}s to {adaptive_mcts['avg_time']:.2f}s.",
            "- The result should be read as a trade-off study, not as a claim that the original AirTalking paper is fully reproduced. The encoder/decoder network is still not public; Cityscapes label maps are used as a public semantic-feature proxy.",
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
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--t-slots", type=int, default=None)
    parser.add_argument("--areas", default=",".join(str(area) for area in AREAS))
    parser.add_argument("--policies", default=",".join(POLICIES))
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    quality_path = Path(args.quality)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper, assumed, metadata = load_base_params(metadata_path)
    repeats = args.repeats or paper.repeats
    t_slots = args.t_slots or paper.t_slots
    paper = PaperParams(**{**paper.__dict__, "repeats": repeats, "t_slots": t_slots})
    quality_rows = load_quality_rows(quality_path)
    fixed_profile, adaptive_profile = build_profiles(quality_rows)

    areas = tuple(int(value.strip()) for value in args.areas.split(",") if value.strip())
    policies = tuple(value.strip() for value in args.policies.split(",") if value.strip())

    started = time.perf_counter()
    results: Dict[str, Dict[int, Dict[str, SimulationResult]]] = {
        "nonsemantic": run_mode("nonsemantic", False, None, paper, assumed, repeats, areas, policies),
        "fixed_paper_like": run_mode("fixed_paper_like", True, fixed_profile, paper, assumed, repeats, areas, policies),
        "adaptive_semantic": run_mode("adaptive_semantic", True, adaptive_profile, paper, assumed, repeats, areas, policies),
    }
    elapsed = time.perf_counter() - started

    summary_csv = write_summary_csv(results, out_dir)
    usage_csv = write_mode_usage_csv(results, out_dir)
    npz_path = write_timeseries_npz(results, out_dir)
    figure_paths = save_figures(results, out_dir)
    report_path = write_analysis(results, quality_rows, out_dir, elapsed, repeats, t_slots)
    metadata_out = out_dir / "run_metadata.json"
    metadata_out.write_text(
        json.dumps(
            {
                "source_metadata": str(metadata_path),
                "source_quality": str(quality_path),
                "base_paper_params": paper.__dict__,
                "base_assumed_params": assumed.__dict__,
                "source_reproduction_metadata": metadata,
                "profiles": {
                    "fixed_paper_like": {
                        "name": fixed_profile.name,
                        "strategy": fixed_profile.strategy,
                        "modes": [mode.__dict__ for mode in fixed_profile.modes],
                        "target_thresholds": fixed_profile.target_thresholds,
                    },
                    "adaptive_semantic": {
                        "name": adaptive_profile.name,
                        "strategy": adaptive_profile.strategy,
                        "modes": [mode.__dict__ for mode in adaptive_profile.modes],
                        "target_thresholds": adaptive_profile.target_thresholds,
                    },
                },
                "areas": areas,
                "policies": policies,
                "elapsed_seconds": elapsed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "summary_csv": str(summary_csv),
                "usage_csv": str(usage_csv),
                "npz": str(npz_path),
                "report": str(report_path),
                "metadata": str(metadata_out),
                "figures": figure_paths,
                "elapsed_seconds": elapsed,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
