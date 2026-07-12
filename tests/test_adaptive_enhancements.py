from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
AIR_TALKING_CODE = WORKSPACE_ROOT / "studies" / "airtalking_reproduction" / "code"
ADAPTIVE_CODE = WORKSPACE_ROOT / "studies" / "adaptive_semantic_compression" / "code"
for code_dir in (AIR_TALKING_CODE, ADAPTIVE_CODE):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

import airtalking_reproduction as reproduction  # noqa: E402
import run_full_adaptive_research as adaptive  # noqa: E402


def quality_rows() -> list[dict[str, float | str]]:
    return [
        {
            "mode": mode,
            "description": f"legacy {mode}",
            "feature_ratio_mean": 0.01 * (index + 1),
            "feature_ratio_median": 0.01 * (index + 1),
            "zlib_ratio_mean": 0.005 * (index + 1),
            "zlib_ratio_median": 0.005 * (index + 1),
            "mean_iou_mean": 0.60 + 0.01 * index,
            "mean_iou_median": 0.60 + 0.01 * index,
        }
        for index, mode in enumerate(adaptive.ADAPTIVE_MODE_ORDER)
    ]


def enhanced_profiles() -> list[dict[str, float | int]]:
    profiles = [
        {"active_channels": 20, "rho_uint8": 0.02, "rho_zlib": 0.012, "mean_iou": 0.71, "psnr_db": 21.0, "ssim": 0.71},
        {"active_channels": 40, "rho_uint8": 0.04, "rho_zlib": 0.022, "mean_iou": 0.76, "psnr_db": 22.0, "ssim": 0.76},
        {"active_channels": 60, "rho_uint8": 0.07, "rho_zlib": 0.035, "mean_iou": 0.81, "psnr_db": 23.0, "ssim": 0.81},
        {"active_channels": 80, "rho_uint8": 0.10, "rho_zlib": 0.048, "mean_iou": 0.86, "psnr_db": 24.0, "ssim": 0.86},
        {"active_channels": 120, "rho_uint8": 0.15, "rho_zlib": 0.070, "mean_iou": 0.91, "psnr_db": 25.0, "ssim": 0.91},
    ]
    return [profiles[index] for index in (3, 0, 4, 1, 2)]


def enhanced_summary() -> dict[str, object]:
    return {
        "source": "enhanced_test_codec",
        "paper_like_active_channels": 80,
        "rho_c_feature_uncompressed_mean": 0.10,
        "semantic_quality_miou_best": 0.86,
        "pixel_accuracy_best": 0.93,
        "feature_encode_bitrate_mbps_median": 200.0,
        "feature_decode_bitrate_mbps_median": 30.0,
        "multi_rate_profiles": enhanced_profiles(),
        "timing": {
            "encode_including_8bit_fake_quantization": {"median_ms": 2.0},
            "decode_from_latent_only": {"median_ms": 3.0},
        },
    }


def low_quality_summary() -> dict[str, object]:
    summary = enhanced_summary()
    profiles = [dict(profile) for profile in summary["multi_rate_profiles"]]  # type: ignore[index]
    by_channels = {20: 0.11, 40: 0.16, 60: 0.21, 80: 0.26, 120: 0.31}
    for profile in profiles:
        profile["mean_iou"] = by_channels[int(profile["active_channels"])]
    summary["multi_rate_profiles"] = profiles
    summary["semantic_quality_miou_best"] = 0.31
    return summary


class MultiRateAnchorTests(unittest.TestCase):
    def test_record_only_uses_neural_ratios_but_preserves_selection_quality(self) -> None:
        updated = adaptive.apply_neural_anchor(quality_rows(), enhanced_summary(), "record_only")

        self.assertEqual([row["mode"] for row in updated], list(adaptive.ADAPTIVE_MODE_ORDER))
        self.assertEqual([row["feature_ratio_mean"] for row in updated], [0.02, 0.04, 0.07, 0.10, 0.15])
        self.assertEqual([row["mean_iou_mean"] for row in updated], [0.60, 0.61, 0.62, 0.63, 0.64])
        self.assertEqual([row["neural_encoder_decoder_miou"] for row in updated], [0.71, 0.76, 0.81, 0.86, 0.91])
        self.assertEqual([row["neural_active_channels"] for row in updated], [20, 40, 60, 80, 120])
        self.assertEqual([row["neural_psnr_db"] for row in updated], [21.0, 22.0, 23.0, 24.0, 25.0])
        self.assertEqual([row["neural_ssim"] for row in updated], [0.71, 0.76, 0.81, 0.86, 0.91])

    def test_selection_uses_all_five_measured_neural_qualities(self) -> None:
        updated = adaptive.apply_neural_anchor(quality_rows(), enhanced_summary(), "selection")
        self.assertEqual([row["mean_iou_mean"] for row in updated], [0.71, 0.76, 0.81, 0.86, 0.91])

    def test_measured_threshold_frontier_reaches_all_five_rates(self) -> None:
        rows = adaptive.apply_neural_anchor(quality_rows(), enhanced_summary(), "selection")
        thresholds, details = adaptive.resolve_adaptive_thresholds(
            rows, enhanced_summary(), "selection", "auto"
        )
        _, profile = adaptive.build_profiles(rows, thresholds)

        self.assertEqual(details["resolved_rule"], "measured_ordered")
        self.assertEqual(details["quality_thresholds"], [0.71, 0.76, 0.81, 0.86, 0.91])
        self.assertEqual(adaptive.reachable_adaptive_modes(profile), list(adaptive.ADAPTIVE_MODE_ORDER))

    def test_record_only_auto_thresholds_follow_preserved_quality_table(self) -> None:
        rows = adaptive.apply_neural_anchor(quality_rows(), enhanced_summary(), "record_only")
        thresholds, details = adaptive.resolve_adaptive_thresholds(
            rows, enhanced_summary(), "record_only", "auto"
        )
        _, profile = adaptive.build_profiles(rows, thresholds)

        self.assertEqual(details["resolved_rule"], "ordered_mode_quality")
        self.assertEqual(adaptive.reachable_adaptive_modes(profile), list(adaptive.ADAPTIVE_MODE_ORDER))

    def test_nonmonotonic_measured_frontier_still_avoids_single_mode_collapse(self) -> None:
        summary = enhanced_summary()
        measured = {20: 0.3048, 40: 0.3058, 60: 0.3050, 80: 0.3054, 120: 0.3063}
        for profile in summary["multi_rate_profiles"]:  # type: ignore[index]
            profile["mean_iou"] = measured[int(profile["active_channels"])]
        rows = adaptive.apply_neural_anchor(quality_rows(), summary, "selection")
        thresholds, _ = adaptive.resolve_adaptive_thresholds(rows, summary, "selection", "auto")
        _, profile = adaptive.build_profiles(rows, thresholds)

        self.assertGreaterEqual(len(adaptive.reachable_adaptive_modes(profile)), 2)

    def test_explicit_threshold_rule_records_exact_cli_values(self) -> None:
        rows = adaptive.apply_neural_anchor(quality_rows(), enhanced_summary(), "selection")
        explicit = adaptive.parse_quality_thresholds("0.10,0.20,0.30,0.40,0.50")
        thresholds, details = adaptive.resolve_adaptive_thresholds(
            rows, enhanced_summary(), "selection", "explicit", explicit
        )

        self.assertEqual(details["resolved_rule"], "explicit")
        self.assertEqual(details["quality_thresholds"], [0.10, 0.20, 0.30, 0.40, 0.50])
        self.assertEqual([quality for _, quality in thresholds], [0.10, 0.20, 0.30, 0.40, 0.50])

    def test_nested_alias_schema_and_zlib_fallback_are_supported(self) -> None:
        profiles = []
        for index, channels in enumerate((20, 40, 60, 80, 120), start=1):
            profiles.append(
                {
                    "channel_count": channels,
                    "payload_ratios": {"zlib": index / 100.0},
                    "quality_metrics": {
                        "semantic_quality": 0.60 + index / 20.0,
                        "psnr": 20.0 + index,
                        "ssim_score": 0.70 + index / 100.0,
                    },
                }
            )
        anchor = {"evaluation": {"operating_points": profiles}, "source": "alias_test"}

        updated = adaptive.apply_neural_anchor(quality_rows(), anchor, "selection")

        self.assertEqual([row["feature_ratio_mean"] for row in updated], [0.01, 0.02, 0.03, 0.04, 0.05])
        self.assertTrue(all(row["neural_payload_ratio_basis"] == "zlib" for row in updated))
        self.assertEqual(updated[-1]["mean_iou_mean"], 0.85)

    def test_single_rate_fallback_keeps_legacy_paper_like_behavior(self) -> None:
        anchor = {
            "source": "legacy_alias_test",
            "payload_ratio_raw": 0.104,
            "semantic_quality": 0.77,
        }

        updated = adaptive.apply_neural_anchor(quality_rows(), anchor, "selection")
        by_mode = {str(row["mode"]): row for row in updated}

        self.assertEqual(by_mode["paper_like"]["feature_ratio_mean"], 0.104)
        self.assertEqual(by_mode["paper_like"]["mean_iou_mean"], 0.77)
        self.assertEqual(by_mode["emergency"]["feature_ratio_mean"], 0.01)
        self.assertEqual(by_mode["emergency"]["mean_iou_mean"], 0.60)

        record_only = adaptive.apply_neural_anchor(quality_rows(), anchor, "record_only")
        record_by_mode = {str(row["mode"]): row for row in record_only}
        self.assertEqual(record_by_mode["paper_like"]["feature_ratio_mean"], 0.104)
        self.assertEqual(record_by_mode["paper_like"]["mean_iou_mean"], 0.63)
        self.assertEqual(record_by_mode["paper_like"]["neural_encoder_decoder_miou"], 0.77)

    def test_invalid_multi_rate_profile_fails_with_profile_location(self) -> None:
        anchor = enhanced_summary()
        anchor["multi_rate_profiles"] = [dict(profile) for profile in enhanced_profiles()]
        del anchor["multi_rate_profiles"][2]["mean_iou"]

        with self.assertRaisesRegex(ValueError, r"multi_rate_profiles\[2\].*mean_iou"):
            adaptive.apply_neural_anchor(quality_rows(), anchor, "selection")


class NeuralBitrateTests(unittest.TestCase):
    def test_measured_bitrates_replace_paper_params(self) -> None:
        paper, details = adaptive.apply_neural_bitrates(reproduction.PaperParams(), enhanced_summary())

        self.assertEqual(paper.enc_bitrate, 200.0e6)
        self.assertEqual(paper.dec_bitrate, 30.0e6)
        self.assertIs(details["applied"], True)

    def test_incomplete_bitrate_pair_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing measured decode"):
            adaptive.apply_neural_bitrates(
                reproduction.PaperParams(),
                {"encoder_bitrate_mbps": 123.0},
            )


class RepeatStatisticsTests(unittest.TestCase):
    def test_repeat_and_student_t_summary_csvs(self) -> None:
        rows = [
            {"mode": "adaptive_semantic", "area": 300, "policy": "Greedy", "repeat": 0, "finished": 10.0, "avg_time": 5.0, "sinr_median_db": math.nan},
            {"mode": "adaptive_semantic", "area": 300, "policy": "Greedy", "repeat": 1, "finished": 14.0, "avg_time": 3.0, "sinr_median_db": math.nan},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            repeat_path = adaptive.write_repeat_metrics_csv(rows, out_dir)
            stats_path = adaptive.write_statistical_summary_csv(rows, out_dir)
            with repeat_path.open(newline="", encoding="utf-8") as fh:
                repeat_rows = list(csv.DictReader(fh))
            with stats_path.open(newline="", encoding="utf-8") as fh:
                stats_rows = list(csv.DictReader(fh))

        self.assertEqual(len(repeat_rows), 2)
        self.assertEqual(len(stats_rows), 1)
        stats = stats_rows[0]
        self.assertEqual(float(stats["finished_mean"]), 12.0)
        self.assertAlmostEqual(float(stats["finished_std"]), math.sqrt(8.0))
        self.assertAlmostEqual(float(stats["finished_ci95_margin"]), 25.412, places=3)
        self.assertEqual(int(stats["sinr_median_db_n"]), 0)
        self.assertEqual(stats["sinr_median_db_mean"], "")


class RunnerIntegrityGateTests(unittest.TestCase):
    def test_failed_integrity_validation_raises_before_completion(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "integrity validation failed"):
            adaptive.require_integrity_validation(
                {
                    "passed": False,
                    "errors": [{"code": "missing_combinations"}],
                }
            )

    def test_successful_integrity_validation_is_accepted(self) -> None:
        adaptive.require_integrity_validation(
            {
                "passed": True,
                "adaptive_vs_fixed": {"all_comparisons_pass": False},
            }
        )


class AdaptiveCliSmokeTests(unittest.TestCase):
    def test_low_quality_custom_area_policy_cli_writes_provenance_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            metadata_path = root / "base_metadata.json"
            quality_path = root / "quality.csv"
            neural_path = root / "neural_summary.json"
            out_dir = root / "out"
            paper = reproduction.PaperParams(n_uav=2, n_device=2, repeats=1, t_slots=60)
            assumed = reproduction.AssumedParams(
                request_probability=1.0,
                device_diffusion=0.0,
                device_speed_cap=0.0,
                workload_mean_bits=1_000.0,
                workload_std_bits=0.0,
                workload_min_bits=1_000.0,
                workload_max_bits=1_000.0,
            )
            metadata_path.write_text(
                json.dumps({"paper_params": paper.__dict__, "assumed_params": assumed.__dict__}),
                encoding="utf-8",
            )
            fields = [
                "mode",
                "description",
                "feature_ratio_mean",
                "feature_ratio_median",
                "zlib_ratio_mean",
                "zlib_ratio_median",
                "mean_iou_mean",
                "mean_iou_median",
            ]
            with quality_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(quality_rows())
            neural_path.write_text(json.dumps(low_quality_summary()), encoding="utf-8")

            command = [
                sys.executable,
                str(ADAPTIVE_CODE / "run_full_adaptive_research.py"),
                "--metadata",
                str(metadata_path),
                "--quality",
                str(quality_path),
                "--neural-summary",
                str(neural_path),
                "--out",
                str(out_dir),
                "--repeats",
                "1",
                "--t-slots",
                "60",
                "--areas",
                "275",
                "--policies",
                "SA",
                "--workers",
                "2",
            ]
            completed = subprocess.run(
                command,
                cwd=WORKSPACE_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if completed.returncode != 0:
                self.fail(f"CLI failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")

            sequential_out = root / "out_sequential"
            sequential_command = list(command)
            sequential_command[sequential_command.index(str(out_dir))] = str(sequential_out)
            sequential_command[-1] = "1"
            sequential = subprocess.run(
                sequential_command,
                cwd=WORKSPACE_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if sequential.returncode != 0:
                self.fail(
                    f"Sequential CLI failed\nSTDOUT:\n{sequential.stdout}\nSTDERR:\n{sequential.stderr}"
                )

            with (out_dir / "repeat_metrics.csv").open(newline="", encoding="utf-8") as fh:
                repeat_rows = list(csv.DictReader(fh))
            with (sequential_out / "repeat_metrics.csv").open(newline="", encoding="utf-8") as fh:
                sequential_repeat_rows = list(csv.DictReader(fh))
            with (out_dir / "statistical_summary.csv").open(newline="", encoding="utf-8") as fh:
                stats_rows = list(csv.DictReader(fh))
            run_metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
            result_validation = json.loads((out_dir / "result_validation.json").read_text(encoding="utf-8"))
            generated_report = (out_dir / "adaptive_followup_research_report.md").read_text(encoding="utf-8")
            figure_names = {path.name for path in (out_dir / "figures").iterdir()}

        self.assertEqual(len(repeat_rows), 3)
        self.assertEqual(repeat_rows, sequential_repeat_rows)
        self.assertEqual(len(stats_rows), 3)
        self.assertEqual(run_metadata["workers"], 2)
        self.assertEqual(run_metadata["base_paper_params"]["enc_bitrate"], 200.0e6)
        self.assertEqual(run_metadata["base_paper_params"]["dec_bitrate"], 30.0e6)
        modes = run_metadata["profiles"]["adaptive_semantic"]["modes"]
        self.assertEqual([mode["rho_c"] for mode in modes], [0.02, 0.04, 0.07, 0.10, 0.15])
        self.assertEqual([mode["quality"] for mode in modes], [0.11, 0.16, 0.21, 0.26, 0.31])
        self.assertEqual(run_metadata["status"], "completed")
        self.assertTrue(run_metadata["started_at_utc"].endswith("Z"))
        self.assertTrue(run_metadata["completed_at_utc"].endswith("Z"))
        self.assertEqual(run_metadata["adaptive_threshold_configuration"]["resolved_rule"], "measured_ordered")
        self.assertEqual(run_metadata["adaptive_threshold_configuration"]["reachable_mode_count"], 5)
        self.assertEqual(run_metadata["runner_source"]["sha256"], run_metadata["runner_source"]["snapshot_sha256"])
        self.assertEqual(run_metadata["validator_source"]["sha256"], run_metadata["validator_source"]["snapshot_sha256"])
        self.assertEqual(run_metadata["validator_source"]["path"], str(Path(adaptive.result_validator.__file__).resolve()))
        self.assertEqual(len(run_metadata["input_files"]["metadata"]["sha256"]), 64)
        self.assertEqual(len(run_metadata["input_files"]["quality"]["sha256"]), 64)
        self.assertEqual(len(run_metadata["input_files"]["neural_summary"]["sha256"]), 64)
        self.assertIn("python_version", run_metadata["environment"])
        self.assertIn("numpy_version", run_metadata["environment"])
        self.assertTrue(run_metadata["command_windows"])
        self.assertIs(run_metadata["neural_encoder_decoder_anchor"]["nested_source_metadata_embedded"], False)
        self.assertEqual(result_validation["schema_version"], 2)
        self.assertEqual(result_validation["expected_combinations"]["areas"], [275])
        self.assertEqual(result_validation["expected_combinations"]["policies"], ["SA"])
        self.assertIn(str(metadata_path.resolve()), generated_report)
        self.assertIn("latency_quality_tradeoff_275m.png", figure_names)
        self.assertIn("adaptive_mode_usage_275m.png", figure_names)
        self.assertIn("validator_source_snapshot", run_metadata["artifacts"])
        for artifact in run_metadata["artifacts"].values():
            self.assertTrue(Path(artifact["path"]).is_absolute())
            self.assertEqual(len(artifact["sha256"]), 64)

    def test_quality_axis_includes_low_neural_miou(self) -> None:
        lower, upper = adaptive.quality_axis_limits([0.11, 0.21, 0.31])
        self.assertLess(lower, 0.11)
        self.assertGreater(upper, 0.31)
        self.assertLess(lower, 0.75)

    def test_cli_rejects_empty_or_nonpositive_dimensions_before_loading_inputs(self) -> None:
        invalid_options = [
            ("--areas", ","),
            ("--policies", ","),
            ("--repeats", "0"),
            ("--t-slots", "0"),
            ("--workers", "0"),
        ]
        for option, value in invalid_options:
            with self.subTest(option=option):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ADAPTIVE_CODE / "run_full_adaptive_research.py"),
                        option,
                        value,
                    ],
                    cwd=WORKSPACE_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
