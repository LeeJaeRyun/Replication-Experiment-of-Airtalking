from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPRODUCTION_CODE = WORKSPACE_ROOT / "studies" / "airtalking_reproduction" / "code"
if str(REPRODUCTION_CODE) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_CODE))

import airtalking_reproduction as reproduction  # noqa: E402


class RepeatStatisticsTests(unittest.TestCase):
    def test_sample_std_student_t_ci_and_repeat_rows(self) -> None:
        rows = [
            {
                "mode": "semantic",
                "area": 300,
                "policy": "Greedy",
                "repeat": 0,
                "finished": 10.0,
                "avg_time": 5.0,
                "sinr_median_db": math.nan,
            },
            {
                "mode": "semantic",
                "area": 300,
                "policy": "Greedy",
                "repeat": 1,
                "finished": 14.0,
                "avg_time": 3.0,
                "sinr_median_db": math.nan,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            repeat_path = reproduction.write_repeat_metrics_csv(rows, out_dir)
            statistics_path = reproduction.write_statistical_summary_csv(rows, out_dir)
            with repeat_path.open(newline="", encoding="utf-8") as fh:
                repeat_rows = list(csv.DictReader(fh))
            with statistics_path.open(newline="", encoding="utf-8") as fh:
                statistics_rows = list(csv.DictReader(fh))

        self.assertEqual(len(repeat_rows), 2)
        self.assertEqual([int(row["repeat"]) for row in repeat_rows], [0, 1])
        self.assertEqual(len(statistics_rows), 1)
        statistics_row = statistics_rows[0]
        self.assertEqual(float(statistics_row["finished_mean"]), 12.0)
        self.assertAlmostEqual(float(statistics_row["finished_std"]), math.sqrt(8.0))
        self.assertAlmostEqual(float(statistics_row["finished_ci95_margin"]), 25.412, places=3)
        self.assertEqual(int(statistics_row["sinr_median_db_n"]), 0)
        self.assertEqual(statistics_row["sinr_median_db_mean"], "")

    def test_workers_one_and_two_are_deterministically_identical(self) -> None:
        paper = reproduction.PaperParams(n_uav=2, n_device=2, repeats=2, t_slots=30)
        assumed = reproduction.AssumedParams(
            request_probability=1.0,
            device_diffusion=0.0,
            device_speed_cap=0.0,
            workload_mean_bits=1_000.0,
            workload_std_bits=0.0,
            workload_min_bits=1_000.0,
            workload_max_bits=1_000.0,
            seed=1234,
        )
        sequential_rows: list[dict[str, object]] = []
        parallel_rows: list[dict[str, object]] = []
        with mock.patch.object(reproduction, "AREAS", (100,)):
            sequential = reproduction.run_experiments(
                paper,
                assumed,
                repeat_metrics=sequential_rows,
                workers=1,
            )
            parallel = reproduction.run_experiments(
                paper,
                assumed,
                repeat_metrics=parallel_rows,
                workers=2,
            )

        self.assertEqual(len(sequential_rows), len(parallel_rows))
        for sequential_row, parallel_row in zip(sequential_rows, parallel_rows):
            self.assertEqual(sequential_row.keys(), parallel_row.keys())
            for key in sequential_row:
                left = sequential_row[key]
                right = parallel_row[key]
                if isinstance(left, float) and math.isnan(left):
                    self.assertIsInstance(right, float)
                    self.assertTrue(math.isnan(right))
                else:
                    self.assertEqual(left, right)

        for mode in sequential:
            for area in sequential[mode]:
                for policy in sequential[mode][area]:
                    left = sequential[mode][area][policy]
                    right = parallel[mode][area][policy]
                    self.assertEqual(left.summary.keys(), right.summary.keys())
                    for key in left.summary:
                        if math.isnan(left.summary[key]):
                            self.assertTrue(math.isnan(right.summary[key]))
                        else:
                            self.assertEqual(left.summary[key], right.summary[key])

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            summary_path = reproduction.write_summary_csv(sequential, out_dir)
            statistics_path = reproduction.write_statistical_summary_csv(sequential_rows, out_dir)
            with summary_path.open(newline="", encoding="utf-8") as fh:
                summary_rows = list(csv.DictReader(fh))
            with statistics_path.open(newline="", encoding="utf-8") as fh:
                statistics_rows = list(csv.DictReader(fh))
        statistics_by_key = {
            (row["mode"], row["area"], row["policy"]): row
            for row in statistics_rows
        }
        for summary_row in summary_rows:
            key = (summary_row["mode"], summary_row["area"], summary_row["policy"])
            statistics_row = statistics_by_key[key]
            for metric in reproduction.REPEAT_METRIC_ORDER:
                if summary_row.get(metric) and statistics_row.get(f"{metric}_mean"):
                    self.assertEqual(
                        float(summary_row[metric]),
                        float(statistics_row[f"{metric}_mean"]),
                    )


class AssumedMetadataTests(unittest.TestCase):
    def test_metadata_is_base_and_explicit_assumed_value_overrides_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "run_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "assumed_params": {
                            **reproduction.AssumedParams().__dict__,
                            "request_probability": 0.123,
                            "seed": 17,
                            "future_unrelated_field": "ignored",
                        }
                    }
                ),
                encoding="utf-8",
            )
            loaded = reproduction.load_assumed_params_from_metadata(metadata_path)

        overridden = reproduction.apply_assumed_overrides(loaded, ["request_probability=0.456"])
        self.assertEqual(loaded.request_probability, 0.123)
        self.assertEqual(overridden.request_probability, 0.456)
        self.assertEqual(overridden.seed, 17)


class EnhancedSummaryContractTests(unittest.TestCase):
    def test_measured_enhanced_profile_is_feature_uncompressed_only(self) -> None:
        enhanced_summary = {
            "schema_version": 2,
            "rho_c_feature_uncompressed_mean": 0.1,
            "feature_encode_bitrate_mbps_median": 200.0,
            "feature_decode_bitrate_mbps_median": 30.0,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "enhanced_summary.json"
            summary_path.write_text(json.dumps(enhanced_summary), encoding="utf-8")
            paper, _ = reproduction.apply_semantic_summary(
                reproduction.PaperParams(),
                summary_path,
                raw_basis="uncompressed",
                encoder_mode="measured",
                decoder_mode="measured",
                profile_kind="feature",
            )
            self.assertEqual(paper.enc_bitrate, 200.0e6)
            self.assertEqual(paper.dec_bitrate, 30.0e6)

            for raw_basis, profile_kind in (("png", "feature"), ("uncompressed", "zlib")):
                with self.subTest(raw_basis=raw_basis, profile_kind=profile_kind):
                    with self.assertRaisesRegex(ValueError, "intentionally not fabricated"):
                        reproduction.apply_semantic_summary(
                            reproduction.PaperParams(),
                            summary_path,
                            raw_basis=raw_basis,
                            encoder_mode="measured",
                            decoder_mode="measured",
                            profile_kind=profile_kind,
                        )


class ReproductionCliSmokeTests(unittest.TestCase):
    def test_two_repeat_short_slot_cli_writes_statistics_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            assumed_metadata_path = root / "base_run_metadata.json"
            out_dir = root / "out"
            assumed_metadata_path.write_text(
                json.dumps(
                    {
                        "assumed_params": {
                            **reproduction.AssumedParams().__dict__,
                            "request_probability": 0.0,
                            "device_diffusion": 0.0,
                            "seed": 91,
                        }
                    }
                ),
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(REPRODUCTION_CODE / "airtalking_reproduction.py"),
                "--out",
                str(out_dir),
                "--repeats",
                "2",
                "--t-slots",
                "2",
                "--workers",
                "2",
                "--assumed-metadata",
                str(assumed_metadata_path),
                "--assumed",
                "seed=92",
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
                self.fail(f"CLI failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")

            with (out_dir / "summary_metrics.csv").open(newline="", encoding="utf-8") as fh:
                summary_rows = list(csv.DictReader(fh))
            with (out_dir / "repeat_metrics.csv").open(newline="", encoding="utf-8") as fh:
                repeat_rows = list(csv.DictReader(fh))
            with (out_dir / "statistical_summary.csv").open(newline="", encoding="utf-8") as fh:
                statistics_rows = list(csv.DictReader(fh))
            metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
            launch = json.loads((out_dir / "launch_manifest.json").read_text(encoding="utf-8"))
            source_snapshot_exists = Path(metadata["run_provenance"]["source"]["snapshot"]).is_file()
            assumed_metadata_hash = reproduction.sha256_file(assumed_metadata_path)

        self.assertEqual(len(summary_rows), 29)
        self.assertEqual(len(repeat_rows), 58)
        self.assertEqual(len(statistics_rows), 29)
        statistics_by_key = {
            (row["mode"], row["area"], row["policy"]): row
            for row in statistics_rows
        }
        for summary_row in summary_rows:
            key = (summary_row["mode"], summary_row["area"], summary_row["policy"])
            statistics_row = statistics_by_key[key]
            self.assertEqual(int(statistics_row["repeat_count"]), 2)
            self.assertEqual(float(statistics_row["finished_mean"]), float(summary_row["finished"]))
            self.assertEqual(float(statistics_row["avg_time_mean"]), float(summary_row["avg_time"]))

        self.assertEqual(metadata["workers"], 2)
        self.assertEqual(metadata["assumed_params"]["seed"], 92)
        self.assertEqual(metadata["input_paths"]["assumed_metadata"], str(assumed_metadata_path))
        self.assertEqual(metadata["artifacts"]["repeat_metrics_csv"], str(out_dir / "repeat_metrics.csv"))
        self.assertEqual(
            metadata["repeat_statistics"]["statistical_summary_csv"],
            str(out_dir / "statistical_summary.csv"),
        )
        self.assertEqual(metadata["assumed_params_provenance"]["cli_overrides"], ["seed=92"])
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue(metadata["run_provenance"]["command_windows"])
        self.assertTrue(metadata["run_provenance"]["source"]["sha256"])
        self.assertTrue(source_snapshot_exists)
        self.assertEqual(
            metadata["run_provenance"]["input_sha256"]["assumed_metadata"],
            assumed_metadata_hash,
        )
        self.assertEqual(launch["status"], "completed")


if __name__ == "__main__":
    unittest.main()
