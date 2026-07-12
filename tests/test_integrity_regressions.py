from __future__ import annotations

import csv
import hashlib
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
import validate_full_adaptive_results as validation  # noqa: E402


class SemanticSummaryTests(unittest.TestCase):
    def test_feature_profile_uses_feature_encoder_bitrate(self) -> None:
        summary = {
            "rho_c_feature_uncompressed_mean": 0.125,
            "rho_c_feature_png_mean": 0.25,
            "rho_r_proxy": 3.0,
            "encode_bitrate_mbps_median": 11.0,
            "decode_bitrate_mbps_median": 12.0,
            "feature_encode_bitrate_mbps_median": 21.0,
            "feature_decode_bitrate_mbps_median": 22.0,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            paper, metadata = reproduction.apply_semantic_summary(
                reproduction.PaperParams(),
                summary_path,
                raw_basis="uncompressed",
                encoder_mode="measured",
                decoder_mode="measured",
                profile_kind="feature",
            )

        self.assertEqual(paper.enc_bitrate, 21.0e6)
        self.assertEqual(paper.dec_bitrate, 22.0e6)
        self.assertEqual(metadata["enc_bitrate"], 21.0e6)


class MetadataJsonTests(unittest.TestCase):
    def test_profile_metadata_is_strict_json_and_round_trips_unbounded_bound(self) -> None:
        profile = reproduction.SemanticProfile(
            name="adaptive_test",
            strategy="adaptive",
            modes=(reproduction.SemanticCompressionMode("mode", 0.1, 0.95),),
        )
        profile_metadata = adaptive.profile_to_metadata(profile)

        self.assertIsNone(profile_metadata["target_thresholds"][-1][0])
        self.assertIs(profile_metadata["null_means_unbounded"], True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "run_metadata.json"
            adaptive.write_metadata_json(path, {"profile": profile_metadata})
            raw_json = path.read_text(encoding="utf-8")
            parsed = json.loads(
                raw_json,
                parse_constant=lambda token: self.fail(f"non-standard JSON token: {token}"),
            )

        self.assertNotIn("Infinity", raw_json)
        restored = adaptive.profile_from_metadata(parsed["profile"])
        self.assertTrue(math.isinf(restored.target_thresholds[-1][0]))
        self.assertGreater(restored.target_thresholds[-1][0], 0.0)

    def test_metadata_writer_rejects_unsanitized_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "run_metadata.json"
            with self.assertRaises(ValueError):
                adaptive.write_metadata_json(path, {"unexpected": math.inf})


class AdaptiveValidationTests(unittest.TestCase):
    @staticmethod
    def complete_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for mode in validation.MODES:
            for area in validation.AREAS:
                for policy in validation.POLICIES:
                    rows.append(
                        {
                            "mode": mode,
                            "area": area,
                            "policy": policy,
                            "finished": 10.0,
                            "avg_time": 5.0,
                            "semantic_quality": 0.95,
                        }
                    )
        return rows

    def test_missing_combination_returns_clear_failure_instead_of_key_error(self) -> None:
        rows = self.complete_rows()
        missing_key = ("adaptive_semantic", 100, "Stochastic")
        rows = [
            row
            for row in rows
            if (row["mode"], row["area"], row["policy"]) != missing_key
        ]

        result = validation.validate(rows)

        self.assertIs(result["passed"], False)
        self.assertIn(missing_key, result["missing_combinations"])
        self.assertIs(result["adaptive_vs_fixed"]["evaluated"], False)
        self.assertIn("1 required", result["adaptive_vs_fixed"]["reason"])

    def test_complete_finite_rows_pass_when_one_hypothesis_comparison_loses(self) -> None:
        rows = self.complete_rows()
        adaptive_row = next(
            row
            for row in rows
            if row["mode"] == "adaptive_semantic"
            and row["area"] == 100
            and row["policy"] == "Stochastic"
        )
        adaptive_row["finished"] = 9.0

        result = validation.validate(rows)
        failed_comparisons = [
            item
            for item in result["adaptive_vs_fixed"]["comparisons"]
            if item["pass"] is False
        ]

        self.assertEqual(len(rows), 75)
        self.assertTrue(
            all(
                math.isfinite(float(row[column]))
                for row in rows
                for column in ("area", *validation.METRIC_COLUMNS)
            )
        )
        self.assertEqual(result["row_count"], 75)
        self.assertEqual(result["expected_row_count"], 75)
        self.assertEqual(result["errors"], [])
        self.assertIs(result["passed"], True)
        self.assertIs(result["adaptive_vs_fixed"]["evaluated"], True)
        self.assertIs(result["adaptive_vs_fixed"]["all_comparisons_pass"], False)
        self.assertEqual(
            [(item["area"], item["policy"]) for item in failed_comparisons],
            [(100, "Stochastic")],
        )

    def test_numeric_canonical_duplicate_is_structured(self) -> None:
        rows = self.complete_rows()
        duplicate = dict(rows[0])
        duplicate["area"] = "100.0"
        rows.append(duplicate)

        result = validation.validate(rows)

        self.assertIs(result["passed"], False)
        self.assertEqual(len(result["duplicates"]), 1)
        self.assertEqual(result["duplicates"][0]["rows"], [1, len(rows)])
        self.assertIn("duplicate_combinations", {error["code"] for error in result["errors"]})

    def test_nonfinite_and_missing_values_never_escape_strict_json(self) -> None:
        rows = self.complete_rows()
        rows[0]["semantic_quality"] = math.nan
        del rows[1]["finished"]
        rows[2]["flight_energy_per_req"] = math.inf

        result = validation.validate(rows)
        encoded = json.dumps(result, allow_nan=False)

        self.assertIs(result["passed"], False)
        self.assertTrue(result["nonfinite_values"])
        self.assertIn("flight_energy_per_req", {item["column"] for item in result["nonfinite_values"]})
        self.assertTrue(result["missing_values"])
        self.assertNotIn("NaN", encoded)

    def test_zero_denominator_is_structured_instead_of_dividing(self) -> None:
        rows = self.complete_rows()
        fixed = next(
            row
            for row in rows
            if row["mode"] == "fixed_paper_like" and row["area"] == 100 and row["policy"] == "Greedy"
        )
        fixed["finished"] = 0.0

        result = validation.validate(rows)

        self.assertIs(result["passed"], False)
        self.assertTrue(result["zero_denominators"])
        self.assertIs(result["adaptive_vs_fixed"]["evaluated"], False)
        self.assertIn("zero_denominator", {error["code"] for error in result["errors"]})

    def test_validator_cli_honors_summary_out_and_records_source_hash(self) -> None:
        rows = self.complete_rows()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = root / "custom_summary.csv"
            output = root / "nested" / "custom_validation.json"
            with summary.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(validation.REQUIRED_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
            expected_hash = hashlib.sha256(summary.read_bytes()).hexdigest()
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ADAPTIVE_CODE / "validate_full_adaptive_results.py"),
                    "--summary",
                    str(summary),
                    "--out",
                    str(output),
                ],
                cwd=WORKSPACE_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if completed.returncode != 0:
                self.fail(f"validator CLI failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["source_summary"]["path"], str(summary.resolve()))
        self.assertEqual(result["source_summary"]["sha256"], expected_hash)
        self.assertEqual(result["row_count"], len(rows))
        self.assertEqual(result["expected_combinations"]["count"], len(rows))
        self.assertIs(result["adaptive_vs_fixed"]["all_comparisons_pass"], True)

    def test_validator_cli_matches_explicit_in_process_bytes_in_first_seen_order(self) -> None:
        areas = (300, 100)
        policies = ("MCTS", "Stochastic")
        rows = [
            {
                "mode": mode,
                "area": area,
                "policy": policy,
                "finished": 10.0,
                "avg_time": 5.0,
                "semantic_quality": 0.95,
            }
            for mode in validation.MODES
            for area in areas
            for policy in policies
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = root / "custom_order_bom.csv"
            in_process_output = root / "in_process_validation.json"
            cli_output = root / "cli_validation.json"
            with summary.open("w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(validation.REQUIRED_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)

            loaded_rows, columns, load_errors = validation.load_rows_with_schema(summary)
            in_process_result = validation.validate(
                loaded_rows,
                available_columns=columns,
                pre_errors=load_errors,
                source_summary_path=summary,
                expected_areas=areas,
                expected_policies=policies,
            )
            validation.write_result(in_process_output, in_process_result)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ADAPTIVE_CODE / "validate_full_adaptive_results.py"),
                    "--summary",
                    str(summary),
                    "--out",
                    str(cli_output),
                ],
                cwd=WORKSPACE_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            cli_result = json.loads(cli_output.read_text(encoding="utf-8"))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(cli_output.read_bytes(), in_process_output.read_bytes())

        self.assertEqual(cli_result["expected_combinations"]["areas"], list(areas))
        self.assertEqual(cli_result["expected_combinations"]["policies"], list(policies))
        self.assertEqual(cli_result["schema"]["required_columns"][0], "mode")

    def test_validator_cli_missing_column_writes_failure_to_requested_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = root / "missing.csv"
            output = root / "validation.json"
            summary.write_text("mode,area,policy,finished,avg_time\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ADAPTIVE_CODE / "validate_full_adaptive_results.py"),
                    "--summary",
                    str(summary),
                    "--out",
                    str(output),
                ],
                cwd=WORKSPACE_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 1)
        self.assertIs(result["passed"], False)
        self.assertIn("missing_required_columns", {error["code"] for error in result["errors"]})


if __name__ == "__main__":
    unittest.main()
