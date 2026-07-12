from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "studies" / "airtalking_reproduction" / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import verify_against_paper as verifier  # noqa: E402


class VerificationProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (
            ROOT
            / "studies"
            / "neural_encoder_decoder"
            / "results"
            / "airtalking_neural_encoder_decoder_timed"
            / "summary_metrics.csv"
        )

    def test_companion_json_binds_verdicts_to_summary_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary_metrics.csv"
            shutil.copyfile(self.source, summary)
            actual = verifier.load_actual(summary)
            rows = verifier.compare_rows(actual)
            qualitative = verifier.qualitative_checks(actual)
            verifier.write_outputs(rows, qualitative, root, "bound", summary)
            companion = json.loads(
                (root / "verification_against_paper_bound.json").read_text(encoding="utf-8")
            )
            expected_hash = hashlib.sha256(summary.read_bytes()).hexdigest()
            self.assertEqual(companion["schema_version"], 2)
            self.assertEqual(companion["source_summary"]["sha256"], expected_hash)
            self.assertEqual(companion["row_count"], len(rows))
            self.assertEqual(sum(companion["verdict_counts"].values()), len(rows))
            self.assertTrue(Path(companion["source_summary"]["path"]).is_file())
            self.assertEqual(set(companion["artifacts"]), {"verification_csv"})
            self.assertTrue((root / "verification_against_paper_bound.csv").is_file())
            self.assertFalse((root / "verification_against_paper_bound.md").exists())

    def test_duplicate_and_nonfinite_summary_rows_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.csv"
            with self.source.open(newline="", encoding="utf-8") as source_handle:
                rows = list(csv.DictReader(source_handle))
            with duplicate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows([*rows, rows[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                verifier.load_actual(duplicate)

            nonfinite = root / "nonfinite.csv"
            rows[0]["finished"] = "NaN"
            with nonfinite.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                verifier.load_actual(nonfinite)


if __name__ == "__main__":
    unittest.main()
