from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = WORKSPACE_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import audit_cityscapes_dataset as audit  # noqa: E402


class CityscapesDatasetAuditTests(unittest.TestCase):
    @staticmethod
    def _write_sample(root: Path, split: str, city: str, frame: str, labels: list[int]) -> None:
        stem = f"{city}_{frame}_000019"
        left_dir = root / "leftImg8bit_trainvaltest" / "leftImg8bit" / split / city
        gt_dir = root / "gtFine_trainvaltest" / "gtFine" / split / city
        left_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        label_array = np.asarray(labels, dtype=np.uint8).reshape(2, 4)
        Image.new("RGB", (4, 2), (10, 20, 30)).save(
            left_dir / f"{stem}_leftImg8bit.png"
        )
        Image.new("RGBA", (4, 2), (1, 2, 3, 255)).save(
            gt_dir / f"{stem}_gtFine_color.png"
        )
        Image.fromarray(label_array.astype(np.uint16)).save(
            gt_dir / f"{stem}_gtFine_instanceIds.png"
        )
        Image.fromarray(label_array, mode="L").save(gt_dir / f"{stem}_gtFine_labelIds.png")
        (gt_dir / f"{stem}_gtFine_polygons.json").write_text(
            json.dumps(
                {
                    "imgHeight": 2,
                    "imgWidth": 4,
                    "objects": [{"label": "road", "polygon": [[0, 0], [3, 0], [3, 1]]}],
                }
            ),
            encoding="utf-8",
        )

    def _make_dataset(self, parent: Path) -> Path:
        root = parent / "dataset"
        self._write_sample(root, "train", "traincity", "000000", [7, 7, 8, 11, 0, 1, 2, 3])
        self._write_sample(root, "val", "valcity", "000001", [24, 24, 26, 33, 0, 1, 2, 3])
        self._write_sample(root, "test", "testcity", "000002", [0, 1, 2, 3, 0, 1, 2, 3])
        return root

    def test_complete_audit_counts_histogram_test_semantics_and_stable_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            dataset = self._make_dataset(parent)
            first = audit.audit_dataset(dataset, parent / "out", quiet=True)
            second = audit.audit_dataset(dataset, parent / "out2", quiet=True)
            paths = audit.write_outputs(first, parent / "out")

            self.assertTrue(first["status"]["strict_pass"])
            self.assertEqual(first["splits"]["train"]["counts"]["rgb"], 1)
            self.assertTrue(
                first["splits"]["train"]["one_to_one"][
                    "all_rgb_have_exactly_one_of_each_gt_type"
                ]
            )
            self.assertFalse(first["leakage"]["has_stem_leakage"])
            self.assertFalse(first["leakage"]["has_city_leakage"])
            self.assertEqual(
                first["splits"]["train"]["semantic_label_pixels"]["train_id_histogram"]["0"],
                2,
            )
            self.assertEqual(first["train_val_19_class_histogram"]["ignored_pixels"], 8)
            self.assertFalse(first["test_ground_truth_semantics"]["has_19_class_evaluation_pixels"])
            self.assertEqual(first["fingerprint"]["digest"], second["fingerprint"]["digest"])
            self.assertEqual(first["fingerprint"]["content_hashed_file_count"], 6)

            raw_manifest = paths["manifest"].read_text(encoding="utf-8")
            parsed = json.loads(
                raw_manifest,
                parse_constant=lambda token: self.fail(f"non-standard JSON token: {token}"),
            )
            self.assertEqual(parsed["fingerprint"]["digest"], first["fingerprint"]["digest"])
            csv_lines = paths["histogram"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(csv_lines), 1 + 3 * 20)

    def test_missing_gt_is_a_strict_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            dataset = self._make_dataset(parent)
            polygon = next(
                (dataset / "gtFine_trainvaltest" / "gtFine" / "val").rglob(
                    "*_gtFine_polygons.json"
                )
            )
            polygon.unlink()

            manifest = audit.audit_dataset(dataset, parent / "out", quiet=True)

            self.assertFalse(manifest["status"]["strict_pass"])
            self.assertFalse(
                manifest["splits"]["val"]["one_to_one"][
                    "all_rgb_have_exactly_one_of_each_gt_type"
                ]
            )
            codes = {error["code"] for error in manifest["status"]["errors"]}
            self.assertIn("one_to_one_correspondence_failed", codes)


if __name__ == "__main__":
    unittest.main()
