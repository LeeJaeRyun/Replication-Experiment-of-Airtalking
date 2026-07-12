from __future__ import annotations

"""Atomically rebuild every reader-facing DOCX from verified final Markdown.

This is a presentation-only refresh. It does not recompute evidence or change
the canonical Markdown. Run the strict finalizer first whenever experiment
artifacts or claims have changed.
"""

import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:  # Package import used by tests and other tools.
    from tools.build_easy_study_reports import REPORT_TARGETS
    from tools.finalize_research_reports import ROOT, _sha256, markdown_to_docx
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    from build_easy_study_reports import REPORT_TARGETS
    from finalize_research_reports import ROOT, _sha256, markdown_to_docx


FINAL_DIR = ROOT / "reports" / "final"
FINALIZATION_MANIFEST = FINAL_DIR / "finalization_manifest.json"
COMPARISON_MANIFEST = FINAL_DIR / "04_이전_현재_컴퓨터_실험_결과_비교_manifest.json"
CANONICAL_REPORTS = tuple(
    FINAL_DIR / name
    for name in (
        "01_인코더_디코더_딥러닝_과정_final.md",
        "02_AirTalking_논문_실험_재현_final.md",
        "03_후속연구_적응형_의미압축_final.md",
        "04_이전_현재_컴퓨터_실험_결과_비교_final.md",
    )
)


def _build_time() -> datetime:
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        return datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc)
    return datetime.now(timezone.utc).replace(microsecond=0)


def _artifact(staged: Path, destination: Path) -> dict[str, Any]:
    return {
        "path": str(destination),
        "sha256": _sha256(staged),
        "size_bytes": staged.stat().st_size,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _finalization_manifest_payload(
    staged_pairs: Sequence[tuple[Path, Path]],
    generated_at: datetime,
) -> dict[str, Any]:
    manifest = json.loads(FINALIZATION_MANIFEST.read_text(encoding="utf-8"))
    first_three = staged_pairs[:3]
    manifest["docx"] = [str(destination) for _, destination in first_three]
    manifest.setdefault("artifacts", {})["docx"] = [
        _artifact(staged, destination) for staged, destination in first_three
    ]

    finalizer_source = ROOT / "tools" / "finalize_research_reports.py"
    renderer_source = ROOT / "tools" / "docx_report_design.py"
    # Keep ``finalizer_source`` untouched: it belongs to the canonical Markdown
    # generation. These two fields describe this presentation-only rebuild.
    manifest["docx_converter_source"] = {
        "path": str(finalizer_source),
        "sha256": _sha256(finalizer_source),
        "function": "markdown_to_docx",
    }
    manifest["docx_renderer_source"] = {
        "path": str(renderer_source),
        "sha256": _sha256(renderer_source),
        "function": "build_docx",
    }
    manifest["docx_generated_at_utc"] = generated_at.isoformat()
    return manifest


def _comparison_manifest_payload(
    staged: Path,
    destination: Path,
    generated_at: datetime,
) -> dict[str, Any]:
    manifest = json.loads(COMPARISON_MANIFEST.read_text(encoding="utf-8"))
    finalizer_source = ROOT / "tools" / "finalize_research_reports.py"
    renderer_source = ROOT / "tools" / "docx_report_design.py"
    manifest["converter_source"] = {
        "path": "tools/finalize_research_reports.py",
        "sha256": _sha256(finalizer_source),
        "function": "markdown_to_docx",
    }
    manifest["docx_renderer_source"] = {
        "path": "tools/docx_report_design.py",
        "sha256": _sha256(renderer_source),
        "function": "build_docx",
    }
    manifest["docx_generated_at_utc"] = generated_at.isoformat()
    manifest.setdefault("output_artifacts", {})["docx"] = {
        "path": "reports/final/04_이전_현재_컴퓨터_실험_결과_비교_final.docx",
        "sha256": _sha256(staged),
        "size_bytes": staged.stat().st_size,
    }
    return manifest


def _validate_docx(path: Path) -> None:
    from docx import Document

    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"DOCX ZIP CRC 실패: {path}: {corrupt}")
    document = Document(path)
    if not document.paragraphs or not document.core_properties.title:
        raise RuntimeError(f"DOCX 구조/제목 검증 실패: {path}")


def _commit_with_rollback(
    staged_pairs: Sequence[tuple[Path, Path]],
    backup_dir: Path,
) -> None:
    backups: dict[Path, Path | None] = {}
    for index, (_, destination) in enumerate(staged_pairs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            backup = backup_dir / f"{index:02d}-{destination.name}"
            shutil.copy2(destination, backup)
            backups[destination] = backup
        else:
            backups[destination] = None

    committed: list[Path] = []
    try:
        for staged, destination in staged_pairs:
            # Copy into the destination inode so Windows/OneDrive keeps the
            # destination directory ACL. Moving a tempfile can carry its
            # restrictive temporary ACL and make the report unreadable.
            shutil.copyfile(staged, destination)
            committed.append(destination)
    except Exception:
        for destination in reversed(committed):
            backup = backups[destination]
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                shutil.copyfile(backup, destination)
        raise


def refresh_all_docx_reports() -> tuple[list[Path], list[str]]:
    missing = [
        path
        for path in (*CANONICAL_REPORTS, FINALIZATION_MANIFEST, COMPARISON_MANIFEST)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "최종 Markdown/manifest가 없습니다. strict finalizer를 먼저 실행하세요: "
            + ", ".join(str(path) for path in missing)
        )

    cache_dir = ROOT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    generated_at = _build_time()
    warnings: list[str] = []

    with tempfile.TemporaryDirectory(prefix="docx-refresh-", dir=cache_dir) as temporary:
        staging = Path(temporary)
        backups = staging / "backups"
        backups.mkdir()

        canonical_pairs: list[tuple[Path, Path]] = []
        markdown_to_staged: dict[Path, Path] = {}
        for index, markdown_path in enumerate(CANONICAL_REPORTS, start=1):
            staged = staging / f"canonical-{index:02d}.docx"
            destination = markdown_path.with_suffix(".docx")
            markdown_to_docx(
                markdown_path,
                staged,
                warnings,
                generated_at=generated_at,
            )
            _validate_docx(staged)
            canonical_pairs.append((staged, destination))
            markdown_to_staged[markdown_path.resolve()] = staged

        if warnings:
            raise RuntimeError(
                "DOCX 생성 경고가 있어 기존 파일을 보존했습니다: " + " | ".join(warnings)
            )

        compatibility_pairs: list[tuple[Path, Path]] = []
        for index, (markdown_path, destination) in enumerate(REPORT_TARGETS, start=1):
            source = markdown_to_staged[markdown_path.resolve()]
            staged = staging / f"compatibility-{index:02d}.docx"
            shutil.copyfile(source, staged)
            _validate_docx(staged)
            compatibility_pairs.append((staged, destination))

        finalization_staged = staging / FINALIZATION_MANIFEST.name
        comparison_staged = staging / COMPARISON_MANIFEST.name
        _write_json(
            finalization_staged,
            _finalization_manifest_payload(canonical_pairs, generated_at),
        )
        _write_json(
            comparison_staged,
            _comparison_manifest_payload(*canonical_pairs[3], generated_at),
        )

        all_pairs = [
            *canonical_pairs,
            *compatibility_pairs,
            (finalization_staged, FINALIZATION_MANIFEST),
            (comparison_staged, COMPARISON_MANIFEST),
        ]
        _commit_with_rollback(all_pairs, backups)

    generated = [destination for _, destination in (*canonical_pairs, *compatibility_pairs)]
    return generated, warnings


def main() -> None:
    generated, warnings = refresh_all_docx_reports()
    for path in generated:
        print(path)
    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
