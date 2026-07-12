from __future__ import annotations

"""Refresh the study-local compatibility DOCX files from verified final reports.

The canonical report sources live in ``reports/*.md`` and the strict finalizer
writes the evidence-backed Markdown/DOCX files to ``reports/final``.  The three
``studies/*/reports/*Easy_Final_KR.docx`` paths are kept for compatibility, but
they must not retain the old single-rate/proxy results. This helper therefore
mirrors the canonical final DOCX bytes instead of maintaining or rendering a
second copy of the research claims.
"""

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_REPORTS = ROOT / "reports" / "final"

REPORT_TARGETS = (
    (
        FINAL_REPORTS / "01_인코더_디코더_딥러닝_과정_final.md",
        ROOT
        / "studies"
        / "neural_encoder_decoder"
        / "reports"
        / "Neural_Encoder_Decoder_Easy_Final_KR.docx",
    ),
    (
        FINAL_REPORTS / "02_AirTalking_논문_실험_재현_final.md",
        ROOT
        / "studies"
        / "airtalking_reproduction"
        / "reports"
        / "AirTalking_Reproduction_Easy_Final_KR.docx",
    ),
    (
        FINAL_REPORTS / "03_후속연구_적응형_의미압축_final.md",
        ROOT
        / "studies"
        / "adaptive_semantic_compression"
        / "reports"
        / "Adaptive_Semantic_Compression_Easy_Final_KR.docx",
    ),
)


def refresh_compatibility_reports() -> tuple[list[Path], list[str]]:
    generated: list[Path] = []
    warnings: list[str] = []
    for markdown_path, docx_path in REPORT_TARGETS:
        if not markdown_path.is_file():
            raise FileNotFoundError(
                f"최신 최종 Markdown이 없습니다. strict finalizer를 먼저 실행하세요: {markdown_path}"
            )
        canonical_docx = markdown_path.with_suffix(".docx")
        if not canonical_docx.is_file():
            raise FileNotFoundError(
                f"최신 최종 DOCX가 없습니다. strict finalizer를 먼저 실행하세요: {canonical_docx}"
            )
        docx_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canonical_docx, docx_path)
        generated.append(docx_path)
    return generated, warnings


def main() -> None:
    generated, warnings = refresh_compatibility_reports()
    for path in generated:
        print(path)
    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
