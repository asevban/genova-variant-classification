"""Teslim öncesi makine-okunur paket doğrulaması ve envanter üretir."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from run_checks import check_data, check_fold_fit, check_splits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    check_data()
    check_fold_fit()
    check_splits()

    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    split_summary = json.loads(
        (ROOT / "data/splits/kanser/validation_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (ROOT / "reports/tables/schema_validation.json").read_text(encoding="utf-8")
    )
    build = json.loads(
        (ROOT / "data/processed/kanser/DATASET_BUILD_SUMMARY.json").read_text(encoding="utf-8")
    )
    required_docs = [
        "README.md",
        "VERI_ON_ISLEME_OZETI_KANSER.md",
        "PROJE_DOSYA_YAPISI_KANSER.md",
        "reports/01_EDA_RAPORU_KANSER.md",
        "reports/02_ON_ISLEME_KARARLARI_KANSER.md",
        "reports/03_PREPROCESSING_VARYANTLARI_KANSER.md",
        "reports/04_MISSINGNESS_PROVENANCE_KONTROLU_KANSER.md",
        "reports/05_TAKIM_PAYLASIMI_VE_UYARLANABILIRLIK_KANSER.md",
        "reports/06_THRESHOLD_NOTU_KANSER.md",
    ]
    missing_docs = [name for name in required_docs if not (ROOT / name).exists()]
    if missing_docs:
        raise AssertionError(f"eksik belgeler: {missing_docs}")
    if not split_summary["ok"] or split_summary["group_split_violations"] != 0:
        raise AssertionError("split doğrulaması başarısız")
    if not schema["ok"]:
        raise AssertionError("şema doğrulaması başarısız")

    validation = {
        "validation_date": date.today().isoformat(),
        "status": "PASS",
        "checks": {
            "schema": "PASS",
            "processed_dataset_shapes": "PASS",
            "identifier_and_target_exclusion": "PASS",
            "numeric_infinity_scan": "PASS",
            "fold_fit_transform_alignment": "PASS",
            "source_resistant_missingness_exclusion": "PASS",
            "outer_split_partition": "PASS",
            "duplicate_group_integrity": "PASS",
            "python_ast_parse": "PASS",
            "required_documentation": "PASS",
        },
        "rows": build["rows"],
        "raw_columns": build["raw_columns"],
        "source_features": build["source_features"],
        "dataset_versions": len(build["versions"]),
        "outer_split_files": 10,
        "inner_split_files": 50,
        "group_split_violations": split_summary["group_split_violations"],
        "raw_sha256": build["source_sha256"],
        "python_files_parsed": len(python_files),
    }
    (ROOT / "PACKAGE_VALIDATION.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    excluded_names = {"PACKAGE_MANIFEST.json", "CHECKSUMS_SHA256.txt"}
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.name in excluded_names or "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        files.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "package": "GENOVA_KANSER_VERI_ON_ISLEME",
        "manifest_version": "1.0",
        "manifest_note": "Manifest kendi dosyasını ve sonradan üretilen CHECKSUMS_SHA256.txt dosyasını listelemez.",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    (ROOT / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
