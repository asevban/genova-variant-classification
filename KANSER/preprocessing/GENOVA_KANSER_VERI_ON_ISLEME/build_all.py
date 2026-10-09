"""KANSER veri ön işleme paketinin bütün üretim adımlarını çalıştırır."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from genova.kanser.dataset_versions import build_all as build_datasets
from genova.kanser.diagnostics import build_all as build_diagnostics
from genova.kanser.split_bank import build_all as build_splits


def main() -> None:
    build_datasets()
    build_splits()
    build_diagnostics()
    print("KANSER paketi başarıyla yeniden üretildi.")


if __name__ == "__main__":
    main()
