from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
PANEL_NAMES = ("MASTER", "KANSER", "PAH", "CFTR")
ID_ALIASES = ("id", "ID", "Variant_ID", "variant_id")


def canonical_panel(value: object) -> str:
    text = str(value).strip().upper()
    aliases = {
        "MASTER": "MASTER",
        "KANSER": "KANSER",
        "CANCER": "KANSER",
        "PAH": "PAH",
        "CFTR": "CFTR",
    }
    if text not in aliases:
        raise ValueError(f"UNKNOWN PANEL: {value!r}. Beklenen: {', '.join(PANEL_NAMES)}")
    return aliases[text]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Desteklenmeyen input dosyasi: {path}")


def find_id_column(columns: Iterable[str]) -> str:
    available = set(columns)
    for alias in ID_ALIASES:
        if alias in available:
            return alias
    raise ValueError(f"ID kolonu bulunamadi. Desteklenen adlar: {', '.join(ID_ALIASES)}")


def normalize_frame(frame: pd.DataFrame, panel: str | None = None) -> pd.DataFrame:
    out = frame.copy()
    id_col = find_id_column(out.columns)
    if id_col != "Variant_ID":
        out["Variant_ID"] = out[id_col].astype(str)
    else:
        out["Variant_ID"] = out["Variant_ID"].astype(str)
    if out["Variant_ID"].isna().any() or out["Variant_ID"].str.strip().eq("").any():
        raise ValueError("Bos id/Variant_ID degeri var.")
    if panel is not None:
        out["panel"] = panel
    elif "panel" in out.columns:
        out["panel"] = out["panel"].map(canonical_panel)
    else:
        raise ValueError("Tek dosya input icin panel kolonu zorunludur.")
    return out


def discover_input_files(input_path: Path) -> Dict[str, Path]:
    files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in DATA_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"Input dosyasi bulunamadi: {input_path}")
    panel_files: Dict[str, Path] = {}
    for file in files:
        upper = file.stem.upper()
        for panel in PANEL_NAMES:
            if panel in upper:
                if panel in panel_files:
                    raise RuntimeError(f"{panel} icin birden fazla input dosyasi bulundu.")
                panel_files[panel] = file
    if panel_files:
        return panel_files
    if len(files) == 1:
        return {"__COMBINED__": files[0]}
    names = ", ".join(p.name for p in files)
    raise RuntimeError(f"Birden fazla input var ama panel adlari anlasilmiyor: {names}")


def load_inputs(input_arg: str | Path) -> Dict[str, pd.DataFrame]:
    path = Path(input_arg)
    if path.is_dir():
        discovered = discover_input_files(path)
        if "__COMBINED__" in discovered:
            combined = normalize_frame(read_table(discovered["__COMBINED__"]))
        else:
            frames = [
                normalize_frame(read_table(file), panel=panel)
                for panel, file in discovered.items()
            ]
            combined = pd.concat(frames, ignore_index=True, sort=False)
    elif path.is_file():
        combined = normalize_frame(read_table(path))
    else:
        raise FileNotFoundError(f"Input yolu bulunamadi: {path}")

    if combined["Variant_ID"].duplicated().any():
        dupes = combined.loc[combined["Variant_ID"].duplicated(), "Variant_ID"].head(10).tolist()
        raise ValueError(f"Duplicate id bulundu: {dupes}")
    return {panel: part.copy() for panel, part in combined.groupby("panel", sort=False)}
