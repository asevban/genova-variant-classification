"""Package-root predictor for either V2 or V3 model artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = pd.read_csv(args.input, low_memory=False)
    if "Variant_ID" not in raw.columns:
        raise ValueError("KANSER input CSV zorunlu Variant_ID kolonunu icermiyor.")
    if not raw["Variant_ID"].is_unique:
        raise ValueError("KANSER input CSV Variant_ID degerleri benzersiz degil.")
    model = joblib.load(args.model)
    score = np.asarray(model.predict_score(raw), dtype=float)
    pred = np.asarray(model.predict(raw), dtype=int)
    if not np.isfinite(score).all() or len(pred) != len(raw):
        raise RuntimeError("Tahmin dogrulamasi basarisiz")
    out = pd.DataFrame({"row_index": np.arange(len(raw)), "pathogenic_score": score,
                        "prediction": pred,
                        "prediction_name": np.where(pred == 1, "PATHOGENIC", "BENIGN")})
    out.insert(0, "Variant_ID", raw["Variant_ID"].astype(str))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"OK: {len(out)} satir, esik={model.threshold:.12f}, patojenik={int(pred.sum())}")


if __name__ == "__main__":
    main()
