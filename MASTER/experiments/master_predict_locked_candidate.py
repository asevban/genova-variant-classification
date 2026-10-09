"""Run inference with the locked MASTER candidate artefact.

Usage:
    python scripts/master_predict_locked_candidate.py --input path/to/test.csv

The input may be label-free.  Output contains Variant_ID, pathogenic
probability, binary prediction and the locked threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import ID_COLUMN, TARGET_COLUMN  # noqa: E402


DEFAULT_ARTEFACT = (
    ROOT
    / "results"
    / "modeling"
    / "locked_final_candidate"
    / "MASTER_M3_XGBoost_no_weight_locked.joblib"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "modeling"
    / "locked_final_candidate"
    / "locked_candidate_test_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw/canonical MASTER CSV to score.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Prediction CSV path.")
    parser.add_argument(
        "--artefact",
        default=str(DEFAULT_ARTEFACT),
        help="Locked candidate joblib artefact.",
    )
    return parser.parse_args()


def optional_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any] | None:
    if TARGET_COLUMN not in frame.columns:
        return None
    y_true = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def main() -> dict[str, Any]:
    args = parse_args()
    artefact = joblib.load(args.artefact)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path, low_memory=False)
    transformed = artefact["processor"].transform(frame)
    score = artefact["model"].predict_proba(transformed)[:, 1].astype(float)
    threshold = float(artefact["threshold"])
    prediction = (score >= threshold).astype(int)

    output = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].astype(str).to_numpy(),
            "score_pathogenic": score,
            "prediction": prediction,
            "threshold": threshold,
        }
    )
    output.to_csv(output_path, index=False)

    metadata = {
        "status": "completed",
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(output)),
        "threshold": threshold,
        "prediction_counts": {
            str(int(label)): int(count)
            for label, count in pd.Series(prediction).value_counts().sort_index().items()
        },
        "label_metrics_if_available": optional_metrics(frame, prediction),
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


if __name__ == "__main__":
    main()
