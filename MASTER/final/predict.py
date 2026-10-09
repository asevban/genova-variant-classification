"""Run inference with the frozen MASTER Hybrid-E candidate artefact."""

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

FINAL_PATHOGENIC_RATE = 500 / 3500


DEFAULT_ARTEFACT = (
    ROOT
    / "final"
    / "models"
    / "MASTER_Hybrid-E_bestMCC_nested_development_train_frozen.joblib"
)
DEFAULT_OUTPUT = (
    ROOT
    / "output"
    / "MASTER_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw/canonical MASTER CSV to score.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Prediction CSV path.")
    parser.add_argument("--artefact", default=str(DEFAULT_ARTEFACT), help="Frozen artefact.")
    return parser.parse_args()


def _logit(values: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(values, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == 1:
        return np.ones(len(x), dtype=float) if int(model.classes_[0]) == 1 else np.zeros(len(x))
    classes = list(model.classes_)
    return proba[:, classes.index(1)].astype(float)


def prior_correct_probability(
    probability: np.ndarray,
    source_positive_rate: float,
    target_positive_rate: float = FINAL_PATHOGENIC_RATE,
) -> np.ndarray:
    shift = float(_logit(target_positive_rate) - _logit(source_positive_rate))
    return _sigmoid(_logit(probability) + shift)


def apply_artifact_scores(artefact: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    transformed = artefact["processor"].transform(frame).to_numpy(dtype=np.float32)
    raw_scores = {}
    for member in artefact["members"]:
        raw_scores[member] = positive_probability(artefact["base_models"][member], transformed)

    calibration = artefact["calibration"]
    calibrated_columns = []
    for member in artefact["members"]:
        score = np.clip(raw_scores[member], 1e-7, 1 - 1e-7)
        if calibration["method"] == "raw":
            calibrated = score
        elif calibration["method"] == "sigmoid":
            calibrator = calibration["calibrators"][member]
            calibrated = calibrator.predict_proba(_logit(score).reshape(-1, 1))[:, 1]
        elif calibration["method"] == "isotonic":
            calibrated = calibration["calibrators"][member].predict(score)
        else:
            raise ValueError(f"Unknown calibration method: {calibration['method']}")
        if calibration["apply_prior"]:
            calibrated = prior_correct_probability(
                calibrated,
                float(calibration["train_positive_rate"]),
            )
        calibrated_columns.append(np.clip(calibrated, 1e-7, 1 - 1e-7))

    matrix = np.column_stack(calibrated_columns)
    return matrix @ np.asarray(artefact["weights"], dtype=float)


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
    score = apply_artifact_scores(artefact, frame)
    threshold = float(artefact["threshold"])
    prediction = (score >= threshold).astype(int)

    output = pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].astype(str).to_numpy(),
            "score_pathogenic": score,
            "prediction": prediction,
            "threshold": threshold,
            "model": artefact["selection_strategy"],
        }
    )
    output.to_csv(output_path, index=False)

    metadata = {
        "status": "completed",
        "artefact": str(args.artefact),
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(output)),
        "threshold": threshold,
        "selection_strategy": artefact["selection_strategy"],
        "calibration_matrix": artefact["calibration_matrix"],
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
