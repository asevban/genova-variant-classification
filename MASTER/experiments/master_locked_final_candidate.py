"""Train and evaluate the locked MASTER modeling candidate.

Locked decision before touching outer validation:

* feature set: M3_missing_aware_compact
* model: XGBoost without class weighting
* threshold: 0.835, selected from repeated-CV OOF scores under the
  şartname final-prevalence assumption (3000 benign / 500 pathogenic)

The script reports a one-time outer-validation check and then fits a final
development artefact on all labeled rows for later label-free test inference.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import (  # noqa: E402
    ID_COLUMN,
    TARGET_COLUMN,
    STRATEGIES,
    MasterPreprocessor,
    load_master_csv,
)


RAW_FILE = ROOT / "data" / "raw" / "YARISMA_TRAIN_MASTER.csv"
OUTER_SPLIT_FILE = ROOT / "artifacts" / "master_outer_split.csv"
OUT_DIR = ROOT / "results" / "modeling" / "locked_final_candidate"
FEATURE_SET = "M3_missing_aware_compact"
MODEL_NAME = "XGBoost_no_weight"
LOCKED_THRESHOLD = 0.835
FINAL_BENIGN_RATE_ASSUMPTION = 3000 / 3500
RANDOM_STATE = 20260815


def make_xgboost(seed: int = RANDOM_STATE) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=3,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )


def positive_probability(model: XGBClassifier, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1].astype(float)


def metric_row(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    clipped = np.clip(y_score, 1e-7, 1 - 1e-7)
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }


def fit_candidate(frame: pd.DataFrame, seed: int = RANDOM_STATE) -> dict[str, Any]:
    processor = MasterPreprocessor(STRATEGIES[FEATURE_SET])
    x = processor.fit_transform(frame)
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)
    model = make_xgboost(seed=seed)
    model.fit(x, y)
    return {
        "processor": processor,
        "model": model,
        "feature_names": list(x.columns),
        "train_rows": int(len(frame)),
        "train_label_counts": {
            str(int(label)): int(count)
            for label, count in frame[TARGET_COLUMN].value_counts().sort_index().items()
        },
    }


def main() -> dict[str, Any]:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_master_csv(RAW_FILE)
    outer = pd.read_csv(OUTER_SPLIT_FILE, dtype={ID_COLUMN: "string"})
    partition = outer.set_index(ID_COLUMN)["partition"]
    outer_train = data.loc[data[ID_COLUMN].map(partition).eq("train")].reset_index(drop=True)
    outer_validation = data.loc[
        data[ID_COLUMN].map(partition).eq("validation")
    ].reset_index(drop=True)

    outer_fit = fit_candidate(outer_train, seed=RANDOM_STATE)
    x_val = outer_fit["processor"].transform(outer_validation)
    y_val = outer_validation[TARGET_COLUMN].to_numpy(dtype=int)
    val_score = positive_probability(outer_fit["model"], x_val)
    validation_metrics = metric_row(y_val, val_score, LOCKED_THRESHOLD)
    validation_predictions = pd.DataFrame(
        {
            ID_COLUMN: outer_validation[ID_COLUMN].astype(str).to_numpy(),
            TARGET_COLUMN: y_val,
            "score_pathogenic": val_score,
            "prediction": (val_score >= LOCKED_THRESHOLD).astype(int),
            "threshold": LOCKED_THRESHOLD,
        }
    )
    validation_predictions.to_csv(
        OUT_DIR / "outer_validation_locked_predictions.csv", index=False
    )

    full_fit = fit_candidate(data, seed=RANDOM_STATE)
    artefact = {
        "panel": "MASTER",
        "feature_set": FEATURE_SET,
        "model_name": MODEL_NAME,
        "threshold": LOCKED_THRESHOLD,
        "threshold_source": (
            "Repeated-CV OOF final-prevalence analysis; selected before "
            "outer-validation evaluation."
        ),
        "final_benign_rate_assumption": FINAL_BENIGN_RATE_ASSUMPTION,
        "processor": full_fit["processor"],
        "model": full_fit["model"],
        "feature_names": full_fit["feature_names"],
    }
    joblib.dump(artefact, OUT_DIR / "MASTER_M3_XGBoost_no_weight_locked.joblib")

    metadata = {
        "status": "completed",
        "feature_set": FEATURE_SET,
        "model_name": MODEL_NAME,
        "threshold": LOCKED_THRESHOLD,
        "final_benign_rate_assumption": FINAL_BENIGN_RATE_ASSUMPTION,
        "outer_validation_used_for_selection": False,
        "outer_validation_use": "one-time locked candidate report only",
        "outer_train_rows": int(len(outer_train)),
        "outer_validation_rows": int(len(outer_validation)),
        "outer_train_label_counts": outer_fit["train_label_counts"],
        "full_development_rows": full_fit["train_rows"],
        "full_development_label_counts": full_fit["train_label_counts"],
        "output_features_outer_train_fit": int(len(outer_fit["feature_names"])),
        "output_features_full_development_fit": int(len(full_fit["feature_names"])),
        "outer_validation_metrics": validation_metrics,
        "elapsed_seconds": float(time.time() - started),
        "outputs": {
            "model_artefact": str(
                (OUT_DIR / "MASTER_M3_XGBoost_no_weight_locked.joblib").relative_to(ROOT)
            ),
            "outer_validation_predictions": str(
                (OUT_DIR / "outer_validation_locked_predictions.csv").relative_to(ROOT)
            ),
            "metadata": str((OUT_DIR / "locked_candidate_metadata.json").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "locked_candidate_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


if __name__ == "__main__":
    main()
