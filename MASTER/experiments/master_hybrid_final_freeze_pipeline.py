"""Build the frozen Hybrid-E final candidate artefact.

This script does not open a new model search. It applies the already frozen
Hybrid-E governance procedure:

* M3_missing_aware_compact preprocessing
* Hybrid-E model family
* inner_best_strict_mcc selection rule
* inner-CV-only selection of weights, calibration/prior correction and threshold

The resulting artefact is for label-free final/test inference. Reported
performance claims should still come from nested governance outputs, not from
the all-labeled inner OOF selection used here to instantiate the final model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_hybrid_nested_governance import (  # noqa: E402
    BASE_MEMBERS,
    BASE_TO_FULL_NAME,
    FEATURE_SET,
    OUT_DIR as NESTED_OUT_DIR,
    _fit_one_base,
    _hybrid_c_weight_grid,
    _hybrid_e_weight_vectors,
    _logit_array,
    _make_inner_oof,
    _select_all_strategies,
)
from master_preprocessing import (  # noqa: E402
    ID_COLUMN,
    STRATEGIES,
    TARGET_COLUMN,
    MasterPreprocessor,
    load_master_csv,
)
from master_targeted_mcc_fp_experiments import (  # noqa: E402
    prior_correct_probability,
    ranking_metrics,
)
from master_tuned_modeling_experiments import (  # noqa: E402
    FINAL_PATHOGENIC_RATE,
    OUTER_SPLIT_FILE,
    RANDOM_STATE,
    RAW_FILE,
    positive_probability,
)


OUT_DIR = ROOT / "results" / "modeling" / "final_hybrid_freeze"
PREDICT_SCRIPT = "scripts/master_predict_hybrid_e_frozen.py"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _load_training_scope(scope: str) -> pd.DataFrame:
    data = load_master_csv(RAW_FILE)
    if scope == "all_labeled":
        return data.reset_index(drop=True)
    if scope == "development_train":
        outer = pd.read_csv(OUTER_SPLIT_FILE, dtype={ID_COLUMN: "string"})
        train_ids = set(outer.loc[outer["partition"].eq("train"), ID_COLUMN].astype(str))
        return data.loc[data[ID_COLUMN].astype(str).isin(train_ids)].reset_index(drop=True)
    raise ValueError(f"Unknown training scope: {scope}")


def _fit_base_models(train_frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    processor = MasterPreprocessor(STRATEGIES[FEATURE_SET])
    x_fit = processor.fit_transform(train_frame).to_numpy(dtype=np.float32)
    y_fit = train_frame[TARGET_COLUMN].to_numpy(dtype=int)
    fold_spw = float((y_fit == 0).sum() / (y_fit == 1).sum())

    models: dict[str, Any] = {}
    hard_counts: dict[str, int] = {}
    fit_seconds: dict[str, float] = {}
    for idx, member in enumerate(BASE_MEMBERS):
        model, sample_weight, hard_count = _fit_one_base(
            member,
            x_fit,
            y_fit,
            seed + idx,
            fold_spw,
            fold_key=(999, 999, idx),
        )
        started = time.time()
        if sample_weight is None:
            model.fit(x_fit, y_fit)
        else:
            model.fit(x_fit, y_fit, sample_weight=sample_weight)
        models[member] = model
        hard_counts[member] = int(hard_count)
        fit_seconds[member] = float(time.time() - started)

    return {
        "processor": processor,
        "models": models,
        "feature_names": list(processor.feature_names_ or []),
        "hard_counts": hard_counts,
        "fit_seconds": fit_seconds,
    }


def _fit_calibrators(
    inner_frame: pd.DataFrame,
    members: list[str],
    matrix_name: str,
) -> dict[str, Any]:
    if matrix_name == "raw":
        return {
            "matrix_name": "raw",
            "method": "raw",
            "apply_prior": False,
            "train_positive_rate": float(inner_frame[TARGET_COLUMN].mean()),
            "calibrators": {},
        }

    method, prior_part = matrix_name.split("_prior")
    apply_prior = prior_part == "True"
    y = inner_frame[TARGET_COLUMN].to_numpy(dtype=int)
    calibrators: dict[str, Any] = {}
    for member in members:
        score = np.clip(inner_frame[member].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
        if method == "sigmoid":
            calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            calibrator.fit(_logit_array(score).reshape(-1, 1), y)
        elif method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(score, y)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        calibrators[member] = calibrator

    return {
        "matrix_name": matrix_name,
        "method": method,
        "apply_prior": bool(apply_prior),
        "train_positive_rate": float(y.mean()),
        "calibrators": calibrators,
    }


def _apply_artifact_scores(artefact: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
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
            calibrated = calibrator.predict_proba(_logit_array(score).reshape(-1, 1))[:, 1]
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


def _strategy_row(strategy: Any) -> dict[str, Any]:
    inner_metrics = strategy.inner_metrics or {}
    weights = {
        BASE_TO_FULL_NAME[member]: float(weight)
        for member, weight in zip(strategy.members, strategy.weights)
        if weight > 0
    }
    return {
        "strategy": strategy.strategy,
        "selected_variant": strategy.selected_variant,
        "selection_status": strategy.selection_status,
        "calibration_matrix": strategy.calibration_matrix,
        "threshold": float(strategy.threshold),
        "weights_json": json.dumps(weights, ensure_ascii=False),
        **{f"inner_{key}": value for key, value in inner_metrics.items()},
    }


def build_frozen_artifact(
    train_scope: str,
    inner_splits: int,
    hye_weight_samples: int,
    selection_strategy: str,
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run_id = f"{_safe_name(selection_strategy)}_{_safe_name(train_scope)}"
    train_frame = _load_training_scope(train_scope)
    inner_frame, inner_fit_audit = _make_inner_oof(
        train_frame,
        outer_seed=202608,
        outer_fold=15,
        inner_splits=inner_splits,
    )

    h_c_grid = _hybrid_c_weight_grid()
    h_e_vectors = _hybrid_e_weight_vectors(hye_weight_samples)
    strategies, f1_floor = _select_all_strategies(inner_frame, h_c_grid, h_e_vectors)
    selection_rows = pd.DataFrame([_strategy_row(strategy) for strategy in strategies])
    selection_path = OUT_DIR / f"{run_id}_final_freeze_selection.csv"
    inner_fit_path = OUT_DIR / f"{run_id}_final_inner_fit_audit.csv"
    selection_rows.to_csv(selection_path, index=False)
    inner_fit_audit.to_csv(inner_fit_path, index=False)

    selected = next(strategy for strategy in strategies if strategy.strategy == selection_strategy)
    base_fit = _fit_base_models(train_frame, seed=RANDOM_STATE + 9090)
    calibration = _fit_calibrators(
        inner_frame,
        selected.members,
        selected.calibration_matrix,
    )
    artefact = {
        "panel": "MASTER",
        "status": "frozen_candidate_not_final_performance_claim",
        "feature_set": FEATURE_SET,
        "model_family": "Hybrid-E",
        "selection_strategy": selected.strategy,
        "selection_rule": (selected.details or {}).get("selection_rule"),
        "selected_variant": selected.selected_variant,
        "train_scope": train_scope,
        "train_rows": int(len(train_frame)),
        "train_label_counts": {
            str(int(label)): int(count)
            for label, count in train_frame[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "inner_splits": int(inner_splits),
        "hye_weight_samples_requested": int(hye_weight_samples),
        "f1_floor": float(f1_floor),
        "threshold": float(selected.threshold),
        "members": list(selected.members),
        "member_full_names": [BASE_TO_FULL_NAME[member] for member in selected.members],
        "weights": [float(weight) for weight in selected.weights],
        "weights_by_full_name": {
            BASE_TO_FULL_NAME[member]: float(weight)
            for member, weight in zip(selected.members, selected.weights)
            if weight > 0
        },
        "calibration_matrix": selected.calibration_matrix,
        "calibration": calibration,
        "processor": base_fit["processor"],
        "base_models": base_fit["models"],
        "feature_names": base_fit["feature_names"],
        "threshold_source": "Frozen inner-CV selection rule applied on labelled training scope.",
        "final_pathogenic_rate_assumption": FINAL_PATHOGENIC_RATE,
        "final_or_test_data_used": False,
        "external_data_used": False,
        "historical_outer_validation_used_for_performance_claim": False,
        "performance_claim_source": "nested_governance outputs",
    }
    artefact_path = OUT_DIR / f"MASTER_{run_id}_frozen.joblib"
    joblib.dump(artefact, artefact_path)

    inner_score = _apply_artifact_scores(artefact, train_frame)
    in_sample_path = OUT_DIR / f"{run_id}_artifact_in_sample_scores.csv"
    pd.DataFrame(
        {
            ID_COLUMN: train_frame[ID_COLUMN].astype(str).to_numpy(),
            TARGET_COLUMN: train_frame[TARGET_COLUMN].to_numpy(dtype=int),
            "score_pathogenic": inner_score,
            "prediction": (inner_score >= float(selected.threshold)).astype(int),
            "threshold": float(selected.threshold),
            "note": "In-sample final artefact score; do not use as performance evidence.",
        }
    ).to_csv(in_sample_path, index=False)

    metadata = {
        "status": "completed",
        "elapsed_seconds": float(time.time() - started),
        "train_scope": train_scope,
        "train_rows": int(len(train_frame)),
        "selection_strategy": selected.strategy,
        "selected_variant": selected.selected_variant,
        "selection_status": selected.selection_status,
        "threshold": float(selected.threshold),
        "calibration_matrix": selected.calibration_matrix,
        "weights_by_full_name": artefact["weights_by_full_name"],
        "inner_selection_metrics": selected.inner_metrics,
        "ranking_metrics_on_inner_oof": ranking_metrics(
            inner_frame[TARGET_COLUMN].to_numpy(dtype=int),
            inner_frame[selected.members].to_numpy(dtype=float) @ selected.weights,
        ),
        "outputs": {
            "artefact": str(artefact_path.relative_to(ROOT)),
            "selection_summary": str(selection_path.relative_to(ROOT)),
            "inner_fit_audit": str(inner_fit_path.relative_to(ROOT)),
            "in_sample_scores": str(in_sample_path.relative_to(ROOT)),
            "prediction_script": PREDICT_SCRIPT,
            "metadata": str((OUT_DIR / f"{run_id}_final_freeze_metadata.json").relative_to(ROOT)),
        },
    }
    (OUT_DIR / f"{run_id}_final_freeze_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-scope",
        default="all_labeled",
        choices=["all_labeled", "development_train"],
    )
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--hye-weight-samples", type=int, default=2500)
    parser.add_argument("--selection-strategy", default="Hybrid-E_bestMCC_nested")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    metadata = build_frozen_artifact(
        train_scope=args.train_scope,
        inner_splits=args.inner_splits,
        hye_weight_samples=args.hye_weight_samples,
        selection_strategy=args.selection_strategy,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


if __name__ == "__main__":
    main()
