"""Hybrid-C and Hybrid-E experiments for the MASTER panel.

Hybrid-C focuses on false-positive controlled weighted ensembles.
Hybrid-E focuses on calibrated/prior-corrected ensembles and CV stacking.

The script uses only previously saved out-of-fold scores from approved
MASTER-panel model-development runs. No outer validation, final/test data,
or external dataset is used for model, weight, calibration, or threshold
selection.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import ID_COLUMN, TARGET_COLUMN  # noqa: E402
from master_targeted_mcc_fp_experiments import (  # noqa: E402
    F1_KEEP_TOLERANCE,
    FP_TARGET,
    OUT_DIR as TARGETED_DIR,
    RELAXED_F1_FLOOR,
    RELAXED_SENSITIVITY_FLOOR,
    STRICT_SENSITIVITY_FLOOR,
    THRESHOLDS,
    _aligned_score_matrix,
    _averaged_from_oof,
    _logit,
    calibrate_scores_cv,
    load_tuned_oof,
    prior_correct_probability,
    prior_sample_weight,
    summarize_score,
)
from master_tuned_modeling_experiments import (  # noqa: E402
    FINAL_PATHOGENIC_RATE,
    RANDOM_STATE,
)


MODELING_DIR = ROOT / "results" / "modeling"
CATBOOST_DIR = MODELING_DIR / "catboost"
LIGHTGBM_DIR = MODELING_DIR / "lightgbm"
OUT_DIR = MODELING_DIR / "hybrid_ce"
FINAL_TOTAL_N = 3500

CATOPT_A = (
    "CATOPT_M3_RF_depth8_balanced:0.4_"
    "M3_XGB_SPW_regularized_x1.5:0.3_"
    "M3_CatBoost_bernoulli:0.3"
)
OPTENS_A = (
    "OPTENS_M3_XGB_NW_shallow_reg:0.2_"
    "M3_XGB_NW_depth4:0.1_"
    "M3_RF_depth8_balanced:0.3_"
    "M3_RF_leaf4:0.1_"
    "M3_XGB_SPW_regularized_x1.5:0.3"
)


@dataclass(frozen=True)
class WeightSearchResult:
    candidate: str
    method_group: str
    details: dict[str, Any]
    status: str
    threshold: float
    f1: float
    mcc: float
    specificity: float
    sensitivity: float
    precision: float
    expected_fp_per_3500: float
    expected_fn_per_3500: float
    ids: np.ndarray
    score: np.ndarray
    y_true: np.ndarray
    sort_key: tuple[int, float, float, float]


def _reference_summary() -> pd.DataFrame:
    for path in [
        LIGHTGBM_DIR / "lightgbm_vs_current_summary.csv",
        CATBOOST_DIR / "catboost_vs_current_summary.csv",
        TARGETED_DIR / "all_targeted_summary.csv",
    ]:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    raise FileNotFoundError("No reference model summary found.")


def _load_score_sources() -> dict[str, pd.DataFrame]:
    sources = load_tuned_oof()
    for path in [
        TARGETED_DIR / "method2_optimized_ensemble_oof.csv",
        TARGETED_DIR / "new_model_oof_predictions.csv",
        TARGETED_DIR / "method4_calibration_prior_oof.csv",
        TARGETED_DIR / "method6_extra_hgb_ensemble_oof.csv",
        CATBOOST_DIR / "catboost_oof_predictions.csv",
        CATBOOST_DIR / "catboost_ensemble_oof.csv",
        LIGHTGBM_DIR / "lightgbm_oof_predictions.csv",
        LIGHTGBM_DIR / "lightgbm_ensemble_oof.csv",
    ]:
        if path.exists() and path.stat().st_size > 0:
            sources.update(_averaged_from_oof(pd.read_csv(path, dtype={ID_COLUMN: "string"})))
    return sources


def _balanced_accuracy_columns(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    if {"target_specificity", "target_sensitivity"}.issubset(summary.columns):
        summary["target_balanced_accuracy"] = (
            summary["target_specificity"] + summary["target_sensitivity"]
        ) / 2.0
    if {"spec88_specificity", "spec88_sensitivity"}.issubset(summary.columns):
        summary["spec88_balanced_accuracy"] = (
            summary["spec88_specificity"] + summary["spec88_sensitivity"]
        ) / 2.0
    return summary


def _select_fast(y_true: np.ndarray, score: np.ndarray, f1_floor: float) -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    thresholds = THRESHOLDS
    positives = np.sort(score[y_true == 1])
    negatives = np.sort(score[y_true == 0])
    pos_n = len(positives)
    neg_n = len(negatives)

    tp_counts = pos_n - np.searchsorted(positives, thresholds, side="left")
    fp_counts = neg_n - np.searchsorted(negatives, thresholds, side="left")
    tpr = tp_counts / pos_n
    fpr = fp_counts / neg_n

    pi = FINAL_PATHOGENIC_RATE
    tp = pi * tpr
    fn = pi * (1.0 - tpr)
    fp = (1.0 - pi) * fpr
    tn = (1.0 - pi) * (1.0 - fpr)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    f1 = np.divide(
        2.0 * precision * tpr,
        precision + tpr,
        out=np.zeros_like(tp),
        where=(precision + tpr) > 0,
    )
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = np.divide(
        tp * tn - fp * fn,
        np.sqrt(denom),
        out=np.zeros_like(tp),
        where=denom > 0,
    )
    specificity = 1.0 - fpr
    expected_fp = FINAL_TOTAL_N * fp
    expected_fn = FINAL_TOTAL_N * fn

    mask = (
        (f1 >= f1_floor)
        & (expected_fp <= FP_TARGET)
        & (tpr >= STRICT_SENSITIVITY_FLOOR)
    )
    status = "strict"
    if not mask.any():
        mask = (
            (f1 >= RELAXED_F1_FLOOR)
            & (expected_fp <= FP_TARGET)
            & (tpr >= RELAXED_SENSITIVITY_FLOOR)
        )
        status = "relaxed"
    if not mask.any():
        mask = np.ones_like(f1, dtype=bool)
        status = "mcc_only"

    candidate_indices = np.flatnonzero(mask)
    order = np.lexsort(
        (
            -tpr[candidate_indices],
            -specificity[candidate_indices],
            -f1[candidate_indices],
            -mcc[candidate_indices],
        )
    )
    idx = candidate_indices[order[0]]
    return {
        "status": status,
        "threshold": float(thresholds[idx]),
        "f1": float(f1[idx]),
        "mcc": float(mcc[idx]),
        "specificity": float(specificity[idx]),
        "sensitivity": float(tpr[idx]),
        "precision": float(precision[idx]),
        "expected_fp_per_3500": float(expected_fp[idx]),
        "expected_fn_per_3500": float(expected_fn[idx]),
    }


def _sort_key(status: str, mcc: float, f1: float, fp: float) -> tuple[int, float, float, float]:
    status_rank = {"strict": 2, "relaxed": 1, "mcc_only": 0}.get(status, 0)
    return (status_rank, mcc, f1, -fp)


def _weight_label(prefix: str, members: list[str], weights: np.ndarray) -> str:
    aliases = {
        "M3_RF_depth8_balanced": "RF8",
        "M3_XGB_SPW_regularized_x1.5": "XGBSPW",
        "M3_CatBoost_bernoulli": "CATB",
        "M3_CatBoost_regularized": "CATR",
        "M3_CatBoost_prior025": "CATP25",
        "M3_LGBM_prior025": "LGBMP25",
        "M3_XGB_NW_shallow_reg": "XGBNW",
        "M3_HardBenign_RF_top30_w2_s025": "HBRF",
    }
    parts = [
        f"{aliases.get(member, member)}{weight:.3f}"
        for member, weight in zip(members, weights)
        if weight > 0
    ]
    return prefix + "_" + "_".join(parts)


def _screening_record(
    candidate: str,
    method_group: str,
    details: dict[str, Any],
    ids: np.ndarray,
    y_true: np.ndarray,
    score: np.ndarray,
    f1_floor: float,
) -> WeightSearchResult:
    selected = _select_fast(y_true, score, f1_floor)
    return WeightSearchResult(
        candidate=candidate,
        method_group=method_group,
        details=details,
        status=str(selected["status"]),
        threshold=float(selected["threshold"]),
        f1=float(selected["f1"]),
        mcc=float(selected["mcc"]),
        specificity=float(selected["specificity"]),
        sensitivity=float(selected["sensitivity"]),
        precision=float(selected["precision"]),
        expected_fp_per_3500=float(selected["expected_fp_per_3500"]),
        expected_fn_per_3500=float(selected["expected_fn_per_3500"]),
        ids=np.asarray(ids).astype(str),
        score=np.asarray(score, dtype=float),
        y_true=np.asarray(y_true, dtype=int),
        sort_key=_sort_key(
            str(selected["status"]),
            float(selected["mcc"]),
            float(selected["f1"]),
            float(selected["expected_fp_per_3500"]),
        ),
    )


def _screening_rows(results: list[WeightSearchResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "candidate": result.candidate,
                "method_group": result.method_group,
                "screen_status": result.status,
                "screen_threshold": result.threshold,
                "screen_f1": result.f1,
                "screen_mcc": result.mcc,
                "screen_specificity": result.specificity,
                "screen_sensitivity": result.sensitivity,
                "screen_precision": result.precision,
                "screen_expected_fp_per_3500": result.expected_fp_per_3500,
                "screen_expected_fn_per_3500": result.expected_fn_per_3500,
                "details": json.dumps(result.details, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def _summarize_results(
    results: list[WeightSearchResult],
    f1_floor: float,
    top_n: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    ordered = sorted(results, key=lambda item: item.sort_key, reverse=True)
    chosen: list[WeightSearchResult] = []
    seen = set()
    for status in ["strict", "relaxed", "mcc_only"]:
        status_rows = [item for item in ordered if item.status == status]
        status_rows = sorted(status_rows, key=lambda item: item.sort_key, reverse=True)[:top_n]
        for item in status_rows:
            if item.candidate not in seen:
                chosen.append(item)
                seen.add(item.candidate)

    summary_rows = []
    oof_frames = []
    for item in chosen:
        row = summarize_score(
            candidate=item.candidate,
            method_group=item.method_group,
            y_true=item.y_true,
            y_score=item.score,
            f1_floor=f1_floor,
            details=json.dumps(item.details, ensure_ascii=False),
        )
        row["screen_status"] = item.status
        summary_rows.append(row)
        oof_frames.append(
            pd.DataFrame(
                {
                    ID_COLUMN: item.ids,
                    TARGET_COLUMN: item.y_true,
                    "score": item.score,
                    "candidate": item.candidate,
                    "method_group": item.method_group,
                }
            )
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["target_status", "target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[True, False, False, True],
    )
    status_order = {"strict": 0, "relaxed": 1, "mcc_only": 2}
    summary["_status_order"] = summary["target_status"].map(status_order).fillna(3)
    summary = summary.sort_values(
        ["_status_order", "target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[True, False, False, True],
    ).drop(columns=["_status_order"])
    return _balanced_accuracy_columns(summary), pd.concat(oof_frames, ignore_index=True)


def build_hybrid_c(
    sources: dict[str, pd.DataFrame],
    f1_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    members = [
        "M3_RF_depth8_balanced",
        "M3_XGB_SPW_regularized_x1.5",
        "M3_CatBoost_bernoulli",
        "M3_LGBM_prior025",
        "M3_XGB_NW_shallow_reg",
        "M3_HardBenign_RF_top30_w2_s025",
    ]
    base, y_true, matrix = _aligned_score_matrix(sources, members)
    ids = base[ID_COLUMN].astype(str).to_numpy()
    results: list[WeightSearchResult] = []
    searched = 0
    units = 40
    for rf_units in range(6, 23):
        for spw_units in range(2, 17):
            for cat_units in range(2, 25):
                for lgbm_units in range(0, 13):
                    for xgbnw_units in range(0, 13):
                        hard_units = (
                            units
                            - rf_units
                            - spw_units
                            - cat_units
                            - lgbm_units
                            - xgbnw_units
                        )
                        if hard_units < 0 or hard_units > 12:
                            continue
                        if lgbm_units + hard_units == 0:
                            continue
                        if rf_units + spw_units + cat_units < 20:
                            continue
                        weights = np.asarray(
                            [
                                rf_units,
                                spw_units,
                                cat_units,
                                lgbm_units,
                                xgbnw_units,
                                hard_units,
                            ],
                            dtype=float,
                        ) / units
                        searched += 1
                        score = matrix @ weights
                        details = {
                            "family": "Hybrid-C",
                            "subtype": "fp_controlled_weight_grid",
                            "members": {
                                member: float(weight)
                                for member, weight in zip(members, weights)
                                if weight > 0
                            },
                        }
                        results.append(
                            _screening_record(
                                _weight_label("HYC_GRID", members, weights),
                                "Hybrid-C FP-controlled ensemble",
                                details,
                                ids,
                                y_true,
                                score,
                                f1_floor,
                            )
                        )

    if CATOPT_A in sources:
        gate_members = [
            CATOPT_A,
            "M3_LGBM_prior025",
            "M3_HardBenign_RF_top30_w2_s025",
            "M3_CatBoost_prior025",
            "M3_XGB_SPW_regularized_x1.5",
        ]
        available_gate_members = [member for member in gate_members if member in sources]
        gate_base, gate_y, gate_matrix = _aligned_score_matrix(sources, available_gate_members)
        gate_ids = gate_base[ID_COLUMN].astype(str).to_numpy()
        gate_lookup = {
            member: gate_matrix[:, idx]
            for idx, member in enumerate(available_gate_members)
        }
        current_score = gate_lookup[CATOPT_A]
        guards = {
            "lgbm": gate_lookup.get("M3_LGBM_prior025"),
            "hard_benign": gate_lookup.get("M3_HardBenign_RF_top30_w2_s025"),
            "catboost_prior": gate_lookup.get("M3_CatBoost_prior025"),
            "spw": gate_lookup.get("M3_XGB_SPW_regularized_x1.5"),
        }
        if guards["lgbm"] is not None and guards["hard_benign"] is not None:
            guards["lgbm_hard_mean"] = (guards["lgbm"] + guards["hard_benign"]) / 2.0
        if guards["lgbm"] is not None and guards["catboost_prior"] is not None:
            guards["lgbm_catprior_mean"] = (
                guards["lgbm"] + guards["catboost_prior"]
            ) / 2.0

        for guard_name, guard_score in guards.items():
            if guard_score is None:
                continue
            for alpha in np.round(np.arange(0.05, 0.81, 0.05), 2):
                candidates = {
                    "softmin": (1.0 - alpha) * current_score
                    + alpha * np.minimum(current_score, guard_score),
                    "penalty": np.clip(
                        current_score - alpha * np.maximum(0.0, current_score - guard_score),
                        1e-7,
                        1.0 - 1e-7,
                    ),
                    "geometric": np.exp(
                        (1.0 - alpha) * np.log(np.clip(current_score, 1e-7, 1.0))
                        + alpha * np.log(np.clip(guard_score, 1e-7, 1.0))
                    ),
                }
                for transform, score in candidates.items():
                    details = {
                        "family": "Hybrid-C",
                        "subtype": "benign_gate",
                        "base": CATOPT_A,
                        "guard": guard_name,
                        "transform": transform,
                        "alpha": float(alpha),
                    }
                    results.append(
                        _screening_record(
                            f"HYC_GATE_{transform}_{guard_name}_a{alpha:.2f}",
                            "Hybrid-C benign gate",
                            details,
                            gate_ids,
                            gate_y,
                            score,
                            f1_floor,
                        )
                    )

    screening = _screening_rows(results)
    top_screening = pd.concat(
        [
            screening.loc[screening["screen_status"].eq(status)]
            .sort_values(
                ["screen_mcc", "screen_f1", "screen_expected_fp_per_3500"],
                ascending=[False, False, True],
            )
            .head(500)
            for status in ["strict", "relaxed", "mcc_only"]
        ],
        ignore_index=True,
    )
    summary, oof = _summarize_results(results, f1_floor)
    metadata = {
        "searched_weight_grid_candidates": searched,
        "screened_total_candidates": len(results),
        "members": members,
    }
    return summary, oof, top_screening, metadata


def _candidate_weight_vectors(members: list[str], samples: int = 9000) -> list[np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    prototypes = [
        [0.40, 0.30, 0.30, 0.00, 0.00, 0.00, 0.00],
        [0.375, 0.25, 0.325, 0.00, 0.025, 0.00, 0.025],
        [0.35, 0.25, 0.25, 0.05, 0.05, 0.00, 0.05],
        [0.35, 0.30, 0.20, 0.05, 0.05, 0.00, 0.05],
        [0.30, 0.25, 0.25, 0.10, 0.05, 0.00, 0.05],
        [0.30, 0.30, 0.20, 0.10, 0.05, 0.00, 0.05],
        [0.30, 0.25, 0.30, 0.00, 0.10, 0.00, 0.05],
        [0.35, 0.20, 0.30, 0.05, 0.05, 0.00, 0.05],
        [0.45, 0.20, 0.25, 0.00, 0.05, 0.00, 0.05],
    ]
    vectors: list[np.ndarray] = []
    for prototype in prototypes:
        vector = np.asarray(prototype, dtype=float)
        vector = vector / vector.sum()
        vectors.append(vector)
        alpha = vector * 120.0 + 1.0
        vectors.extend(rng.dirichlet(alpha, size=samples // len(prototypes)))
    vectors.extend(rng.dirichlet(np.ones(len(members)) * 2.0, size=samples // 3))

    deduped: dict[tuple[float, ...], np.ndarray] = {}
    for vector in vectors:
        vector = np.asarray(vector, dtype=float)
        vector = vector / vector.sum()
        if not (0.12 <= vector[0] <= 0.60):
            continue
        if not (0.03 <= vector[1] <= 0.45):
            continue
        if not (0.05 <= vector[2] <= 0.65):
            continue
        if vector[3] > 0.35 or vector[4] > 0.35 or vector[5] > 0.35:
            continue
        if vector[6] > 0.25:
            continue
        if vector[3] + vector[4] + vector[6] < 0.02:
            continue
        rounded = np.round(vector, 3)
        rounded = rounded / rounded.sum()
        key = tuple(np.round(rounded, 3))
        deduped[key] = rounded
    return list(deduped.values())


def _stack_cv_scores(
    y_true: np.ndarray,
    matrix: np.ndarray,
    c_value: float,
    prior_strength: float,
    class_weight: str | None,
) -> np.ndarray:
    raw = np.clip(matrix, 1e-7, 1.0 - 1e-7)
    features = np.hstack(
        [
            _logit(raw),
            raw,
            np.max(raw, axis=1, keepdims=True),
            np.min(raw, axis=1, keepdims=True),
            np.std(raw, axis=1, keepdims=True),
        ]
    )
    output = np.zeros(len(y_true), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, hold_idx in splitter.split(features, y_true):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=2000,
                class_weight=class_weight,
            ),
        )
        sample_weight = prior_sample_weight(y_true[train_idx], prior_strength)
        if np.allclose(sample_weight, 1.0):
            sample_weight = None
        if sample_weight is None:
            model.fit(features[train_idx], y_true[train_idx])
        else:
            model.fit(
                features[train_idx],
                y_true[train_idx],
                logisticregression__sample_weight=sample_weight,
            )
        output[hold_idx] = model.predict_proba(features[hold_idx])[:, 1]
    return np.clip(output, 1e-7, 1.0 - 1e-7)


def build_hybrid_e(
    sources: dict[str, pd.DataFrame],
    f1_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    members = [
        "M3_RF_depth8_balanced",
        "M3_XGB_SPW_regularized_x1.5",
        "M3_CatBoost_bernoulli",
        "M3_CatBoost_regularized",
        "M3_LGBM_prior025",
        "M3_XGB_NW_shallow_reg",
        "M3_HardBenign_RF_top30_w2_s025",
    ]
    members = [member for member in members if member in sources]
    base, y_true, raw_matrix = _aligned_score_matrix(sources, members)
    ids = base[ID_COLUMN].astype(str).to_numpy()
    calibrated_matrices = {"raw": raw_matrix}
    for method in ["sigmoid", "isotonic"]:
        for apply_prior in [False, True]:
            calibrated_columns = []
            for idx, _member in enumerate(members):
                calibrated_columns.append(
                    calibrate_scores_cv(
                        y_true,
                        raw_matrix[:, idx],
                        method=method,
                        apply_prior_correction=apply_prior,
                    )
                )
            calibrated_matrices[f"{method}_prior{apply_prior}"] = np.column_stack(
                calibrated_columns
            )

    results: list[WeightSearchResult] = []
    weight_vectors = _candidate_weight_vectors(members)
    for matrix_name, matrix in calibrated_matrices.items():
        for weights in weight_vectors:
            score = matrix @ weights
            details = {
                "family": "Hybrid-E",
                "subtype": "calibrated_weighted_ensemble",
                "calibration_matrix": matrix_name,
                "members": {
                    member: float(weight)
                    for member, weight in zip(members, weights)
                    if weight > 0
                },
            }
            results.append(
                _screening_record(
                    _weight_label(f"HYE_CAL_{matrix_name}", members, weights),
                    "Hybrid-E calibrated ensemble",
                    details,
                    ids,
                    y_true,
                    score,
                    f1_floor,
                )
            )

    c_values = [0.03, 0.10, 0.30, 1.0, 3.0, 10.0]
    prior_strengths = [0.0, 0.25, 0.50, 0.75, 1.0]
    class_weights: list[str | None] = [None, "balanced"]
    for c_value in c_values:
        for prior_strength in prior_strengths:
            for class_weight in class_weights:
                score = _stack_cv_scores(
                    y_true,
                    raw_matrix,
                    c_value=c_value,
                    prior_strength=prior_strength,
                    class_weight=class_weight,
                )
                for apply_prior in [False, True]:
                    final_score = (
                        prior_correct_probability(score, float(y_true.mean()))
                        if apply_prior
                        else score
                    )
                    details = {
                        "family": "Hybrid-E",
                        "subtype": "inner_cv_logistic_stacking",
                        "members": members,
                        "C": c_value,
                        "prior_strength": prior_strength,
                        "class_weight": class_weight,
                        "post_prior_correction": apply_prior,
                    }
                    class_weight_label = "none" if class_weight is None else class_weight
                    results.append(
                        _screening_record(
                            (
                                f"HYE_STACK_C{c_value:g}_ps{prior_strength:g}_"
                                f"cw{class_weight_label}_prior{apply_prior}"
                            ),
                            "Hybrid-E inner-CV stacked ensemble",
                            details,
                            ids,
                            y_true,
                            final_score,
                            f1_floor,
                        )
                    )

    if CATOPT_A in sources:
        current_table = sources[CATOPT_A].sort_values(ID_COLUMN).reset_index(drop=True)
        current_ids = current_table[ID_COLUMN].astype(str).to_numpy()
        current_y = current_table[TARGET_COLUMN].to_numpy(dtype=int)
        current_score = current_table["score"].to_numpy(dtype=float)
        for method in ["sigmoid", "isotonic"]:
            for apply_prior in [False, True]:
                score = calibrate_scores_cv(
                    current_y,
                    current_score,
                    method=method,
                    apply_prior_correction=apply_prior,
                )
                details = {
                    "family": "Hybrid-E",
                    "subtype": "catopt_a_cv_calibrated",
                    "source": CATOPT_A,
                    "calibration": method,
                    "prior_correction": apply_prior,
                }
                results.append(
                    _screening_record(
                        f"HYE_CATOPT_A_{method}_prior{apply_prior}",
                        "Hybrid-E calibrated CATOPT-A",
                        details,
                        current_ids,
                        current_y,
                        score,
                        f1_floor,
                    )
                )

    screening = _screening_rows(results)
    top_screening = pd.concat(
        [
            screening.loc[screening["screen_status"].eq(status)]
            .sort_values(
                ["screen_mcc", "screen_f1", "screen_expected_fp_per_3500"],
                ascending=[False, False, True],
            )
            .head(500)
            for status in ["strict", "relaxed", "mcc_only"]
        ],
        ignore_index=True,
    )
    summary, oof = _summarize_results(results, f1_floor)
    metadata = {
        "screened_weight_vectors": len(weight_vectors),
        "screened_total_candidates": len(results),
        "members": members,
        "calibration_matrices": list(calibrated_matrices),
    }
    return summary, oof, top_screening, metadata


def _write_top_table(summary: pd.DataFrame, path: Path, top_n: int = 25) -> None:
    columns = [
        "candidate",
        "method_group",
        "target_status",
        "oof_auroc",
        "oof_auprc",
        "final_weighted_auprc",
        "target_threshold",
        "target_f1",
        "target_mcc",
        "target_balanced_accuracy",
        "target_specificity",
        "target_sensitivity",
        "target_precision",
        "target_expected_fp_per_3500",
        "target_expected_fn_per_3500",
        "details",
    ]
    available = [column for column in columns if column in summary.columns]
    if not summary.empty:
        summary.loc[:, available].head(top_n).to_csv(path, index=False)


def main() -> dict[str, Any]:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reference = _balanced_accuracy_columns(_reference_summary())
    strict_reference = reference.loc[reference["target_status"].eq("strict")].copy()
    if strict_reference.empty:
        strict_reference = reference.copy()
    current_best = strict_reference.sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    ).iloc[0]
    f1_floor = float(strict_reference["target_f1"].max()) - F1_KEEP_TOLERANCE

    print("Loading score sources", flush=True)
    sources = _load_score_sources()
    required = {
        "M3_RF_depth8_balanced",
        "M3_XGB_SPW_regularized_x1.5",
        "M3_CatBoost_bernoulli",
        "M3_LGBM_prior025",
        "M3_HardBenign_RF_top30_w2_s025",
    }
    missing = sorted(required - set(sources))
    if missing:
        raise FileNotFoundError(f"Missing required OOF sources: {missing}")

    print("Hybrid-C: FP-controlled weighted/gated search", flush=True)
    hybrid_c_summary, hybrid_c_oof, hybrid_c_screening, hybrid_c_meta = build_hybrid_c(
        sources,
        f1_floor,
    )
    hybrid_c_summary.to_csv(OUT_DIR / "hybrid_c_summary.csv", index=False)
    hybrid_c_oof.to_csv(OUT_DIR / "hybrid_c_top_oof.csv", index=False)
    hybrid_c_screening.to_csv(OUT_DIR / "hybrid_c_screening_top.csv", index=False)

    print("Hybrid-E: calibrated ensemble and inner-CV stacking", flush=True)
    hybrid_e_summary, hybrid_e_oof, hybrid_e_screening, hybrid_e_meta = build_hybrid_e(
        sources,
        f1_floor,
    )
    hybrid_e_summary.to_csv(OUT_DIR / "hybrid_e_summary.csv", index=False)
    hybrid_e_oof.to_csv(OUT_DIR / "hybrid_e_top_oof.csv", index=False)
    hybrid_e_screening.to_csv(OUT_DIR / "hybrid_e_screening_top.csv", index=False)

    hybrid_summary = pd.concat(
        [
            hybrid_c_summary.assign(experiment_block="Hybrid-C"),
            hybrid_e_summary.assign(experiment_block="Hybrid-E"),
        ],
        ignore_index=True,
    )
    status_order = {"strict": 0, "relaxed": 1, "mcc_only": 2}
    hybrid_summary["_status_order"] = (
        hybrid_summary["target_status"].map(status_order).fillna(3)
    )
    hybrid_summary = hybrid_summary.sort_values(
        ["_status_order", "target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[True, False, False, True],
    ).drop(columns=["_status_order"])
    hybrid_summary.to_csv(OUT_DIR / "hybrid_ce_summary.csv", index=False)

    comparison = pd.concat(
        [
            reference.assign(experiment_block="current_reference"),
            hybrid_summary.assign(experiment_block=hybrid_summary["experiment_block"]),
        ],
        ignore_index=True,
    )
    comparison["_status_order"] = comparison["target_status"].map(status_order).fillna(3)
    comparison = comparison.sort_values(
        ["_status_order", "target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[True, False, False, True],
    ).drop(columns=["_status_order"])
    comparison.to_csv(OUT_DIR / "hybrid_ce_vs_current_summary.csv", index=False)

    _write_top_table(hybrid_c_summary, OUT_DIR / "hybrid_c_top.csv")
    _write_top_table(hybrid_e_summary, OUT_DIR / "hybrid_e_top.csv")
    _write_top_table(hybrid_summary, OUT_DIR / "hybrid_ce_top.csv")
    _write_top_table(comparison, OUT_DIR / "hybrid_ce_vs_current_top.csv", top_n=40)

    metadata = {
        "status": "completed",
        "outer_validation_used": False,
        "external_data_used": False,
        "score_sources_loaded": len(sources),
        "f1_floor": f1_floor,
        "fp_target_per_3500": FP_TARGET,
        "strict_sensitivity_floor": STRICT_SENSITIVITY_FLOOR,
        "current_strict_reference": {
            "candidate": str(current_best["candidate"]),
            "target_f1": float(current_best["target_f1"]),
            "target_mcc": float(current_best["target_mcc"]),
            "target_expected_fp_per_3500": float(
                current_best["target_expected_fp_per_3500"]
            ),
        },
        "hybrid_c": hybrid_c_meta,
        "hybrid_e": hybrid_e_meta,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "hybrid_c": "results/modeling/hybrid_ce/hybrid_c_summary.csv",
            "hybrid_e": "results/modeling/hybrid_ce/hybrid_e_summary.csv",
            "hybrid_summary": "results/modeling/hybrid_ce/hybrid_ce_summary.csv",
            "comparison": "results/modeling/hybrid_ce/hybrid_ce_vs_current_summary.csv",
        },
    }
    (OUT_DIR / "hybrid_ce_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display_cols = [
        "candidate",
        "experiment_block",
        "method_group",
        "target_status",
        "oof_auroc",
        "oof_auprc",
        "final_weighted_auprc",
        "target_threshold",
        "target_f1",
        "target_mcc",
        "target_balanced_accuracy",
        "target_specificity",
        "target_sensitivity",
        "target_precision",
        "target_expected_fp_per_3500",
        "target_expected_fn_per_3500",
    ]
    print("\n=== HYBRID CE VS CURRENT TOP ===")
    print(
        comparison.loc[:, [col for col in display_cols if col in comparison.columns]]
        .head(30)
        .round(4)
        .to_string(index=False)
    )
    return metadata


if __name__ == "__main__":
    main()
