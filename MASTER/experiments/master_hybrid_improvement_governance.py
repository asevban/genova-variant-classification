"""Nested governance experiments for safer Hybrid-E improvements.

This script evaluates challenger procedures without changing the frozen final
artefact.  The guardrails are intentionally strict:

* use only the labelled MASTER training table,
* fit preprocessing separately inside every inner/outer fold,
* select weights/calibration/threshold/fallback policy only on inner OOF rows,
* use outer holdout rows once for evaluation,
* do not use final/test data, external data, genomic lookup, or saved outer FP
  IDs for model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_hybrid_nested_governance import (  # noqa: E402
    AUPRC_KEEP_TOLERANCE,
    BASE_MEMBERS,
    BASE_TO_FULL_NAME,
    FEATURE_SET,
    FINAL_TOTAL_N,
    FOLD_FILE,
    SelectedStrategy,
    _calibrated_inner_matrices,
    _evaluate_selected_strategy,
    _fit_apply_calibration,
    _fit_one_base,
    _fit_score_base_frame,
    _hybrid_c_weight_grid,
    _hybrid_e_weight_vectors,
    _load_development_frame,
    _make_inner_oof,
    _parse_outer_seeds,
    _ranking_for_score,
    _select_all_strategies,
)
from master_preprocessing import (  # noqa: E402
    ID_COLUMN,
    STRATEGIES,
    TARGET_COLUMN,
    MasterPreprocessor,
    feature_columns,
)
from master_targeted_mcc_fp_experiments import (  # noqa: E402
    FP_TARGET,
    RELAXED_F1_FLOOR,
    STRICT_SENSITIVITY_FLOOR,
    THRESHOLDS,
    final_prevalence_metrics,
)
from master_tuned_modeling_experiments import (  # noqa: E402
    FINAL_PATHOGENIC_RATE,
    RANDOM_STATE,
    positive_probability,
)


OUT_DIR = ROOT / "results" / "modeling" / "hybrid_improvement_governance"
INNER_FP_BUFFERS = [375.0, 350.0]
MIN_SENSITIVITY = STRICT_SENSITIVITY_FLOOR
AUX_FEATURE_SETS = ["M2_compact", "M4_high_missing_ablation"]
AUX_MEMBERS = ["RF8", "XGBNW"]
RISK_BINS = {"Q2", "Q4_high_missing"}
FALLBACK_TRANSFORMS = [
    {
        "label": "Q2Q4_M2_blend50",
        "fallback_col": "M2_compact__fallback",
        "mode": "blend",
        "base_weight": 0.50,
    },
    {
        "label": "Q2Q4_M4_blend50",
        "fallback_col": "M4_high_missing_ablation__fallback",
        "mode": "blend",
        "base_weight": 0.50,
    },
    {
        "label": "Q2Q4_M2M4_blend50",
        "fallback_col": "M2M4__fallback_mean",
        "mode": "blend",
        "base_weight": 0.50,
    },
    {
        "label": "Q2Q4_M2M4_blend65",
        "fallback_col": "M2M4__fallback_mean",
        "mode": "blend",
        "base_weight": 0.65,
    },
    {
        "label": "Q2Q4_M2M4_min",
        "fallback_col": "M2M4__fallback_mean",
        "mode": "min",
        "base_weight": 0.0,
    },
]


def _threshold_metrics_with_buffer(
    y_true: np.ndarray,
    score: np.ndarray,
    f1_floor: float,
    inner_fp_target: float,
) -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = np.sort(score[y_true == 1])
    negatives = np.sort(score[y_true == 0])
    pos_n = len(positives)
    neg_n = len(negatives)
    tp_counts = pos_n - np.searchsorted(positives, THRESHOLDS, side="left")
    fp_counts = neg_n - np.searchsorted(negatives, THRESHOLDS, side="left")
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

    strict = (
        (f1 >= f1_floor)
        & (expected_fp <= inner_fp_target)
        & (tpr >= MIN_SENSITIVITY)
    )
    relaxed = (
        (f1 >= RELAXED_F1_FLOOR)
        & (expected_fp <= inner_fp_target)
        & (tpr >= MIN_SENSITIVITY)
    )
    sensitivity_guard = tpr >= MIN_SENSITIVITY

    if strict.any():
        mask = strict
        status = "strict_buffer"
    elif relaxed.any():
        mask = relaxed
        status = "relaxed_f1_buffer"
    elif sensitivity_guard.any():
        mask = sensitivity_guard
        status = "sensitivity_guard"
    else:
        mask = np.ones_like(f1, dtype=bool)
        status = "mcc_only"

    candidate_indices = np.flatnonzero(mask)
    order = np.lexsort(
        (
            expected_fp[candidate_indices],
            -tpr[candidate_indices],
            -specificity[candidate_indices],
            -f1[candidate_indices],
            -mcc[candidate_indices],
        )
    )
    idx = candidate_indices[order[0]]
    return {
        "status": status,
        "threshold": float(THRESHOLDS[idx]),
        "f1": float(f1[idx]),
        "mcc": float(mcc[idx]),
        "specificity": float(specificity[idx]),
        "sensitivity": float(tpr[idx]),
        "precision": float(precision[idx]),
        "balanced_accuracy": float((specificity[idx] + tpr[idx]) / 2.0),
        "expected_fp_per_3500": float(expected_fp[idx]),
        "expected_fn_per_3500": float(expected_fn[idx]),
        "inner_fp_target": float(inner_fp_target),
        "inner_sensitivity_floor": float(MIN_SENSITIVITY),
    }


def _selection_sort_key(metrics: dict[str, Any]) -> tuple[int, float, float, float]:
    status_rank = {
        "strict_buffer": 3,
        "relaxed_f1_buffer": 2,
        "sensitivity_guard": 1,
        "mcc_only": 0,
    }.get(str(metrics["status"]), 0)
    return (
        status_rank,
        float(metrics["mcc"]),
        float(metrics["f1"]),
        -float(metrics["expected_fp_per_3500"]),
    )


def _remember_top(
    pool: list[tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray]],
    item: tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray],
    max_size: int,
) -> None:
    pool.append(item)
    if len(pool) > max_size * 2:
        pool.sort(key=lambda value: value[0], reverse=True)
        del pool[max_size:]


def _strategy_from_selection(
    strategy_name: str,
    selection_rule: str,
    matrix_name: str,
    weights: np.ndarray,
    metrics: dict[str, Any],
    score: np.ndarray,
    y_true: np.ndarray,
    details: dict[str, Any] | None = None,
) -> SelectedStrategy:
    rank = _ranking_for_score(y_true, score)
    full_metrics = dict(metrics)
    full_metrics.update(
        {
            "inner_oof_auroc": float(rank["oof_auroc"]),
            "inner_oof_auprc": float(rank["oof_auprc"]),
            "inner_final_weighted_auprc": float(rank["final_weighted_auprc"]),
        }
    )
    nonzero = {
        BASE_TO_FULL_NAME[member]: float(weight)
        for member, weight in zip(BASE_MEMBERS, weights)
        if weight > 0
    }
    detail_payload = {
        "members": nonzero,
        "calibration_matrix": matrix_name,
        "selection_rule": selection_rule,
        "no_outer_fp_id_learning": True,
        "outer_validation_used_for_selection": False,
    }
    if details:
        detail_payload.update(details)
    return SelectedStrategy(
        strategy=strategy_name,
        method_group="Nested Hybrid-E improvement challenger",
        selected_variant=f"{matrix_name}:{selection_rule}",
        members=list(BASE_MEMBERS),
        weights=np.asarray(weights, dtype=float),
        threshold=float(full_metrics["threshold"]),
        calibration_matrix=matrix_name,
        selection_status=str(full_metrics["status"]),
        inner_metrics={k: float(v) for k, v in full_metrics.items() if k != "status"},
        details=detail_payload,
    )


def _hard_benign_weight_vectors(weight_vectors: list[np.ndarray]) -> list[np.ndarray]:
    prototypes = [
        [0.20, 0.08, 0.10, 0.10, 0.12, 0.05, 0.35],
        [0.18, 0.08, 0.07, 0.12, 0.20, 0.05, 0.30],
        [0.15, 0.05, 0.05, 0.15, 0.20, 0.10, 0.30],
        [0.25, 0.10, 0.10, 0.05, 0.15, 0.05, 0.30],
    ]
    candidates = [np.asarray(v, dtype=float) for v in weight_vectors if float(v[-1]) >= 0.20]
    candidates.extend(np.asarray(v, dtype=float) for v in prototypes)
    deduped: dict[tuple[float, ...], np.ndarray] = {}
    for vector in candidates:
        vector = vector / vector.sum()
        rounded = np.round(vector, 3)
        rounded = rounded / rounded.sum()
        deduped[tuple(np.round(rounded, 3))] = rounded
    return list(deduped.values())


def _select_buffered_hybrid_e(
    inner_frame: pd.DataFrame,
    f1_floor: float,
    weight_vectors: list[np.ndarray],
    inner_fp_target: float,
    strategy_name: str,
    selection_rule: str,
    only_hard_benign_heavy: bool = False,
) -> SelectedStrategy:
    y_true = inner_frame[TARGET_COLUMN].to_numpy(dtype=int)
    matrices = _calibrated_inner_matrices(inner_frame, list(BASE_MEMBERS))
    vectors = _hard_benign_weight_vectors(weight_vectors) if only_hard_benign_heavy else weight_vectors
    best: tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray] | None = None
    for matrix_name, matrix in matrices.items():
        for weights in vectors:
            score = matrix @ weights
            metrics = _threshold_metrics_with_buffer(
                y_true,
                score,
                f1_floor=f1_floor,
                inner_fp_target=inner_fp_target,
            )
            candidate = (_selection_sort_key(metrics), matrix_name, weights, metrics, score)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise RuntimeError(f"{strategy_name} produced no candidate.")
    _, matrix_name, weights, metrics, score = best
    return _strategy_from_selection(
        strategy_name=strategy_name,
        selection_rule=selection_rule,
        matrix_name=matrix_name,
        weights=weights,
        metrics=metrics,
        score=score,
        y_true=y_true,
        details={
            "inner_fp_buffer_target": float(inner_fp_target),
            "hard_benign_weight_floor": 0.20 if only_hard_benign_heavy else 0.0,
            "weight_vector_count": int(len(vectors)),
        },
    )


def _select_auprc_buffered_hybrid_e(
    inner_frame: pd.DataFrame,
    f1_floor: float,
    catopt_inner_metrics: dict[str, float],
    weight_vectors: list[np.ndarray],
    inner_fp_target: float,
) -> SelectedStrategy:
    y_true = inner_frame[TARGET_COLUMN].to_numpy(dtype=int)
    matrices = _calibrated_inner_matrices(inner_frame, list(BASE_MEMBERS))
    best: tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray] | None = None
    auprc_candidates: list[
        tuple[float, tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray]
    ] = []
    for matrix_name, matrix in matrices.items():
        for weights in weight_vectors:
            score = matrix @ weights
            metrics = _threshold_metrics_with_buffer(
                y_true,
                score,
                f1_floor=f1_floor,
                inner_fp_target=inner_fp_target,
            )
            sort_key = _selection_sort_key(metrics)
            candidate = (sort_key, matrix_name, weights, metrics, score)
            if best is None or sort_key > best[0]:
                best = candidate
            if metrics["status"] in {"strict_buffer", "relaxed_f1_buffer"}:
                if float(metrics["mcc"]) >= catopt_inner_metrics["mcc"] - 0.010:
                    rank = _ranking_for_score(y_true, score)
                    auprc_candidates.append(
                        (
                            float(rank["final_weighted_auprc"]),
                            sort_key,
                            matrix_name,
                            weights,
                            metrics,
                            score,
                        )
                    )
    if best is None:
        raise RuntimeError("AUPRC buffered Hybrid-E produced no candidate.")

    catopt_auprc = float(catopt_inner_metrics["inner_final_weighted_auprc"])
    eligible = [
        item for item in auprc_candidates if item[0] >= catopt_auprc - AUPRC_KEEP_TOLERANCE
    ]
    if eligible:
        eligible.sort(key=lambda value: (value[0], value[1]), reverse=True)
        _, _, matrix_name, weights, metrics, score = eligible[0]
    else:
        _, matrix_name, weights, metrics, score = best
    return _strategy_from_selection(
        strategy_name="Hybrid-E_AUPRC_FP375_nested",
        selection_rule="inner_auprc_guard_with_fp375_buffer",
        matrix_name=matrix_name,
        weights=weights,
        metrics=metrics,
        score=score,
        y_true=y_true,
        details={
            "inner_fp_buffer_target": float(inner_fp_target),
            "catopt_inner_final_weighted_auprc": catopt_auprc,
            "auprc_keep_tolerance": float(AUPRC_KEEP_TOLERANCE),
            "eligible_auprc_candidates": int(len(eligible)),
        },
    )


def _fit_score_feature_set_members(
    feature_set: str,
    members: list[str],
    fit_frame: pd.DataFrame,
    hold_frame: pd.DataFrame,
    seed: int,
    fold_key: tuple[int, int, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    processor = MasterPreprocessor(STRATEGIES[feature_set])
    x_fit = processor.fit_transform(fit_frame).to_numpy(dtype=np.float32)
    x_hold = processor.transform(hold_frame).to_numpy(dtype=np.float32)
    y_fit = fit_frame[TARGET_COLUMN].to_numpy(dtype=int)
    y_hold = hold_frame[TARGET_COLUMN].to_numpy(dtype=int)
    fold_spw = float((y_fit == 0).sum() / (y_fit == 1).sum())
    output = pd.DataFrame(
        {
            ID_COLUMN: hold_frame[ID_COLUMN].astype(str).to_numpy(),
            TARGET_COLUMN: y_hold,
        }
    )
    fit_seconds: dict[str, float] = {}
    for idx, member in enumerate(members):
        started = time.time()
        model, sample_weight, _ = _fit_one_base(
            member,
            x_fit,
            y_fit,
            seed + idx,
            fold_spw,
            fold_key=(fold_key[0], fold_key[1], fold_key[2] * 10 + idx),
        )
        if sample_weight is None:
            model.fit(x_fit, y_fit)
        else:
            model.fit(x_fit, y_fit, sample_weight=sample_weight)
        output[f"{feature_set}__{member}"] = positive_probability(model, x_hold)
        fit_seconds[f"{feature_set}__{member}"] = float(time.time() - started)
    metadata = {
        "feature_set": feature_set,
        "fit_rows": int(len(fit_frame)),
        "hold_rows": int(len(hold_frame)),
        "fit_positive_rate": float(y_fit.mean()),
        "hold_positive_rate": float(y_hold.mean()),
        "output_features": int(x_fit.shape[1]),
        "fit_seconds": fit_seconds,
    }
    return output, metadata


def _add_aux_fallback_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for feature_set in AUX_FEATURE_SETS:
        member_cols = [f"{feature_set}__{member}" for member in AUX_MEMBERS]
        frame[f"{feature_set}__fallback"] = frame[member_cols].mean(axis=1)
    frame["M2M4__fallback_mean"] = frame[
        [f"{feature_set}__fallback" for feature_set in AUX_FEATURE_SETS]
    ].mean(axis=1)
    return frame


def _make_inner_aux_oof(
    outer_train: pd.DataFrame,
    outer_seed: int,
    outer_fold: int,
    inner_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=RANDOM_STATE + outer_seed * 10 + outer_fold,
    )
    y = outer_train[TARGET_COLUMN].to_numpy(dtype=int)
    frames = []
    meta_rows = []
    for inner_fold, (fit_idx, hold_idx) in enumerate(splitter.split(outer_train, y)):
        fit_frame = outer_train.iloc[fit_idx].reset_index(drop=True)
        hold_frame = outer_train.iloc[hold_idx].reset_index(drop=True)
        merged: pd.DataFrame | None = None
        for feature_idx, feature_set in enumerate(AUX_FEATURE_SETS):
            scores, metadata = _fit_score_feature_set_members(
                feature_set,
                AUX_MEMBERS,
                fit_frame,
                hold_frame,
                seed=RANDOM_STATE
                + outer_seed * 100
                + outer_fold * 10
                + inner_fold
                + 1000 * (feature_idx + 1),
                fold_key=(outer_seed, outer_fold, inner_fold + 100 * (feature_idx + 1)),
            )
            if merged is None:
                merged = scores
            else:
                merged = merged.merge(scores, on=[ID_COLUMN, TARGET_COLUMN], how="inner")
            meta_rows.append(
                {
                    "outer_seed": int(outer_seed),
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "auxiliary": True,
                    **metadata,
                }
            )
        if merged is None:
            raise RuntimeError("Auxiliary score generation failed.")
        merged["inner_fold"] = int(inner_fold)
        frames.append(_add_aux_fallback_columns(merged))
    inner_aux = pd.concat(frames, ignore_index=True)
    return inner_aux.sort_values(ID_COLUMN).reset_index(drop=True), pd.DataFrame(meta_rows)


def _make_outer_aux_scores(
    outer_train: pd.DataFrame,
    outer_hold: pd.DataFrame,
    outer_seed: int,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged: pd.DataFrame | None = None
    meta_rows = []
    for feature_idx, feature_set in enumerate(AUX_FEATURE_SETS):
        scores, metadata = _fit_score_feature_set_members(
            feature_set,
            AUX_MEMBERS,
            outer_train,
            outer_hold,
            seed=RANDOM_STATE + outer_seed * 100 + outer_fold + 5000 * (feature_idx + 1),
            fold_key=(outer_seed, outer_fold, 900 + feature_idx),
        )
        if merged is None:
            merged = scores
        else:
            merged = merged.merge(scores, on=[ID_COLUMN, TARGET_COLUMN], how="inner")
        meta_rows.append(
            {
                "outer_seed": int(outer_seed),
                "outer_fold": int(outer_fold),
                "inner_fold": -1,
                "auxiliary": True,
                **metadata,
            }
        )
    if merged is None:
        raise RuntimeError("Outer auxiliary score generation failed.")
    return _add_aux_fallback_columns(merged), pd.DataFrame(meta_rows)


def _missing_rates(frame: pd.DataFrame) -> pd.DataFrame:
    features = feature_columns(frame)
    return pd.DataFrame(
        {
            ID_COLUMN: frame[ID_COLUMN].astype(str).to_numpy(),
            "missing_rate": frame[features].isna().mean(axis=1).to_numpy(dtype=float),
        }
    )


def _missing_cutpoints(frame: pd.DataFrame) -> tuple[float, float, float]:
    rates = _missing_rates(frame)["missing_rate"].to_numpy(dtype=float)
    return tuple(float(value) for value in np.quantile(rates, [0.25, 0.50, 0.75]))


def _assign_missingness_bin(rates: np.ndarray, cutpoints: tuple[float, float, float]) -> np.ndarray:
    q1, q2, q3 = cutpoints
    bins = np.full(len(rates), "Q4_high_missing", dtype=object)
    bins[rates <= q3] = "Q3"
    bins[rates <= q2] = "Q2"
    bins[rates <= q1] = "Q1_low_missing"
    return bins


def _attach_missingness(
    score_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    cutpoints: tuple[float, float, float],
) -> pd.DataFrame:
    rates = _missing_rates(source_frame)
    output = score_frame.merge(rates, on=ID_COLUMN, how="left")
    output["missingness_bin"] = _assign_missingness_bin(
        output["missing_rate"].to_numpy(dtype=float),
        cutpoints,
    )
    return output


def _apply_missingness_fallback_transform(
    base_score: np.ndarray,
    fallback_score: np.ndarray,
    missingness_bin: np.ndarray,
    mode: str,
    base_weight: float,
) -> np.ndarray:
    transformed = np.asarray(base_score, dtype=float).copy()
    fallback_score = np.asarray(fallback_score, dtype=float)
    risk_mask = np.isin(missingness_bin.astype(str), list(RISK_BINS))
    if mode == "blend":
        transformed[risk_mask] = (
            base_weight * transformed[risk_mask]
            + (1.0 - base_weight) * fallback_score[risk_mask]
        )
    elif mode == "min":
        transformed[risk_mask] = np.minimum(transformed[risk_mask], fallback_score[risk_mask])
    else:
        raise ValueError(f"Unknown fallback mode: {mode}")
    return np.clip(transformed, 1e-7, 1 - 1e-7)


def _select_missingness_fallback_strategy(
    inner_frame: pd.DataFrame,
    f1_floor: float,
    weight_vectors: list[np.ndarray],
    inner_fp_target: float,
    max_base_pool: int = 80,
) -> SelectedStrategy:
    y_true = inner_frame[TARGET_COLUMN].to_numpy(dtype=int)
    matrices = _calibrated_inner_matrices(inner_frame, list(BASE_MEMBERS))
    base_pool: list[
        tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray]
    ] = []
    for matrix_name, matrix in matrices.items():
        for weights in weight_vectors:
            score = matrix @ weights
            metrics = _threshold_metrics_with_buffer(
                y_true,
                score,
                f1_floor=f1_floor,
                inner_fp_target=inner_fp_target,
            )
            _remember_top(
                base_pool,
                (_selection_sort_key(metrics), matrix_name, weights, metrics, score),
                max_size=max_base_pool,
            )
    if not base_pool:
        raise RuntimeError("Missingness fallback base pool is empty.")
    base_pool.sort(key=lambda value: value[0], reverse=True)
    base_pool = base_pool[:max_base_pool]

    missingness_bin = inner_frame["missingness_bin"].astype(str).to_numpy()
    best: tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]] | None = None
    for _, matrix_name, weights, _, base_score in base_pool:
        for transform in FALLBACK_TRANSFORMS:
            fallback_score = inner_frame[str(transform["fallback_col"])].to_numpy(dtype=float)
            transformed = _apply_missingness_fallback_transform(
                base_score,
                fallback_score,
                missingness_bin,
                mode=str(transform["mode"]),
                base_weight=float(transform["base_weight"]),
            )
            metrics = _threshold_metrics_with_buffer(
                y_true,
                transformed,
                f1_floor=f1_floor,
                inner_fp_target=inner_fp_target,
            )
            key = _selection_sort_key(metrics)
            candidate = (key, matrix_name, weights, metrics, transformed, dict(transform))
            if best is None or key > best[0]:
                best = candidate
    if best is None:
        raise RuntimeError("Missingness fallback search produced no candidate.")
    _, matrix_name, weights, metrics, score, transform = best
    return _strategy_from_selection(
        strategy_name="Hybrid-E_Q2Q4_M2M4_fallback_FP375_nested",
        selection_rule="inner_q2q4_missingness_fallback_with_fp375_buffer",
        matrix_name=matrix_name,
        weights=weights,
        metrics=metrics,
        score=score,
        y_true=y_true,
        details={
            "score_transform": "missingness_fallback",
            "fallback_transform": transform,
            "risk_bins": sorted(RISK_BINS),
            "inner_fp_buffer_target": float(inner_fp_target),
            "base_pool_size": int(max_base_pool),
            "fallback_fit_feature_sets": AUX_FEATURE_SETS,
            "fallback_members": AUX_MEMBERS,
        },
    )


def _evaluate_improvement_strategy(
    strategy: SelectedStrategy,
    inner_frame: pd.DataFrame,
    outer_frame: pd.DataFrame,
    f1_floor: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    details = strategy.details or {}
    if details.get("score_transform") != "missingness_fallback":
        row, predictions = _evaluate_selected_strategy(strategy, inner_frame, outer_frame, f1_floor)
    else:
        y_true = outer_frame[TARGET_COLUMN].to_numpy(dtype=int)
        matrix = _fit_apply_calibration(
            inner_frame,
            outer_frame,
            strategy.members,
            strategy.calibration_matrix,
        )
        base_score = matrix @ strategy.weights
        transform = dict(details["fallback_transform"])
        fallback_score = outer_frame[str(transform["fallback_col"])].to_numpy(dtype=float)
        score = _apply_missingness_fallback_transform(
            base_score,
            fallback_score,
            outer_frame["missingness_bin"].astype(str).to_numpy(),
            mode=str(transform["mode"]),
            base_weight=float(transform["base_weight"]),
        )
        metrics = _outer_metrics_from_score(y_true, score, strategy.threshold, f1_floor)
        inner_metrics = strategy.inner_metrics or {}
        row = {
            "strategy": strategy.strategy,
            "method_group": strategy.method_group,
            "selected_variant": strategy.selected_variant,
            "calibration_matrix": strategy.calibration_matrix,
            "threshold": strategy.threshold,
            "selection_status": strategy.selection_status,
            "weights_json": json.dumps(
                {
                    BASE_TO_FULL_NAME[member]: float(weight)
                    for member, weight in zip(strategy.members, strategy.weights)
                    if weight > 0
                },
                ensure_ascii=False,
            ),
            "details_json": json.dumps(details, ensure_ascii=False),
            **{f"inner_{key}": value for key, value in inner_metrics.items()},
            **metrics,
        }
        predictions = pd.DataFrame(
            {
                ID_COLUMN: outer_frame[ID_COLUMN].astype(str).to_numpy(),
                TARGET_COLUMN: y_true,
                "score": score,
                "threshold": strategy.threshold,
                "prediction": (score >= strategy.threshold).astype(int),
                "strategy": strategy.strategy,
                "method_group": strategy.method_group,
                "selected_variant": strategy.selected_variant,
            }
        )
    predictions = predictions.merge(
        outer_frame[[ID_COLUMN, "missing_rate", "missingness_bin"]],
        on=ID_COLUMN,
        how="left",
    )
    return row, predictions


def _outer_metrics_from_score(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    f1_floor: float,
) -> dict[str, float | int | bool]:
    final_metrics = final_prevalence_metrics(
        y_true,
        score,
        threshold=threshold,
        pathogenic_rate=FINAL_PATHOGENIC_RATE,
    )
    y_pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ranked = _ranking_for_score(y_true, score)
    balanced_accuracy = (
        final_metrics["final_specificity"] + final_metrics["final_sensitivity"]
    ) / 2.0
    strict_pass = (
        final_metrics["final_f1"] >= f1_floor
        and final_metrics["expected_fp_per_3500"] <= FP_TARGET
        and final_metrics["final_sensitivity"] >= STRICT_SENSITIVITY_FLOOR
    )
    return {
        "outer_f1": float(final_metrics["final_f1"]),
        "outer_mcc": float(final_metrics["final_mcc"]),
        "outer_specificity": float(final_metrics["final_specificity"]),
        "outer_sensitivity": float(final_metrics["final_sensitivity"]),
        "outer_precision": float(final_metrics["final_precision"]),
        "outer_balanced_accuracy": float(balanced_accuracy),
        "outer_expected_fp_per_3500": float(final_metrics["expected_fp_per_3500"]),
        "outer_expected_fn_per_3500": float(final_metrics["expected_fn_per_3500"]),
        "outer_oof_auroc": float(ranked["oof_auroc"]),
        "outer_oof_auprc": float(ranked["oof_auprc"]),
        "outer_final_weighted_auprc": float(ranked["final_weighted_auprc"]),
        "outer_actual_tp": int(tp),
        "outer_actual_tn": int(tn),
        "outer_actual_fp": int(fp),
        "outer_actual_fn": int(fn),
        "outer_strict_pass": bool(strict_pass),
    }


def _select_improvement_strategies(
    inner_frame: pd.DataFrame,
    f1_floor: float,
    catopt_inner_metrics: dict[str, float],
    weight_vectors: list[np.ndarray],
    max_fallback_base_pool: int,
) -> list[SelectedStrategy]:
    strategies = [
        _select_buffered_hybrid_e(
            inner_frame,
            f1_floor,
            weight_vectors,
            inner_fp_target=375.0,
            strategy_name="Hybrid-E_FP375_buffer_nested",
            selection_rule="inner_best_mcc_with_fp375_buffer",
        ),
        _select_buffered_hybrid_e(
            inner_frame,
            f1_floor,
            weight_vectors,
            inner_fp_target=350.0,
            strategy_name="Hybrid-E_FP350_buffer_nested",
            selection_rule="inner_best_mcc_with_fp350_buffer",
        ),
        _select_buffered_hybrid_e(
            inner_frame,
            f1_floor,
            weight_vectors,
            inner_fp_target=375.0,
            strategy_name="Hybrid-E_hardBenign_FP375_nested",
            selection_rule="inner_hard_benign_heavy_with_fp375_buffer",
            only_hard_benign_heavy=True,
        ),
        _select_auprc_buffered_hybrid_e(
            inner_frame,
            f1_floor,
            catopt_inner_metrics,
            weight_vectors,
            inner_fp_target=375.0,
        ),
        _select_missingness_fallback_strategy(
            inner_frame,
            f1_floor,
            weight_vectors,
            inner_fp_target=375.0,
            max_base_pool=max_fallback_base_pool,
        ),
    ]
    return strategies


def _summarize_outer_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "threshold",
        "outer_f1",
        "outer_mcc",
        "outer_specificity",
        "outer_sensitivity",
        "outer_precision",
        "outer_balanced_accuracy",
        "outer_expected_fp_per_3500",
        "outer_expected_fn_per_3500",
        "outer_oof_auroc",
        "outer_oof_auprc",
        "outer_final_weighted_auprc",
    ]
    rows = []
    for strategy, group in results.groupby("strategy", sort=False):
        row: dict[str, Any] = {
            "strategy": strategy,
            "method_group": str(group["method_group"].iloc[0]),
            "outer_folds": int(len(group)),
            "strict_pass_rate": float(group["outer_strict_pass"].mean()),
            "selection_strict_rate": float(
                group["selection_status"].isin(["strict", "strict_buffer"]).mean()
            ),
        }
        for column in metric_cols:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        [
            "strict_pass_rate",
            "outer_mcc_mean",
            "outer_f1_mean",
            "outer_expected_fp_per_3500_mean",
        ],
        ascending=[False, False, False, True],
    )


def _final_prior_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    positives = y_true == 1
    negatives = y_true == 0
    tp = int(((y_pred == 1) & positives).sum())
    tn = int(((y_pred == 0) & negatives).sum())
    fp = int(((y_pred == 1) & negatives).sum())
    fn = int(((y_pred == 0) & positives).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = 1.0 - fpr
    pi = FINAL_PATHOGENIC_RATE
    final_tp = pi * tpr
    final_fn = pi * (1.0 - tpr)
    final_fp = (1.0 - pi) * fpr
    final_tn = (1.0 - pi) * specificity
    precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) else 0.0
    f1 = (
        2.0 * precision * tpr / (precision + tpr)
        if (precision + tpr)
        else 0.0
    )
    denom = (final_tp + final_fp) * (final_tp + final_fn) * (final_tn + final_fp) * (
        final_tn + final_fn
    )
    mcc = (
        (final_tp * final_tn - final_fp * final_fn) / np.sqrt(denom)
        if denom > 0
        else 0.0
    )
    return {
        "n_rows": int(len(y_true)),
        "positives": int(positives.sum()),
        "negatives": int(negatives.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": float(tpr),
        "specificity": float(specificity),
        "precision_final_prior": float(precision),
        "f1_final_prior": float(f1),
        "mcc_final_prior": float(mcc),
        "fp_rate": float(fpr),
    }


def _missingness_subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, missing_bin), group in predictions.groupby(
        ["strategy", "missingness_bin"],
        observed=True,
    ):
        metrics = _final_prior_metrics_from_predictions(
            group[TARGET_COLUMN].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
        )
        rows.append(
            {
                "strategy": strategy,
                "missingness_bin": str(missing_bin),
                "mean_missing_rate": float(group["missing_rate"].mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy", "missingness_bin"])


def _paired_delta(results: pd.DataFrame, reference_strategy: str) -> pd.DataFrame:
    key = ["outer_seed", "outer_fold"]
    ref = results.loc[results["strategy"].eq(reference_strategy)].set_index(key).sort_index()
    rows = []
    for strategy in sorted(set(results["strategy"]) - {reference_strategy}):
        group = results.loc[results["strategy"].eq(strategy)].set_index(key).sort_index()
        group = group.loc[ref.index]
        f1_delta = group["outer_f1"] - ref["outer_f1"]
        mcc_delta = group["outer_mcc"] - ref["outer_mcc"]
        fp_delta = group["outer_expected_fp_per_3500"] - ref["outer_expected_fp_per_3500"]
        auprc_delta = group["outer_final_weighted_auprc"] - ref["outer_final_weighted_auprc"]
        rows.append(
            {
                "reference_strategy": reference_strategy,
                "strategy": strategy,
                "delta_f1_mean": float(f1_delta.mean()),
                "delta_mcc_mean": float(mcc_delta.mean()),
                "delta_fp_per_3500_mean": float(fp_delta.mean()),
                "delta_final_auprc_mean": float(auprc_delta.mean()),
                "f1_win_rate": float((f1_delta > 0).mean()),
                "mcc_win_rate": float((mcc_delta > 0).mean()),
                "fp_le_reference_rate": float((fp_delta <= 0).mean()),
                "all_three_win_rate": float(
                    ((f1_delta >= 0) & (mcc_delta >= 0) & (fp_delta <= 0)).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["delta_mcc_mean", "delta_f1_mean"],
        ascending=[False, False],
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    table = frame.loc[:, columns].copy()
    for column in table.select_dtypes(include=[np.number]).columns:
        table[column] = table[column].map(lambda value: f"{value:.{digits}f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return lines


def _write_report(
    summary: pd.DataFrame,
    deltas_current: pd.DataFrame,
    deltas_catopt: pd.DataFrame,
    missingness: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    best = summary.iloc[0]
    selected_rows = summary.loc[
        summary["strategy"].isin(
            [
                "Hybrid-E_bestMCC_nested",
                "Hybrid-E_AUPRC_guard_nested",
                "Hybrid-E_lowFP_nested",
                "Hybrid-E_FP375_buffer_nested",
                "Hybrid-E_FP350_buffer_nested",
                "Hybrid-E_hardBenign_FP375_nested",
                "Hybrid-E_Q2Q4_M2M4_fallback_FP375_nested",
                "Hybrid-E_AUPRC_FP375_nested",
            ]
        )
    ]
    best_missingness = missingness.loc[missingness["strategy"].eq(best["strategy"])]
    lines = [
        "# MASTER Hybrid-E Geliştirme Nested Governance Raporu",
        "",
        "Bu rapor, kilitli Hybrid-E final adayını bozmadan denenebilecek geliştirme yollarını nested governance altında ölçer.",
        "",
        "## Şartname / Governance Sınırı",
        "",
        "- Kullanılan veri: yalnız sağlanan MASTER etiketli eğitim tablosu.",
        "- Final/test veri kullanılmadı.",
        "- Dış veri, genomic adres çözme/tersine arama veya yarışma dışı label lookup kullanılmadı.",
        "- Preprocessing her inner/outer fold içinde yeniden fit edildi.",
        "- Ağırlık, kalibrasyon, threshold ve fallback politikası yalnız inner OOF üzerinde seçildi.",
        "- Outer holdout sadece seçilmiş kararın tek seferlik değerlendirmesinde kullanıldı.",
        "- Tekrarlayan outer FP ID listesi model seçimi veya eğitim ağırlığı için kullanılmadı.",
        "",
        "## Denenen Adımlar",
        "",
        "1. FP tamponu: inner seçimde 400 yerine 375 ve 350 FP/3500 hedefleri denendi; sensitivity tabanı 0.62 olarak korundu.",
        "2. Missingness guard: Q2/Q4 missingness binlerinde M2/M4 fallback skorları yalnız inner-CV seçilen bir skor dönüşümü olarak denendi.",
        "3. Hard-benign güçlendirme: mevcut HBRF bileşeninin ağırlığı artırılmış aday vektörleri denendi; outer FP ID öğrenmesi yapılmadı.",
        "4. AUPRC guard: mevcut AUPRC koruması ve FP375 tamponlu AUPRC challenger birlikte raporlandı.",
        "5. TabPFN final modele dahil edilmedi.",
        "",
        "## Kısa Karar",
        "",
        f"Özet sıralamada en yüksek aday: **{best['strategy']}**. "
        f"Strict pass={best['strict_pass_rate']:.2f}, F1={best['outer_f1_mean']:.4f}, "
        f"MCC={best['outer_mcc_mean']:.4f}, FP/3500={best['outer_expected_fp_per_3500_mean']:.1f}.",
        "",
        "## Final Uygulama Kararı",
        "",
        "Mevcut `Hybrid-E_bestMCC_nested` freeze korunmalı. Denenen challenger'ların hiçbiri mevcut adayı aynı anda F1, MCC, strict pass ve FP kuyruğu açısından geçmedi.",
        "",
        "- `Hybrid-E_lowFP_nested` ortalama FP'yi düşürdü; ancak F1, MCC ve strict pass geriledi.",
        "- `Hybrid-E_AUPRC_guard_nested` final weighted AUPRC'yi artırdı; fakat F1/MCC tarafında mevcut BestMCC'nin altında kaldı.",
        "- `Hybrid-E_Q2Q4_M2M4_fallback_FP375_nested` AUPRC ve bazı missingness risklerini iyileştirme sinyali verdi; ama F1/MCC ve strict pass yeterli değil.",
        "- FP350/FP375 buffer ve hard-benign-heavy challenger'lar kötü fold FP kuyruğunu güvenilir biçimde kesemedi.",
        "",
        "## Ortalama Outer Fold Sonuçları",
        "",
        *_markdown_table(
            selected_rows,
            [
                "strategy",
                "outer_folds",
                "strict_pass_rate",
                "outer_f1_mean",
                "outer_mcc_mean",
                "outer_expected_fp_per_3500_mean",
                "outer_expected_fp_per_3500_max",
                "outer_sensitivity_mean",
                "outer_specificity_mean",
                "outer_final_weighted_auprc_mean",
            ],
            digits=4,
        ),
        "",
        "## Mevcut Hybrid-E BestMCC'ye Göre Fark",
        "",
        *_markdown_table(
            deltas_current,
            [
                "strategy",
                "delta_f1_mean",
                "delta_mcc_mean",
                "delta_fp_per_3500_mean",
                "delta_final_auprc_mean",
                "f1_win_rate",
                "mcc_win_rate",
                "fp_le_reference_rate",
            ],
            digits=4,
        ),
        "",
        "## CATOPT-A Referansına Göre Fark",
        "",
        *_markdown_table(
            deltas_catopt,
            [
                "strategy",
                "delta_f1_mean",
                "delta_mcc_mean",
                "delta_fp_per_3500_mean",
                "delta_final_auprc_mean",
                "f1_win_rate",
                "mcc_win_rate",
                "fp_le_reference_rate",
            ],
            digits=4,
        ),
        "",
        "## En İyi Aday Missingness Dağılımı",
        "",
        *_markdown_table(
            best_missingness,
            [
                "missingness_bin",
                "mean_missing_rate",
                "f1_final_prior",
                "mcc_final_prior",
                "specificity",
                "sensitivity",
                "fp_rate",
            ],
            digits=4,
        ),
        "",
        "## Yorum",
        "",
        "- Bir challenger mevcut Hybrid-E'yi hem F1/MCC hem FP kuyruğu bakımından geçmiyorsa final freeze bozulmamalı.",
        "- FP tamponu ortalama FP'yi düşürüp F1/MCC'yi belirgin azaltıyorsa sadece risk raporunda tutulmalı.",
        "- Q2/Q4 fallback, missingness shortcut riskini azaltırsa bile yalnız bu nested sonuçla freeze adayı olabilir.",
        "- Hard-benign ağırlığı güçlenmiş adaylar outer FP ID'lerinden öğrenmediği için yöntemsel olarak savunulabilir; karar tamamen bu rapordaki nested metriklere bağlıdır.",
        "",
        "## Çıktılar",
        "",
        "- `results/modeling/hybrid_improvement_governance/improvement_outer_fold_results.csv`",
        "- `results/modeling/hybrid_improvement_governance/improvement_outer_predictions.csv`",
        "- `results/modeling/hybrid_improvement_governance/improvement_selection_details.csv`",
        "- `results/modeling/hybrid_improvement_governance/improvement_summary.csv`",
        "- `results/modeling/hybrid_improvement_governance/improvement_missingness_subgroup_summary.csv`",
        "- `results/modeling/hybrid_improvement_governance/improvement_delta_vs_current.csv`",
        "- `results/modeling/hybrid_improvement_governance/improvement_delta_vs_catopt.csv`",
        "- `results/modeling/hybrid_improvement_governance/improvement_governance_metadata.json`",
        "",
        "## Run Metadata",
        "",
        f"- Outer seedler: `{metadata['outer_seeds']}`",
        f"- Outer fold sayısı: `{metadata['outer_fold_count']}`",
        f"- Inner fold sayısı: `{metadata['inner_splits']}`",
        f"- Hybrid-E weight vector sayısı: `{metadata['hybrid_e_weight_vector_count']}`",
        f"- Süre saniye: `{metadata['elapsed_seconds']:.1f}`",
    ]
    (OUT_DIR / "HYBRID_E_GELISTIRME_NESTED_RAPORU.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-seeds", default="all")
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--hye-weight-samples", type=int, default=2500)
    parser.add_argument("--max-fallback-base-pool", type=int, default=80)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outer_seeds = _parse_outer_seeds(args.outer_seeds)
    development = _load_development_frame()
    folds = pd.read_csv(FOLD_FILE, dtype={ID_COLUMN: "string"})
    folds[ID_COLUMN] = folds[ID_COLUMN].astype(str)
    h_c_grid = _hybrid_c_weight_grid()
    weight_vectors = _hybrid_e_weight_vectors(args.hye_weight_samples)

    outer_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fit_audit_rows: list[pd.DataFrame] = []

    print(
        f"Prepared Hybrid-E improvement governance: vectors={len(weight_vectors)}, "
        f"outer_seeds={outer_seeds}",
        flush=True,
    )

    for outer_seed in outer_seeds:
        seed_folds = folds.loc[folds["repeat_seed"].eq(outer_seed)].copy()
        for outer_fold in sorted(seed_folds["fold"].unique()):
            fold_started = time.time()
            hold_ids = set(
                seed_folds.loc[seed_folds["fold"].eq(outer_fold), ID_COLUMN].astype(str)
            )
            outer_train = development.loc[
                ~development[ID_COLUMN].astype(str).isin(hold_ids)
            ].reset_index(drop=True)
            outer_hold = development.loc[
                development[ID_COLUMN].astype(str).isin(hold_ids)
            ].reset_index(drop=True)
            cutpoints = _missing_cutpoints(outer_train)

            print(
                f"[outer seed={outer_seed} fold={outer_fold}] "
                f"train={len(outer_train)} holdout={len(outer_hold)}",
                flush=True,
            )
            inner_frame, inner_meta = _make_inner_oof(
                outer_train,
                outer_seed=int(outer_seed),
                outer_fold=int(outer_fold),
                inner_splits=args.inner_splits,
            )
            inner_aux, inner_aux_meta = _make_inner_aux_oof(
                outer_train,
                outer_seed=int(outer_seed),
                outer_fold=int(outer_fold),
                inner_splits=args.inner_splits,
            )
            inner_frame = inner_frame.merge(
                inner_aux.drop(columns=["inner_fold"]),
                on=[ID_COLUMN, TARGET_COLUMN],
                how="inner",
            )
            inner_frame = _attach_missingness(inner_frame, outer_train, cutpoints)
            fit_audit_rows.extend([inner_meta, inner_aux_meta])

            baseline_strategies, f1_floor = _select_all_strategies(
                inner_frame,
                h_c_grid=h_c_grid,
                h_e_vectors=weight_vectors,
            )
            catopt = next(
                strategy
                for strategy in baseline_strategies
                if strategy.strategy == "CATOPT-A_nested_reference"
            )
            catopt_inner_metrics = catopt.inner_metrics or {}
            improvement_strategies = _select_improvement_strategies(
                inner_frame,
                f1_floor,
                catopt_inner_metrics,
                weight_vectors,
                max_fallback_base_pool=int(args.max_fallback_base_pool),
            )
            strategies = [
                strategy
                for strategy in baseline_strategies
                if strategy.strategy
                in {
                    "CATOPT-A_nested_reference",
                    "Hybrid-E_bestMCC_nested",
                    "Hybrid-E_AUPRC_guard_nested",
                    "Hybrid-E_lowFP_nested",
                }
            ]
            strategies.extend(improvement_strategies)

            outer_score_frame, outer_meta = _fit_score_base_frame(
                outer_train,
                outer_hold,
                seed=RANDOM_STATE + int(outer_seed) * 100 + int(outer_fold),
                fold_key=(int(outer_seed), int(outer_fold), 99),
            )
            outer_aux, outer_aux_meta = _make_outer_aux_scores(
                outer_train,
                outer_hold,
                outer_seed=int(outer_seed),
                outer_fold=int(outer_fold),
            )
            outer_score_frame = outer_score_frame.merge(
                outer_aux,
                on=[ID_COLUMN, TARGET_COLUMN],
                how="inner",
            )
            outer_score_frame = _attach_missingness(outer_score_frame, outer_hold, cutpoints)
            fit_audit_rows.append(outer_aux_meta)

            for strategy in strategies:
                row, predictions = _evaluate_improvement_strategy(
                    strategy,
                    inner_frame,
                    outer_score_frame,
                    f1_floor,
                )
                row.update(
                    {
                        "outer_seed": int(outer_seed),
                        "outer_fold": int(outer_fold),
                        "outer_train_rows": int(len(outer_train)),
                        "outer_hold_rows": int(len(outer_hold)),
                        "selection_f1_floor": float(f1_floor),
                        "outer_output_features": int(outer_meta["output_features"]),
                    }
                )
                outer_rows.append(row)
                predictions["outer_seed"] = int(outer_seed)
                predictions["outer_fold"] = int(outer_fold)
                prediction_frames.append(predictions)
                selection_rows.append(
                    {
                        "outer_seed": int(outer_seed),
                        "outer_fold": int(outer_fold),
                        "strategy": strategy.strategy,
                        "selected_variant": strategy.selected_variant,
                        "selection_status": strategy.selection_status,
                        "threshold": strategy.threshold,
                        "selection_f1_floor": float(f1_floor),
                        "calibration_matrix": strategy.calibration_matrix,
                        "weights_json": json.dumps(
                            {
                                BASE_TO_FULL_NAME[member]: float(weight)
                                for member, weight in zip(strategy.members, strategy.weights)
                                if weight > 0
                            },
                            ensure_ascii=False,
                        ),
                        "details_json": json.dumps(strategy.details or {}, ensure_ascii=False),
                    }
                )
            print(
                f"[outer seed={outer_seed} fold={outer_fold}] "
                f"completed in {time.time() - fold_started:.1f}s",
                flush=True,
            )

    outer_results = pd.DataFrame(outer_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    fit_audit = pd.concat(fit_audit_rows, ignore_index=True)
    summary = _summarize_outer_results(outer_results)
    missingness = _missingness_subgroups(predictions)
    deltas_current = _paired_delta(outer_results, "Hybrid-E_bestMCC_nested")
    deltas_catopt = _paired_delta(outer_results, "CATOPT-A_nested_reference")

    outer_results.to_csv(OUT_DIR / "improvement_outer_fold_results.csv", index=False)
    predictions.to_csv(OUT_DIR / "improvement_outer_predictions.csv", index=False)
    selections.to_csv(OUT_DIR / "improvement_selection_details.csv", index=False)
    fit_audit.to_csv(OUT_DIR / "improvement_inner_fit_audit.csv", index=False)
    summary.to_csv(OUT_DIR / "improvement_summary.csv", index=False)
    missingness.to_csv(OUT_DIR / "improvement_missingness_subgroup_summary.csv", index=False)
    deltas_current.to_csv(OUT_DIR / "improvement_delta_vs_current.csv", index=False)
    deltas_catopt.to_csv(OUT_DIR / "improvement_delta_vs_catopt.csv", index=False)

    metadata = {
        "status": "completed",
        "outer_seeds": outer_seeds,
        "outer_fold_count": int(
            folds.loc[folds["repeat_seed"].isin(outer_seeds), ["repeat_seed", "fold"]]
            .drop_duplicates()
            .shape[0]
        ),
        "inner_splits": int(args.inner_splits),
        "hye_weight_samples_requested": int(args.hye_weight_samples),
        "hybrid_e_weight_vector_count": int(len(weight_vectors)),
        "max_fallback_base_pool": int(args.max_fallback_base_pool),
        "feature_set_primary": FEATURE_SET,
        "auxiliary_feature_sets": AUX_FEATURE_SETS,
        "auxiliary_members": AUX_MEMBERS,
        "inner_fp_buffers": INNER_FP_BUFFERS,
        "minimum_sensitivity_floor": float(MIN_SENSITIVITY),
        "official_fp_target_for_outer_strict_pass": float(FP_TARGET),
        "final_pathogenic_rate_assumption": float(FINAL_PATHOGENIC_RATE),
        "final_or_test_data_used": False,
        "external_data_used": False,
        "genomic_lookup_used": False,
        "outer_fp_ids_used_for_training_or_selection": False,
        "outer_validation_used_for_selection": False,
        "frozen_final_artifact_modified": False,
        "elapsed_seconds": float(time.time() - started),
        "outputs": {
            "outer_results": str(
                (OUT_DIR / "improvement_outer_fold_results.csv").relative_to(ROOT)
            ),
            "predictions": str(
                (OUT_DIR / "improvement_outer_predictions.csv").relative_to(ROOT)
            ),
            "selections": str(
                (OUT_DIR / "improvement_selection_details.csv").relative_to(ROOT)
            ),
            "fit_audit": str(
                (OUT_DIR / "improvement_inner_fit_audit.csv").relative_to(ROOT)
            ),
            "summary": str((OUT_DIR / "improvement_summary.csv").relative_to(ROOT)),
            "missingness": str(
                (OUT_DIR / "improvement_missingness_subgroup_summary.csv").relative_to(ROOT)
            ),
            "delta_vs_current": str(
                (OUT_DIR / "improvement_delta_vs_current.csv").relative_to(ROOT)
            ),
            "delta_vs_catopt": str(
                (OUT_DIR / "improvement_delta_vs_catopt.csv").relative_to(ROOT)
            ),
            "report": str((OUT_DIR / "HYBRID_E_GELISTIRME_NESTED_RAPORU.md").relative_to(ROOT)),
        },
    }
    _write_report(summary, deltas_current, deltas_catopt, missingness, metadata)
    (OUT_DIR / "improvement_governance_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display_cols = [
        "strategy",
        "outer_folds",
        "strict_pass_rate",
        "outer_f1_mean",
        "outer_mcc_mean",
        "outer_expected_fp_per_3500_mean",
        "outer_expected_fp_per_3500_max",
        "outer_sensitivity_mean",
        "outer_specificity_mean",
        "outer_final_weighted_auprc_mean",
    ]
    print("\n=== HYBRID-E IMPROVEMENT GOVERNANCE SUMMARY ===")
    print(summary.loc[:, display_cols].round(4).to_string(index=False))
    return metadata


if __name__ == "__main__":
    main()
