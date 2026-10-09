"""Targeted MASTER modeling experiments for MCC and false-positive control.

This is a model-development script. It does not use the outer-validation
partition for model/threshold selection and it does not add any external data.
All model fits reuse the approved MASTER raw table and the fold-safe
``MasterPreprocessor`` pipeline.
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import ID_COLUMN, TARGET_COLUMN  # noqa: E402
from master_tuned_modeling_experiments import (  # noqa: E402
    FINAL_PATHOGENIC_RATE,
    RANDOM_STATE,
    final_prevalence_metrics,
    metric_row,
    positive_probability,
    prepare_fold_cache,
    rf_params,
    xgb_params,
)


TUNED_DIR = ROOT / "results" / "modeling" / "tuned"
OUT_DIR = ROOT / "results" / "modeling" / "targeted_mcc_fp"
FEATURE_SET = "M3_missing_aware_compact"
FINAL_TOTAL_N = 3500
FINAL_BENIGN_N = 3000
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.001), 3)
F1_KEEP_TOLERANCE = 0.001
FP_TARGET = 400.0
STRICT_SENSITIVITY_FLOOR = 0.62
RELAXED_F1_FLOOR = 0.530
RELAXED_SENSITIVITY_FLOOR = 0.60


@dataclass(frozen=True)
class TargetModelCandidate:
    name: str
    method_group: str
    family: str
    params: dict[str, Any]
    prior_strength: float = 0.0
    hard_top_fraction: float = 0.0
    hard_multiplier: float = 1.0


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def prior_correct_probability(
    probability: np.ndarray,
    source_positive_rate: float,
    target_positive_rate: float = FINAL_PATHOGENIC_RATE,
) -> np.ndarray:
    shift = float(_logit(target_positive_rate) - _logit(source_positive_rate))
    return _sigmoid(_logit(probability) + shift)


def final_metric_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    positives = np.sort(y_score[y_true == 1])
    negatives = np.sort(y_score[y_true == 0])
    pos_n = len(positives)
    neg_n = len(negatives)
    tp_counts = pos_n - np.searchsorted(positives, thresholds, side="left")
    fp_counts = neg_n - np.searchsorted(negatives, thresholds, side="left")
    tpr = tp_counts / pos_n if pos_n else np.zeros_like(thresholds, dtype=float)
    fpr = fp_counts / neg_n if neg_n else np.zeros_like(thresholds, dtype=float)

    pi = FINAL_PATHOGENIC_RATE
    tp = pi * tpr
    fn = pi * (1.0 - tpr)
    fp = (1.0 - pi) * fpr
    tn = (1.0 - pi) * (1.0 - fpr)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    final_f1 = np.divide(
        2 * precision * tpr,
        precision + tpr,
        out=np.zeros_like(tp),
        where=(precision + tpr) > 0,
    )
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    final_mcc = np.divide(
        tp * tn - fp * fn,
        np.sqrt(denom),
        out=np.zeros_like(tp),
        where=denom > 0,
    )
    return pd.DataFrame(
        {
            "threshold": thresholds.astype(float),
            "final_f1": final_f1.astype(float),
            "final_mcc": final_mcc.astype(float),
            "final_specificity": (1.0 - fpr).astype(float),
            "final_sensitivity": tpr.astype(float),
            "final_precision": precision.astype(float),
            "expected_fp_per_3500": (FINAL_TOTAL_N * fp).astype(float),
            "expected_fn_per_3500": (FINAL_TOTAL_N * fn).astype(float),
        }
    )


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {
            "oof_auroc": float("nan"),
            "oof_auprc": float("nan"),
            "final_weighted_auroc": float("nan"),
            "final_weighted_auprc": float("nan"),
        }

    source_pi = float(y_true.mean())
    sample_weight = np.where(
        y_true == 1,
        FINAL_PATHOGENIC_RATE / source_pi,
        (1.0 - FINAL_PATHOGENIC_RATE) / (1.0 - source_pi),
    )
    return {
        "oof_auroc": float(roc_auc_score(y_true, y_score)),
        "oof_auprc": float(average_precision_score(y_true, y_score)),
        "final_weighted_auroc": float(
            roc_auc_score(y_true, y_score, sample_weight=sample_weight)
        ),
        "final_weighted_auprc": float(
            average_precision_score(y_true, y_score, sample_weight=sample_weight)
        ),
    }


def _select_from_thresholds(
    table: pd.DataFrame,
    f1_floor: float,
    fp_max: float | None = None,
    specificity_min: float | None = None,
    sensitivity_min: float | None = None,
) -> tuple[pd.Series, str]:
    eligible = table.loc[table["final_f1"].ge(f1_floor)].copy()
    if fp_max is not None:
        eligible = eligible.loc[eligible["expected_fp_per_3500"].le(fp_max)]
    if specificity_min is not None:
        eligible = eligible.loc[eligible["final_specificity"].ge(specificity_min)]
    if sensitivity_min is not None:
        eligible = eligible.loc[eligible["final_sensitivity"].ge(sensitivity_min)]
    if len(eligible):
        selected = eligible.sort_values(
            ["final_mcc", "final_f1", "final_specificity", "final_sensitivity"],
            ascending=False,
        ).iloc[0]
        return selected, "strict"

    relaxed = table.loc[
        table["final_f1"].ge(RELAXED_F1_FLOOR)
        & table["final_sensitivity"].ge(RELAXED_SENSITIVITY_FLOOR)
    ].copy()
    if fp_max is not None:
        relaxed = relaxed.loc[relaxed["expected_fp_per_3500"].le(fp_max)]
    if specificity_min is not None:
        relaxed = relaxed.loc[relaxed["final_specificity"].ge(specificity_min)]
    if len(relaxed):
        selected = relaxed.sort_values(
            ["final_mcc", "final_f1", "final_specificity", "final_sensitivity"],
            ascending=False,
        ).iloc[0]
        return selected, "relaxed"

    selected = table.sort_values(
        ["final_mcc", "final_f1", "final_specificity"], ascending=False
    ).iloc[0]
    return selected, "mcc_only"


def summarize_score(
    candidate: str,
    method_group: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    f1_floor: float,
    details: str = "",
) -> dict[str, Any]:
    table = final_metric_table(y_true, y_score)
    rank_metrics = ranking_metrics(y_true, y_score)
    best_f1 = table.sort_values(
        ["final_f1", "final_mcc", "final_specificity"], ascending=False
    ).iloc[0]
    best_mcc = table.sort_values(
        ["final_mcc", "final_f1", "final_specificity"], ascending=False
    ).iloc[0]
    targeted, status = _select_from_thresholds(
        table,
        f1_floor=f1_floor,
        fp_max=FP_TARGET,
        sensitivity_min=STRICT_SENSITIVITY_FLOOR,
    )
    spec88, spec88_status = _select_from_thresholds(
        table,
        f1_floor=f1_floor,
        specificity_min=0.88,
        sensitivity_min=STRICT_SENSITIVITY_FLOOR,
    )
    default_metrics = final_prevalence_metrics(
        y_true, y_score, threshold=0.50, pathogenic_rate=FINAL_PATHOGENIC_RATE
    )
    return {
        "candidate": candidate,
        "method_group": method_group,
        "details": details,
        "n": int(len(y_true)),
        **rank_metrics,
        "f1_floor_used": float(f1_floor),
        "target_status": status,
        "target_threshold": float(targeted["threshold"]),
        "target_f1": float(targeted["final_f1"]),
        "target_mcc": float(targeted["final_mcc"]),
        "target_specificity": float(targeted["final_specificity"]),
        "target_sensitivity": float(targeted["final_sensitivity"]),
        "target_precision": float(targeted["final_precision"]),
        "target_expected_fp_per_3500": float(targeted["expected_fp_per_3500"]),
        "target_expected_fn_per_3500": float(targeted["expected_fn_per_3500"]),
        "spec88_status": spec88_status,
        "spec88_threshold": float(spec88["threshold"]),
        "spec88_f1": float(spec88["final_f1"]),
        "spec88_mcc": float(spec88["final_mcc"]),
        "spec88_specificity": float(spec88["final_specificity"]),
        "spec88_sensitivity": float(spec88["final_sensitivity"]),
        "spec88_precision": float(spec88["final_precision"]),
        "spec88_expected_fp_per_3500": float(spec88["expected_fp_per_3500"]),
        "spec88_expected_fn_per_3500": float(spec88["expected_fn_per_3500"]),
        "best_f1_threshold": float(best_f1["threshold"]),
        "best_f1": float(best_f1["final_f1"]),
        "best_f1_mcc": float(best_f1["final_mcc"]),
        "best_f1_expected_fp_per_3500": float(best_f1["expected_fp_per_3500"]),
        "best_f1_expected_fn_per_3500": float(best_f1["expected_fn_per_3500"]),
        "best_mcc_threshold": float(best_mcc["threshold"]),
        "best_mcc_f1": float(best_mcc["final_f1"]),
        "best_mcc": float(best_mcc["final_mcc"]),
        "best_mcc_expected_fp_per_3500": float(best_mcc["expected_fp_per_3500"]),
        "best_mcc_expected_fn_per_3500": float(best_mcc["expected_fn_per_3500"]),
        "default_f1": float(default_metrics["final_f1"]),
        "default_mcc": float(default_metrics["final_mcc"]),
        "default_expected_fp_per_3500": float(default_metrics["expected_fp_per_3500"]),
        "default_expected_fn_per_3500": float(default_metrics["expected_fn_per_3500"]),
    }


def load_tuned_oof() -> dict[str, pd.DataFrame]:
    frames = []
    for path in [
        TUNED_DIR / "master_tuned_oof_predictions.csv",
        TUNED_DIR / "master_tuned_ensemble_oof_predictions.csv",
    ]:
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path, dtype={ID_COLUMN: "string"}))
    if not frames:
        raise FileNotFoundError("No tuned OOF files found. Run master_tuned_modeling_experiments.py first.")
    data = pd.concat(frames, ignore_index=True)
    result: dict[str, pd.DataFrame] = {}
    for candidate, group in data.groupby("candidate"):
        avg = (
            group.groupby([ID_COLUMN, TARGET_COLUMN], as_index=False)["score"]
            .mean()
            .sort_values(ID_COLUMN)
            .reset_index(drop=True)
        )
        result[str(candidate)] = avg
    return result


def method1_constrained_thresholds(
    averaged_oof: dict[str, pd.DataFrame],
    f1_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    grid_rows = []
    for candidate, table in averaged_oof.items():
        y = table[TARGET_COLUMN].to_numpy(dtype=int)
        score = table["score"].to_numpy(dtype=float)
        metrics = final_metric_table(y, score)
        keep = metrics.copy()
        keep.insert(0, "candidate", candidate)
        grid_rows.append(keep)
        rows.append(summarize_score(candidate, "MCC + FP constrained threshold", y, score, f1_floor))
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    grid = pd.concat(grid_rows, ignore_index=True)
    return summary, grid


def _simplex_weights(n: int, units: int) -> list[tuple[float, ...]]:
    result: list[tuple[float, ...]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            result.append(tuple((prefix + [remaining])[i] / units for i in range(n)))
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], units, n)
    return result


def _aligned_score_matrix(
    averaged_oof: dict[str, pd.DataFrame],
    members: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    base = averaged_oof[members[0]][[ID_COLUMN, TARGET_COLUMN, "score"]].rename(
        columns={"score": members[0]}
    )
    for member in members[1:]:
        scores = averaged_oof[member][[ID_COLUMN, TARGET_COLUMN, "score"]].rename(
            columns={"score": member}
        )
        base = base.merge(scores, on=[ID_COLUMN, TARGET_COLUMN], how="inner")
    y = base[TARGET_COLUMN].to_numpy(dtype=int)
    matrix = base[members].to_numpy(dtype=float)
    return base, y, matrix


def method2_optimized_ensembles(
    averaged_oof: dict[str, pd.DataFrame],
    f1_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preferred_members = [
        "M3_XGB_NW_shallow_reg",
        "M3_XGB_NW_depth4",
        "M3_RF_depth8_balanced",
        "M3_RF_leaf4",
        "M3_XGB_SPW_regularized_x1.5",
        "M3_XGB_NW_low_fp_reg",
    ]
    members = [name for name in preferred_members if name in averaged_oof]
    if len(members) < 3:
        raise ValueError(f"Not enough candidate scores for ensemble search: {members}")

    base, y, matrix = _aligned_score_matrix(averaged_oof, members)
    rows = []
    oof_frames = []
    for weights in _simplex_weights(len(members), units=10):
        if sum(weight > 0 for weight in weights) < 2:
            continue
        score = matrix @ np.asarray(weights, dtype=float)
        nonzero = {member: weight for member, weight in zip(members, weights) if weight > 0}
        name = "OPTENS_" + "_".join(f"{member}:{weight:.1f}" for member, weight in nonzero.items())
        row = summarize_score(
            candidate=name,
            method_group="Optimized ensemble weights",
            y_true=y,
            y_score=score,
            f1_floor=f1_floor,
            details=json.dumps(nonzero, ensure_ascii=False),
        )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )

    for _, row in summary.head(30).iterrows():
        weights = json.loads(row["details"])
        score = np.zeros(len(base), dtype=float)
        for member, weight in weights.items():
            score += float(weight) * base[member].to_numpy(dtype=float)
        oof_frames.append(
            pd.DataFrame(
                {
                    ID_COLUMN: base[ID_COLUMN].astype(str).to_numpy(),
                    TARGET_COLUMN: y,
                    "score": score,
                    "candidate": row["candidate"],
                    "method_group": "Optimized ensemble weights",
                }
            )
        )
    oof = pd.concat(oof_frames, ignore_index=True)
    return summary, oof


def prior_sample_weight(y: np.ndarray, strength: float) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    train_pi = float(y.mean())
    if strength <= 0:
        return np.ones(len(y), dtype=float)
    pos_full = FINAL_PATHOGENIC_RATE / train_pi
    neg_full = (1.0 - FINAL_PATHOGENIC_RATE) / (1.0 - train_pi)
    weights = np.ones(len(y), dtype=float)
    weights[y == 1] = pos_full
    weights[y == 0] = neg_full
    weights = np.exp(strength * np.log(weights))
    return weights / weights.mean()


def make_target_model(candidate: TargetModelCandidate):
    params = dict(candidate.params)
    if candidate.family == "xgboost":
        return XGBClassifier(**params)
    if candidate.family == "random_forest":
        return RandomForestClassifier(**params)
    if candidate.family == "extra_trees":
        return ExtraTreesClassifier(**params)
    if candidate.family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params)
    raise ValueError(f"Unknown candidate family: {candidate.family}")


def inner_hard_benign_scores(x_fit: np.ndarray, y_fit: np.ndarray, seed: int) -> np.ndarray:
    scores = np.zeros(len(y_fit), dtype=float)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for train_idx, hold_idx in splitter.split(x_fit, y_fit):
        probe = XGBClassifier(
            **xgb_params(
                n_estimators=220,
                max_depth=2,
                learning_rate=0.04,
                min_child_weight=5,
                gamma=0.05,
                reg_alpha=0.10,
                reg_lambda=5.0,
                subsample=0.85,
                colsample_bytree=0.80,
                random_state=seed,
            )
        )
        probe.fit(x_fit[train_idx], y_fit[train_idx])
        scores[hold_idx] = positive_probability(probe, x_fit[hold_idx])
    return scores


def sample_weight_for_candidate(
    candidate: TargetModelCandidate,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    seed: int,
    hard_cache: dict[tuple[int, int], np.ndarray],
    fold_key: tuple[int, int],
) -> tuple[np.ndarray | None, int]:
    weight = prior_sample_weight(y_fit, candidate.prior_strength)
    hard_count = 0
    if candidate.hard_top_fraction > 0:
        if fold_key not in hard_cache:
            hard_cache[fold_key] = inner_hard_benign_scores(x_fit, y_fit, seed)
        hard_scores = hard_cache[fold_key]
        benign_scores = hard_scores[y_fit == 0]
        cutoff = np.quantile(benign_scores, 1.0 - candidate.hard_top_fraction)
        hard_mask = (y_fit == 0) & (hard_scores >= cutoff)
        hard_count = int(hard_mask.sum())
        weight[hard_mask] *= candidate.hard_multiplier
        weight = weight / weight.mean()
    if np.allclose(weight, 1.0):
        return None, hard_count
    return weight.astype(float), hard_count


def build_new_model_candidates() -> list[TargetModelCandidate]:
    xgb_shallow = xgb_params(
        n_estimators=550,
        max_depth=2,
        learning_rate=0.025,
        min_child_weight=5,
        gamma=0.05,
        reg_alpha=0.10,
        reg_lambda=5.0,
        colsample_bytree=0.80,
    )
    rf_depth8 = rf_params(
        n_estimators=650,
        max_depth=8,
        min_samples_leaf=3,
        min_samples_split=8,
        class_weight=None,
    )
    extra_base = {
        "n_estimators": 750,
        "criterion": "gini",
        "max_depth": 10,
        "min_samples_leaf": 3,
        "min_samples_split": 8,
        "max_features": "sqrt",
        "bootstrap": False,
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    hgb_base = {
        "max_iter": 320,
        "learning_rate": 0.035,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 25,
        "l2_regularization": 0.25,
        "early_stopping": False,
        "random_state": RANDOM_STATE,
    }

    candidates: list[TargetModelCandidate] = []
    for strength in [0.25, 0.50, 0.75, 1.00]:
        suffix = str(int(strength * 100)).zfill(3)
        candidates.extend(
            [
                TargetModelCandidate(
                    f"M3_FPrior_XGB_s{suffix}",
                    "Final-prior weighted XGBoost/RF",
                    "xgboost",
                    xgb_shallow,
                    prior_strength=strength,
                ),
                TargetModelCandidate(
                    f"M3_FPrior_RF_s{suffix}",
                    "Final-prior weighted XGBoost/RF",
                    "random_forest",
                    rf_depth8,
                    prior_strength=strength,
                ),
            ]
        )

    candidates.extend(
        [
            TargetModelCandidate(
                "M3_HardBenign_XGB_top20_w2_s025",
                "Hard benign focused model",
                "xgboost",
                xgb_shallow,
                prior_strength=0.25,
                hard_top_fraction=0.20,
                hard_multiplier=2.0,
            ),
            TargetModelCandidate(
                "M3_HardBenign_XGB_top30_w2_s025",
                "Hard benign focused model",
                "xgboost",
                xgb_shallow,
                prior_strength=0.25,
                hard_top_fraction=0.30,
                hard_multiplier=2.0,
            ),
            TargetModelCandidate(
                "M3_HardBenign_RF_top20_w3_s025",
                "Hard benign focused model",
                "random_forest",
                rf_depth8,
                prior_strength=0.25,
                hard_top_fraction=0.20,
                hard_multiplier=3.0,
            ),
            TargetModelCandidate(
                "M3_HardBenign_RF_top30_w2_s025",
                "Hard benign focused model",
                "random_forest",
                rf_depth8,
                prior_strength=0.25,
                hard_top_fraction=0.30,
                hard_multiplier=2.0,
            ),
        ]
    )

    candidates.extend(
        [
            TargetModelCandidate(
                "M3_ExtraTrees_no_weight",
                "ExtraTrees / HistGradientBoosting ensemble",
                "extra_trees",
                extra_base,
            ),
            TargetModelCandidate(
                "M3_ExtraTrees_prior50",
                "ExtraTrees / HistGradientBoosting ensemble",
                "extra_trees",
                extra_base,
                prior_strength=0.50,
            ),
            TargetModelCandidate(
                "M3_ExtraTrees_prior75",
                "ExtraTrees / HistGradientBoosting ensemble",
                "extra_trees",
                extra_base,
                prior_strength=0.75,
            ),
            TargetModelCandidate(
                "M3_HGB_no_weight",
                "ExtraTrees / HistGradientBoosting ensemble",
                "hist_gradient_boosting",
                hgb_base,
            ),
            TargetModelCandidate(
                "M3_HGB_prior50",
                "ExtraTrees / HistGradientBoosting ensemble",
                "hist_gradient_boosting",
                hgb_base,
                prior_strength=0.50,
            ),
            TargetModelCandidate(
                "M3_HGB_prior75",
                "ExtraTrees / HistGradientBoosting ensemble",
                "hist_gradient_boosting",
                hgb_base,
                prior_strength=0.75,
            ),
        ]
    )
    return candidates


def run_new_model_candidates(
    candidates: list[TargetModelCandidate],
    f1_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = prepare_fold_cache([FEATURE_SET])[FEATURE_SET]
    hard_cache: dict[tuple[int, int], np.ndarray] = {}
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    total = len(candidates)

    for idx, candidate in enumerate(candidates, start=1):
        started = time.time()
        for fold_data in cache:
            seed = int(fold_data["repeat_seed"])
            fold = int(fold_data["fold"])
            x_fit = fold_data["x_fit"]
            y_fit = fold_data["y_fit"]
            model = make_target_model(candidate)
            sample_weight, hard_count = sample_weight_for_candidate(
                candidate,
                x_fit,
                y_fit,
                seed,
                hard_cache,
                fold_key=(seed, fold),
            )
            fit_started = time.time()
            if sample_weight is None:
                model.fit(x_fit, y_fit)
            else:
                model.fit(x_fit, y_fit, sample_weight=sample_weight)
            score = positive_probability(model, fold_data["x_hold"])
            fit_seconds = time.time() - fit_started
            fold_metrics = metric_row(fold_data["y_hold"], score, threshold=0.50)
            fold_rows.append(
                {
                    "candidate": candidate.name,
                    "method_group": candidate.method_group,
                    "family": candidate.family,
                    "feature_set": FEATURE_SET,
                    "repeat_seed": seed,
                    "fold": fold,
                    "prior_strength": candidate.prior_strength,
                    "hard_top_fraction": candidate.hard_top_fraction,
                    "hard_multiplier": candidate.hard_multiplier,
                    "hard_benign_count": hard_count,
                    "fit_seconds": fit_seconds,
                    **fold_metrics,
                }
            )
            oof_rows.append(
                pd.DataFrame(
                    {
                        ID_COLUMN: fold_data["hold_ids"],
                        TARGET_COLUMN: fold_data["y_hold"],
                        "score": score,
                        "candidate": candidate.name,
                        "method_group": candidate.method_group,
                        "family": candidate.family,
                        "feature_set": FEATURE_SET,
                        "repeat_seed": seed,
                        "fold": fold,
                    }
                )
            )
        elapsed = time.time() - started
        print(f"[new {idx}/{total}] {candidate.name} completed in {elapsed:.1f}s", flush=True)

    fold_results = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_rows, ignore_index=True)
    summary_rows = []
    for candidate, group in oof.groupby("candidate"):
        avg = (
            group.groupby([ID_COLUMN, TARGET_COLUMN], as_index=False)["score"]
            .mean()
            .sort_values(ID_COLUMN)
            .reset_index(drop=True)
        )
        first = group.iloc[0]
        details = (
            f"family={first['family']}; "
            f"prior_strength={fold_results.loc[fold_results['candidate'].eq(candidate), 'prior_strength'].iloc[0]}; "
            f"hard_top_fraction={fold_results.loc[fold_results['candidate'].eq(candidate), 'hard_top_fraction'].iloc[0]}"
        )
        summary_rows.append(
            summarize_score(
                candidate=str(candidate),
                method_group=str(first["method_group"]),
                y_true=avg[TARGET_COLUMN].to_numpy(dtype=int),
                y_score=avg["score"].to_numpy(dtype=float),
                f1_floor=f1_floor,
                details=details,
            )
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    return fold_results, oof, summary


def calibrate_scores_cv(
    y: np.ndarray,
    score: np.ndarray,
    method: str,
    apply_prior_correction: bool,
) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    output = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, hold_idx in splitter.split(score.reshape(-1, 1), y):
        train_score = score[train_idx]
        hold_score = score[hold_idx]
        if method == "sigmoid":
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            model.fit(_logit(train_score).reshape(-1, 1), y[train_idx])
            calibrated = model.predict_proba(_logit(hold_score).reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(train_score, y[train_idx])
            calibrated = model.predict(hold_score)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        if apply_prior_correction:
            calibrated = prior_correct_probability(calibrated, float(y[train_idx].mean()))
        output[hold_idx] = calibrated
    return np.clip(output, 1e-7, 1 - 1e-7)


def method4_calibration_prior(
    score_sources: dict[str, pd.DataFrame],
    f1_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preferred = [
        "ENS_XGB_NW_RF_mean",
        "ENS_XGB_NW60_SPW20_RF20",
        "M3_RF_depth8_balanced",
        "M3_XGB_NW_shallow_reg",
        "M3_XGB_SPW_regularized_x1.5",
    ]
    available = [candidate for candidate in preferred if candidate in score_sources]
    rows = []
    oof_frames = []
    for candidate in available:
        table = score_sources[candidate].sort_values(ID_COLUMN).reset_index(drop=True)
        y = table[TARGET_COLUMN].to_numpy(dtype=int)
        raw_score = table["score"].to_numpy(dtype=float)
        variants = {
            f"{candidate}__raw_prior": prior_correct_probability(raw_score, float(y.mean())),
            f"{candidate}__sigmoid_cv": calibrate_scores_cv(
                y, raw_score, method="sigmoid", apply_prior_correction=False
            ),
            f"{candidate}__sigmoid_cv_prior": calibrate_scores_cv(
                y, raw_score, method="sigmoid", apply_prior_correction=True
            ),
            f"{candidate}__isotonic_cv": calibrate_scores_cv(
                y, raw_score, method="isotonic", apply_prior_correction=False
            ),
            f"{candidate}__isotonic_cv_prior": calibrate_scores_cv(
                y, raw_score, method="isotonic", apply_prior_correction=True
            ),
        }
        for name, score in variants.items():
            rows.append(
                summarize_score(
                    candidate=name,
                    method_group="Calibration + prior correction",
                    y_true=y,
                    y_score=score,
                    f1_floor=f1_floor,
                    details=f"source={candidate}",
                )
            )
            oof_frames.append(
                pd.DataFrame(
                    {
                        ID_COLUMN: table[ID_COLUMN].astype(str).to_numpy(),
                        TARGET_COLUMN: y,
                        "score": score,
                        "candidate": name,
                        "method_group": "Calibration + prior correction",
                    }
                )
            )
    summary = pd.DataFrame(rows).sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    return summary, pd.concat(oof_frames, ignore_index=True)


def method6_extra_hgb_ensembles(
    score_sources: dict[str, pd.DataFrame],
    f1_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preferred = [
        "M3_XGB_NW_shallow_reg",
        "M3_RF_depth8_balanced",
        "M3_ExtraTrees_prior50",
        "M3_HGB_prior50",
        "M3_XGB_SPW_regularized_x1.5",
    ]
    members = [candidate for candidate in preferred if candidate in score_sources]
    if len(members) < 3:
        return pd.DataFrame(), pd.DataFrame()
    base, y, matrix = _aligned_score_matrix(score_sources, members)
    extra_or_hgb = {"M3_ExtraTrees_prior50", "M3_HGB_prior50"}
    rows = []
    oof_frames = []
    for weights in _simplex_weights(len(members), units=10):
        if sum(weight > 0 for weight in weights) < 2:
            continue
        nonzero = {member: weight for member, weight in zip(members, weights) if weight > 0}
        if not any(nonzero.get(member, 0.0) > 0 for member in extra_or_hgb):
            continue
        score = matrix @ np.asarray(weights, dtype=float)
        name = "EXHGBENS_" + "_".join(f"{member}:{weight:.1f}" for member, weight in nonzero.items())
        rows.append(
            summarize_score(
                candidate=name,
                method_group="ExtraTrees / HistGradientBoosting ensemble",
                y_true=y,
                y_score=score,
                f1_floor=f1_floor,
                details=json.dumps(nonzero, ensure_ascii=False),
            )
        )
    summary = pd.DataFrame(rows).sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    for _, row in summary.head(30).iterrows():
        weights = json.loads(row["details"])
        score = np.zeros(len(base), dtype=float)
        for member, weight in weights.items():
            score += float(weight) * base[member].to_numpy(dtype=float)
        oof_frames.append(
            pd.DataFrame(
                {
                    ID_COLUMN: base[ID_COLUMN].astype(str).to_numpy(),
                    TARGET_COLUMN: y,
                    "score": score,
                    "candidate": row["candidate"],
                    "method_group": "ExtraTrees / HistGradientBoosting ensemble",
                }
            )
        )
    return summary, pd.concat(oof_frames, ignore_index=True)


def _averaged_from_oof(oof: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for candidate, group in oof.groupby("candidate"):
        avg = (
            group.groupby([ID_COLUMN, TARGET_COLUMN], as_index=False)["score"]
            .mean()
            .sort_values(ID_COLUMN)
            .reset_index(drop=True)
        )
        result[str(candidate)] = avg
    return result


def _write_top_table(summary: pd.DataFrame, path: Path, top_n: int = 12) -> None:
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
        "target_specificity",
        "target_sensitivity",
        "target_precision",
        "target_expected_fp_per_3500",
        "target_expected_fn_per_3500",
        "details",
    ]
    available = [column for column in columns if column in summary.columns]
    summary.loc[:, available].head(top_n).to_csv(path, index=False)


def main() -> dict[str, Any]:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tuned_oof = load_tuned_oof()

    if not (TUNED_DIR / "master_tuned_combined_summary.csv").exists():
        raise FileNotFoundError("Missing tuned summary file.")
    tuned_summary = pd.read_csv(TUNED_DIR / "master_tuned_combined_summary.csv")
    current_best_f1 = float(tuned_summary["best_f1_final_f1"].max())
    f1_floor = current_best_f1 - F1_KEEP_TOLERANCE

    print("Step 1/6: MCC + FP constrained threshold", flush=True)
    method1_summary, method1_grid = method1_constrained_thresholds(tuned_oof, f1_floor)
    method1_summary.to_csv(OUT_DIR / "method1_mcc_fp_constrained_threshold.csv", index=False)
    method1_grid.to_csv(OUT_DIR / "method1_threshold_grid.csv", index=False)

    print("Step 2/6: Optimized ensemble weights", flush=True)
    method2_summary, method2_oof = method2_optimized_ensembles(tuned_oof, f1_floor)
    method2_summary.to_csv(OUT_DIR / "method2_optimized_ensemble_weights.csv", index=False)
    method2_oof.to_csv(OUT_DIR / "method2_optimized_ensemble_oof.csv", index=False)

    print("Step 3/6, 5/6, 6/6: New fold-safe model families", flush=True)
    new_candidates = build_new_model_candidates()
    new_fold_results, new_oof, new_summary = run_new_model_candidates(new_candidates, f1_floor)
    new_fold_results.to_csv(OUT_DIR / "new_model_fold_results.csv", index=False)
    new_oof.to_csv(OUT_DIR / "new_model_oof_predictions.csv", index=False)
    new_summary.to_csv(OUT_DIR / "new_model_summary.csv", index=False)
    method3_summary = new_summary.loc[
        new_summary["method_group"].eq("Final-prior weighted XGBoost/RF")
    ].copy()
    method5_summary = new_summary.loc[
        new_summary["method_group"].eq("Hard benign focused model")
    ].copy()
    method6_base_summary = new_summary.loc[
        new_summary["method_group"].eq("ExtraTrees / HistGradientBoosting ensemble")
    ].copy()
    method3_summary.to_csv(OUT_DIR / "method3_final_prior_weighted_summary.csv", index=False)
    method5_summary.to_csv(OUT_DIR / "method5_hard_benign_summary.csv", index=False)
    method6_base_summary.to_csv(OUT_DIR / "method6_extra_hgb_base_summary.csv", index=False)

    score_sources = dict(tuned_oof)
    score_sources.update(_averaged_from_oof(method2_oof))
    score_sources.update(_averaged_from_oof(new_oof))

    print("Step 4/6: Calibration + prior correction", flush=True)
    method4_summary, method4_oof = method4_calibration_prior(score_sources, f1_floor)
    method4_summary.to_csv(OUT_DIR / "method4_calibration_prior_summary.csv", index=False)
    method4_oof.to_csv(OUT_DIR / "method4_calibration_prior_oof.csv", index=False)

    print("Step 6/6: ExtraTrees / HistGradientBoosting ensembles", flush=True)
    method6_ensemble_summary, method6_ensemble_oof = method6_extra_hgb_ensembles(
        score_sources, f1_floor
    )
    method6_ensemble_summary.to_csv(
        OUT_DIR / "method6_extra_hgb_ensemble_summary.csv", index=False
    )
    method6_ensemble_oof.to_csv(OUT_DIR / "method6_extra_hgb_ensemble_oof.csv", index=False)
    method6_summary = pd.concat(
        [method6_base_summary, method6_ensemble_summary], ignore_index=True
    ).sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    method6_summary.to_csv(OUT_DIR / "method6_extra_hgb_summary.csv", index=False)

    combined = pd.concat(
        [
            method1_summary.assign(experiment_block="1_constrained_threshold"),
            method2_summary.assign(experiment_block="2_optimized_ensemble"),
            method3_summary.assign(experiment_block="3_final_prior_weighted"),
            method4_summary.assign(experiment_block="4_calibration_prior"),
            method5_summary.assign(experiment_block="5_hard_benign"),
            method6_summary.assign(experiment_block="6_extra_hgb"),
        ],
        ignore_index=True,
    ).sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    combined.to_csv(OUT_DIR / "all_targeted_summary.csv", index=False)

    for filename, frame in [
        ("method1_top.csv", method1_summary),
        ("method2_top.csv", method2_summary),
        ("method3_top.csv", method3_summary),
        ("method4_top.csv", method4_summary),
        ("method5_top.csv", method5_summary),
        ("method6_top.csv", method6_summary),
        ("all_targeted_top.csv", combined),
    ]:
        _write_top_table(frame, OUT_DIR / filename)

    metadata = {
        "status": "completed",
        "outer_validation_used": False,
        "external_data_used": False,
        "feature_set_for_new_models": FEATURE_SET,
        "final_pathogenic_rate_assumption": FINAL_PATHOGENIC_RATE,
        "final_benign_count_assumption": FINAL_BENIGN_N,
        "final_total_n_assumption": FINAL_TOTAL_N,
        "current_best_f1_reference": current_best_f1,
        "f1_keep_floor": f1_floor,
        "fp_target_per_3500": FP_TARGET,
        "strict_sensitivity_floor": STRICT_SENSITIVITY_FLOOR,
        "new_model_candidate_count": len(new_candidates),
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "method1": "results/modeling/targeted_mcc_fp/method1_mcc_fp_constrained_threshold.csv",
            "method2": "results/modeling/targeted_mcc_fp/method2_optimized_ensemble_weights.csv",
            "method3": "results/modeling/targeted_mcc_fp/method3_final_prior_weighted_summary.csv",
            "method4": "results/modeling/targeted_mcc_fp/method4_calibration_prior_summary.csv",
            "method5": "results/modeling/targeted_mcc_fp/method5_hard_benign_summary.csv",
            "method6": "results/modeling/targeted_mcc_fp/method6_extra_hgb_summary.csv",
            "combined": "results/modeling/targeted_mcc_fp/all_targeted_summary.csv",
        },
    }
    (OUT_DIR / "targeted_mcc_fp_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    display_cols = [
        "candidate",
        "experiment_block",
        "target_status",
        "target_threshold",
        "target_f1",
        "target_mcc",
        "target_specificity",
        "target_sensitivity",
        "target_precision",
        "target_expected_fp_per_3500",
        "target_expected_fn_per_3500",
    ]
    print("\n=== TARGETED OVERALL TOP ===")
    print(combined.loc[:, display_cols].head(20).round(4).to_string(index=False))
    return metadata


if __name__ == "__main__":
    main()
