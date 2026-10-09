"""LightGBM experiments for the MASTER panel.

The experiment uses only the approved MASTER data and the existing
fold-safe M3 preprocessing path. Outer validation is not used for model,
threshold, or ensemble-weight selection.
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
from lightgbm import LGBMClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import ID_COLUMN, TARGET_COLUMN  # noqa: E402
from master_targeted_mcc_fp_experiments import (  # noqa: E402
    F1_KEEP_TOLERANCE,
    OUT_DIR as TARGETED_DIR,
    _aligned_score_matrix,
    _averaged_from_oof,
    _simplex_weights,
    load_tuned_oof,
    prior_sample_weight,
    summarize_score,
)
from master_tuned_modeling_experiments import (  # noqa: E402
    RANDOM_STATE,
    metric_row,
    positive_probability,
    prepare_fold_cache,
)


FEATURE_SET = "M3_missing_aware_compact"
OUT_DIR = ROOT / "results" / "modeling" / "lightgbm"
CATBOOST_DIR = ROOT / "results" / "modeling" / "catboost"


@dataclass(frozen=True)
class LightGBMCandidate:
    name: str
    params: dict[str, Any]
    prior_strength: float = 0.0


def lgbm_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 35,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.10,
        "reg_lambda": 6.0,
        "objective": "binary",
        "boosting_type": "gbdt",
        "random_state": RANDOM_STATE,
        "n_jobs": 4,
        "verbosity": -1,
        "force_col_wise": True,
    }
    params.update(overrides)
    return params


def build_candidates() -> list[LightGBMCandidate]:
    regularized = lgbm_params(
        n_estimators=550,
        learning_rate=0.028,
        num_leaves=15,
        min_child_samples=45,
        reg_alpha=0.20,
        reg_lambda=9.0,
        colsample_bytree=0.75,
    )
    shallow_fp = lgbm_params(
        n_estimators=700,
        learning_rate=0.022,
        num_leaves=7,
        max_depth=3,
        min_child_samples=65,
        reg_alpha=0.35,
        reg_lambda=12.0,
        colsample_bytree=0.70,
    )
    leaf31 = lgbm_params(
        n_estimators=420,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=35,
        reg_alpha=0.15,
        reg_lambda=8.0,
        colsample_bytree=0.75,
    )
    dart = lgbm_params(
        boosting_type="dart",
        n_estimators=520,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=45,
        reg_alpha=0.20,
        reg_lambda=9.0,
        colsample_bytree=0.75,
        drop_rate=0.08,
        skip_drop=0.50,
    )

    candidates = [
        LightGBMCandidate("M3_LGBM_regularized", regularized),
        LightGBMCandidate("M3_LGBM_shallow_fp", shallow_fp),
        LightGBMCandidate("M3_LGBM_leaf31", leaf31),
        LightGBMCandidate("M3_LGBM_dart", dart),
    ]
    for strength in [0.25, 0.50, 0.75, 1.00]:
        suffix = str(int(strength * 100)).zfill(3)
        candidates.append(
            LightGBMCandidate(
                f"M3_LGBM_prior{suffix}",
                shallow_fp,
                prior_strength=strength,
            )
        )
    return candidates


def run_lightgbm_candidates(
    candidates: list[LightGBMCandidate],
    f1_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = prepare_fold_cache([FEATURE_SET])[FEATURE_SET]
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    for idx, candidate in enumerate(candidates, start=1):
        started = time.time()
        for fold_data in cache:
            seed = int(fold_data["repeat_seed"])
            params = dict(candidate.params)
            params["random_state"] = seed
            model = LGBMClassifier(**params)
            sample_weight = prior_sample_weight(fold_data["y_fit"], candidate.prior_strength)
            if np.allclose(sample_weight, 1.0):
                sample_weight = None
            fit_started = time.time()
            if sample_weight is None:
                model.fit(fold_data["x_fit"], fold_data["y_fit"])
            else:
                model.fit(
                    fold_data["x_fit"],
                    fold_data["y_fit"],
                    sample_weight=sample_weight,
                )
            score = positive_probability(model, fold_data["x_hold"])
            fit_seconds = time.time() - fit_started
            fold_metrics = metric_row(fold_data["y_hold"], score, threshold=0.50)
            fold_rows.append(
                {
                    "candidate": candidate.name,
                    "method_group": "LightGBM",
                    "family": "lightgbm",
                    "feature_set": FEATURE_SET,
                    "repeat_seed": seed,
                    "fold": int(fold_data["fold"]),
                    "prior_strength": candidate.prior_strength,
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
                        "method_group": "LightGBM",
                        "family": "lightgbm",
                        "feature_set": FEATURE_SET,
                        "repeat_seed": seed,
                        "fold": int(fold_data["fold"]),
                    }
                )
            )
        print(
            f"[LightGBM {idx}/{len(candidates)}] {candidate.name} "
            f"completed in {time.time() - started:.1f}s",
            flush=True,
        )

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
        strength = float(
            fold_results.loc[fold_results["candidate"].eq(candidate), "prior_strength"].iloc[0]
        )
        summary_rows.append(
            summarize_score(
                candidate=str(candidate),
                method_group="LightGBM",
                y_true=avg[TARGET_COLUMN].to_numpy(dtype=int),
                y_score=avg["score"].to_numpy(dtype=float),
                f1_floor=f1_floor,
                details=f"prior_strength={strength}",
            )
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    return fold_results, oof, summary


def _load_score_sources(lightgbm_oof: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sources = load_tuned_oof()
    for path in [
        TARGETED_DIR / "method2_optimized_ensemble_oof.csv",
        TARGETED_DIR / "new_model_oof_predictions.csv",
        CATBOOST_DIR / "catboost_oof_predictions.csv",
        CATBOOST_DIR / "catboost_ensemble_oof.csv",
    ]:
        if path.exists() and path.stat().st_size > 0:
            sources.update(_averaged_from_oof(pd.read_csv(path, dtype={ID_COLUMN: "string"})))
    sources.update(_averaged_from_oof(lightgbm_oof))
    return sources


def _current_reference_summary() -> pd.DataFrame:
    if (CATBOOST_DIR / "catboost_vs_current_summary.csv").exists():
        return pd.read_csv(CATBOOST_DIR / "catboost_vs_current_summary.csv")
    return pd.read_csv(TARGETED_DIR / "all_targeted_summary.csv")


def build_lightgbm_ensembles(
    score_sources: dict[str, pd.DataFrame],
    lightgbm_summary: pd.DataFrame,
    f1_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = _current_reference_summary()
    strict = reference.loc[reference["target_status"].eq("strict")].copy()
    if strict.empty:
        strict = reference.copy()
    ordered_current = strict.sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    current_best = next(
        (
            str(row["candidate"])
            for _, row in ordered_current.iterrows()
            if str(row["candidate"]) in score_sources
        ),
        None,
    )

    best_lgbm = str(
        lightgbm_summary.sort_values(
            ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
            ascending=[False, False, True],
        ).iloc[0]["candidate"]
    )

    rows = []
    oof_frames = []
    if current_best is not None:
        pair_members = [current_best, best_lgbm]
        base, y, matrix = _aligned_score_matrix(score_sources, pair_members)
        for lgbm_weight in np.round(np.arange(0.1, 0.61, 0.1), 1):
            weights = {
                current_best: float(1.0 - lgbm_weight),
                best_lgbm: float(lgbm_weight),
            }
            score = matrix @ np.asarray([weights[current_best], weights[best_lgbm]])
            name = f"LGBMENS_current:{1.0 - lgbm_weight:.1f}_{best_lgbm}:{lgbm_weight:.1f}"
            rows.append(
                summarize_score(
                    candidate=name,
                    method_group="LightGBM added ensemble",
                    y_true=y,
                    y_score=score,
                    f1_floor=f1_floor,
                    details=json.dumps(weights, ensure_ascii=False),
                )
            )

    simplex_members = [
        "M3_RF_depth8_balanced",
        "M3_XGB_SPW_regularized_x1.5",
        "M3_CatBoost_bernoulli",
        best_lgbm,
        "M3_XGB_NW_shallow_reg",
    ]
    simplex_members = [member for member in simplex_members if member in score_sources]
    if len(simplex_members) >= 3 and best_lgbm in simplex_members:
        base, y, matrix = _aligned_score_matrix(score_sources, simplex_members)
        for weights_tuple in _simplex_weights(len(simplex_members), units=10):
            weights = {
                member: weight
                for member, weight in zip(simplex_members, weights_tuple)
                if weight > 0
            }
            if weights.get(best_lgbm, 0.0) <= 0:
                continue
            if sum(value > 0 for value in weights.values()) < 2:
                continue
            score = matrix @ np.asarray(weights_tuple, dtype=float)
            name = "LGBMOPT_" + "_".join(
                f"{member}:{weight:.1f}" for member, weight in weights.items()
            )
            rows.append(
                summarize_score(
                    candidate=name,
                    method_group="LightGBM added ensemble",
                    y_true=y,
                    y_score=score,
                    f1_floor=f1_floor,
                    details=json.dumps(weights, ensure_ascii=False),
                )
            )

    summary = pd.DataFrame(rows).sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    for _, row in summary.head(30).iterrows():
        weights = json.loads(row["details"])
        members = list(weights)
        base, y, matrix = _aligned_score_matrix(score_sources, members)
        score = matrix @ np.asarray([weights[member] for member in members], dtype=float)
        oof_frames.append(
            pd.DataFrame(
                {
                    ID_COLUMN: base[ID_COLUMN].astype(str).to_numpy(),
                    TARGET_COLUMN: y,
                    "score": score,
                    "candidate": row["candidate"],
                    "method_group": "LightGBM added ensemble",
                }
            )
        )
    ensemble_oof = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    return summary, ensemble_oof


def _write_top_table(summary: pd.DataFrame, path: Path, top_n: int = 20) -> None:
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
    summary.loc[:, [column for column in columns if column in summary.columns]].head(top_n).to_csv(
        path,
        index=False,
    )


def main() -> dict[str, Any]:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reference = _current_reference_summary()
    strict = reference.loc[reference["target_status"].eq("strict")]
    reference_best_f1 = float((strict if len(strict) else reference)["target_f1"].max())
    f1_floor = reference_best_f1 - F1_KEEP_TOLERANCE

    candidates = build_candidates()
    fold_results, lgbm_oof, lgbm_summary = run_lightgbm_candidates(candidates, f1_floor)
    fold_results.to_csv(OUT_DIR / "lightgbm_fold_results.csv", index=False)
    lgbm_oof.to_csv(OUT_DIR / "lightgbm_oof_predictions.csv", index=False)
    lgbm_summary.to_csv(OUT_DIR / "lightgbm_candidate_summary.csv", index=False)

    score_sources = _load_score_sources(lgbm_oof)
    ensemble_summary, ensemble_oof = build_lightgbm_ensembles(
        score_sources,
        lgbm_summary,
        f1_floor,
    )
    ensemble_summary.to_csv(OUT_DIR / "lightgbm_ensemble_summary.csv", index=False)
    ensemble_oof.to_csv(OUT_DIR / "lightgbm_ensemble_oof.csv", index=False)

    comparison = pd.concat(
        [
            reference.assign(method_group="Current best before LightGBM"),
            lgbm_summary.assign(experiment_block="8_lightgbm_single"),
            ensemble_summary.assign(experiment_block="8_lightgbm_ensemble"),
        ],
        ignore_index=True,
    ).sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    comparison.to_csv(OUT_DIR / "lightgbm_vs_current_summary.csv", index=False)
    _write_top_table(lgbm_summary, OUT_DIR / "lightgbm_candidate_top.csv")
    _write_top_table(ensemble_summary, OUT_DIR / "lightgbm_ensemble_top.csv")
    _write_top_table(comparison, OUT_DIR / "lightgbm_vs_current_top.csv")

    metadata = {
        "status": "completed",
        "outer_validation_used": False,
        "external_data_used": False,
        "feature_set": FEATURE_SET,
        "candidate_count": len(candidates),
        "folds_per_candidate": 25,
        "f1_floor": f1_floor,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "fold_results": "results/modeling/lightgbm/lightgbm_fold_results.csv",
            "oof_predictions": "results/modeling/lightgbm/lightgbm_oof_predictions.csv",
            "candidate_summary": "results/modeling/lightgbm/lightgbm_candidate_summary.csv",
            "ensemble_summary": "results/modeling/lightgbm/lightgbm_ensemble_summary.csv",
            "comparison": "results/modeling/lightgbm/lightgbm_vs_current_summary.csv",
        },
    }
    (OUT_DIR / "lightgbm_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display_cols = [
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
    ]
    print("\n=== LIGHTGBM VS CURRENT TOP ===")
    print(comparison.loc[:, display_cols].head(20).round(4).to_string(index=False))
    return metadata


if __name__ == "__main__":
    main()
