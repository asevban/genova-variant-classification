"""CatBoost experiments for the MASTER panel.

The experiment stays inside the approved MASTER modeling setup:
no external data, no preprocessing change, and no outer-validation use for
model or threshold selection.
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
from catboost import CatBoostClassifier


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
OUT_DIR = ROOT / "results" / "modeling" / "catboost"


@dataclass(frozen=True)
class CatBoostCandidate:
    name: str
    params: dict[str, Any]
    prior_strength: float = 0.0


def cat_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "iterations": 450,
        "depth": 4,
        "learning_rate": 0.035,
        "l2_leaf_reg": 8.0,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
        "border_count": 128,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": 4,
        "random_seed": RANDOM_STATE,
    }
    params.update(overrides)
    return params


def build_candidates() -> list[CatBoostCandidate]:
    base_regularized = cat_params(
        iterations=550,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=10.0,
        bagging_temperature=0.8,
        random_strength=1.5,
    )
    shallow_fp = cat_params(
        iterations=700,
        depth=3,
        learning_rate=0.025,
        l2_leaf_reg=14.0,
        bagging_temperature=1.0,
        random_strength=2.0,
        border_count=96,
    )
    depth5 = cat_params(
        iterations=450,
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=12.0,
        bagging_temperature=0.6,
        random_strength=1.5,
        border_count=128,
    )
    bernoulli = cat_params(
        iterations=520,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=10.0,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        random_strength=1.5,
    )
    bernoulli.pop("bagging_temperature", None)

    candidates = [
        CatBoostCandidate("M3_CatBoost_regularized", base_regularized),
        CatBoostCandidate("M3_CatBoost_shallow_fp", shallow_fp),
        CatBoostCandidate("M3_CatBoost_depth5", depth5),
        CatBoostCandidate("M3_CatBoost_bernoulli", bernoulli),
    ]
    for strength in [0.25, 0.50, 0.75, 1.00]:
        suffix = str(int(strength * 100)).zfill(3)
        candidates.append(
            CatBoostCandidate(
                f"M3_CatBoost_prior{suffix}",
                shallow_fp,
                prior_strength=strength,
            )
        )
    return candidates


def run_catboost_candidates(
    candidates: list[CatBoostCandidate],
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
            params["random_seed"] = seed
            model = CatBoostClassifier(**params)
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
                    "method_group": "CatBoost",
                    "family": "catboost",
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
                        "method_group": "CatBoost",
                        "family": "catboost",
                        "feature_set": FEATURE_SET,
                        "repeat_seed": seed,
                        "fold": int(fold_data["fold"]),
                    }
                )
            )
        print(
            f"[CatBoost {idx}/{len(candidates)}] {candidate.name} "
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
                method_group="CatBoost",
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


def build_catboost_ensembles(
    score_sources: dict[str, pd.DataFrame],
    f1_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_summary = pd.read_csv(TARGETED_DIR / "all_targeted_summary.csv")
    strict = target_summary.loc[target_summary["target_status"].eq("strict")].copy()
    if strict.empty:
        strict = target_summary.copy()
    current_best = str(
        strict.sort_values(
            ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
            ascending=[False, False, True],
        ).iloc[0]["candidate"]
    )

    cat_summary = pd.read_csv(OUT_DIR / "catboost_candidate_summary.csv")
    best_cat = str(
        cat_summary.sort_values(
            ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
            ascending=[False, False, True],
        ).iloc[0]["candidate"]
    )

    rows = []
    oof_frames = []

    pair_members = [current_best, best_cat]
    base, y, matrix = _aligned_score_matrix(score_sources, pair_members)
    for cat_weight in np.round(np.arange(0.1, 0.61, 0.1), 1):
        weights = {current_best: float(1.0 - cat_weight), best_cat: float(cat_weight)}
        score = matrix @ np.asarray([weights[current_best], weights[best_cat]], dtype=float)
        name = f"CATENS_current:{1.0 - cat_weight:.1f}_{best_cat}:{cat_weight:.1f}"
        rows.append(
            summarize_score(
                candidate=name,
                method_group="CatBoost added ensemble",
                y_true=y,
                y_score=score,
                f1_floor=f1_floor,
                details=json.dumps(weights, ensure_ascii=False),
            )
        )

    simplex_members = [
        "M3_XGB_NW_shallow_reg",
        "M3_RF_depth8_balanced",
        "M3_XGB_SPW_regularized_x1.5",
        best_cat,
    ]
    simplex_members = [member for member in simplex_members if member in score_sources]
    if len(simplex_members) >= 3 and best_cat in simplex_members:
        base, y, matrix = _aligned_score_matrix(score_sources, simplex_members)
        for weights_tuple in _simplex_weights(len(simplex_members), units=10):
            weights = {
                member: weight
                for member, weight in zip(simplex_members, weights_tuple)
                if weight > 0
            }
            if weights.get(best_cat, 0.0) <= 0:
                continue
            if sum(value > 0 for value in weights.values()) < 2:
                continue
            score = matrix @ np.asarray(weights_tuple, dtype=float)
            name = "CATOPT_" + "_".join(
                f"{member}:{weight:.1f}" for member, weight in weights.items()
            )
            rows.append(
                summarize_score(
                    candidate=name,
                    method_group="CatBoost added ensemble",
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
                    "method_group": "CatBoost added ensemble",
                }
            )
        )

    ensemble_oof = pd.concat(oof_frames, ignore_index=True)
    return summary, ensemble_oof


def _write_top_table(summary: pd.DataFrame, path: Path, top_n: int = 15) -> None:
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
    summary.loc[:, [c for c in columns if c in summary.columns]].head(top_n).to_csv(
        path,
        index=False,
    )


def main() -> dict[str, Any]:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target_summary = pd.read_csv(TARGETED_DIR / "all_targeted_summary.csv")
    current_best_f1 = float(target_summary["target_f1"].max())
    f1_floor = current_best_f1 - F1_KEEP_TOLERANCE

    candidates = build_candidates()
    fold_results, cat_oof, cat_summary = run_catboost_candidates(candidates, f1_floor)
    fold_results.to_csv(OUT_DIR / "catboost_fold_results.csv", index=False)
    cat_oof.to_csv(OUT_DIR / "catboost_oof_predictions.csv", index=False)
    cat_summary.to_csv(OUT_DIR / "catboost_candidate_summary.csv", index=False)

    score_sources = load_tuned_oof()
    for extra_file in [
        TARGETED_DIR / "method2_optimized_ensemble_oof.csv",
        TARGETED_DIR / "new_model_oof_predictions.csv",
        cat_oof,
    ]:
        if isinstance(extra_file, pd.DataFrame):
            score_sources.update(_averaged_from_oof(extra_file))
        elif extra_file.exists():
            score_sources.update(
                _averaged_from_oof(pd.read_csv(extra_file, dtype={ID_COLUMN: "string"}))
            )

    ensemble_summary, ensemble_oof = build_catboost_ensembles(score_sources, f1_floor)
    ensemble_summary.to_csv(OUT_DIR / "catboost_ensemble_summary.csv", index=False)
    ensemble_oof.to_csv(OUT_DIR / "catboost_ensemble_oof.csv", index=False)

    current_rows = target_summary.assign(method_group="Current best before CatBoost")
    comparison = pd.concat(
        [
            current_rows,
            cat_summary.assign(experiment_block="7_catboost_single"),
            ensemble_summary.assign(experiment_block="7_catboost_ensemble"),
        ],
        ignore_index=True,
    ).sort_values(
        ["target_mcc", "target_f1", "target_expected_fp_per_3500"],
        ascending=[False, False, True],
    )
    comparison.to_csv(OUT_DIR / "catboost_vs_current_summary.csv", index=False)
    _write_top_table(cat_summary, OUT_DIR / "catboost_candidate_top.csv")
    _write_top_table(ensemble_summary, OUT_DIR / "catboost_ensemble_top.csv")
    _write_top_table(comparison, OUT_DIR / "catboost_vs_current_top.csv", top_n=20)

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
            "fold_results": "results/modeling/catboost/catboost_fold_results.csv",
            "oof_predictions": "results/modeling/catboost/catboost_oof_predictions.csv",
            "candidate_summary": "results/modeling/catboost/catboost_candidate_summary.csv",
            "ensemble_summary": "results/modeling/catboost/catboost_ensemble_summary.csv",
            "comparison": "results/modeling/catboost/catboost_vs_current_summary.csv",
        },
    }
    (OUT_DIR / "catboost_metadata.json").write_text(
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
    print("\n=== CATBOOST VS CURRENT TOP ===")
    print(comparison.loc[:, display_cols].head(20).round(4).to_string(index=False))
    return metadata


if __name__ == "__main__":
    main()
