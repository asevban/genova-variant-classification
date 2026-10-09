"""PAH-style governance audits for the MASTER final modeling package.

This script does not fit a new production model. It reuses existing nested
governance predictions, final-freeze metadata and raw labelled rows to add the
audits that were useful in the PAH package:

* one explicit official artefact registry
* fold-level fragility counts
* Hybrid-E vs CATOPT-A paired deltas with Nadeau-Bengio correction
* cluster bootstrap confidence intervals over Variant_ID
* missingness/provenance stress summaries
* repeated false-positive and OOF leave-one-Variant_ID-out influence tables
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

ID_COLUMN = "Variant_ID"
TARGET_COLUMN = "Label"
FINAL_PATHOGENIC_RATE = 500 / 3500
FINAL_TOTAL_N = 3500
FP_TARGET = 400.0
STRICT_SENSITIVITY_FLOOR = 0.62
RELAXED_F1_FLOOR = 0.530

RAW_FILE = ROOT / "data" / "raw" / "YARISMA_TRAIN_MASTER.csv"
NESTED_DIR = ROOT / "results" / "modeling" / "nested_governance"
FREEZE_DIR = ROOT / "results" / "modeling" / "final_hybrid_freeze"
OUT_DIR = ROOT / "results" / "modeling" / "pah_style_governance_audits"

PRIMARY_STRATEGY = "Hybrid-E_bestMCC_nested"
REFERENCE_STRATEGY = "CATOPT-A_nested_reference"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return pd.read_csv(path, low_memory=False)


def _safe_auc(y_true: np.ndarray, score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score, sample_weight=sample_weight))


def _safe_auprc(
    y_true: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score, sample_weight=sample_weight))


def _metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    pi = FINAL_PATHOGENIC_RATE
    final_tp = pi * sensitivity
    final_fn = pi * (1.0 - sensitivity)
    final_fp = (1.0 - pi) * (1.0 - specificity)
    final_tn = (1.0 - pi) * specificity
    precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) else 0.0
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )
    denom = (final_tp + final_fp) * (final_tp + final_fn) * (final_tn + final_fp) * (
        final_tn + final_fn
    )
    mcc = ((final_tp * final_tn - final_fp * final_fn) / math.sqrt(denom)) if denom else 0.0
    return {
        "final_f1": float(f1),
        "final_mcc": float(mcc),
        "final_specificity": float(specificity),
        "final_sensitivity": float(sensitivity),
        "final_precision": float(precision),
        "expected_fp_per_3500": float(FINAL_TOTAL_N * final_fp),
        "expected_fn_per_3500": float(FINAL_TOTAL_N * final_fn),
    }


def _count_components(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _final_metrics_from_rows(frame: pd.DataFrame) -> dict[str, float]:
    counts = _count_components(
        frame[TARGET_COLUMN].to_numpy(dtype=int),
        frame["prediction"].to_numpy(dtype=int),
    )
    metrics = _metrics_from_counts(**counts)
    y_true = frame[TARGET_COLUMN].to_numpy(dtype=int)
    score = frame["score"].to_numpy(dtype=float)
    source_pi = float(y_true.mean()) if len(y_true) else 0.0
    if 0.0 < source_pi < 1.0:
        sample_weight = np.where(
            y_true == 1,
            FINAL_PATHOGENIC_RATE / source_pi,
            (1.0 - FINAL_PATHOGENIC_RATE) / (1.0 - source_pi),
        )
    else:
        sample_weight = None
    metrics.update(
        {
            **counts,
            "oof_auroc": _safe_auc(y_true, score),
            "oof_auprc": _safe_auprc(y_true, score),
            "final_weighted_auprc": _safe_auprc(y_true, score, sample_weight),
            "rows": int(len(frame)),
            "unique_variants": int(frame[ID_COLUMN].nunique()),
        }
    )
    return metrics


def _raw_missingness_profile(raw: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in raw.columns if col not in {ID_COLUMN, TARGET_COLUMN}]
    missing = raw[feature_cols].isna()
    profile = pd.DataFrame(
        {
            ID_COLUMN: raw[ID_COLUMN].astype(str),
            "raw_feature_count": len(feature_cols),
            "missing_feature_count": missing.sum(axis=1).astype(int),
            "missing_rate": missing.mean(axis=1).astype(float),
            "all_features_missing": missing.all(axis=1).astype(int),
        }
    )
    q1, q2, q3 = profile["missing_rate"].quantile([0.25, 0.50, 0.75]).to_list()

    def bin_name(value: float) -> str:
        if value <= q1:
            return "Q1_low_missing"
        if value <= q2:
            return "Q2"
        if value <= q3:
            return "Q3"
        return "Q4_high_missing"

    profile["missingness_bin"] = profile["missing_rate"].map(bin_name)
    return profile


def _write_artifact_registry() -> dict[str, Any]:
    registry = {
        "decision_date": "2026-09-02",
        "panel": "MASTER",
        "primary_official_inference_artifact": (
            "results/modeling/final_hybrid_freeze/"
            "MASTER_Hybrid-E_bestMCC_nested_development_train_frozen.joblib"
        ),
        "primary_strategy": PRIMARY_STRATEGY,
        "primary_reason": (
            "Current package default. Its final-freeze metadata has strict inner "
            "selection status and the nested governance report keeps the performance "
            "claim separate from the inference artefact."
        ),
        "prediction_script": "scripts/master_predict_hybrid_e_frozen.py",
        "performance_claim_source": (
            "results/modeling/nested_governance/HYBRID_NESTED_GOVERNANCE_REPORT.md"
        ),
        "accepted_fallback_artifact": (
            "results/modeling/final_hybrid_freeze/"
            "MASTER_Hybrid-C_nested_selected_all_labeled_frozen.joblib"
        ),
        "fallback_reason": (
            "Use only when an all-labelled strict artefact is explicitly required; "
            "the all-labelled Hybrid-E artefact selected an mcc_only configuration."
        ),
        "non_primary_artifacts": [
            {
                "path": (
                    "results/modeling/final_hybrid_freeze/"
                    "MASTER_Hybrid-E_bestMCC_nested_all_labeled_frozen.joblib"
                ),
                "status": "non_primary",
                "reason": "All-labelled Hybrid-E metadata selected mcc_only, not strict.",
            },
            {
                "path": (
                    "results/modeling/final_hybrid_freeze/"
                    "MASTER_HybridE_bestMCC_nested_frozen.joblib"
                ),
                "status": "legacy_name",
                "reason": "Older hyphenless artefact name; do not use as the default interface.",
            },
        ],
    }
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    (FREEZE_DIR / "MASTER_ARTIFACT_REGISTRY.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "MASTER_ARTIFACT_REGISTRY.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return registry


def _fold_fragility(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, group in fold_results.groupby("strategy", sort=False):
        rows.append(
            {
                "strategy": strategy,
                "outer_folds": int(len(group)),
                "strict_pass_count": int(group["outer_strict_pass"].astype(bool).sum()),
                "strict_pass_rate": float(group["outer_strict_pass"].astype(bool).mean()),
                "fp_over_400_count": int(group["outer_expected_fp_per_3500"].gt(FP_TARGET).sum()),
                "sensitivity_below_0_62_count": int(
                    group["outer_sensitivity"].lt(STRICT_SENSITIVITY_FLOOR).sum()
                ),
                "f1_below_0_530_count": int(group["outer_f1"].lt(RELAXED_F1_FLOOR).sum()),
                "mcc_below_0_40_count": int(group["outer_mcc"].lt(0.40).sum()),
                "fp_over_400_and_sens_below_count": int(
                    (
                        group["outer_expected_fp_per_3500"].gt(FP_TARGET)
                        & group["outer_sensitivity"].lt(STRICT_SENSITIVITY_FLOOR)
                    ).sum()
                ),
                "worst_f1": float(group["outer_f1"].min()),
                "worst_mcc": float(group["outer_mcc"].min()),
                "max_fp_per_3500": float(group["outer_expected_fp_per_3500"].max()),
                "min_sensitivity": float(group["outer_sensitivity"].min()),
                "mean_fp_per_3500": float(group["outer_expected_fp_per_3500"].mean()),
                "mean_mcc": float(group["outer_mcc"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strict_pass_rate", "mean_mcc", "mean_fp_per_3500"],
        ascending=[False, False, True],
    )


def _paired_nb_tests(
    fold_results: pd.DataFrame,
    candidate: str,
    reference: str,
) -> pd.DataFrame:
    metrics = [
        ("outer_f1", False),
        ("outer_mcc", False),
        ("outer_expected_fp_per_3500", True),
        ("outer_sensitivity", False),
        ("outer_specificity", False),
        ("outer_oof_auroc", False),
        ("outer_final_weighted_auprc", False),
    ]
    key_cols = ["outer_seed", "outer_fold"]
    candidate_rows = fold_results.loc[fold_results["strategy"].eq(candidate), key_cols + [m for m, _ in metrics]]
    reference_rows = fold_results.loc[fold_results["strategy"].eq(reference), key_cols + [m for m, _ in metrics]]
    merged = candidate_rows.merge(reference_rows, on=key_cols, suffixes=("_candidate", "_reference"))
    folds_per_repeat = (
        fold_results.groupby("outer_seed")["outer_fold"].nunique().median()
        if "outer_seed" in fold_results.columns
        else 5
    )
    k = max(2, int(round(float(folds_per_repeat))))
    train_test_ratio = 1.0 / (k - 1)
    m = len(merged)
    rows: list[dict[str, Any]] = []
    for metric, lower_is_better in metrics:
        delta = (
            merged[f"{metric}_candidate"].to_numpy(dtype=float)
            - merged[f"{metric}_reference"].to_numpy(dtype=float)
        )
        mean_delta = float(np.mean(delta))
        sd_delta = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
        corrected_factor = (1.0 / m) + train_test_ratio if m else float("nan")
        corrected_se = math.sqrt(corrected_factor * (sd_delta**2)) if sd_delta else 0.0
        t_stat = mean_delta / corrected_se if corrected_se else float("inf") if mean_delta else 0.0
        p_approx = math.erfc(abs(t_stat) / math.sqrt(2.0)) if math.isfinite(t_stat) else 0.0
        if lower_is_better:
            win_rate = float((delta <= 0).mean())
        else:
            win_rate = float((delta >= 0).mean())
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "metric": metric,
                "delta_mean_candidate_minus_reference": mean_delta,
                "delta_sd": sd_delta,
                "paired_outer_folds": int(m),
                "folds_per_repeat_assumption": int(k),
                "nadeau_bengio_train_test_ratio": float(train_test_ratio),
                "nadeau_bengio_corrected_se": float(corrected_se),
                "nadeau_bengio_t": float(t_stat),
                "normal_approx_two_sided_p": float(p_approx),
                "candidate_win_rate": win_rate,
                "lower_is_better": bool(lower_is_better),
            }
        )
    return pd.DataFrame(rows)


def _cluster_bootstrap_ci(
    predictions: pd.DataFrame,
    candidate: str,
    reference: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    cand = predictions.loc[predictions["strategy"].eq(candidate)].reset_index(drop=True)
    ref = predictions.loc[predictions["strategy"].eq(reference)].reset_index(drop=True)
    ids = sorted(set(cand[ID_COLUMN].astype(str)).intersection(ref[ID_COLUMN].astype(str)))
    cand_groups = {key: values.index.to_numpy() for key, values in cand.groupby(ID_COLUMN)}
    ref_groups = {key: values.index.to_numpy() for key, values in ref.groupby(ID_COLUMN)}
    rng = np.random.default_rng(seed)
    metric_names = [
        "final_f1",
        "final_mcc",
        "expected_fp_per_3500",
        "final_sensitivity",
        "final_specificity",
    ]
    deltas = {metric: [] for metric in metric_names}
    id_array = np.asarray(ids, dtype=object)
    for _ in range(iterations):
        sampled_ids = rng.choice(id_array, size=len(id_array), replace=True)
        cand_idx = np.concatenate([cand_groups[item] for item in sampled_ids])
        ref_idx = np.concatenate([ref_groups[item] for item in sampled_ids])
        cand_metrics = _final_metrics_from_rows(cand.iloc[cand_idx])
        ref_metrics = _final_metrics_from_rows(ref.iloc[ref_idx])
        for metric in metric_names:
            deltas[metric].append(cand_metrics[metric] - ref_metrics[metric])

    rows = []
    for metric, values in deltas.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "metric": metric,
                "bootstrap_iterations": int(iterations),
                "cluster_unit": ID_COLUMN,
                "delta_mean": float(np.nanmean(arr)),
                "delta_ci025": float(np.nanpercentile(arr, 2.5)),
                "delta_ci500": float(np.nanpercentile(arr, 50.0)),
                "delta_ci975": float(np.nanpercentile(arr, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def _subgroup_metrics(predictions: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    joined = predictions.merge(profile, on=ID_COLUMN, how="left")
    rows: list[dict[str, Any]] = []
    for (strategy, missing_bin), group in joined.groupby(["strategy", "missingness_bin"], sort=False):
        metrics = _final_metrics_from_rows(group)
        rows.append(
            {
                "strategy": strategy,
                "subgroup": missing_bin,
                "mean_missing_rate": float(group["missing_rate"].mean()),
                **metrics,
            }
        )
    for strategy, group in joined.groupby("strategy", sort=False):
        for flag_value, subgroup in [(1, "all_features_missing"), (0, "not_all_features_missing")]:
            subset = group.loc[group["all_features_missing"].eq(flag_value)]
            if subset.empty:
                continue
            metrics = _final_metrics_from_rows(subset)
            rows.append(
                {
                    "strategy": strategy,
                    "subgroup": subgroup,
                    "mean_missing_rate": float(subset["missing_rate"].mean()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _adversarial_missingness_audit(
    raw: pd.DataFrame,
    profile: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_cols = [col for col in raw.columns if col not in {ID_COLUMN, TARGET_COLUMN}]
    y_label = pd.to_numeric(raw[TARGET_COLUMN], errors="raise").astype(int).to_numpy()
    missing_rate = profile["missing_rate"].to_numpy(dtype=float)
    rows.append(
        {
            "audit": "missing_rate_predicts_label",
            "strategy": "raw_missingness_only",
            "auc": _safe_auc(y_label, missing_rate),
            "separation_auc": max(_safe_auc(y_label, missing_rate), 1.0 - _safe_auc(y_label, missing_rate)),
            "rows": int(len(raw)),
            "note": "AUC of raw missing-rate alone against the pathogenic label.",
        }
    )

    missing_signature = raw[feature_cols].isna().astype(np.int8).to_numpy()
    if len(np.unique(y_label)) == 2:
        oof = np.zeros(len(y_label), dtype=float)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260815)
        for train_idx, hold_idx in splitter.split(missing_signature, y_label):
            model = LogisticRegression(
                C=1.0,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
            )
            model.fit(missing_signature[train_idx], y_label[train_idx])
            oof[hold_idx] = model.predict_proba(missing_signature[hold_idx])[:, 1]
        auc = _safe_auc(y_label, oof)
        rows.append(
            {
                "audit": "missingness_signature_predicts_label",
                "strategy": "raw_missingness_signature_cv",
                "auc": auc,
                "separation_auc": max(auc, 1.0 - auc) if math.isfinite(auc) else float("nan"),
                "rows": int(len(raw)),
                "note": "5-fold OOF logistic model using only feature-missing indicators.",
            }
        )

    joined = predictions.merge(profile, on=ID_COLUMN, how="left")
    for strategy, group in joined.groupby("strategy", sort=False):
        q14 = group.loc[group["missingness_bin"].isin(["Q1_low_missing", "Q4_high_missing"])]
        if not q14.empty:
            y_adv = q14["missingness_bin"].eq("Q4_high_missing").astype(int).to_numpy()
            auc = _safe_auc(y_adv, q14["score"].to_numpy(dtype=float))
            rows.append(
                {
                    "audit": "score_predicts_q4_vs_q1_missingness",
                    "strategy": strategy,
                    "auc": auc,
                    "separation_auc": max(auc, 1.0 - auc) if math.isfinite(auc) else float("nan"),
                    "rows": int(len(q14)),
                    "note": "AUC of model score for separating Q4 high-missing from Q1 low-missing rows.",
                }
            )
        y_high = group["missingness_bin"].eq("Q4_high_missing").astype(int).to_numpy()
        auc = _safe_auc(y_high, group["score"].to_numpy(dtype=float))
        rows.append(
            {
                "audit": "score_predicts_q4_vs_rest_missingness",
                "strategy": strategy,
                "auc": auc,
                "separation_auc": max(auc, 1.0 - auc) if math.isfinite(auc) else float("nan"),
                "rows": int(len(group)),
                "note": "AUC of model score for separating Q4 high-missing from all other rows.",
            }
        )
    return pd.DataFrame(rows)


def _repeated_fp_and_loo(
    predictions: pd.DataFrame,
    profile: pd.DataFrame,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.loc[predictions["strategy"].eq(strategy)].merge(profile, on=ID_COLUMN, how="left")
    base_counts = _count_components(frame[TARGET_COLUMN].to_numpy(), frame["prediction"].to_numpy())
    base_metrics = _metrics_from_counts(**base_counts)
    total_counts = pd.Series(base_counts)

    influence_rows: list[dict[str, Any]] = []
    for variant_id, group in frame.groupby(ID_COLUMN, sort=False):
        removed_counts = _count_components(group[TARGET_COLUMN].to_numpy(), group["prediction"].to_numpy())
        loo_counts = (total_counts - pd.Series(removed_counts)).astype(int).to_dict()
        loo_metrics = _metrics_from_counts(**loo_counts)
        influence_rows.append(
            {
                ID_COLUMN: variant_id,
                "label": int(group[TARGET_COLUMN].iloc[0]),
                "evaluations": int(len(group)),
                "positive_prediction_count": int(group["prediction"].sum()),
                "fp_count": int(((group[TARGET_COLUMN] == 0) & (group["prediction"] == 1)).sum()),
                "fn_count": int(((group[TARGET_COLUMN] == 1) & (group["prediction"] == 0)).sum()),
                "mean_score": float(group["score"].mean()),
                "max_score": float(group["score"].max()),
                "mean_threshold": float(group["threshold"].mean()),
                "missing_rate": float(group["missing_rate"].iloc[0]),
                "missingness_bin": str(group["missingness_bin"].iloc[0]),
                "all_features_missing": int(group["all_features_missing"].iloc[0]),
                "delta_f1_if_removed": float(loo_metrics["final_f1"] - base_metrics["final_f1"]),
                "delta_mcc_if_removed": float(loo_metrics["final_mcc"] - base_metrics["final_mcc"]),
                "delta_fp_per_3500_if_removed": float(
                    loo_metrics["expected_fp_per_3500"] - base_metrics["expected_fp_per_3500"]
                ),
            }
        )
    influence = pd.DataFrame(influence_rows).sort_values(
        ["delta_mcc_if_removed", "delta_f1_if_removed"],
        ascending=[False, False],
    )
    repeated_fp = influence.loc[influence["fp_count"].gt(0)].copy()
    repeated_fp = repeated_fp.sort_values(
        ["fp_count", "mean_score", "delta_mcc_if_removed"],
        ascending=[False, False, False],
    )
    return repeated_fp, influence


def _write_report(
    registry: dict[str, Any],
    fold_fragility: pd.DataFrame,
    nb_tests: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    subgroup: pd.DataFrame,
    adversarial: pd.DataFrame,
    repeated_fp: pd.DataFrame,
    influence: pd.DataFrame,
    path: Path,
) -> None:
    primary_fragility = fold_fragility.loc[fold_fragility["strategy"].eq(PRIMARY_STRATEGY)].iloc[0]
    ref_fragility = fold_fragility.loc[fold_fragility["strategy"].eq(REFERENCE_STRATEGY)].iloc[0]
    mcc_nb = nb_tests.loc[nb_tests["metric"].eq("outer_mcc")].iloc[0]
    f1_nb = nb_tests.loc[nb_tests["metric"].eq("outer_f1")].iloc[0]
    fp_nb = nb_tests.loc[nb_tests["metric"].eq("outer_expected_fp_per_3500")].iloc[0]
    primary_q4 = subgroup.loc[
        subgroup["strategy"].eq(PRIMARY_STRATEGY) & subgroup["subgroup"].eq("Q4_high_missing")
    ]
    primary_missing_score = adversarial.loc[
        adversarial["strategy"].eq(PRIMARY_STRATEGY)
        & adversarial["audit"].eq("score_predicts_q4_vs_q1_missingness")
    ]
    signature_auc = adversarial.loc[
        adversarial["audit"].eq("missingness_signature_predicts_label")
    ]

    lines = [
        "# MASTER PAH-Style Governance Audit Report",
        "",
        "Bu rapor PAH paketindeki kalite-guvence fikirlerini MASTER modelleme paketine uygular. Yeni bir final model egitmez; mevcut nested governance ve frozen artefact ciktisini denetler.",
        "",
        "## Resmi Artefact Karari",
        "",
        f"- Primary official inference artefact: `{registry['primary_official_inference_artifact']}`",
        f"- Prediction script: `{registry['prediction_script']}`",
        f"- Performance claim source: `{registry['performance_claim_source']}`",
        f"- Accepted fallback: `{registry['accepted_fallback_artifact']}`",
        "",
        "## Hybrid-E vs CATOPT-A Istatistikleri",
        "",
        "| Test | Delta | Win rate | NB corrected SE | Normal approx p |",
        "|---|---:|---:|---:|---:|",
        f"| F1 | {f1_nb['delta_mean_candidate_minus_reference']:+.4f} | {f1_nb['candidate_win_rate']:.2f} | {f1_nb['nadeau_bengio_corrected_se']:.4f} | {f1_nb['normal_approx_two_sided_p']:.3f} |",
        f"| MCC | {mcc_nb['delta_mean_candidate_minus_reference']:+.4f} | {mcc_nb['candidate_win_rate']:.2f} | {mcc_nb['nadeau_bengio_corrected_se']:.4f} | {mcc_nb['normal_approx_two_sided_p']:.3f} |",
        f"| FP/3500 | {fp_nb['delta_mean_candidate_minus_reference']:+.1f} | {fp_nb['candidate_win_rate']:.2f} | {fp_nb['nadeau_bengio_corrected_se']:.1f} | {fp_nb['normal_approx_two_sided_p']:.3f} |",
        "",
        "Yorum: Hybrid-E ortalamada CATOPT-A'dan biraz iyi gorunuyor; fakat Nadeau-Bengio duzeltmesi kucuk deltalari temkinli yorumlamayi gerektirir.",
        "",
        "## Fold Fragility",
        "",
        "| Strategy | Strict pass | FP>400 | Sens<0.62 | F1<0.530 | Worst MCC | Max FP/3500 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {PRIMARY_STRATEGY} | {int(primary_fragility['strict_pass_count'])}/{int(primary_fragility['outer_folds'])} | {int(primary_fragility['fp_over_400_count'])} | {int(primary_fragility['sensitivity_below_0_62_count'])} | {int(primary_fragility['f1_below_0_530_count'])} | {primary_fragility['worst_mcc']:.4f} | {primary_fragility['max_fp_per_3500']:.1f} |",
        f"| {REFERENCE_STRATEGY} | {int(ref_fragility['strict_pass_count'])}/{int(ref_fragility['outer_folds'])} | {int(ref_fragility['fp_over_400_count'])} | {int(ref_fragility['sensitivity_below_0_62_count'])} | {int(ref_fragility['f1_below_0_530_count'])} | {ref_fragility['worst_mcc']:.4f} | {ref_fragility['max_fp_per_3500']:.1f} |",
        "",
        "## Missingness / Provenance",
        "",
    ]
    if not primary_q4.empty:
        q4 = primary_q4.iloc[0]
        lines.extend(
            [
                f"- Hybrid-E Q4 high-missing subgroup: F1={q4['final_f1']:.4f}, MCC={q4['final_mcc']:.4f}, specificity={q4['final_specificity']:.4f}, FP/3500={q4['expected_fp_per_3500']:.1f}.",
            ]
        )
    if not primary_missing_score.empty:
        row = primary_missing_score.iloc[0]
        lines.append(
            f"- Hybrid-E score Q4-vs-Q1 missingness separation AUC={row['separation_auc']:.4f}."
        )
    if not signature_auc.empty:
        row = signature_auc.iloc[0]
        lines.append(
            f"- Missingness-signature-only label AUC={row['auc']:.4f}; bu deger provenance shortcut riskini canli tutar."
        )
    lines.extend(
        [
            "",
            "## Repeated FP / LOO Influence",
            "",
            f"- Hybrid-E repeated FP variant count: `{int(repeated_fp[ID_COLUMN].nunique())}`",
            f"- Top LOO MCC improvement if removed: `{influence['delta_mcc_if_removed'].max():+.6f}`",
            "- LOO influence tablosu yeniden egitim yapmadan OOF tahminlerinden hesaplanir; row deletion karari yerine domain incelemesi icin oncelik listesi olarak kullanilmalidir.",
            "",
            "## Output Files",
            "",
            "- `results/modeling/final_hybrid_freeze/MASTER_ARTIFACT_REGISTRY.json`",
            "- `results/modeling/pah_style_governance_audits/fold_fragility_summary.csv`",
            "- `results/modeling/pah_style_governance_audits/hybrid_e_vs_catopt_nb_tests.csv`",
            "- `results/modeling/pah_style_governance_audits/hybrid_e_vs_catopt_cluster_bootstrap_ci.csv`",
            "- `results/modeling/pah_style_governance_audits/missingness_subgroup_stress.csv`",
            "- `results/modeling/pah_style_governance_audits/missingness_adversarial_audit.csv`",
            "- `results/modeling/pah_style_governance_audits/hybrid_e_repeated_false_positives.csv`",
            "- `results/modeling/pah_style_governance_audits/hybrid_e_oof_loo_influence.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = _read_csv(RAW_FILE)
    raw[ID_COLUMN] = raw[ID_COLUMN].astype(str)
    fold_results = _read_csv(NESTED_DIR / "nested_outer_fold_results.csv")
    predictions = _read_csv(NESTED_DIR / "nested_outer_predictions.csv")
    predictions[ID_COLUMN] = predictions[ID_COLUMN].astype(str)
    predictions[TARGET_COLUMN] = pd.to_numeric(predictions[TARGET_COLUMN], errors="raise").astype(int)
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="raise").astype(int)

    registry = _write_artifact_registry()
    profile = _raw_missingness_profile(raw)
    fold_fragility = _fold_fragility(fold_results)
    nb_tests = _paired_nb_tests(fold_results, PRIMARY_STRATEGY, REFERENCE_STRATEGY)
    bootstrap_ci = _cluster_bootstrap_ci(
        predictions,
        PRIMARY_STRATEGY,
        REFERENCE_STRATEGY,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    subgroup = _subgroup_metrics(predictions, profile)
    adversarial = _adversarial_missingness_audit(raw, profile, predictions)
    repeated_fp, influence = _repeated_fp_and_loo(predictions, profile, PRIMARY_STRATEGY)

    fold_fragility.to_csv(OUT_DIR / "fold_fragility_summary.csv", index=False)
    nb_tests.to_csv(OUT_DIR / "hybrid_e_vs_catopt_nb_tests.csv", index=False)
    bootstrap_ci.to_csv(OUT_DIR / "hybrid_e_vs_catopt_cluster_bootstrap_ci.csv", index=False)
    subgroup.to_csv(OUT_DIR / "missingness_subgroup_stress.csv", index=False)
    adversarial.to_csv(OUT_DIR / "missingness_adversarial_audit.csv", index=False)
    repeated_fp.to_csv(OUT_DIR / "hybrid_e_repeated_false_positives.csv", index=False)
    influence.to_csv(OUT_DIR / "hybrid_e_oof_loo_influence.csv", index=False)

    report_path = OUT_DIR / "MASTER_PAH_STYLE_GOVERNANCE_AUDIT_REPORT.md"
    _write_report(
        registry,
        fold_fragility,
        nb_tests,
        bootstrap_ci,
        subgroup,
        adversarial,
        repeated_fp,
        influence,
        report_path,
    )

    metadata = {
        "status": "completed",
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "primary_strategy": PRIMARY_STRATEGY,
        "reference_strategy": REFERENCE_STRATEGY,
        "outputs": {
            "report": str(report_path.relative_to(ROOT)),
            "artifact_registry": str(
                (FREEZE_DIR / "MASTER_ARTIFACT_REGISTRY.json").relative_to(ROOT)
            ),
            "fold_fragility": str((OUT_DIR / "fold_fragility_summary.csv").relative_to(ROOT)),
            "nb_tests": str((OUT_DIR / "hybrid_e_vs_catopt_nb_tests.csv").relative_to(ROOT)),
            "bootstrap_ci": str(
                (OUT_DIR / "hybrid_e_vs_catopt_cluster_bootstrap_ci.csv").relative_to(ROOT)
            ),
            "missingness_subgroup_stress": str(
                (OUT_DIR / "missingness_subgroup_stress.csv").relative_to(ROOT)
            ),
            "adversarial_missingness": str(
                (OUT_DIR / "missingness_adversarial_audit.csv").relative_to(ROOT)
            ),
            "repeated_false_positives": str(
                (OUT_DIR / "hybrid_e_repeated_false_positives.csv").relative_to(ROOT)
            ),
            "loo_influence": str((OUT_DIR / "hybrid_e_oof_loo_influence.csv").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "master_pah_style_governance_audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


if __name__ == "__main__":
    main()
