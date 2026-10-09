"""Post-freeze safety audits for the MASTER Hybrid-E candidate.

These audits do not train new models and do not perform new model selection.
They summarize existing nested-governance outputs so the frozen Hybrid-E
candidate can be reported with clearer calibration, FP and stability evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import (  # noqa: E402
    ID_COLUMN,
    TARGET_COLUMN,
    feature_columns,
    load_master_csv,
)
from master_tuned_modeling_experiments import FINAL_PATHOGENIC_RATE, RAW_FILE  # noqa: E402


NESTED_DIR = ROOT / "results" / "modeling" / "nested_governance"
OUT_DIR = ROOT / "results" / "modeling" / "final_hybrid_freeze"
SELECTED = "Hybrid-E_bestMCC_nested"
REFERENCE = "CATOPT-A_nested_reference"
AUDIT_STRATEGIES = [
    "Hybrid-E_bestMCC_nested",
    "Hybrid-E_AUPRC_guard_nested",
    "Hybrid-E_lowFP_nested",
    "CATOPT-A_nested_reference",
]


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


def _weighted_brier(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    observed_pi = float(y_true.mean())
    pos_weight = FINAL_PATHOGENIC_RATE / observed_pi
    neg_weight = (1.0 - FINAL_PATHOGENIC_RATE) / (1.0 - observed_pi)
    weights = np.where(y_true == 1, pos_weight, neg_weight)
    weights = weights / weights.mean()
    return float(np.average((score - y_true) ** 2, weights=weights))


def _ece(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(score)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (score >= lower) & (score <= upper)
        else:
            mask = (score >= lower) & (score < upper)
        if not mask.any():
            continue
        confidence = float(score[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (mask.sum() / total) * abs(confidence - accuracy)
    return float(ece)


def _calibration_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in AUDIT_STRATEGIES:
        group = predictions.loc[predictions["strategy"].eq(strategy)]
        y = group[TARGET_COLUMN].to_numpy(dtype=int)
        score = np.clip(group["score"].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
        rows.append(
            {
                "strategy": strategy,
                "rows": int(len(group)),
                "brier": float(brier_score_loss(y, score)),
                "final_prior_weighted_brier": _weighted_brier(y, score),
                "log_loss": float(log_loss(y, score, labels=[0, 1])),
                "ece_10bin": _ece(y, score, bins=10),
                "mean_score": float(score.mean()),
                "label_rate": float(y.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("final_prior_weighted_brier")


def _paired_delta(outer: pd.DataFrame) -> pd.DataFrame:
    key = ["outer_seed", "outer_fold"]
    ref = outer.loc[outer["strategy"].eq(REFERENCE)].set_index(key).sort_index()
    rows = []
    for strategy in [item for item in AUDIT_STRATEGIES if item != REFERENCE]:
        group = outer.loc[outer["strategy"].eq(strategy)].set_index(key).sort_index().loc[ref.index]
        f1_delta = group["outer_f1"] - ref["outer_f1"]
        mcc_delta = group["outer_mcc"] - ref["outer_mcc"]
        fp_delta = group["outer_expected_fp_per_3500"] - ref["outer_expected_fp_per_3500"]
        auprc_delta = group["outer_final_weighted_auprc"] - ref["outer_final_weighted_auprc"]
        rows.append(
            {
                "strategy": strategy,
                "delta_f1_mean": float(f1_delta.mean()),
                "delta_mcc_mean": float(mcc_delta.mean()),
                "delta_fp_per_3500_mean": float(fp_delta.mean()),
                "delta_final_auprc_mean": float(auprc_delta.mean()),
                "f1_win_rate": float((f1_delta > 0).mean()),
                "mcc_win_rate": float((mcc_delta > 0).mean()),
                "fp_le_reference_rate": float((fp_delta <= 0).mean()),
                "f1_mcc_fp_all_win_rate": float(
                    ((f1_delta >= 0) & (mcc_delta >= 0) & (fp_delta <= 0)).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["delta_mcc_mean", "delta_f1_mean"],
        ascending=[False, False],
    )


def _fold_stability(outer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in AUDIT_STRATEGIES:
        group = outer.loc[outer["strategy"].eq(strategy)]
        rows.append(
            {
                "strategy": strategy,
                "outer_folds": int(len(group)),
                "strict_pass_rate": float(group["outer_strict_pass"].mean()),
                "f1_mean": float(group["outer_f1"].mean()),
                "f1_std": float(group["outer_f1"].std(ddof=1)),
                "f1_min": float(group["outer_f1"].min()),
                "mcc_mean": float(group["outer_mcc"].mean()),
                "mcc_std": float(group["outer_mcc"].std(ddof=1)),
                "mcc_min": float(group["outer_mcc"].min()),
                "fp_per_3500_mean": float(group["outer_expected_fp_per_3500"].mean()),
                "fp_per_3500_std": float(group["outer_expected_fp_per_3500"].std(ddof=1)),
                "fp_per_3500_max": float(group["outer_expected_fp_per_3500"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strict_pass_rate", "mcc_mean"],
        ascending=[False, False],
    )


def _missingness_table() -> pd.DataFrame:
    raw = load_master_csv(RAW_FILE)
    features = feature_columns(raw)
    missing_rate = raw[features].isna().mean(axis=1)
    result = pd.DataFrame(
        {
            ID_COLUMN: raw[ID_COLUMN].astype(str).to_numpy(),
            "missing_rate": missing_rate.to_numpy(dtype=float),
        }
    )
    result["missingness_bin"] = pd.qcut(
        result["missing_rate"].rank(method="first"),
        q=4,
        labels=["Q1_low_missing", "Q2", "Q3", "Q4_high_missing"],
    )
    return result


def _missingness_subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = _missingness_table()
    joined = predictions.merge(missing, on=ID_COLUMN, how="left")
    rows = []
    for (strategy, missing_bin), group in joined.groupby(["strategy", "missingness_bin"], observed=True):
        if strategy not in AUDIT_STRATEGIES:
            continue
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


def _fp_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = _missingness_table()
    selected = predictions.loc[predictions["strategy"].eq(SELECTED)].copy()
    selected["is_fp"] = (
        selected[TARGET_COLUMN].eq(0) & selected["prediction"].eq(1)
    ).astype(int)
    selected["is_tn"] = (
        selected[TARGET_COLUMN].eq(0) & selected["prediction"].eq(0)
    ).astype(int)
    grouped = (
        selected.groupby(ID_COLUMN)
        .agg(
            label=(TARGET_COLUMN, "first"),
            evaluations=("is_fp", "size"),
            fp_count=("is_fp", "sum"),
            tn_count=("is_tn", "sum"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
            mean_threshold=("threshold", "mean"),
        )
        .reset_index()
    )
    grouped["fp_rate"] = grouped["fp_count"] / grouped["evaluations"]
    grouped = grouped.merge(missing, on=ID_COLUMN, how="left")
    return grouped.loc[grouped["fp_count"].gt(0)].sort_values(
        ["fp_count", "fp_rate", "mean_score"],
        ascending=[False, False, False],
    )


def _action_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": "Frozen Hybrid-E artefact",
                "status": "applied",
                "governance_note": "Frozen Hybrid-E procedure instantiated for final/test inference.",
            },
            {
                "step": "Inner-CV final selection",
                "status": "applied",
                "governance_note": "Weights, calibration/prior correction and threshold selected only inside inner CV.",
            },
            {
                "step": "Calibration/Brier audit",
                "status": "applied",
                "governance_note": "Existing nested predictions summarized; no new model search.",
            },
            {
                "step": "FP error analysis",
                "status": "applied",
                "governance_note": "Repeated false positives identified from nested outer predictions.",
            },
            {
                "step": "Missingness subgroup stability",
                "status": "applied",
                "governance_note": "Nested predictions checked across missingness quartiles.",
            },
            {
                "step": "CATOPT-A paired comparison",
                "status": "applied",
                "governance_note": "Same outer seed/fold pairs compared against reference.",
            },
            {
                "step": "Multi-seed bagging",
                "status": "not_applied_to_final",
                "governance_note": "Would be a new model variant; should be nested-validated before final use.",
            },
        ]
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> list[str]:
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
    calibration: pd.DataFrame,
    paired: pd.DataFrame,
    stability: pd.DataFrame,
    missingness: pd.DataFrame,
    fp_errors: pd.DataFrame,
    actions: pd.DataFrame,
) -> None:
    lines = [
        "# MASTER Hybrid-E Strengthening Audit Raporu",
        "",
        "Bu rapor, freeze edilen Hybrid-E adayını yeni bir model araması açmadan güçlendirmek için yapılan güvenlik ve stabilite kontrollerini özetler.",
        "",
        "## Uygulanan Adımlar",
        "",
        *_markdown_table(actions, ["step", "status", "governance_note"], digits=4),
        "",
        "## Calibration / Brier Özeti",
        "",
        *_markdown_table(
            calibration,
            [
                "strategy",
                "brier",
                "final_prior_weighted_brier",
                "log_loss",
                "ece_10bin",
                "mean_score",
            ],
            digits=4,
        ),
        "",
        "## CATOPT-A Paired Fark Özeti",
        "",
        *_markdown_table(
            paired,
            [
                "strategy",
                "delta_f1_mean",
                "delta_mcc_mean",
                "delta_fp_per_3500_mean",
                "delta_final_auprc_mean",
                "f1_win_rate",
                "mcc_win_rate",
            ],
            digits=4,
        ),
        "",
        "## Fold Stabilitesi",
        "",
        *_markdown_table(
            stability,
            [
                "strategy",
                "strict_pass_rate",
                "f1_mean",
                "f1_std",
                "mcc_mean",
                "mcc_std",
                "fp_per_3500_mean",
                "fp_per_3500_std",
            ],
            digits=4,
        ),
        "",
        "## Missingness Subgroup Özeti",
        "",
        *_markdown_table(
            missingness.loc[missingness["strategy"].eq(SELECTED)],
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
        "## FP Hata Analizi",
        "",
        f"Hybrid-E_bestMCC_nested için en az bir outer değerlendirmede FP olan benzersiz varyant sayısı: **{len(fp_errors)}**.",
        "",
        "En sık tekrar eden FP örnekleri ayrı CSV dosyasında saklandı.",
        "",
        "## Çıktılar",
        "",
        "- `results/modeling/final_hybrid_freeze/hybrid_e_strengthening_actions.csv`",
        "- `results/modeling/final_hybrid_freeze/hybrid_e_calibration_brier_summary.csv`",
        "- `results/modeling/final_hybrid_freeze/hybrid_e_paired_delta_vs_catopt.csv`",
        "- `results/modeling/final_hybrid_freeze/hybrid_e_fold_stability_summary.csv`",
        "- `results/modeling/final_hybrid_freeze/hybrid_e_missingness_subgroup_summary.csv`",
        "- `results/modeling/final_hybrid_freeze/hybrid_e_fp_error_analysis.csv`",
    ]
    (OUT_DIR / "HYBRID_E_STRENGTHENING_AUDIT_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(NESTED_DIR / "nested_outer_predictions.csv", dtype={ID_COLUMN: "string"})
    outer = pd.read_csv(NESTED_DIR / "nested_outer_fold_results.csv", dtype={ID_COLUMN: "string"})

    calibration = _calibration_summary(predictions)
    paired = _paired_delta(outer)
    stability = _fold_stability(outer)
    missingness = _missingness_subgroups(predictions)
    fp_errors = _fp_error_analysis(predictions)
    actions = _action_summary()

    calibration.to_csv(OUT_DIR / "hybrid_e_calibration_brier_summary.csv", index=False)
    paired.to_csv(OUT_DIR / "hybrid_e_paired_delta_vs_catopt.csv", index=False)
    stability.to_csv(OUT_DIR / "hybrid_e_fold_stability_summary.csv", index=False)
    missingness.to_csv(OUT_DIR / "hybrid_e_missingness_subgroup_summary.csv", index=False)
    fp_errors.to_csv(OUT_DIR / "hybrid_e_fp_error_analysis.csv", index=False)
    actions.to_csv(OUT_DIR / "hybrid_e_strengthening_actions.csv", index=False)
    _write_report(calibration, paired, stability, missingness, fp_errors, actions)

    metadata = {
        "status": "completed",
        "selected_strategy": SELECTED,
        "reference_strategy": REFERENCE,
        "new_model_search_performed": False,
        "final_or_test_data_used": False,
        "outputs": {
            "actions": str((OUT_DIR / "hybrid_e_strengthening_actions.csv").relative_to(ROOT)),
            "calibration": str(
                (OUT_DIR / "hybrid_e_calibration_brier_summary.csv").relative_to(ROOT)
            ),
            "paired_delta": str(
                (OUT_DIR / "hybrid_e_paired_delta_vs_catopt.csv").relative_to(ROOT)
            ),
            "fold_stability": str(
                (OUT_DIR / "hybrid_e_fold_stability_summary.csv").relative_to(ROOT)
            ),
            "missingness": str(
                (OUT_DIR / "hybrid_e_missingness_subgroup_summary.csv").relative_to(ROOT)
            ),
            "fp_errors": str((OUT_DIR / "hybrid_e_fp_error_analysis.csv").relative_to(ROOT)),
            "report": str(
                (OUT_DIR / "HYBRID_E_STRENGTHENING_AUDIT_REPORT.md").relative_to(ROOT)
            ),
        },
    }
    (OUT_DIR / "hybrid_e_strengthening_audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


if __name__ == "__main__":
    main()
