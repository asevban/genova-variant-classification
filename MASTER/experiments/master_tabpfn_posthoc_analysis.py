"""Post-hoc screening for TabPFN threshold and Hybrid-E blending.

This analysis uses saved seed-42 outer-fold OOF scores. It is useful for
deciding whether a more expensive nested TabPFN governance run is worth doing,
but the weight/threshold selections here are not official final-selection
evidence because they are selected on already observed outer-fold labels.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from master_preprocessing import ID_COLUMN, TARGET_COLUMN

TABPFN_DIR = ROOT / "results" / "modeling" / "tabpfn_api"
NESTED_DIR = ROOT / "results" / "modeling" / "nested_governance"
OUT_DIR = TABPFN_DIR
FINAL_PATHOGENIC_RATE = 500 / 3500
FINAL_TOTAL_N = 3500
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.001), 3)
F1_FLOOR = 0.530
FP_MAX = 400.0
SENSITIVITY_FLOOR = 0.62
BASE_STRATEGIES = [
    "Hybrid-E_bestMCC_nested",
    "Hybrid-E_AUPRC_guard_nested",
    "Hybrid-E_lowFP_nested",
]
BLEND_WEIGHTS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]


def final_metric_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    positives = np.sort(y_score[y_true == 1])
    negatives = np.sort(y_score[y_true == 0])
    pos_n = len(positives)
    neg_n = len(negatives)
    tp_counts = pos_n - np.searchsorted(positives, THRESHOLDS, side="left")
    fp_counts = neg_n - np.searchsorted(negatives, THRESHOLDS, side="left")
    tpr = tp_counts / pos_n if pos_n else np.zeros_like(THRESHOLDS, dtype=float)
    fpr = fp_counts / neg_n if neg_n else np.zeros_like(THRESHOLDS, dtype=float)

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
            "threshold": THRESHOLDS.astype(float),
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


def select_screening_threshold(table: pd.DataFrame) -> tuple[pd.Series, str]:
    strict = table.loc[
        table["final_f1"].ge(F1_FLOOR)
        & table["expected_fp_per_3500"].le(FP_MAX)
        & table["final_sensitivity"].ge(SENSITIVITY_FLOOR)
    ].copy()
    if len(strict):
        return (
            strict.sort_values(
                ["final_mcc", "final_f1", "final_specificity", "final_sensitivity"],
                ascending=False,
            ).iloc[0],
            "strict",
        )

    fp_forced = table.loc[
        table["expected_fp_per_3500"].le(FP_MAX)
        & table["final_sensitivity"].ge(SENSITIVITY_FLOOR)
    ].copy()
    if len(fp_forced):
        return (
            fp_forced.sort_values(
                ["final_mcc", "final_f1", "final_specificity", "final_sensitivity"],
                ascending=False,
            ).iloc[0],
            "fp_forced",
        )

    return (
        table.sort_values(["final_mcc", "final_f1", "final_specificity"], ascending=False).iloc[0],
        "mcc_only",
    )


def summarize_threshold_cases(tabpfn_oof: pd.DataFrame) -> pd.DataFrame:
    y_true = tabpfn_oof[TARGET_COLUMN].to_numpy(dtype=int)
    y_score = tabpfn_oof["score"].to_numpy(dtype=float)
    table = final_metric_table(y_true, y_score)
    rows: list[dict[str, Any]] = []

    def append_case(case: str, frame: pd.DataFrame, sort_cols: list[str]) -> None:
        if frame.empty:
            rows.append({"case": case, "available": False})
            return
        selected = frame.sort_values(sort_cols, ascending=False).iloc[0]
        rows.append({"case": case, "available": True, **selected.to_dict()})

    append_case("threshold_0_50", table.loc[table["threshold"].eq(0.50)], ["final_mcc"])
    append_case("best_mcc_overall", table, ["final_mcc", "final_f1", "final_specificity"])
    append_case(
        "fp_le_400_best_mcc",
        table.loc[table["expected_fp_per_3500"].le(FP_MAX)],
        ["final_mcc", "final_f1", "final_specificity"],
    )
    append_case(
        "strict_gate_best_mcc",
        table.loc[
            table["final_f1"].ge(F1_FLOOR)
            & table["expected_fp_per_3500"].le(FP_MAX)
            & table["final_sensitivity"].ge(SENSITIVITY_FLOOR)
        ],
        ["final_mcc", "final_f1", "final_specificity"],
    )
    return pd.DataFrame(rows)


def summarize_governance_seed42() -> pd.DataFrame:
    fold_results = pd.read_csv(NESTED_DIR / "nested_outer_fold_results.csv")
    fold_results = fold_results.loc[fold_results["outer_seed"].eq(42)].copy()
    rows: list[dict[str, Any]] = []
    for strategy in BASE_STRATEGIES + ["CATOPT-A_nested_reference"]:
        group = fold_results.loc[fold_results["strategy"].eq(strategy)].copy()
        rows.append(
            {
                "candidate": strategy,
                "folds": len(group),
                "f1": float(group["outer_f1"].mean()),
                "mcc": float(group["outer_mcc"].mean()),
                "specificity": float(group["outer_specificity"].mean()),
                "sensitivity": float(group["outer_sensitivity"].mean()),
                "expected_fp_per_3500": float(group["outer_expected_fp_per_3500"].mean()),
                "final_weighted_auprc": float(
                    group["outer_final_weighted_auprc"].mean()
                ),
                "strict_pass_rate": float(group["outer_strict_pass"].mean()),
            }
        )
    return pd.DataFrame(rows)


def merge_base_with_tabpfn(strategy: str, tabpfn_oof: pd.DataFrame) -> pd.DataFrame:
    nested = pd.read_csv(NESTED_DIR / "nested_outer_predictions.csv")
    nested = nested.loc[
        nested["strategy"].eq(strategy) & nested["outer_seed"].eq(42),
        [ID_COLUMN, TARGET_COLUMN, "outer_fold", "score"],
    ].rename(columns={"outer_fold": "fold", "score": "hybrid_score"})
    merged = nested.merge(
        tabpfn_oof.loc[:, [ID_COLUMN, TARGET_COLUMN, "fold", "score"]].rename(
            columns={"score": "tabpfn_score"}
        ),
        on=[ID_COLUMN, TARGET_COLUMN, "fold"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(tabpfn_oof):
        raise ValueError(
            f"Merge mismatch for {strategy}: merged={len(merged)} tabpfn={len(tabpfn_oof)}"
        )
    return merged


def blend_screening(tabpfn_oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy in BASE_STRATEGIES:
        merged = merge_base_with_tabpfn(strategy, tabpfn_oof)
        y_true = merged[TARGET_COLUMN].to_numpy(dtype=int)
        hybrid_score = merged["hybrid_score"].to_numpy(dtype=float)
        tabpfn_score = merged["tabpfn_score"].to_numpy(dtype=float)
        for weight in BLEND_WEIGHTS:
            blended_score = (1.0 - weight) * hybrid_score + weight * tabpfn_score
            threshold_table = final_metric_table(y_true, blended_score)
            selected, status = select_screening_threshold(threshold_table)
            rows.append(
                {
                    "base_strategy": strategy,
                    "tabpfn_weight": float(weight),
                    "hybrid_weight": float(1.0 - weight),
                    "selection_status": status,
                    **ranking_metrics(y_true, blended_score),
                    **selected.to_dict(),
                }
            )
    result = pd.DataFrame(rows)
    delta_cols = [
        "final_f1",
        "final_mcc",
        "final_specificity",
        "final_sensitivity",
        "expected_fp_per_3500",
        "expected_fn_per_3500",
        "final_weighted_auprc",
    ]
    for strategy in BASE_STRATEGIES:
        base = result.loc[
            result["base_strategy"].eq(strategy) & result["tabpfn_weight"].eq(0.0)
        ].iloc[0]
        mask = result["base_strategy"].eq(strategy)
        for column in delta_cols:
            result.loc[mask, f"delta_{column}"] = result.loc[mask, column] - float(
                base[column]
            )
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in headers:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    threshold_summary: pd.DataFrame,
    governance_reference: pd.DataFrame,
    blend_results: pd.DataFrame,
) -> None:
    best_constrained = blend_results.loc[
        blend_results["expected_fp_per_3500"].le(FP_MAX)
        & blend_results["final_sensitivity"].ge(SENSITIVITY_FLOOR)
    ].sort_values(["final_mcc", "final_f1", "final_weighted_auprc"], ascending=False)
    best_strict = blend_results.loc[blend_results["selection_status"].eq("strict")].sort_values(
        ["final_mcc", "final_f1", "final_weighted_auprc"], ascending=False
    )
    top_constrained = best_constrained.head(8)
    top_strict = best_strict.head(8)
    threshold_cols = [
        "case",
        "available",
        "threshold",
        "final_f1",
        "final_mcc",
        "final_specificity",
        "final_sensitivity",
        "expected_fp_per_3500",
    ]
    governance_cols = [
        "candidate",
        "folds",
        "f1",
        "mcc",
        "specificity",
        "sensitivity",
        "expected_fp_per_3500",
        "final_weighted_auprc",
        "strict_pass_rate",
    ]
    blend_cols = [
        "base_strategy",
        "tabpfn_weight",
        "selection_status",
        "threshold",
        "final_f1",
        "final_mcc",
        "final_specificity",
        "final_sensitivity",
        "expected_fp_per_3500",
        "final_weighted_auprc",
        "delta_final_mcc",
        "delta_expected_fp_per_3500",
    ]

    lines = [
        "# TabPFN API Smoke Test and Hybrid-E Blend Screening",
        "",
        "Bu rapor yalnız ara tarama çıktısıdır. TabPFN API ile üretilen seed=42, 5-fold OOF skorları kullanıldı; final/test etiketi kullanılmadı. Ağırlık ve threshold kararları burada outer OOF üzerinde seçildiği için bu bölüm resmi final governance kanıtı değildir.",
        "",
        "## TabPFN Threshold Kontrolü",
        "",
        markdown_table(threshold_summary.loc[:, threshold_cols]),
        "",
        "## Seed-42 Governance Referansı",
        "",
        markdown_table(governance_reference.loc[:, governance_cols]),
        "",
        "## Hybrid-E + TabPFN Küçük Ağırlık Taraması",
        "",
        "Aşağıdaki tablo FP <= 400 ve sensitivity >= 0.62 koşulu altında en iyi post-hoc karışımları gösterir.",
        "",
        markdown_table(top_constrained.loc[:, blend_cols]),
        "",
        "## Strict Geçen Karışımlar",
        "",
    ]
    if len(top_strict):
        lines.append(markdown_table(top_strict.loc[:, blend_cols]))
    else:
        lines.append("Strict koşulunu geçen Hybrid-E + TabPFN karışımı bulunmadı.")
    lines.extend(
        [
            "",
            "## Kısa Karar",
            "",
            "TabPFN tek başına güçlü sensitivity/F1 sinyali verdi, ancak benign-heavy hedefte FP yükü yüksek kaldı. FP <= 400 zorlandığında TabPFN MCC/F1 düşüyor. Küçük ağırlıklı Hybrid-E karışımları bu seed-42 taramada sınırlı iyileşme gösterebilse de, resmi aday yapılmadan önce TabPFN skorunun inner seçimli nested governance akışına dahil edilmesi gerekir.",
        ]
    )
    (OUT_DIR / "TABPFN_API_SCREENING_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> dict[str, Any]:
    started = time.time()
    tabpfn_oof = pd.read_csv(TABPFN_DIR / "tabpfn_api_oof_predictions.csv")
    threshold_summary = summarize_threshold_cases(tabpfn_oof)
    governance_reference = summarize_governance_seed42()
    blend_results = blend_screening(tabpfn_oof)

    threshold_summary.to_csv(
        OUT_DIR / "tabpfn_api_fp_forced_threshold_summary.csv", index=False
    )
    governance_reference.to_csv(
        OUT_DIR / "tabpfn_seed42_governance_reference.csv", index=False
    )
    blend_results.to_csv(OUT_DIR / "tabpfn_hybrid_e_blend_screening.csv", index=False)
    write_report(threshold_summary, governance_reference, blend_results)
    metadata = {
        "status": "completed",
        "analysis_type": "posthoc_seed42_screening_not_final_governance",
        "base_strategies": BASE_STRATEGIES,
        "tabpfn_weights": BLEND_WEIGHTS,
        "f1_floor": F1_FLOOR,
        "fp_max": FP_MAX,
        "sensitivity_floor": SENSITIVITY_FLOOR,
        "api_calls_made": False,
        "outer_validation_used": False,
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "threshold_summary": "results/modeling/tabpfn_api/tabpfn_api_fp_forced_threshold_summary.csv",
            "governance_reference": "results/modeling/tabpfn_api/tabpfn_seed42_governance_reference.csv",
            "blend_screening": "results/modeling/tabpfn_api/tabpfn_hybrid_e_blend_screening.csv",
            "report": "results/modeling/tabpfn_api/TABPFN_API_SCREENING_REPORT.md",
            "metadata": "results/modeling/tabpfn_api/tabpfn_posthoc_metadata.json",
        },
    }
    (OUT_DIR / "tabpfn_posthoc_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== TABPFN POSTHOC THRESHOLD SUMMARY ===")
    print(threshold_summary.to_string(index=False))
    print("\n=== TOP HYBRID-E + TABPFN CONSTRAINED BLENDS ===")
    top = blend_results.loc[
        blend_results["expected_fp_per_3500"].le(FP_MAX)
        & blend_results["final_sensitivity"].ge(SENSITIVITY_FLOOR)
    ].sort_values(["final_mcc", "final_f1", "final_weighted_auprc"], ascending=False)
    display_cols = [
        "base_strategy",
        "tabpfn_weight",
        "selection_status",
        "threshold",
        "final_f1",
        "final_mcc",
        "final_specificity",
        "final_sensitivity",
        "expected_fp_per_3500",
        "final_weighted_auprc",
    ]
    print(top.loc[:, display_cols].head(12).to_string(index=False))
    return metadata


if __name__ == "__main__":
    main()
