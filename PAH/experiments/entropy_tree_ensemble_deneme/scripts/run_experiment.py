"""Bagimsiz entropy-tree ensemble deneyi -- ana orkestrasyon.

Bu script YALNIZCA experiments/entropy_tree_ensemble_deneme/results/ altina
yazar. data/, configs/, reports/tables/, src/genova/pah/*.py -- hicbirine
yazmaz; yalnizca `genova.pah.fold_features.build_fold_features` ve
`genova.pah.split_bank`'in urettigi split bankasini import/okuma yoluyla
kullanir.

Modeller:
  A -- 25 ozellik (mevcut havuz), tek entropy tree, esik aramasi YOK (basit
       referans -- native .predict(), class_weight="balanced" zaten dahili
       dengesizlik duzeltmesi sagliyor).
  B -- 25 ozellik, 21-agac dengeli-bootstrap hard-voting ensemble, nested
       oy-esigi aramasi (11-19/21).
  C -- 405 (tam aday evren), agac basina rastgele 70 ozellik, 21-agac
       ensemble, nested oy-esigi aramasi.
  D -- Model B'nin CAT_1-cikarilmis hali (24 ozellik) -- YALNIZCA Model B
       uzerinde ablasyon (Model B onceden, tasarim geregi ana hipotez
       oldugu icin secildi -- outer sonuclara bakilarak degil).

Nested prosedur (her dis fold icin, B/C/D):
  1. Dis-fold'un 4 ic fold'unda ayri ayri: ic-egitimde ensemble fit, ic-
     val'de oy-esigi 11..19 arasinda F1 taranir.
  2. 4 ic fold'un F1'leri esik-basina ortalanir, en iyi esik secilir.
  3. TUM dis-egitimde YENI bir ensemble fit edilir (ic-egitim ensemble'lari
     ATILIR, yalnizca esik secimi icin kullanildilar).
  4. Dis-test, adim-2'de secilen sabit esikle TEK SEFER skorlanir.

Calistirma: python run_experiment.py (scripts/ icinden)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from genova.pah.fold_features import build_fold_features  # noqa: E402

from entropy_tree import make_tree  # noqa: E402
from ensemble import EntropyTreeEnsemble  # noqa: E402
from threshold_search import f1_by_threshold, best_threshold_from_inner_folds  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "pah"
POOL_PATH = PROJECT_ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
V1_PATH = PROJECT_ROOT / "data" / "processed" / "pah" / "v1.parquet"

SEED = 42  # 03c'nin kullandigi taban seed ile tutarli
N_REPEATS = 10
N_OUTER_SPLITS = 5
N_FEATURES_MODEL_C = 70
TARGET_PREVALENCE = 0.286  # final yarisma beklenen patojenik orani


def outer_seed(repeat_idx, fold_idx):
    """03c/id3_feature_selection.py ile ayni seed formulu."""
    return SEED + repeat_idx * 100 + fold_idx


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0,
        "recall": recall,
        "specificity": specificity,
    }


def prevalence_adjust(recall, specificity, target_prevalence=TARGET_PREVALENCE):
    """Standart prevalence-adjustment (Saerens, Latinne & Decaestecker, 2002
    usulu): TPR (recall) ve TNR (specificity)'nin sinif onceliginden bagimsiz
    (intrinsik siniflandirici ozelligi) oldugu varsayimiyla, hedef prevalans
    altinda beklenen precision/F1 yeniden hesaplanir.

    NOT: Gorev metninin bahsettigi "E-F dokumani"/"E4 formulu" bu repoda
    ARANDI, BULUNAMADI (grep ile dogrulandi) -- bu yuzden bu standart,
    dogrulanabilir formul dogrudan uygulandi, uydurma bir "E4" referansi
    kullanilmadi.
    """
    tp_rate = target_prevalence * recall
    fn_rate = target_prevalence * (1 - recall)
    fp_rate = (1 - target_prevalence) * (1 - specificity)
    precision_adj = tp_rate / (tp_rate + fp_rate) if (tp_rate + fp_rate) > 0 else 0.0
    f1_adj = (2 * precision_adj * recall / (precision_adj + recall)
              if (precision_adj + recall) > 0 else 0.0)
    fp_per_100_benign = (1 - specificity) * 100
    return f1_adj, fp_per_100_benign


def select_cols(X, wanted):
    cols = [c for c in wanted if c in X.columns]
    missing = set(wanted) - set(cols)
    if missing:
        raise ValueError(f"beklenen kolonlar aday evrende yok: {missing}")
    return X[cols]


def run_model_a(X_train, y_train, X_test, y_test, seed, feature_list):
    tree = make_tree(random_state=seed)
    tree.fit(select_cols(X_train, feature_list), y_train)
    y_pred = tree.predict(select_cols(X_test, feature_list))
    metrics = compute_metrics(y_test, y_pred)
    return metrics, None, None


def _inner_threshold_search(v1, al_columns, inner_folds, feature_list, base_seed, n_features_per_tree):
    per_inner_f1 = []
    for i, inner in enumerate(inner_folds):
        Xi_tr, yi_tr, Xi_val, yi_val = build_fold_features(
            v1, al_columns, inner["train_variant_ids"], inner["val_variant_ids"],
        )
        Xi_tr_sel = Xi_tr if feature_list is None else select_cols(Xi_tr, feature_list)
        Xi_val_sel = Xi_val if feature_list is None else select_cols(Xi_val, feature_list)

        inner_seed = base_seed * 10 + i
        ens = EntropyTreeEnsemble(base_seed=inner_seed, n_features_per_tree=n_features_per_tree)
        ens.fit(Xi_tr_sel, yi_tr)
        votes = ens.vote_counts(Xi_val_sel)
        per_inner_f1.append(f1_by_threshold(votes, yi_val))
    return best_threshold_from_inner_folds(per_inner_f1)


def run_ensemble_model(v1, al_columns, inner_folds, X_train, y_train, X_test, y_test,
                        base_seed, feature_list, n_features_per_tree=None):
    """B/C/D icin ortak nested prosedur. feature_list=None ise Model C'nin
    tam 405-aday-evreni kullanilir (X_train/X_test zaten build_fold_features
    ciktisi, ek filtre yok).
    """
    best_t, mean_f1_by_t = _inner_threshold_search(
        v1, al_columns, inner_folds, feature_list, base_seed, n_features_per_tree,
    )

    X_train_sel = X_train if feature_list is None else select_cols(X_train, feature_list)
    X_test_sel = X_test if feature_list is None else select_cols(X_test, feature_list)

    final_ens = EntropyTreeEnsemble(base_seed=base_seed, n_features_per_tree=n_features_per_tree)
    final_ens.fit(X_train_sel, y_train)
    y_pred = final_ens.predict(X_test_sel, threshold=best_t)
    metrics = compute_metrics(y_test, y_pred)
    return metrics, best_t, mean_f1_by_t


def main():
    v1 = pd.read_parquet(V1_PATH)
    al_columns = [c for c in v1.columns if c.startswith("AL_")]

    pool = json.loads(POOL_PATH.read_text())
    features_25 = pool["features"]
    assert len(features_25) == 25, f"beklenmedik havuz boyutu: {len(features_25)}"
    features_24_no_cat1 = [c for c in features_25 if c != "CAT_1"]
    assert len(features_24_no_cat1) == 24

    comparison_rows = []
    threshold_rows = []
    ablation_rows = []
    n_folds_done = 0
    total_folds = N_REPEATS * N_OUTER_SPLITS

    for repeat_idx in range(N_REPEATS):
        outer_data = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer in outer_data["folds"]:
            fold_idx = outer["fold"]
            base_seed = outer_seed(repeat_idx, fold_idx)

            inner_data = json.loads(
                (SPLITS_DIR / f"inner_fold_repeat{repeat_idx:02d}_outer{fold_idx}.json").read_text()
            )
            inner_folds = inner_data["folds"]

            X_train, y_train, X_test, y_test = build_fold_features(
                v1, al_columns, outer["train_variant_ids"], outer["test_variant_ids"],
            )

            # --- Model A: 25 ozellik, tek agac, esik aramasi yok ---
            metrics_a, _, _ = run_model_a(X_train, y_train, X_test, y_test, base_seed, features_25)
            comparison_rows.append({"repeat": repeat_idx, "fold": fold_idx, "model": "A", **metrics_a})

            # --- Model B: 25 ozellik, 21-agac ensemble, nested esik ---
            metrics_b, best_t_b, mean_f1_b = run_ensemble_model(
                v1, al_columns, inner_folds, X_train, y_train, X_test, y_test,
                base_seed, features_25, n_features_per_tree=None,
            )
            comparison_rows.append({"repeat": repeat_idx, "fold": fold_idx, "model": "B",
                                     "chosen_threshold": best_t_b, **metrics_b})
            threshold_rows.append({"repeat": repeat_idx, "fold": fold_idx, "model": "B",
                                    "chosen_threshold": best_t_b,
                                    **{f"inner_mean_f1_t{t}": v for t, v in mean_f1_b.items()}})

            # --- Model C: 405 ozellik (tam aday evren), agac basina rastgele 70 ---
            metrics_c, best_t_c, mean_f1_c = run_ensemble_model(
                v1, al_columns, inner_folds, X_train, y_train, X_test, y_test,
                base_seed, feature_list=None, n_features_per_tree=N_FEATURES_MODEL_C,
            )
            comparison_rows.append({"repeat": repeat_idx, "fold": fold_idx, "model": "C",
                                     "chosen_threshold": best_t_c, **metrics_c})
            threshold_rows.append({"repeat": repeat_idx, "fold": fold_idx, "model": "C",
                                    "chosen_threshold": best_t_c,
                                    **{f"inner_mean_f1_t{t}": v for t, v in mean_f1_c.items()}})

            # --- Model D: Model B'nin CAT_1-ablasyonu (24 ozellik) ---
            metrics_d, best_t_d, mean_f1_d = run_ensemble_model(
                v1, al_columns, inner_folds, X_train, y_train, X_test, y_test,
                base_seed, features_24_no_cat1, n_features_per_tree=None,
            )
            comparison_rows.append({"repeat": repeat_idx, "fold": fold_idx, "model": "D",
                                     "chosen_threshold": best_t_d, **metrics_d})
            threshold_rows.append({"repeat": repeat_idx, "fold": fold_idx, "model": "D",
                                    "chosen_threshold": best_t_d,
                                    **{f"inner_mean_f1_t{t}": v for t, v in mean_f1_d.items()}})

            ablation_rows.append({
                "repeat": repeat_idx, "fold": fold_idx,
                "model_B_f1": metrics_b["f1"], "model_D_f1": metrics_d["f1"],
                "delta_f1_B_minus_D": metrics_b["f1"] - metrics_d["f1"],
                "model_B_mcc": metrics_b["mcc"], "model_D_mcc": metrics_d["mcc"],
                "delta_mcc_B_minus_D": metrics_b["mcc"] - metrics_d["mcc"],
            })

            n_folds_done += 1
            print(f"repeat {repeat_idx} fold {fold_idx} tamamlandi ({n_folds_done}/{total_folds})")

    comparison_df = pd.DataFrame(comparison_rows)
    f1_adj, fp_adj = zip(*[prevalence_adjust(r["recall"], r["specificity"]) for _, r in comparison_df.iterrows()])
    comparison_df["f1_adjusted_prevalence_28_6pct"] = f1_adj
    comparison_df["fp_per_100_benign"] = fp_adj

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(RESULTS_DIR / "threshold_results.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(RESULTS_DIR / "cat1_ablation.csv", index=False)

    print("\n=== Model basina ozet (50 dis fold ortalama +- std) ===")
    summary = comparison_df.groupby("model")[
        ["f1", "mcc", "recall", "specificity", "f1_adjusted_prevalence_28_6pct", "fp_per_100_benign"]
    ].agg(["mean", "std"])
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "model_comparison_summary.csv")

    print("\nTum sonuclar kaydedildi:", RESULTS_DIR)


if __name__ == "__main__":
    main()
