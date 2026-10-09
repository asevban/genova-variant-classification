"""ADIM 13 -- Nested Cross-Validation (ASIL GUVENILIR SONUC).

Dis: split bankasinin 50 outer fold'u. Ic: 4-fold (split bankasinin ic
yapisi). Ic dongu Hard Voting esigini secer (11-19/21), dis dongu
GORULMEMIS veride olcer. Faz 1'in EntropyTreeEnsemble/threshold_search'u
DOGRUDAN import edilir (yeniden yazilmaz).

Karsilastirilan varyantlar:
  A: Temel model -- 405 ozellik (tum aday evren, agac basina rastgele 70)
  B: Resmi 25-ozellik havuzu (referans, Faz 1'in Model B'siyle ayni kurulum)
  C: Adim 3'un eleme HIPOTEZI (~220 kolon, agac basina rastgele 70)
  D: Adim 8'in redundancy-budanmis seti (405 - 9 zayif-taraf kolon, agac basina rastgele 70)
  E: 25-havuz + log1p(AL_ kolonlari) -- Adim 12'nin genel-log1p kontrolu
  F: 25-havuz + Adim 11'in 4 ortalama-ozelligi (29 kolon)

Bu script YALNIZCA experiments/tam_16_adim_deneme/results/ altina yazar.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, recall_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, SPLITS_DIR, load_v1, load_pool_features, safe_print, SEED  # noqa: E402
from ensemble import EntropyTreeEnsemble  # noqa: E402  (Faz 1'den)
from threshold_search import f1_by_threshold, best_threshold_from_inner_folds  # noqa: E402
from genova.pah.fold_features import build_fold_features  # noqa: E402
from genova.pah.transforms import classify_al_columns  # noqa: E402  (resmi src/ -- yalniz IMPORT, degistirilmedi)
from step11_12_transform_candidates import AVERAGED_FEATURE_CANDIDATES, add_zscore_and_averaged_features  # noqa: E402

N_REPEATS, N_OUTER_SPLITS = 10, 5
TARGET_PREVALENCE = 0.286


def outer_seed(repeat_idx, fold_idx):
    return SEED + repeat_idx * 100 + fold_idx


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_label1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0,
        "recall": recall, "specificity": specificity,
    }


def prevalence_adjust(recall, specificity, target=TARGET_PREVALENCE):
    tp_rate, fp_rate = target * recall, (1 - target) * (1 - specificity)
    prec = tp_rate / (tp_rate + fp_rate) if (tp_rate + fp_rate) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return f1, (1 - specificity) * 100


def apply_variant_transform(X_train, X_test, variant, al_types_train, extra_cols_needed=None):
    """Her varyant icin (X_train, X_test) -> (X_train_sel, X_test_sel).
    Kolon SECIMI/DONUSUMU yalnizca X_train'den ogrenilen bilgiyle (al_types
    -- yapisal, etiket-bagimsiz -- veya sabit kolon adlariyla) yapilir."""
    if variant["transform"] == "select":
        cols = [c for c in variant["columns"] if c in X_train.columns]
        return X_train[cols], X_test[cols]
    if variant["transform"] == "select_log1p_al":
        cols = [c for c in variant["columns"] if c in X_train.columns]
        Xtr, Xte = X_train[cols].copy(), X_test[cols].copy()
        al_freq_cols = [c for c in cols if c in al_types_train.get("frequency_type", []) + al_types_train.get("ratio_type", [])]
        for c in al_freq_cols:
            Xtr[c] = np.log1p(Xtr[c].clip(lower=0))
            Xte[c] = np.log1p(Xte[c].clip(lower=0))
        return Xtr, Xte
    if variant["transform"] == "select_plus_averaged":
        # ONEMLI: ortalama-ozellikler (ZORT_AL88_AL121 vb.) HAM ciftlerden
        # (AL_88, AL_121, ...) turetiliyor -- bu ciftlerin cogu 25-havuzda
        # DEGIL. Bu yuzden once TAM X_train/X_test uzerinde (405 kolon)
        # ZORT_* hesaplanir, SONRA 25-havuz + ZORT_* kolonlari secilir.
        Xtr_full, Xte_full = add_zscore_and_averaged_features(X_train, X_test)
        zort_names = [name for _, _, name in AVERAGED_FEATURE_CANDIDATES]
        cols = [c for c in (variant["columns"] + zort_names) if c in Xtr_full.columns]
        return Xtr_full[cols], Xte_full[cols]
    raise ValueError(variant["transform"])


def run_variant_nested(v1, al_columns, variant, n_features_per_tree=None):
    per_fold_rows = []
    for repeat_idx in range(N_REPEATS):
        outer_data = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer in outer_data["folds"]:
            fold_idx = outer["fold"]
            base_seed = outer_seed(repeat_idx, fold_idx)
            inner_data = json.loads(
                (SPLITS_DIR / f"inner_fold_repeat{repeat_idx:02d}_outer{fold_idx}.json").read_text()
            )

            per_inner_f1 = []
            for i, inner in enumerate(inner_data["folds"]):
                Xi_tr, yi_tr, Xi_val, yi_val = build_fold_features(
                    v1, al_columns, inner["train_variant_ids"], inner["val_variant_ids"],
                )
                al_types_tr = classify_al_columns(v1[v1["Variant_ID"].isin(inner["train_variant_ids"])], al_columns)
                Xi_tr_sel, Xi_val_sel = apply_variant_transform(Xi_tr, Xi_val, variant, al_types_tr)
                inner_seed = base_seed * 10 + i
                ens = EntropyTreeEnsemble(base_seed=inner_seed, n_features_per_tree=n_features_per_tree)
                ens.fit(Xi_tr_sel, yi_tr)
                votes = ens.vote_counts(Xi_val_sel)
                per_inner_f1.append(f1_by_threshold(votes, yi_val))
            best_t, _ = best_threshold_from_inner_folds(per_inner_f1)

            X_train, y_train, X_test, y_test = build_fold_features(
                v1, al_columns, outer["train_variant_ids"], outer["test_variant_ids"],
            )
            al_types_train = classify_al_columns(v1[v1["Variant_ID"].isin(outer["train_variant_ids"])], al_columns)
            X_train_sel, X_test_sel = apply_variant_transform(X_train, X_test, variant, al_types_train)

            final_ens = EntropyTreeEnsemble(base_seed=base_seed, n_features_per_tree=n_features_per_tree)
            final_ens.fit(X_train_sel, y_train)
            y_pred = final_ens.predict(X_test_sel, threshold=best_t)
            m = compute_metrics(y_test, y_pred)
            f1_adj, fp100 = prevalence_adjust(m["recall"], m["specificity"])
            per_fold_rows.append({
                "repeat": repeat_idx, "fold": fold_idx, "chosen_threshold": best_t,
                "n_features_used": X_train_sel.shape[1], **m,
                "f1_adjusted_prevalence_28_6pct": f1_adj, "fp_per_100_benign": fp100,
            })
    return pd.DataFrame(per_fold_rows)


def main():
    v1 = load_v1()
    al_columns = [c for c in v1.columns if c.startswith("AL_")]
    pool_25 = load_pool_features()

    elim_df = pd.read_csv(RESULTS_DIR / "step3_elimination_hypothesis.csv")
    eliminated_cols = set(elim_df["column"])
    all_raw_feature_cols = [c for c in v1.columns if c not in ("Variant_ID", "Label", "group_id")]
    # step3 353-kolonluk HAM sema uzerinden calisiyordu; build_fold_features
    # ciktisi farkli (encode edilmis, ~405 kolon) -- ortak isimli kolonlari
    # (AL_/EK_ ham adlari + al_all_missing gibi turetilenler HARIC) kesisim al
    step3_reduced_raw_names = set(all_raw_feature_cols) - eliminated_cols

    redundancy_df = pd.read_csv(RESULTS_DIR / "step8_redundancy_candidates.csv")
    step8_weak_cols = set(redundancy_df.loc[redundancy_df["IG_farki"] > 0.005, "aday_cikarilacak_zayif_taraf"])

    variants = {
        "A_temel_405": {"transform": "select", "columns": None, "n_features_per_tree": 70, "desc": "Temel model (405, tam aday evren)"},
        "B_resmi_25havuz": {"transform": "select", "columns": pool_25, "n_features_per_tree": None, "desc": "Resmi 25-özellik havuzu (referans)"},
        "C_adim3_hipotez": {"transform": "select", "columns": None, "n_features_per_tree": 70, "desc": "Adım 3 eleme hipotezi (~220 ham kolon kesişimi)", "raw_filter": step3_reduced_raw_names},
        "D_adim8_redundancy_pruned": {"transform": "select", "columns": None, "n_features_per_tree": 70, "desc": "Adım 8 redundancy-budanmış (405 - zayıf taraflar)", "drop_cols": step8_weak_cols},
        "E_25havuz_log1p": {"transform": "select_log1p_al", "columns": pool_25, "n_features_per_tree": None, "desc": "25-havuz + log1p(AL_ kolonları) (Adım 12 genel kontrolü)"},
        "F_25havuz_ortalama": {"transform": "select_plus_averaged", "columns": pool_25, "n_features_per_tree": None, "desc": "25-havuz + 4 ortalama-özellik (Adım 11, 29 kolon)"},
    }

    all_results = {}
    summary_rows = []
    for name, variant in variants.items():
        safe_print(f"\n=== Varyant {name}: {variant['desc']} ===")
        # A/C/D icin kolon listesi build_fold_features ciktisindan (dinamik,
        # ~405 kolon) turetilmesi gerekiyor -- ilk outer fold'da kolonlari
        # kesfedip filtre kuruyoruz (yapisal, etiket-bagimsiz).
        if variant["columns"] is None:
            probe_outer = json.loads((SPLITS_DIR / "outer_fold_repeat00.json").read_text())["folds"][0]
            X_probe, _, _, _ = build_fold_features(v1, al_columns, probe_outer["train_variant_ids"], probe_outer["test_variant_ids"])
            all_cols = list(X_probe.columns)
            if "raw_filter" in variant:
                raw_filter = variant["raw_filter"]
                cols = [c for c in all_cols if any(c == rf or c.startswith(rf + "_") for rf in raw_filter)]
            elif "drop_cols" in variant:
                cols = [c for c in all_cols if c not in variant["drop_cols"]]
            else:
                cols = all_cols
            variant["columns"] = cols
            safe_print(f"  -> {len(cols)} kolon seçildi (dinamik, ilk fold'dan keşfedildi)")

        df = run_variant_nested(v1, al_columns, variant, n_features_per_tree=variant["n_features_per_tree"])
        df.to_csv(RESULTS_DIR / f"step13_nested_{name}.csv", index=False)
        all_results[name] = df
        summary_rows.append({
            "variant": name, "aciklama": variant["desc"], "n_features_final": int(df["n_features_used"].iloc[0]),
            "macro_f1_mean": df["macro_f1"].mean(), "macro_f1_std": df["macro_f1"].std(),
            "f1_label1_mean": df["f1_label1"].mean(), "f1_label1_std": df["f1_label1"].std(),
            "mcc_mean": df["mcc"].mean(), "mcc_std": df["mcc"].std(),
            "recall_mean": df["recall"].mean(), "specificity_mean": df["specificity"].mean(),
            "f1_adjusted_prevalence_mean": df["f1_adjusted_prevalence_28_6pct"].mean(),
            "fp_per_100_benign_mean": df["fp_per_100_benign"].mean(),
        })
        safe_print(f"  Macro-F1={df['macro_f1'].mean():.4f}±{df['macro_f1'].std():.4f}, "
                   f"F1(label=1)={df['f1_label1'].mean():.4f}, MCC={df['mcc'].mean():.4f}, "
                   f"F1_adj={df['f1_adjusted_prevalence_28_6pct'].mean():.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "step13_nested_summary.csv", index=False)
    safe_print("\n\n=== ADIM 13 NIHAI (RESMI) SONUC TABLOSU ===")
    safe_print(summary_df.to_string(index=False))
    safe_print(f"\nTüm dış-fold ham sonuçlar: results/step13_nested_<varyant>.csv")


if __name__ == "__main__":
    main()
