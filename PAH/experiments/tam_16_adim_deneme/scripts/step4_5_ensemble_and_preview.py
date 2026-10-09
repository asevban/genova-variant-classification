"""ADIM 4 -- ID3 Agaci ve Ensemble (Hard + Soft Voting).
ADIM 5 -- On-izleme CV (DUZELTME 2: nested DEGIL, split bankasinin outer
fold'lari uzerinde, kendi CV'si KURULMADAN).

Faz 1'in EntropyTreeEnsemble sinifi DOGRUDAN import edilir (yeniden
yazilmaz) -- bu script yalnizca soft-voting yetenegini disaridan, mevcut
sinifin fit ettigi agaclari (`.trees_`, `.tree_feature_subsets_`) kullanarak
EKLIYOR (subclass degil, sinifi degistirmeden).

Adim 4, Faz 1'in Model C'siyle (405 ozellik, agac basina rastgele 70,
21-agac hard-voting) BUYUK OLCUDE ORTUSUYOR -- Faz 1 sonuclarina referans
verilir (`entropy_tree_ensemble_deneme/results/model_comparison.csv`),
tamamen tekrar hesaplanmaz. Bu script'in kendi katkisi: (a) SOFT voting
karsilastirmasi (Faz 1'de yoktu), (b) Adim 3'un eleme hipotezine uygun
YENI bir ozellik seti versiyonu icin on-izleme.

DUZELTME 2: Adim 5'in ciktisi kesin sonuc DEGILDIR -- yalnizca "bu esik
bolgesi umut verici mi" sorusuna kaba bir on-izleme. Asil guvenilir sonuc
Adim 13'un nested prosedurunde.

Bu script YALNIZCA experiments/tam_16_adim_deneme/results/ altina yazar.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, PROJECT_ROOT, SPLITS_DIR, load_v1, load_pool_features, safe_print, SEED  # noqa: E402
from ensemble import EntropyTreeEnsemble  # noqa: E402  (Faz 1'den, degistirilmeden)
from genova.pah.fold_features import build_fold_features  # noqa: E402

N_REPEATS, N_OUTER_SPLITS = 10, 5
THRESHOLD_GRID = list(range(11, 20))
TARGET_PREVALENCE = 0.286


def outer_seed(repeat_idx, fold_idx):
    return SEED + repeat_idx * 100 + fold_idx


def soft_vote_proba(ensemble, X):
    """Ensemble'daki her agacin predict_proba'sini ortalar (soft voting).
    Faz 1'in EntropyTreeEnsemble'ini DEGISTIRMEDEN, disaridan kullanir."""
    probas = np.zeros(len(X))
    for tree, cols in zip(ensemble.trees_, ensemble.tree_feature_subsets_):
        probas += tree.predict_proba(X[cols])[:, 1]
    return probas / len(ensemble.trees_)


def full_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_label1": recall1,
        "specificity": specificity,
        "f1_label1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_label0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def prevalence_adjust(recall, specificity, target=TARGET_PREVALENCE):
    tp_rate, fp_rate = target * recall, (1 - target) * (1 - specificity)
    prec = tp_rate / (tp_rate + fp_rate) if (tp_rate + fp_rate) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    return f1, (1 - specificity) * 100


def main():
    v1 = load_v1()
    al_columns = [c for c in v1.columns if c.startswith("AL_")]
    pool_25 = load_pool_features()

    safe_print("=== ADIM 4: Ensemble kurulumu (Faz 1'in EntropyTreeEnsemble'i import edildi) ===")
    safe_print("Model C (405 özellik, ağaç başına rastgele 70) zaten Faz 1'de ölçüldü -- "
               "bkz. entropy_tree_ensemble_deneme/results/model_comparison.csv. "
               "Burada YENİ katkı: soft-voting karşılaştırması + Adım 3 hipotez-seti önizlemesi.")

    hard_rows, soft_rows = [], []
    n_folds = 0
    for repeat_idx in range(N_REPEATS):
        outer_data = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer in outer_data["folds"]:
            fold_idx = outer["fold"]
            seed = outer_seed(repeat_idx, fold_idx)
            X_train, y_train, X_test, y_test = build_fold_features(
                v1, al_columns, outer["train_variant_ids"], outer["test_variant_ids"],
            )
            ens = EntropyTreeEnsemble(base_seed=seed)  # 25-ozellik havuzu, hard/soft ikisi icin ayni agaclar
            ens.fit(X_train[pool_25], y_train)

            y_pred_hard = ens.predict(X_test[pool_25], threshold=11)  # basit cogunluk, on-izleme
            hard_rows.append({"repeat": repeat_idx, "fold": fold_idx, **full_metrics(y_test, y_pred_hard)})

            proba = soft_vote_proba(ens, X_test[pool_25])
            y_pred_soft = (proba >= 0.5).astype(int)
            soft_rows.append({"repeat": repeat_idx, "fold": fold_idx, **full_metrics(y_test, y_pred_soft)})
            n_folds += 1

    hard_df, soft_df = pd.DataFrame(hard_rows), pd.DataFrame(soft_rows)
    hard_df.to_csv(RESULTS_DIR / "step4_hard_voting_by_fold.csv", index=False)
    soft_df.to_csv(RESULTS_DIR / "step4_soft_voting_by_fold.csv", index=False)

    safe_print(f"\nHard voting (eşik=11/21, {n_folds} fold): F1={hard_df['f1_label1'].mean():.4f}±{hard_df['f1_label1'].std():.4f}, "
               f"MCC={hard_df['mcc'].mean():.4f}")
    safe_print(f"Soft voting (P>=0.5, {n_folds} fold): F1={soft_df['f1_label1'].mean():.4f}±{soft_df['f1_label1'].std():.4f}, "
               f"MCC={soft_df['mcc'].mean():.4f}")
    safe_print("NOT: n=295 egitim (~50 benign) gibi kucuk bir ornekte agac-basina yaprak olasiliklari "
               "(predict_proba) az sayida ornege dayaniyor -- kalibrasyonun zayif olmasi beklenir; "
               "bu karsilastirma yalnizca kaba bir sinyal, kesin sonuc degil.")

    # --- ADIM 5: on-izleme esik taramasi (NESTED DEGIL) ---
    safe_print("\n=== ADIM 5: Ön-izleme eşik taraması (NESTED DEĞİL -- yalnızca outer fold'larda) ===")
    threshold_rows = []
    for repeat_idx in range(N_REPEATS):
        outer_data = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer in outer_data["folds"]:
            fold_idx = outer["fold"]
            seed = outer_seed(repeat_idx, fold_idx)
            X_train, y_train, X_test, y_test = build_fold_features(
                v1, al_columns, outer["train_variant_ids"], outer["test_variant_ids"],
            )
            ens = EntropyTreeEnsemble(base_seed=seed)
            ens.fit(X_train[pool_25], y_train)
            votes = ens.vote_counts(X_test[pool_25])
            for t in THRESHOLD_GRID:
                y_pred = (votes >= t).astype(int)
                m = full_metrics(y_test, y_pred)
                f1_adj, fp100 = prevalence_adjust(m["recall_label1"], m["specificity"])
                threshold_rows.append({
                    "repeat": repeat_idx, "fold": fold_idx, "threshold": t,
                    **m, "f1_adjusted_prevalence_28_6pct": f1_adj, "fp_per_100_benign": fp100,
                })
    preview_df = pd.DataFrame(threshold_rows)
    preview_df.to_csv(RESULTS_DIR / "step5_preview_threshold_sweep.csv", index=False)

    preview_summary = preview_df.groupby("threshold")[
        ["accuracy", "balanced_accuracy", "precision", "recall_label1", "specificity",
         "f1_label1", "f1_label0", "macro_f1", "mcc", "f1_adjusted_prevalence_28_6pct", "fp_per_100_benign"]
    ].mean()
    preview_summary.to_csv(RESULTS_DIR / "step5_preview_threshold_summary.csv")
    safe_print(preview_summary.to_string())
    safe_print("\n*** BU BİR ÖN-İZLEMEDİR, NESTED DEĞİL *** -- her eşik AYNI outer-test fold'ları üzerinde "
               "tekrar tekrar denendi, bu yüzden en iyi görünen eşik burada 'seçilemez' (seçim kendisi "
               "outer-test'e bakarak yapılmış olur = sızıntı). Kesin eşik seçimi yalnızca Adım 13'te, "
               "iç fold'larla yapılıyor.")


if __name__ == "__main__":
    main()
