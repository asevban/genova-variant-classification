"""Bolum A -- Aykiri Gozlem / Etkili Nokta Analizi.

Faz 1'in Model B'sini (25-ozellik, 21-agac entropy ensemble, nested oy-esigi
prosedur) yeniden kullanir -- `entropy_tree.py`/`ensemble.py`/
`threshold_search.py`, Faz 1'in KENDI klasorunden import edilir (kod
kopyalanmiyor/yeniden yazilmiyor).

Iki asama:
  1. Ekstremlik taramasi (hizli, betimsel): 25-ozellik uzayinda her satirin
     robust z-skoru (medyan/MAD) ile ne kadar "uc" oldugu olculur -- tum
     369 satir icin, tam LOO-nested prosedurunu calistirmadan ONCE bir
     aday kisa-liste cikarmak icin. Bu adim TAMAMEN betimsel/siralama
     amacli -- hicbir CV metriginde veya ozellik seciminde kullanilmiyor,
     bu yuzden sizinti riski YOK.
  2. Fold-koruyan Leave-One-Out: kisa listedeki her aday satir icin, split
     bankasinin MEVCUT fold uyeliklerini (yeniden uretmeden) korur, yalnizca
     o satiri (train/test/val nerede gorunuyorsa) filtreler, ayni nested
     prosedurle 50 dis fold'u yeniden skorlar.

METODOLOJIK NOT (fold-koruyan vs yeniden-split): Gorev metni "split
bankasinin fold yapisini o satir olmadan yeniden turet" diyor. Bu, iki
sekilde okunabilir: (a) StratifiedGroupKFold'u 368 satirla SIFIRDAN yeniden
calistirmak, ya da (b) MEVCUT fold uyeliklerini koruyup yalnizca o satiri
cikarmak. (a) secilirse, gozlenen performans farki hem satirin cikarilmasindan
HEM DE tamamen farkli bir rastgele fold-atamasindan kaynaklanir -- bu ikisini
ayirt etmek imkansiz, olcum gurultuye bogulur. (b) bu karisikligi onluyor:
tek degisken satirin kendisi oluyor, fold yapisinin geri kalani sabit kaliyor.
Split bankasi zaten "yalnizca okunur, yeniden uretilmez" kuralina da (b)
daha uygun. Bu yuzden (b) -- fold-koruyan cikarma -- uygulandi; bu tasarim
karari acikca burada belgeleniyor.

`conflict_group_1` ozel durumu: iki uyesinden (VAR_003238/VAR_003234) biri
kisa listeye girerse, ATLANMIYOR -- ama not ediliyor. Fold-koruyan tasarimda
bu bir sorun teskil etmiyor: grup-butunlugu kurali split uretilirken
(StratifiedGroupKFold asamasinda) uygulaniyordu; burada fold'lar zaten sabit,
yalnizca bir satir cikariliyor -- kalan uye oldugu fold'da kalmaya devam
ediyor, hicbir butunluk ihlali olusmuyor.

Bu script YALNIZCA experiments/outlier_partial_corr_analizi/results/ altina
yazar.
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
PHASE1_SCRIPTS = EXPERIMENT_DIR.parents[0] / "entropy_tree_ensemble_deneme" / "scripts"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PHASE1_SCRIPTS))  # Faz 1'in entropy_tree/ensemble/threshold_search'u -- YENIDEN YAZILMIYOR

from genova.pah.fold_features import build_fold_features  # noqa: E402
from entropy_tree import make_tree  # noqa: E402  (bu deneyde dogrudan kullanilmiyor, Model B ensemble uzerinden)
from ensemble import EntropyTreeEnsemble  # noqa: E402
from threshold_search import f1_by_threshold, best_threshold_from_inner_folds  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "pah"
POOL_PATH = PROJECT_ROOT / "reports" / "tables" / "v4_final_feature_pool.json"
V1_PATH = PROJECT_ROOT / "data" / "processed" / "pah" / "v1.parquet"

SEED = 42
N_REPEATS = 10
N_OUTER_SPLITS = 5
CONFLICT_GROUP_IDS = {"VAR_003238", "VAR_003234"}
N_SCREENING_SHORTLIST = 12


def outer_seed(repeat_idx, fold_idx):
    return SEED + repeat_idx * 100 + fold_idx


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Asama 1 -- ekstremlik taramasi (betimsel, sizintisiz -- tum veri uzerinde)
# ---------------------------------------------------------------------------
def screen_extremity(v1, al_columns, features_25):
    """Rank-tabanli (percentile) ekstremlik skoru -- robust z-skoru (medyan/MAD)
    DENENDI ama reddedildi: `AL_` frekans-tipi kolonlar sifir-siskin (CLAUDE.md,
    EDA A.6) oldugu icin MAD sifira yakin cikiyor, bu da z-skorlarini
    anlamsiz/patlamis degerlere (yuzlerce milyon) tasiyor -- yalnizca AL_8/
    AL_12 gibi birkac kolonun MAD-dejenerasyonuyla domine edilen, gercek
    coklu-degiskenli ekstremligi yansitmayan bir siralama uretiyordu (ampirik
    olarak dogrulandi, ilk calistirmada gozlendi). Rank/percentile-tabanli
    olcum olcek-bagimsiz oldugu icin bu dejenerasyona karsi bagisik.
    """
    all_ids = v1["Variant_ID"].tolist()
    X_all, y_all, _, _ = build_fold_features(v1, al_columns, all_ids, all_ids)
    X_pool = X_all[features_25]

    pct = X_pool.rank(pct=True)
    edge_distance = pd.concat({c: pd.concat([pct[c], 1 - pct[c]], axis=1).max(axis=1) for c in X_pool.columns}, axis=1)

    extremity = edge_distance.max(axis=1)
    screening = pd.DataFrame({
        "Variant_ID": v1["Variant_ID"].values,
        "Label": y_all.values,
        "max_percentile_extremity": extremity.values,
        "most_extreme_feature": edge_distance.idxmax(axis=1).values,
    }).sort_values("max_percentile_extremity", ascending=False).reset_index(drop=True)

    # AL_49 x EK_7 -- 03d'nin en guclu yapisal cifti; ikisinin de kendi
    # dagiliminin %5/%95 kuyruginda oldugu satirlar
    pct_al49 = X_pool["AL_49"].rank(pct=True)
    pct_ek7 = X_pool["EK_7"].rank(pct=True)
    al49_tail = (pct_al49 <= 0.05) | (pct_al49 >= 0.95)
    ek7_tail = (pct_ek7 <= 0.05) | (pct_ek7 >= 0.95)
    joint_tail_mask = al49_tail & ek7_tail
    joint_tail = pd.DataFrame({
        "Variant_ID": v1["Variant_ID"].values,
        "Label": y_all.values,
        "AL_49": X_pool["AL_49"].values,
        "AL_49_percentile": pct_al49.values,
        "EK_7": X_pool["EK_7"].values,
        "EK_7_percentile": pct_ek7.values,
    })[joint_tail_mask.values].reset_index(drop=True)

    return screening, joint_tail, X_pool


def build_shortlist(screening, joint_tail):
    top_extremity = screening.head(N_SCREENING_SHORTLIST)["Variant_ID"].tolist()
    joint_ids = joint_tail["Variant_ID"].tolist()
    shortlist = list(dict.fromkeys(top_extremity + joint_ids))  # sira korunarak dedup
    flagged_conflict = [v for v in shortlist if v in CONFLICT_GROUP_IDS]
    return shortlist, flagged_conflict


# ---------------------------------------------------------------------------
# Asama 2 -- fold-koruyan LOO (Model B, nested esik prosedürü)
# ---------------------------------------------------------------------------
def _filter_ids(ids, excluded):
    return [i for i in ids if i not in excluded]


def run_model_b_fold_preserving(v1, al_columns, features_25, excluded_ids=frozenset()):
    """Faz 1'in run_experiment.py::run_ensemble_model'iyla AYNI nested
    mantik (bkz. entropy_tree_ensemble_deneme/scripts/run_experiment.py) --
    tek fark: her outer/inner fold'un train/test/val id listelerinden
    `excluded_ids` cikarilarak fold uyelikleri KORUNUYOR, split bankasi
    yeniden uretilmiyor.
    """
    per_fold_rows = []
    for repeat_idx in range(N_REPEATS):
        outer_data = json.loads((SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json").read_text())
        for outer in outer_data["folds"]:
            fold_idx = outer["fold"]
            base_seed = outer_seed(repeat_idx, fold_idx)

            train_ids = _filter_ids(outer["train_variant_ids"], excluded_ids)
            test_ids = _filter_ids(outer["test_variant_ids"], excluded_ids)
            if len(test_ids) == 0 or len(train_ids) == 0:
                continue  # bu deneyde hic olmuyor (>=1 satir cikariliyor, fold'lar ~74/295 buyuklugunde)

            inner_data = json.loads(
                (SPLITS_DIR / f"inner_fold_repeat{repeat_idx:02d}_outer{fold_idx}.json").read_text()
            )
            per_inner_f1 = []
            for i, inner in enumerate(inner_data["folds"]):
                itr = _filter_ids(inner["train_variant_ids"], excluded_ids)
                ival = _filter_ids(inner["val_variant_ids"], excluded_ids)
                if len(itr) == 0 or len(ival) == 0:
                    continue
                Xi_tr, yi_tr, Xi_val, yi_val = build_fold_features(v1, al_columns, itr, ival)
                inner_seed = base_seed * 10 + i
                ens = EntropyTreeEnsemble(base_seed=inner_seed)
                ens.fit(Xi_tr[features_25], yi_tr)
                votes = ens.vote_counts(Xi_val[features_25])
                per_inner_f1.append(f1_by_threshold(votes, yi_val))

            best_t, _ = best_threshold_from_inner_folds(per_inner_f1)

            X_train, y_train, X_test, y_test = build_fold_features(v1, al_columns, train_ids, test_ids)
            final_ens = EntropyTreeEnsemble(base_seed=base_seed)
            final_ens.fit(X_train[features_25], y_train)
            y_pred = final_ens.predict(X_test[features_25], threshold=best_t)
            metrics = compute_metrics(y_test, y_pred)
            metrics.update({"repeat": repeat_idx, "fold": fold_idx, "chosen_threshold": best_t})
            per_fold_rows.append(metrics)

    df = pd.DataFrame(per_fold_rows)
    return {
        "f1_mean": df["f1"].mean(), "f1_std": df["f1"].std(),
        "mcc_mean": df["mcc"].mean(), "mcc_std": df["mcc"].std(),
        "n_folds": len(df),
    }, df


def main():
    v1 = pd.read_parquet(V1_PATH)
    al_columns = [c for c in v1.columns if c.startswith("AL_")]
    pool = json.loads(POOL_PATH.read_text())
    features_25 = pool["features"]

    print("=== Asama 1: ekstremlik taramasi (tum 369 satir, betimsel) ===")
    screening, joint_tail, X_pool = screen_extremity(v1, al_columns, features_25)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    screening.to_csv(RESULTS_DIR / "influence_screening.csv", index=False)
    joint_tail.to_csv(RESULTS_DIR / "al49_ek7_joint_tail_rows.csv", index=False)
    print(f"En ekstrem 5 satir:\n{screening.head(5).to_string(index=False)}")
    print(f"\nAL_49 x EK_7 ortak-kuyruk satir sayisi: {len(joint_tail)}")

    shortlist, flagged_conflict = build_shortlist(screening, joint_tail)
    print(f"\nAday kisa liste ({len(shortlist)} satir): {shortlist}")
    if flagged_conflict:
        print(f"UYARI/NOT -- conflict_group_1 uyesi kisa listede: {flagged_conflict} "
              f"(atlanmiyor, fold-koruyan tasarimda butunluk sorunu yok -- bkz. modul docstring'i)")

    print("\n=== Asama 2a: referans (hicbir satir cikarilmamis) ===")
    ref_summary, ref_df = run_model_b_fold_preserving(v1, al_columns, features_25, excluded_ids=frozenset())
    print(ref_summary)

    print(f"\n=== Asama 2b: kisa listedeki {len(shortlist)} adayin tek-tek LOO etkisi ===")
    loo_rows = []
    for i, vid in enumerate(shortlist):
        summary, _ = run_model_b_fold_preserving(v1, al_columns, features_25, excluded_ids={vid})
        row_info = v1.loc[v1["Variant_ID"] == vid].iloc[0]
        loo_rows.append({
            "Variant_ID": vid,
            "Label": int(row_info["Label"]),
            "is_conflict_group_member": vid in CONFLICT_GROUP_IDS,
            "f1_before": ref_summary["f1_mean"], "f1_after": summary["f1_mean"],
            "delta_f1": summary["f1_mean"] - ref_summary["f1_mean"],
            "mcc_before": ref_summary["mcc_mean"], "mcc_after": summary["mcc_mean"],
            "delta_mcc": summary["mcc_mean"] - ref_summary["mcc_mean"],
        })
        print(f"  [{i+1}/{len(shortlist)}] {vid}: dF1={loo_rows[-1]['delta_f1']:+.4f}, "
              f"dMCC={loo_rows[-1]['delta_mcc']:+.4f}")

    loo_df = pd.DataFrame(loo_rows).sort_values("delta_f1", key=lambda s: s.abs(), ascending=False)

    # kisa listedeki satirlarin 25-ozellik havuzundaki en ekstrem degerleri
    extreme_feature_notes = []
    for vid in loo_df["Variant_ID"]:
        z_row = ((X_pool.loc[v1["Variant_ID"] == vid].iloc[0] - X_pool.median()) /
                 (1.4826 * (X_pool - X_pool.median()).abs().median().replace(0, np.nan))).fillna(0.0)
        top3 = z_row.abs().sort_values(ascending=False).head(3)
        extreme_feature_notes.append("; ".join(f"{feat}(z={z_row[feat]:+.2f})" for feat in top3.index))
    loo_df["en_ekstrem_3_ozellik"] = extreme_feature_notes
    loo_df.to_csv(RESULTS_DIR / "loo_top10.csv", index=False)

    print("\n=== Asama 2c: en etkili 5 satir BIRLIKTE cikarilirsa ===")
    top5_ids = set(loo_df.head(5)["Variant_ID"].tolist())
    top1_id = {loo_df.iloc[0]["Variant_ID"]}
    top5_summary, _ = run_model_b_fold_preserving(v1, al_columns, features_25, excluded_ids=top5_ids)
    print(top5_summary)

    scenario_rows = [
        {"senaryo": "(a) hicbir satir cikarilmamis (referans)", "n_cikarilan": 0,
         "f1_mean": ref_summary["f1_mean"], "f1_std": ref_summary["f1_std"],
         "mcc_mean": ref_summary["mcc_mean"], "mcc_std": ref_summary["mcc_std"]},
        {"senaryo": f"(b) en etkili tekil satir cikarilmis ({list(top1_id)[0]})", "n_cikarilan": 1,
         "f1_mean": loo_df.iloc[0]["f1_after"], "f1_std": None,
         "mcc_mean": loo_df.iloc[0]["mcc_after"], "mcc_std": None},
        {"senaryo": "(c) en etkili ilk 5 satir birlikte cikarilmis", "n_cikarilan": 5,
         "f1_mean": top5_summary["f1_mean"], "f1_std": top5_summary["f1_std"],
         "mcc_mean": top5_summary["mcc_mean"], "mcc_std": top5_summary["mcc_std"]},
    ]
    pd.DataFrame(scenario_rows).to_csv(RESULTS_DIR / "loo_scenario_comparison.csv", index=False)

    print("\n=== 3-senaryo ozet ===")
    print(pd.DataFrame(scenario_rows).to_string(index=False))
    print("\nTum Bolum A sonuclari kaydedildi:", RESULTS_DIR)


if __name__ == "__main__":
    main()
