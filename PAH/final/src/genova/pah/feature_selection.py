"""Aşama D.1: nested özellik-seçimi yöntem karşılaştırması, grup ablation
ve izole-grup modelleri. Buradaki her şey sabit split bankasını
(data/splits/pah/) okur ve her dış tekrar için fold_features.
build_fold_features ile fold-güvenli özellikleri yeniden fit eder --
hiçbir adım aynı anda birden fazla dış-eğitim fold'unun satırlarında fit
edilmez.

Burada eğitilen yardımcı modeller (LightGBM, ElasticNet-cezalı lojistik
regresyon) yalnızca seçim/diagnostik araçlarıdır, asla final model adayı
değildir -- bu modülde hiçbir yerde F1 hesaplanmaz, yalnızca AUC (diagnostik).

Çalıştırma: python -m genova.pah.feature_selection
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import lightgbm as lgb

from genova.pah.fold_features import build_fold_features, columns_for_group

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
V1_PATH = ROOT / "data" / "processed" / "pah" / "v1.parquet"
SPLITS_DIR = ROOT / "data" / "splits" / "pah"
TAB_DIR = ROOT / "reports" / "tables"

N_REPEATS = 10
N_OUTER_SPLITS = 5
STABILITY_THRESHOLD = 0.60
TOP_K_MI = 30
TOP_K_GBDT = 30
ELASTICNET_N_BOOTSTRAP = 10
ELASTICNET_SUBSAMPLE_FRAC = 0.8
ELASTICNET_MAX_ITER = 500
PERMUTATION_N_REPEATS = 5
SEED = 42

GROUPS = ["AL", "EK", "AA", "CAT"]


def _lgb_model(seed):
    return lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, num_leaves=15, learning_rate=0.1,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbose=-1,
    )


def mi_selected(X_train, y_train, seed):
    scores = mutual_info_classif(X_train, y_train, random_state=seed)
    order = np.argsort(scores)[::-1][:TOP_K_MI]
    return set(X_train.columns[order]), pd.Series(scores, index=X_train.columns)


def gbdt_selected(X_train, y_train, seed):
    model = _lgb_model(seed).fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top = importances.sort_values(ascending=False).head(TOP_K_GBDT).index
    return set(top), importances


def elasticnet_selected(X_train, y_train, seed):
    """Tekrarlı-alt-örnekleme stability selection: bir özellik, eğitim
    fold'unun bootstrap alt-örneklemelerinin >=%50'sinde elastic-net
    katsayısı sıfırdan farklıysa bu fold için seçilmiş sayılır.
    """
    scaler = StandardScaler().fit(X_train)
    X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    rng = np.random.RandomState(seed)
    nonzero_counts = pd.Series(0, index=X_train.columns)
    n_samples = int(len(X_scaled) * ELASTICNET_SUBSAMPLE_FRAC)
    for b in range(ELASTICNET_N_BOOTSTRAP):
        idx = rng.choice(len(X_scaled), size=n_samples, replace=False)
        model = LogisticRegression(
            penalty="elasticnet", l1_ratio=0.5, solver="saga", C=0.5,
            max_iter=ELASTICNET_MAX_ITER, random_state=seed + b,
        )
        model.fit(X_scaled.iloc[idx], y_train.iloc[idx])
        nonzero_counts += (np.abs(model.coef_[0]) > 1e-6).astype(int)
    selection_rate = nonzero_counts / ELASTICNET_N_BOOTSTRAP
    selected = set(selection_rate[selection_rate >= 0.5].index)
    return selected, selection_rate


def permutation_selected(X_train, y_train, seed, val_frac=0.2):
    """DÜZELTME (P0-1, bkz. reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md):
    önceki sürüm dış-fold'un TEST kısmını (X_test, y_test) doğrudan
    alıyordu -- dış-test etiketleri permutation-importance skoruna, dolayısıyla
    özellik-seçim kararına doğrudan giriyordu. Bu KRİTİK bir sızıntıydı.

    Yeni tasarım: yalnızca dış-eğitim fold'unu (`X_train`, `y_train`) alır.
    Dış-test bu fonksiyona hiçbir şekilde girmez -- kendi basit iç
    bölmesini (`train_test_split`, stratified, %(1-val_frac)/%val_frac)
    ayırır, yardımcı GBDT'yi iç-fit kısmında eğitir, permutation-importance'ı
    iç-val kısmında ölçer.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=val_frac, random_state=seed, stratify=y_train,
    )
    model = _lgb_model(seed).fit(X_fit, y_fit)
    result = permutation_importance(
        model, X_val, y_val, scoring="roc_auc", n_repeats=PERMUTATION_N_REPEATS, random_state=seed,
    )
    importances = pd.Series(result.importances_mean, index=X_val.columns)
    selected = set(importances[importances > 0].index)
    return selected, importances


N_STABILITY_RESAMPLES = 10
STABILITY_SUBSAMPLE_FRAC = 0.8


def _resampled_hit_rate(X_train, y_train, select_fn, seed, n_resamples=N_STABILITY_RESAMPLES,
                         subsample_frac=STABILITY_SUBSAMPLE_FRAC):
    """MI/GBDT/permutation icin, `elasticnet_selected`'in KENDI ic bootstrap
    mantigiyla ayni felsefede, "yalnizca bu dis-fold'un train'inden" bir
    stabilite orani uretir -- n_resamples adet %subsample_frac'lik (yerine
    koymadan) alt-orneklemde `select_fn(X_sub, y_sub, seed)` calistirilir
    (permutation_selected dahil -- kendi ic train_test_split'ini KENDISI
    yapar, bu yuzden ayni 3-parametreli imza tum yontemlerde ortak); her
    ozelligin kac resample'da secildigi oranlanir. Dis-test'e (`X_train`'in
    disindaki hicbir veriye) hicbir yontem erismiyor.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(len(X_train) * subsample_frac)
    hits = pd.Series(0, index=X_train.columns, dtype=float)
    for b in range(n_resamples):
        idx = rng.choice(len(X_train), size=n_samples, replace=False)
        X_sub, y_sub = X_train.iloc[idx], y_train.iloc[idx]
        selected, _ = select_fn(X_sub, y_sub, seed + b)
        for c in selected:
            hits[c] += 1
    return hits / n_resamples


def compute_fold_local_pool(X_train, y_train, seed, rule_threshold=None, stability_threshold=None):
    """DÜZELTME (P0-2): Aşama D.1'in "50 dış-fold'un birleşimiyle TEK
    global havuz" mimarisi yerine -- her dış-fold (ya da herhangi bir
    train_ids kümesi, örn. bir iç-fold) YALNIZCA KENDİ `X_train`/`y_train`'inden
    türetilmiş kendi lokal havuzunu üretir. `fold_versions.py::build_v4_
    from_v2/v3` zaten bir `pool` parametresi alıyordu (mimari olarak
    fold-local'a hazırdı) -- eksik olan, bu havuzun HER dış-fold için AYRI
    ayrı, yalnızca o fold'un train_ids'inden türetilmesiydi; önceden tüm
    50 dış-fold'un hit-oranları TEK bir global tabloda birleştirilip TEK
    bir sabit havuz üretiliyordu.

    Dört yöntemin dördü de artık BU fold'un train_ids'i içinde kendi
    resampling'iyle bir stabilite oranı üretir (MI/GBDT/permutation için
    `_resampled_hit_rate`, elasticnet için zaten var olan kendi 10-bootstrap
    mekanizması) -- dış-test'e (ya da bu fonksiyona YALNIZCA train_ids'i
    verilen çağıran tarafın dış-test'ine) hiçbir yöntem erişmiyor.
    """
    rule_threshold = FINAL_POOL_RULE_THRESHOLD if rule_threshold is None else rule_threshold
    stability_threshold = STABILITY_THRESHOLD if stability_threshold is None else stability_threshold

    mi_rate = _resampled_hit_rate(X_train, y_train, mi_selected, seed)
    gbdt_rate = _resampled_hit_rate(X_train, y_train, gbdt_selected, seed + 1000)
    perm_rate = _resampled_hit_rate(X_train, y_train, permutation_selected, seed + 2000)
    _, enet_rate_raw = elasticnet_selected(X_train, y_train, seed + 3000)
    enet_rate = enet_rate_raw.reindex(X_train.columns, fill_value=0.0)

    stability = pd.DataFrame({
        "mutual_information": mi_rate,
        "gbdt_importance": gbdt_rate,
        "elasticnet_stability": enet_rate,
        "permutation_importance": perm_rate,
    })
    stability["n_methods_stable"] = (stability >= stability_threshold).sum(axis=1)
    pool_features = stability[stability["n_methods_stable"] >= rule_threshold].index.tolist()
    return pool_features, stability


def run_group_ablation(X_train, y_train, X_test, y_test, seed):
    """AL_/EK_/AA_/CAT_'ten her biri için o grubun kolonlarını düşürür ve
    yardımcı modelin dış-fold AUC'sinin tam modele göre ne kadar düştüğünü
    ölçer.
    """
    full_model = _lgb_model(seed).fit(X_train, y_train)
    full_auc = roc_auc_score(y_test, full_model.predict_proba(X_test)[:, 1])
    results = {"full": full_auc}
    for group in GROUPS:
        drop_cols = columns_for_group(X_train.columns, group)
        keep_cols = [c for c in X_train.columns if c not in drop_cols]
        model = _lgb_model(seed).fit(X_train[keep_cols], y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test[keep_cols])[:, 1])
        results[f"without_{group}"] = auc
    return results


def run_isolated_group_models(X_train, y_train, X_test, y_test, seed):
    """Her grup için, YALNIZCA o grubun kolonlarını kullanarak eğitir ve
    tek başına dış-fold AUC'sini raporlar.
    """
    results = {}
    for group in GROUPS:
        cols = columns_for_group(X_train.columns, group)
        if not cols:
            results[group] = None
            continue
        model = _lgb_model(seed).fit(X_train[cols], y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test[cols])[:, 1])
        results[group] = auc
    return results


FINAL_POOL_RULE_THRESHOLD = 2


def build_v4_final_pool(stability, rule_threshold=FINAL_POOL_RULE_THRESHOLD):
    """ESKİ/ARŞİV (P0-2, bkz. reports/09_ASAMA_F_ONCESI_YAPILACAKLAR_PAH.md):
    50 dış-fold'un hit-oranlarını TEK global tabloda birleştirip TEK sabit
    havuz üreten mimari -- `stability` girdisi (`main()`'in eski sürümünün
    ürettiği global cross-fold aggregation) `permutation_selected`'ın eski
    (sızıntılı, dış-test'i doğrudan kullanan) sürümüne dayanıyordu.

    Artık ÇAĞRILMIYOR -- `main()` bunun yerine `compute_fold_local_pool`'u
    her dış-fold için AYRI AYRI çağırıyor (bkz. o fonksiyonun docstring'i).
    Yalnızca tarihsel izlenebilirlik için (CLAUDE.md "her karar izlenebilir")
    ve eski `v4_final_feature_pool_ARCHIVED_leaky.json`'ın nasıl üretildiğini
    belgelemek için tanımlı bırakıldı -- SİLİNMEDİ.
    """
    mask = stability["n_methods_stable_at_60pct"] >= rule_threshold
    stable_features = stability[mask].index.tolist()
    pool = {
        "rule": (
            f"n_methods_stable_at_60pct >= {rule_threshold} "
            f"(4 bağımsız özellik seçimi yönteminden en az {rule_threshold}'sinde uzlaşma)"
        ),
        "rationale": (
            "elasticnet_stability tek başına, 405 özelliğin 165'inde kendi %60 eşiğini "
            "geçiyor (MI=4, GBDT=20, permutation=22'ye göre çok daha gevşek) -- bu, "
            "çoklu-doğrusallıktan çok az-belirlenmiş (n≈p) bir rejimi yansıtıyor (bkz. "
            "reports/03_OZELLIK_SECIMI_PAH.md). Bu NİHAİ, resmi koşumdur: AL_296_missing "
            "kaldırıldı, AL_ korelasyonunun min_periods değeri 30'a düzeltildi (bu "
            "pipeline'a etkisi yok, hem yapısal hem ampirik olarak doğrulandı), ve "
            "feature_selection.py::main()'deki fold'lar-arası birleşim (all_features "
            "union-across-folds) hatası düzeltildi (doğrulandı: bu koşumu etkilemedi -- "
            "son işlenen fold zaten 405 kolonluk birleşimin tamamını kapsıyordu). Bu dosya "
            "artık feature_selection.py::main() tarafından otomatik/izlenebilir şekilde "
            "üretilir (reports/06_ASAMA_E_ONCESI_DENETIM_PAH.md, bulgu K1 düzeltmesi)."
        ),
        "n_features": len(stable_features),
        "features": stable_features,
    }
    with open(TAB_DIR / "v4_final_feature_pool.json", "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False)
    return pool


def main():
    """DÜZELTME (P0-3): eski sürüm burada 50 dış-fold'un hit-oranlarını TEK
    global tabloda birleştirip `build_v4_final_pool` ile TEK sabit havuz
    yazıyordu (ve eski `permutation_selected` dış-test alıyordu). Yeni
    sürüm her dış-fold için `compute_fold_local_pool`'u AYRI AYRI çağırır
    -- havuzlar fold'lar arasında aynı boyutta/kimlikte olmak ZORUNDA
    değildir, bu beklenen ve doğru bir sonuçtur (bkz. compute_fold_local_
    pool docstring'i). `run_group_ablation`/`run_isolated_group_models`
    değişmedi -- ikisi de yalnızca diagnostik AUC ölçüyor, hiçbir seçim
    kararına girmiyor.
    """
    v1 = pd.read_parquet(V1_PATH)
    al_columns = [c for c in v1.columns if c.startswith("AL_")]

    fold_pools = {}
    stability_rows = []
    ablation_rows, isolated_rows = [], []
    n_folds_run = 0

    for repeat_idx in range(N_REPEATS):
        outer = json.load(open(SPLITS_DIR / f"outer_fold_repeat{repeat_idx:02d}.json"))
        for fold in outer["folds"]:
            seed = SEED + repeat_idx * 100 + fold["fold"]
            X_train, y_train, X_test, y_test = build_fold_features(
                v1, al_columns, fold["train_variant_ids"], fold["test_variant_ids"],
            )

            pool_features, stability = compute_fold_local_pool(X_train, y_train, seed)
            fold_key = f"repeat{repeat_idx:02d}_fold{fold['fold']}"
            fold_pools[fold_key] = {"n_features": len(pool_features), "features": sorted(pool_features)}

            stab_long = stability.reset_index(names="feature")
            stab_long["repeat"], stab_long["fold"] = repeat_idx, fold["fold"]
            stability_rows.append(stab_long)

            ablation = run_group_ablation(X_train, y_train, X_test, y_test, seed)
            ablation["repeat"], ablation["fold"] = repeat_idx, fold["fold"]
            ablation_rows.append(ablation)

            isolated = run_isolated_group_models(X_train, y_train, X_test, y_test, seed)
            isolated["repeat"], isolated["fold"] = repeat_idx, fold["fold"]
            isolated_rows.append(isolated)

            n_folds_run += 1
            print(f"repeat {repeat_idx} fold {fold['fold']} done "
                  f"({n_folds_run}/{N_REPEATS * N_OUTER_SPLITS}) -- pool={len(pool_features)} ozellik")

    with open(TAB_DIR / "v4_fold_local_pools.json", "w", encoding="utf-8") as f:
        json.dump({
            "rule": (
                f"n_methods_stable >= {FINAL_POOL_RULE_THRESHOLD} (4 bagimsiz yontemden en az "
                f"{FINAL_POOL_RULE_THRESHOLD}'sinde uzlasma), HER YONTEM yalnizca bu fold'un kendi "
                f"train_ids'i icindeki resampling'den turer -- dis-test hicbir yontemde kullanilmiyor"
            ),
            "n_folds": n_folds_run,
            "pools": fold_pools,
        }, f, indent=2, ensure_ascii=False)

    stability_long = pd.concat(stability_rows, ignore_index=True)
    stability_long.to_csv(TAB_DIR / "feature_selection_stability_fold_local.csv", index=False)

    pool_sizes = [v["n_features"] for v in fold_pools.values()]
    print(f"\nfold-lokal havuz buyuklugu: ort={np.mean(pool_sizes):.2f}, "
          f"std={np.std(pool_sizes):.2f}, min={min(pool_sizes)}, max={max(pool_sizes)}")
    print(f"kaydedildi: {TAB_DIR / 'v4_fold_local_pools.json'}, "
          f"{TAB_DIR / 'feature_selection_stability_fold_local.csv'}")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(TAB_DIR / "group_ablation_raw.csv", index=False)
    ablation_summary = ablation_df.drop(columns=["repeat", "fold"]).mean().to_frame("mean_auc")
    ablation_summary["std_auc"] = ablation_df.drop(columns=["repeat", "fold"]).std()
    ablation_summary.to_csv(TAB_DIR / "group_ablation_summary.csv")

    isolated_df = pd.DataFrame(isolated_rows)
    isolated_df.to_csv(TAB_DIR / "isolated_group_models_raw.csv", index=False)
    isolated_summary = isolated_df.drop(columns=["repeat", "fold"]).mean().to_frame("mean_auc")
    isolated_summary["std_auc"] = isolated_df.drop(columns=["repeat", "fold"]).std()
    isolated_summary.to_csv(TAB_DIR / "isolated_group_models_summary.csv")

    print("\n=== fold-lokal havuz buyuklugu dagilimi ===")
    print(pd.Series(pool_sizes).describe())
    print("\n=== grup ablation ===")
    print(ablation_summary)
    print("\n=== izole-grup modelleri ===")
    print(isolated_summary)


if __name__ == "__main__":
    main()
