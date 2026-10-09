"""P2 madde 16: CatBoost izgarasini `depth x learning_rate` = 6 adaya
genisletir (eski: 2 aday, `depth in {3,5}`, `lr=0.05` sabit). Madde 7'nin
dersi burada da uygulanir: (1) DOGRU OLCUT -- onsel-duzeltilmis +
uyarlanabilir esikli weighted-F1 (`f0_final_model_v3.py::prior_weighted_
inner_score`, DEGISTIRILMEDEN yeniden kullanildi), esik=0,5 ham F1 DEGIL;
(2) herhangi bir yeni aday "kazanirsa" tek bir 4-fold bolunmeye
guvenilmez -- madde 7'nin tekrarli-bolme + NB-zorunlu-kriter deseni
birebir tekrarlanir.

Early stopping: `iterations=100` SABIT (izgara boyutunu artirmiyor);
CatBoost'un kendi `early_stopping_rounds` mekanizmasi, egitim verisinin
ICINDEN ayrilmis kucuk bir dogrulama dilimiyle (val_frac=0,15, stratified)
uygulanir -- fold-guvenli (yalnizca o fold'un kendi train'i icinde).
Grup-farkindaligi (conflict_group_1, yalnizca 2 satir) burada BILEREK
atlanmis kucuk bir basitlestirme -- bu bir genelleme tahmini degil,
egitim-ici otomatik iterasyon-ayari teknigi.

Adim 4 (NB testi), yalnizca Adim 3'un tekrarli-bolme teshisi YENI adayi
GERCEK bir sinyal olarak dogrularsa (madde 7'nin kendi esigiyle ayni:
>=7/10) calistirilir -- yalnizca Adim2'nin tek-bolme "kazanani" degismis
olmasi yeterli degil (madde 7'nin kendi onceki tecrubesi: tek-bolme
kazanani her zaman gurultuye karsi kirilgan olabiliyor).

Calistirma: python -m genova.pah.f0_madde16_grid_search
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from genova.pah import fold_versions as fv
from genova.pah import split_bank as sb
from genova.pah import models as m
from genova.pah.f0_final_model import V1_PATH, F0_SEED, E5_FOLD_THRESHOLD_RANGE
from genova.pah.f0_final_model_v3 import prior_weighted_inner_score, CURRENT_BUNDLE_PATH
from genova.pah.calibration import BetaCalibrator
from genova.pah.e4_prior_correction import sld_correct, W1, W0
from genova.pah.e5_threshold_selection import select_threshold
from genova.statistics import nadeau_bengio_corrected_ttest

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
GRID_SEARCH_OUT = TAB_DIR / "f0_madde16_grid_search.csv"
STABILITY_OUT = TAB_DIR / "f0_madde16_stability_check.csv"
NB_OUT = TAB_DIR / "f0_madde16_nb_decision.csv"

EXPANDED_CATBOOST_GRID = [{"depth": d, "learning_rate": lr} for d in (3, 5) for lr in (0.03, 0.05, 0.1)]
EARLY_STOPPING_ROUNDS = 20
VAL_FRAC = 0.15
N_STABILITY_SEEDS = 10
N_STABILITY_SPLITS = 4
STABILITY_WIN_THRESHOLD = 7  # madde 7'nin kendi esigiyle ayni


def fit_predict_catboost_early_stopping(X_train, y_train, X_test, params, val_frac=VAL_FRAC,
                                         early_stopping_rounds=EARLY_STOPPING_ROUNDS, seed=F0_SEED):
    """`models.py::fit_predict_catboost` ile AYNI mimari -- ek olarak
    `X_train`'in ICINDEN (fold-guvenli, dis bir veriye erismeden) kucuk
    bir dogrulama dilimi ayirip `early_stopping_rounds` ile `iterations`'i
    otomatik kisitlar."""
    weights_full = m._minority_sample_weight(y_train)
    cat_cols = m._category_columns(X_train)

    def _stringify(X):
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].astype(object).fillna("__MISSING__").astype(str)
        return X

    idx_fit, idx_val = train_test_split(
        np.arange(len(y_train)), test_size=val_frac, random_state=seed, stratify=y_train,
    )
    X_fit, y_fit, w_fit = X_train.iloc[idx_fit], y_train.iloc[idx_fit], weights_full[idx_fit]
    X_val, y_val = X_train.iloc[idx_val], y_train.iloc[idx_val]

    model = CatBoostClassifier(
        iterations=100, cat_features=cat_cols, random_seed=m.RANDOM_STATE,
        verbose=False, thread_count=8, allow_writing_files=False, **params,
    )
    model.fit(
        _stringify(X_fit), y_fit, sample_weight=w_fit,
        eval_set=(_stringify(X_val), y_val), early_stopping_rounds=early_stopping_rounds,
    )
    return model.predict_proba(_stringify(X_test))[:, 1]


def grid_search_single_split(v1_df, pool, all_ids, grid, seed=F0_SEED):
    inner_folds = sb.build_inner_folds(v1_df, all_ids, seed=seed)
    scores = []
    for params in grid:
        fold_scores = []
        for fold in inner_folds:
            X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(
                v1_df, fold["train_variant_ids"], fold["val_variant_ids"], pool,
            )
            raw_proba = fit_predict_catboost_early_stopping(X_tr, y_tr, X_va, params)
            fold_scores.append(prior_weighted_inner_score(raw_proba, y_va))
        scores.append(float(np.mean(fold_scores)))
    return scores


def stability_check(v1_df, pool, all_ids, old_params, new_params, n_seeds=N_STABILITY_SEEDS):
    """Madde 7'nin `f0_madde7_stability_check.py` deseniyle BIREBIR AYNI --
    yalnizca eski/yeni kazanan (2 aday) 10 farkli rastgele 4-fold
    bolunmede (`StratifiedKFold`, split bankasindan bagimsiz) tekrar
    karsilastirilir."""
    rows = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=N_STABILITY_SPLITS, shuffle=True, random_state=seed)
        scores_by_params = {json.dumps(old_params): [], json.dumps(new_params): []}
        for train_idx, val_idx in skf.split(v1_df, v1_df["Label"]):
            train_ids = v1_df.iloc[train_idx]["Variant_ID"].tolist()
            val_ids = v1_df.iloc[val_idx]["Variant_ID"].tolist()
            for params in (old_params, new_params):
                X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, train_ids, val_ids, pool)
                raw_proba = fit_predict_catboost_early_stopping(X_tr, y_tr, X_va, params)
                scores_by_params[json.dumps(params)].append(prior_weighted_inner_score(raw_proba, y_va))

        mean_scores = {k: float(np.mean(v)) for k, v in scores_by_params.items()}
        winner_key = max(mean_scores, key=mean_scores.get)
        winner_params = json.loads(winner_key)
        sorted_scores = sorted(mean_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        is_new_winner = winner_params == new_params
        rows.append({"seed": seed, "winner_is_new": is_new_winner, "margin": margin,
                      "score_old": mean_scores[json.dumps(old_params)], "score_new": mean_scores[json.dumps(new_params)]})
        print(f"  seed={seed}: kazanan={'YENI' if is_new_winner else 'ESKI'} marj={margin:.4f}", flush=True)
    return pd.DataFrame(rows)


def per_fold_scores(v1_df, pool, params, calibrator, threshold):
    """Madde 7'nin `_per_fold_scores_for_bundle`'iyla AYNI 5-fold split
    (`sb.build_outer_folds(v1_df, seed=F0_SEED+1)`), ama bu adayin KENDI
    (henuz bundle'a paketlenmemis) hiperparametre/kalibratorunu kullanir."""
    from genova.metrics import f1_binary_positive_weighted
    from genova.pah.e5_threshold_selection import _prior_weights, FINAL_PATHOGENIC_PRIOR
    outer_folds = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    rows = []
    for fold in outer_folds:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, fold["train_variant_ids"], fold["test_variant_ids"], pool)
        raw_proba = fit_predict_catboost_early_stopping(X_tr, y_tr, X_va, params)
        calibrated = calibrator.transform(raw_proba)
        sld = sld_correct(calibrated, w1=W1, w0=W0)
        pred = (sld >= threshold).astype(int)
        weights = _prior_weights(y_va, FINAL_PATHOGENIC_PRIOR)
        rows.append({
            "fold": fold["fold"], "n_train": len(y_tr), "n_test": len(y_va),
            "weighted_f1": f1_binary_positive_weighted(y_va, pred, weights),
        })
    return pd.DataFrame(rows)


def main():
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()
    current_bundle = joblib.load(CURRENT_BUNDLE_PATH)
    pool = current_bundle["pool"]
    old_params = current_bundle["model_best_params"]
    print(f"mevcut model: {old_params}, havuz={len(pool)} ozellik", flush=True)

    print(f"\n=== Adim 1-2: genisletilmis izgara ({len(EXPANDED_CATBOOST_GRID)} aday), tek 4-fold bolunme, dogru olcut ===", flush=True)
    scores = grid_search_single_split(v1_df, pool, all_ids, EXPANDED_CATBOOST_GRID)
    for params, score in zip(EXPANDED_CATBOOST_GRID, scores):
        print(f"  {params}: onsel-agirlikli ic-F1={score:.4f}", flush=True)
    best_idx = int(np.argmax(scores))
    new_params = EXPANDED_CATBOOST_GRID[best_idx]
    pd.DataFrame([{"params": json.dumps(p), "score": s} for p, s in zip(EXPANDED_CATBOOST_GRID, scores)]).to_csv(GRID_SEARCH_OUT, index=False)
    print(f"kaydedildi: {GRID_SEARCH_OUT}")
    print(f"\nyeni kazanan: {new_params} (skor={scores[best_idx]:.4f})  |  mevcut: {old_params}", flush=True)

    if new_params == old_params:
        print("\n=== SONUC: genisletilmis izgarada da AYNI aday kazandi -- mevcut model daha genis "
              "bir aday havuzuna karsi da SAGLAM. DURULUYOR. ===", flush=True)
        return

    print(f"\n=== Adim 3: tekrarli-bolme teshisi (10 tekrar, eski vs yeni) ===", flush=True)
    stability_df = stability_check(v1_df, pool, all_ids, old_params, new_params)
    stability_df.to_csv(STABILITY_OUT, index=False)
    n_new_wins = int(stability_df["winner_is_new"].sum())
    print(f"\nyeni aday kazandi: {n_new_wins}/{N_STABILITY_SEEDS}  "
          f"(marj: ort={stability_df['margin'].mean():.4f} std={stability_df['margin'].std():.4f})", flush=True)
    print(f"kaydedildi: {STABILITY_OUT}")

    if n_new_wins < STABILITY_WIN_THRESHOLD:
        print(f"\n=== SONUC: {n_new_wins}/10 < {STABILITY_WIN_THRESHOLD} esigi -- GURULTU. "
              "Mevcut model korunuyor. DURULUYOR. ===", flush=True)
        return

    print(f"\n=== Adim 4: {n_new_wins}/10 >= {STABILITY_WIN_THRESHOLD} -- gercek sinyal, NB testine geciliyor ===", flush=True)
    outer_folds_5fold = sb.build_outer_folds(v1_df, repeat_idx=0, seed=F0_SEED + 1)
    proba_by_vid, y_by_vid = {}, {}
    for fold in outer_folds_5fold:
        X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, fold["train_variant_ids"], fold["test_variant_ids"], pool)
        proba = fit_predict_catboost_early_stopping(X_tr, y_tr, X_va, new_params)
        from genova.pah.e3_calibration_run import _variant_ids_in_row_order
        val_vids = _variant_ids_in_row_order(v1_df, fold["test_variant_ids"])
        for vid, p, y in zip(val_vids, proba, y_va):
            proba_by_vid[vid] = p
            y_by_vid[vid] = y
    ordered_ids = v1_df["Variant_ID"].tolist()
    oof_proba = np.array([proba_by_vid[v] for v in ordered_ids])
    y_full = np.array([y_by_vid[v] for v in ordered_ids])

    new_calibrator = BetaCalibrator().fit(oof_proba, y_full)
    calibrated_oof = new_calibrator.transform(oof_proba)
    sld_oof = sld_correct(calibrated_oof)
    new_threshold, _ = select_threshold(sld_oof, y_full)
    lo, hi = E5_FOLD_THRESHOLD_RANGE
    print(f"yeni esik={new_threshold:.2f}", flush=True)
    if not (lo <= new_threshold <= hi):
        raise RuntimeError(f"Saglamlik kontrolu basarisiz: esik={new_threshold:.2f} [{lo},{hi}] disinda -- DURDURULDU.")

    old_folds = per_fold_scores(v1_df, pool, old_params, current_bundle["calibrator"], current_bundle["threshold"])
    new_folds = per_fold_scores(v1_df, pool, new_params, new_calibrator, new_threshold)
    n_train, n_test = old_folds["n_train"].to_numpy(), old_folds["n_test"].to_numpy()
    nb = nadeau_bengio_corrected_ttest(new_folds["weighted_f1"].to_numpy(), old_folds["weighted_f1"].to_numpy(), n_train, n_test)
    favors_new = (nb["mean_diff"] > 0) and (nb["p_value"] < 0.05)
    print(f"NB (weighted-F1, yeni-eski): mean_diff={nb['mean_diff']:+.4f} p={nb['p_value']:.4f} yeni-lehine-anlamli={favors_new}", flush=True)
    pd.DataFrame([{
        "old_params": json.dumps(old_params), "new_params": json.dumps(new_params),
        "nb_mean_diff": nb["mean_diff"], "nb_p_value": nb["p_value"], "favors_new": favors_new,
    }]).to_csv(NB_OUT, index=False)
    print(f"kaydedildi: {NB_OUT}")

    if favors_new:
        print("\n=== SONUC: yeni aday NB testinde anlamli ustunluk gosteriyor -- DURDURULDU, "
              "kullaniciya bildirilmeli, otomatik benimseme YOK. Hicbir bundle/predict.py degistirilmedi. ===", flush=True)
    else:
        print("\n=== SONUC: NB testinde anlamli degil -- mevcut model korunuyor. DURULUYOR. ===", flush=True)


if __name__ == "__main__":
    main()
