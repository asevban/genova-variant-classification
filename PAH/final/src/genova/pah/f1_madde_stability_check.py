"""Asama F1 Ek: `AL_26`/`AL_12`/`AL_7`/`AL_49` (provenance-riskli, F1
adversarial validation'in buldugu) ablasyonunun tekrarli-bolme testi --
madde 7'nin AYNI deseniyle (10 farkli rastgele `StratifiedKFold` bolunmesi,
split bankasindan bagimsiz, `i=0..9`). Tek bir ablasyon karsilastirmasi
(bkz. `f1_ablation_comparison.py`) belirsiz cikmisti (CI ortusuyor, NB
anlamsiz, ama tum metrikler tutarli yonde kotulesiyor) -- bu, madde 7'deki
`depth=3` vs `depth=5` belirsizligiyle AYNI yapida, ayni cozum uygulaniyor.

Her tekrarda HEM 26-ozellik (mevcut) HEM 22-ozellik (temizlenmis) icin
Adim 2 (hiperparametre secimi, `prior_weighted_inner_score` -- madde 7'nin
DUZELTILMIS olcutu) + Adim 3 (cross-fit OOF -> Beta kalibrasyon -> SLD ->
esik) yeniden calistirilir, nihai onsel-agirlikli F1 karsilastirilir.

Calistirma: python -m genova.pah.f1_madde_stability_check
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from genova.pah import fold_versions as fv
from genova.pah.models import CATBOOST_GRID, fit_predict_catboost
from genova.pah.f0_final_model import V1_PATH, MODEL_DIR
from genova.pah.f0_final_model_v3 import prior_weighted_inner_score, CURRENT_BUNDLE_PATH
from genova.pah.calibration import BetaCalibrator
from genova.pah.e4_prior_correction import sld_correct
from genova.pah.e5_threshold_selection import select_threshold

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
NEW_POOL_PATH = TAB_DIR / "f1_final_feature_pool_v3.json"
OUT_CSV = TAB_DIR / "f1_madde_stability_check.csv"

N_SEEDS = 10
N_HP_SPLITS = 4
N_OOF_SPLITS = 5
WIN_THRESHOLD = 7
MARGIN_THRESHOLD = 0.01


def select_hp(v1_df, pool, seed):
    skf = StratifiedKFold(n_splits=N_HP_SPLITS, shuffle=True, random_state=seed)
    scores = []
    for params in CATBOOST_GRID:
        fold_scores = []
        for train_idx, val_idx in skf.split(v1_df, v1_df["Label"]):
            train_ids = v1_df.iloc[train_idx]["Variant_ID"].tolist()
            val_ids = v1_df.iloc[val_idx]["Variant_ID"].tolist()
            X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, train_ids, val_ids, pool)
            raw_proba = fit_predict_catboost(X_tr, y_tr, X_va, params)
            fold_scores.append(prior_weighted_inner_score(raw_proba, y_va))
        scores.append(float(np.mean(fold_scores)))
    best_idx = int(np.argmax(scores))
    return CATBOOST_GRID[best_idx]


def cross_fit_oof_plain(v1_df, pool, best_params, seed):
    skf = StratifiedKFold(n_splits=N_OOF_SPLITS, shuffle=True, random_state=seed)
    proba_by_pos, y_by_pos = {}, {}
    for train_idx, test_idx in skf.split(v1_df, v1_df["Label"]):
        train_ids = v1_df.iloc[train_idx]["Variant_ID"].tolist()
        test_ids = v1_df.iloc[test_idx]["Variant_ID"].tolist()
        X_tr, y_tr, X_te, y_te = fv.build_v4_from_v2(v1_df, train_ids, test_ids, pool)
        proba = fit_predict_catboost(X_tr, y_tr, X_te, best_params)
        for idx, p, y in zip(test_idx, proba, y_te):
            proba_by_pos[idx] = p
            y_by_pos[idx] = y
    ordered = sorted(proba_by_pos.keys())
    oof_proba = np.array([proba_by_pos[i] for i in ordered])
    y_full = np.array([y_by_pos[i] for i in ordered])
    return oof_proba, y_full


def pool_score(v1_df, pool, seed):
    best_params = select_hp(v1_df, pool, seed)
    oof_proba, y_full = cross_fit_oof_plain(v1_df, pool, best_params, seed)
    cal = BetaCalibrator().fit(oof_proba, y_full)
    calibrated = cal.transform(oof_proba)
    sld = sld_correct(calibrated)
    _, best_wf1 = select_threshold(sld, y_full)
    return best_wf1, best_params


def main():
    v1_df = pd.read_parquet(V1_PATH)
    current_bundle = joblib.load(CURRENT_BUNDLE_PATH)
    pool_26 = sorted(current_bundle["pool"])
    pool_22 = sorted(json.loads(NEW_POOL_PATH.read_text(encoding="utf-8"))["features"])
    print(f"26-ozellik (mevcut): {len(pool_26)}  |  22-ozellik (temizlenmis): {len(pool_22)}", flush=True)

    rows = []
    for seed in range(N_SEEDS):
        score_26, params_26 = pool_score(v1_df, pool_26, seed)
        score_22, params_22 = pool_score(v1_df, pool_22, seed)
        winner = "22" if score_22 > score_26 else "26"
        margin_signed = score_22 - score_26  # pozitif -> 22 lehine
        rows.append({
            "seed": seed, "score_26": score_26, "score_22": score_22,
            "winner": winner, "margin_signed_22_minus_26": margin_signed,
            "params_26": json.dumps(params_26), "params_22": json.dumps(params_22),
        })
        print(f"  seed={seed}: 26->{score_26:.4f}  22->{score_22:.4f}  kazanan={winner}  "
              f"marj(22-26)={margin_signed:+.4f}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nkaydedildi: {OUT_CSV}", flush=True)

    n_22_wins = int((result["winner"] == "22").sum())
    n_26_wins = N_SEEDS - n_22_wins
    mean_signed_margin = result["margin_signed_22_minus_26"].mean()
    std_signed_margin = result["margin_signed_22_minus_26"].std()

    print(f"\n=== SONUC ===")
    print(f"22-ozellik (temizlenmis) kazandi: {n_22_wins}/{N_SEEDS}")
    print(f"26-ozellik (mevcut) kazandi: {n_26_wins}/{N_SEEDS}")
    print(f"yon-isaretli marj (22-26): ort={mean_signed_margin:+.4f} std={std_signed_margin:.4f}")

    if n_22_wins >= WIN_THRESHOLD or (4 <= n_22_wins <= 6 and abs(mean_signed_margin) < MARGIN_THRESHOLD):
        verdict = "22-OZELLIK KAZANDI, MALIYET DUSUK/YOK -- yeni resmi final yapilmali (madde 11 disipliniyle)"
    elif n_26_wins >= WIN_THRESHOLD and abs(mean_signed_margin) >= MARGIN_THRESHOLD:
        verdict = "26-OZELLIK (MEVCUT) KAZANDI, GERCEK MALIYET VAR -- DURDURULMALI, kullaniciya bildirilmeli"
    else:
        verdict = "BELIRSIZ (4-6/10 VE |marj|>=0,01) -- DURDURULMALI, her iki taraf da raporlanmali"

    print(f"\nKARAR KURALINA GORE: {verdict}")


if __name__ == "__main__":
    main()
