"""P1 madde 7 teshis turu: `depth=3` vs `depth=5` kazananinin, TEK bir
4-fold bolunmesinin sans eseri sonucu mu, yoksa tekrarli bolunmelerde
tutarli bir sinyal mi oldugunu olcer. HICBIR KALICI MODEL/BUNDLE URETMEZ
-- yalnizca 10 farkli rastgele 4-fold bolunmesinde (`StratifiedKFold(
shuffle=True, random_state=i)`, i=0..9 -- split bankasindan BAGIMSIZ,
grup-farkinda DEGIL, bilerek boyle: bu saf bir gurultu/kararlilik testi,
nested genelleme iddiasi degil) hangi hiperparametrenin kazandigini ve
kazanma marjini toplar.

`f0_final_model_v3.py::prior_weighted_inner_score` (P1 madde 7'nin
duzeltilmis ic secim olcutu) DEGISTIRILMEDEN yeniden kullanilir.

Calistirma: python -m genova.pah.f0_madde7_stability_check
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from genova.pah import fold_versions as fv
from genova.pah.models import CATBOOST_GRID, fit_predict_catboost
from genova.pah.f0_final_model import MODEL_DIR, V1_PATH
from genova.pah.f0_final_model_v3 import prior_weighted_inner_score, CURRENT_BUNDLE_PATH

ROOT = Path(__file__).resolve().parents[3]
TAB_DIR = ROOT / "reports" / "tables"
OUT_CSV = TAB_DIR / "f0_madde7_stability_check.csv"

N_SEEDS = 10
N_SPLITS = 4


def _repeated_selection(v1_df, pool, all_ids):
    rows = []
    for seed in range(N_SEEDS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        scores_by_params = {json.dumps(p): [] for p in CATBOOST_GRID}
        for train_idx, val_idx in skf.split(v1_df, v1_df["Label"]):
            train_ids = v1_df.iloc[train_idx]["Variant_ID"].tolist()
            val_ids = v1_df.iloc[val_idx]["Variant_ID"].tolist()
            for params in CATBOOST_GRID:
                X_tr, y_tr, X_va, y_va = fv.build_v4_from_v2(v1_df, train_ids, val_ids, pool)
                raw_proba = fit_predict_catboost(X_tr, y_tr, X_va, params)
                scores_by_params[json.dumps(params)].append(prior_weighted_inner_score(raw_proba, y_va))

        mean_scores = {k: float(np.mean(v)) for k, v in scores_by_params.items()}
        winner_key = max(mean_scores, key=mean_scores.get)
        winner_params = json.loads(winner_key)
        sorted_scores = sorted(mean_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]

        rows.append({
            "seed": seed,
            "winner_depth": winner_params["depth"], "winner_lr": winner_params["learning_rate"],
            "margin": margin,
            **{f"score_depth{json.loads(k)['depth']}": v for k, v in mean_scores.items()},
        })
        print(f"  seed={seed}: kazanan=depth={winner_params['depth']} marj={margin:.4f} "
              f"skorlar={mean_scores}", flush=True)
    return pd.DataFrame(rows)


def main():
    current_bundle = joblib.load(CURRENT_BUNDLE_PATH)
    pool = current_bundle["pool"]
    v1_df = pd.read_parquet(V1_PATH)
    all_ids = v1_df["Variant_ID"].tolist()

    print(f"havuz: {len(pool)} ozellik (mevcut resmi bundle'dan, final_model_bundle_v2.pkl)", flush=True)
    print(f"\n=== {N_SEEDS} tekrarli 4-fold hiperparametre secimi (StratifiedKFold, split bankasindan bagimsiz) ===", flush=True)
    result = _repeated_selection(v1_df, pool, all_ids)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    n_depth3 = int((result["winner_depth"] == 3).sum())
    n_depth5 = int((result["winner_depth"] == 5).sum())
    print(f"\n=== SONUC ===")
    print(f"depth=3 kazandi: {n_depth3}/{N_SEEDS}")
    print(f"depth=5 kazandi: {n_depth5}/{N_SEEDS}")
    print(f"marj: ort={result['margin'].mean():.4f} std={result['margin'].std():.4f} "
          f"min={result['margin'].min():.4f} max={result['margin'].max():.4f}")

    if n_depth3 >= 7:
        verdict = "GERCEK SINYAL -- depth=3 tutarli kazaniyor, Adim 3'e (yeniden fit + karsilastirma) gecilmeli"
    elif n_depth3 <= 3:
        verdict = "GURULTU -- depth=5 (mevcut final_model_bundle_v2.pkl) zaten dogru secimdi, korunuyor"
    else:
        verdict = "GURULTU (yazi-tura civari) -- mevcut final_model_bundle_v2.pkl (depth=5) korunuyor"
    print(f"\nKARAR KURALINA GORE: {verdict}")

    print(f"\nkaydedildi: {OUT_CSV}")


if __name__ == "__main__":
    main()
