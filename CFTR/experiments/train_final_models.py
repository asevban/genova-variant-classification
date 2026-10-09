from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "YARISMA_TRAIN_CFTR_REDUCED.csv"
ART = ROOT / "final_package" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
GROUPS = {"GRUP_AL6_AL251": ["AL_6", "AL_251"], "GRUP_AL1_AL211": ["AL_1", "AL_211"]}
REMOVE = ["CAT_6", "AL_6", "AL_251", "AL_1", "AL_211"]
SEEDS = [41001, 41002, 41003, 41004, 41005]

df = pd.read_csv(DATA)
y = df["Label"].astype(int)
raw = df.drop(columns=["Variant_ID", "Label"])

def fit_group_params(frame):
    out = {}
    for new, members in GROUPS.items():
        out[new] = {}
        for col in members:
            s = pd.to_numeric(frame[col], errors="coerce")
            med = float(s.median())
            scale = float(s.quantile(.75) - s.quantile(.25))
            if not np.isfinite(scale) or scale == 0:
                scale = float(s.std(ddof=0))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            out[new][col] = {"median": med, "scale": scale}
    return out

def transform(frame, params):
    z = frame.drop(columns=REMOVE).copy()
    for new, members in GROUPS.items():
        vals = []
        for col in members:
            p = params[new][col]
            vals.append((pd.to_numeric(frame[col], errors="coerce") - p["median"]) / p["scale"])
        z[new] = pd.concat(vals, axis=1).mean(axis=1, skipna=True)
    for col in z.columns:
        if col.startswith("CAT_") or col.startswith("AA_"):
            z[col] = z[col].fillna("__MISSING__").astype(str)
    return z

params = fit_group_params(raw)
X = transform(raw, params)
cats = [c for c in X.columns if c.startswith("CAT_") or c.startswith("AA_")]
nums = [c for c in X.columns if c not in cats]

for i, seed in enumerate(SEEDS, 1):
    cat = CatBoostClassifier(iterations=40, depth=4, learning_rate=.05, l2_leaf_reg=5,
        loss_function="Logloss", auto_class_weights="Balanced", random_seed=seed,
        verbose=False, allow_writing_files=False, thread_count=-1)
    cat.fit(X, y, cat_features=[X.columns.get_loc(c) for c in cats])
    cat.save_model(ART / f"catboost_repeat_{i}.cbm")

    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), nums),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), cats),
    ])
    rf = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=3,
        max_features=.3, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    pipe = Pipeline([("prep", prep), ("model", rf)])
    pipe.fit(X, y)
    joblib.dump(pipe, ART / f"randomforest_repeat_{i}.joblib", compress=3)

metadata = {
    "raw_required_features": list(raw.columns), "transformed_features": list(X.columns),
    "categorical_features": cats, "numeric_features": nums, "groups": GROUPS,
    "removed": REMOVE, "group_parameters": params, "seeds": SEEDS,
}
(ART / "preprocessing.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved 5 CatBoost + 5 RF models; transformed feature count={X.shape[1]}")
