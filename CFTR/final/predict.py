from pathlib import Path
import argparse, json, hashlib
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
ART = ROOT / "models" / "artifacts" if (ROOT / "models" / "artifacts").exists() else ROOT / "artifacts"
MAIN_THRESHOLD = 0.7224
BACKUP_THRESHOLD = 0.780952380952381

def transform(raw, meta):
    z = raw.drop(columns=meta["removed"]).copy()
    for new, members in meta["groups"].items():
        vals=[]
        for col in members:
            p=meta["group_parameters"][new][col]
            vals.append((pd.to_numeric(raw[col], errors="coerce")-p["median"])/p["scale"])
        z[new]=pd.concat(vals,axis=1).mean(axis=1,skipna=True)
    for col in meta["categorical_features"]:
        z[col]=z[col].fillna("__MISSING__").astype(str)
    return z[meta["transformed_features"]]

def id3_probability(X, file):
    model=json.loads(Path(file).read_text(encoding="utf-8")); probs=[]
    def value(row,name,typ):
        v=row[name]
        if typ=="categorical": return "__MISSING__" if pd.isna(v) or str(v)=="" else str(v)
        try: return float(v)
        except: return np.nan
    def walk(tree,row):
        n=tree
        while "feature" in n:
            v=value(row,n["feature"],n["type"])
            if n["type"]=="numeric":
                if n["kind"]=="missingness": key="present" if np.isfinite(v) else "missing"
                else: key="missing" if not np.isfinite(v) else ("le" if v<=n["threshold"] else "gt")
            else: key=v
            if key not in n["children"]: break
            n=n["children"][key]
        return n["prediction"]
    for _,row in X.iterrows(): probs.append(np.mean([walk(t,row) for t in model["trees"]]))
    return np.asarray(probs,float)

def drift_report(raw, meta, extras, missing, duplicate_ids):
    report={"rows":len(raw),"missing_required_columns":missing,"extra_columns_ignored":extras,
            "duplicate_variant_ids":duplicate_ids,"completely_empty_required_columns":[],
            "numeric_coercion_failures":{},"unseen_categories":{}}
    train=pd.read_csv(ROOT.parent/"YARISMA_TRAIN_CFTR_REDUCED.csv") if (ROOT.parent/"YARISMA_TRAIN_CFTR_REDUCED.csv").exists() else None
    for col in meta["raw_required_features"]:
        if raw[col].isna().all(): report["completely_empty_required_columns"].append(col)
        if col not in meta["categorical_features"]:
            fail=int((raw[col].notna() & pd.to_numeric(raw[col],errors="coerce").isna()).sum())
            if fail: report["numeric_coercion_failures"][col]=fail
        elif train is not None and col in train:
            known=set(train[col].dropna().astype(str)); seen=set(raw[col].dropna().astype(str)); new=sorted(seen-known)
            if new: report["unseen_categories"][col]=new[:20]
    return report

def main():
    ap=argparse.ArgumentParser(description="CFTR final inference")
    ap.add_argument("--input",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--mode",choices=["main","backup"],default="main")
    ap.add_argument("--audit"); ap.add_argument("--drift-report")
    args=ap.parse_args()
    meta=json.loads((ART/"preprocessing.json").read_text(encoding="utf-8"))
    df=pd.read_csv(args.input)
    if "Variant_ID" not in df: raise ValueError("Variant_ID sütunu zorunludur.")
    if df["Variant_ID"].isna().any() or (df["Variant_ID"].astype(str).str.strip()=="").any(): raise ValueError("Variant_ID boş olamaz.")
    missing=[c for c in meta["raw_required_features"] if c not in df.columns]
    if missing: raise ValueError(f"Eksik gerekli sütunlar: {missing}")
    extras=[c for c in df.columns if c not in meta["raw_required_features"]+["Variant_ID","Label"]]
    raw=df[meta["raw_required_features"]].copy(); X=transform(raw,meta)
    id3=np.mean([id3_probability(X,ART/f"id3_repeat_{i}.json") for i in range(1,6)],axis=0)
    if args.mode=="backup": prob=id3; threshold=BACKUP_THRESHOLD
    else:
        cat=[]; rf=[]
        for i in range(1,6):
            m=CatBoostClassifier(); m.load_model(ART/f"catboost_repeat_{i}.cbm"); cat.append(m.predict_proba(X)[:,1])
            rf.append(joblib.load(ART/f"randomforest_repeat_{i}.joblib").predict_proba(X)[:,1])
        prob=(id3+np.mean(cat,axis=0)+np.mean(rf,axis=0))/3; threshold=MAIN_THRESHOLD
    pred=(prob>=threshold).astype(int)
    out=pd.DataFrame({"Variant_ID":df["Variant_ID"],"Label":pred}); out.to_csv(args.output,index=False)
    if args.audit: pd.DataFrame({"Variant_ID":df["Variant_ID"],"probability":prob,"threshold":threshold,"Label":pred}).to_csv(args.audit,index=False)
    report=drift_report(raw,meta,extras,missing,int(df["Variant_ID"].duplicated().sum()))
    if args.drift_report: Path(args.drift_report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"mode={args.mode} rows={len(out)} positives={int(pred.sum())} threshold={threshold:.6f} output={args.output}")

if __name__=="__main__": main()
