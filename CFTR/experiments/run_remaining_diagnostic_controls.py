from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)
DATA=ROOT/"YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv"
PRED=OUT/"randomforest_319_hybrid_predictions.csv"
df=pd.read_csv(DATA); ids=df.Variant_ID.astype(str); y=df.Label.astype(int).to_numpy(); X=df.drop(columns=["Variant_ID","Label"])

def metrics(a,p):
    a=np.asarray(a,int); p=np.asarray(p,int)
    tn=int(((a==0)&(p==0)).sum()); fp=int(((a==0)&(p==1)).sum()); fn=int(((a==1)&(p==0)).sum()); tp=int(((a==1)&(p==1)).sum())
    sp=tn/(tn+fp) if tn+fp else 0.; re=tp/(tp+fn) if tp+fn else 0.; den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)); mcc=(tp*tn-fp*fn)/den if den else 0.
    ptp=20*re; pfp=100*(1-sp); prec=ptp/(ptp+pfp) if ptp+pfp else 0.; pf1=2*prec*re/(prec+re) if prec+re else 0.
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,specificity=sp,recall=re,mcc=mcc,projected_f1=pf1,fp_per_100_benign=pfp)

# 1) Exact and near-duplicate audit.
norm=X.copy()
for c in norm:
    if c.startswith("CAT_") or c.startswith("AA_"): norm[c]=norm[c].fillna("__MISSING__").astype(str)
    else: norm[c]=pd.to_numeric(norm[c],errors="coerce")
exact_key=norm.astype(str).agg("\x1f".join,axis=1)
exact=[]
for key,idx in exact_key.groupby(exact_key).groups.items():
    if len(idx)>1: exact.append({"rows":";".join(map(str,idx)),"Variant_IDs":";".join(ids.iloc[list(idx)]),"count":len(idx),"labels":";".join(map(str,sorted(set(y[list(idx)]))))})

cats=[c for c in X if c.startswith("CAT_") or c.startswith("AA_")]; nums=[c for c in X if c not in cats]
Z=np.empty((len(X),len(nums)),float)
for j,c in enumerate(nums):
    s=pd.to_numeric(X[c],errors="coerce"); med=s.median(); scale=s.quantile(.75)-s.quantile(.25)
    if not np.isfinite(scale) or scale==0: scale=s.std(ddof=0)
    if not np.isfinite(scale) or scale==0: scale=1.
    Z[:,j]=(s-med)/scale
C=X[cats].fillna("__MISSING__").astype(str).to_numpy() if cats else np.empty((len(X),0))
near=[]
for i in range(len(X)):
    for j in range(i+1,len(X)):
        both=np.isfinite(Z[i])&np.isfinite(Z[j]); one=np.isfinite(Z[i])^np.isfinite(Z[j]); parts=[]
        if both.any(): parts.extend(np.minimum(np.abs(Z[i,both]-Z[j,both]),1.).tolist())
        if one.any(): parts.extend(np.ones(one.sum()).tolist())
        if C.shape[1]: parts.extend((C[i]!=C[j]).astype(float).tolist())
        dist=float(np.mean(parts)) if parts else 1.; sim=1-dist
        near.append({"row_a":i,"row_b":j,"Variant_ID_a":ids.iloc[i],"Variant_ID_b":ids.iloc[j],"label_a":int(y[i]),"label_b":int(y[j]),"similarity":sim,"conflicting_label":int(y[i]!=y[j])})
near=sorted(near,key=lambda r:r["similarity"],reverse=True)
pd.DataFrame(exact).to_csv(OUT/"exact_duplicate_audit.csv",index=False)
pd.DataFrame(near[:100]).to_csv(OUT/"near_duplicate_top100.csv",index=False)

# 2) Missingness-only diagnostic, no feature/model selection.
M=X.isna().astype(np.uint8); M=M.loc[:,M.nunique()>1]
miss_rows=[]; miss_pred={"LogisticRegression":np.zeros((5,len(y))),"RandomForest":np.zeros((5,len(y)))}
for rep in range(5):
    cv=StratifiedKFold(5,shuffle=True,random_state=62000+rep)
    for fold,(tr,te) in enumerate(cv.split(M,y),1):
        models={"LogisticRegression":LogisticRegression(C=1,max_iter=2000,class_weight="balanced",random_state=rep),
                "RandomForest":RandomForestClassifier(n_estimators=150,max_depth=4,min_samples_leaf=3,class_weight="balanced_subsample",random_state=63000+rep*10+fold,n_jobs=-1)}
        for name,m in models.items():
            m.fit(M.iloc[tr],y[tr]); s=m.predict_proba(M.iloc[te])[:,1]; miss_pred[name][rep,te]=s
            z=metrics(y[te],(s>=.5).astype(int)); miss_rows.append({"model":name,"repeat":rep+1,"fold":fold,**z})
miss_summary=[]
for name,A in miss_pred.items():
    pred=(A.ravel()>=.5).astype(int); actual=np.tile(y,5); z=metrics(actual,pred)
    z.update(model=name,roc_auc=roc_auc_score(actual,A.ravel()),nonconstant_missing_masks=M.shape[1]); miss_summary.append(z)
pd.DataFrame(miss_rows).to_csv(OUT/"missing_only_nested_folds.csv",index=False)
pd.DataFrame(miss_summary).to_csv(OUT/"missing_only_nested_scores.csv",index=False)

# 3) Label permutation against frozen, fully out-of-fold main predictions.
p=pd.read_csv(PRED); p=p[p.recipe=="ID3_CAT_RF_equal"].copy()
observed=metrics(p.actual,p.predicted); rng=np.random.default_rng(20260904); B=100000
null_mcc=np.empty(B); null_f1=np.empty(B)
actual=p.actual.to_numpy(int); predicted=p.predicted.to_numpy(int)
for b in range(B):
    shuffled=rng.permutation(actual); z=metrics(shuffled,predicted); null_mcc[b]=z["mcc"]; null_f1[b]=z["projected_f1"]
perm={"method":"frozen_out_of_fold_prediction_label_permutation","permutations":B,"observed_mcc":observed["mcc"],"mcc_null_mean":float(null_mcc.mean()),"mcc_p_value":float((1+(null_mcc>=observed["mcc"]).sum())/(B+1)),"observed_projected_f1":observed["projected_f1"],"projected_f1_null_mean":float(null_f1.mean()),"projected_f1_p_value":float((1+(null_f1>=observed["projected_f1"]).sum())/(B+1)),"limitation":"Labels were permuted against frozen OOF predictions; the full training pipeline was not refit for every permutation."}
(OUT/"label_permutation_frozen_oof.json").write_text(json.dumps(perm,indent=2),encoding="utf-8")

high_near=[r for r in near if r["similarity"]>=.98]; conflicts=[r for r in high_near if r["conflicting_label"]]
md="# Kalan Tanısal Teknik Kontroller\n\nBu kontroller modeli, özellikleri veya eşiği değiştirmek için kullanılmamıştır.\n\n"
md+=f"## 1. Aynı/yakın satır denetimi\n\n- Tam aynı özellik satırı grubu: **{len(exact)}**\n- Benzerliği ≥0.98 olan çift: **{len(high_near)}**\n- Bunların içinde farklı etiketli çift: **{len(conflicts)}**\n- En yakın çift benzerliği: **{near[0]['similarity']:.4f}** (`{near[0]['Variant_ID_a']}`–`{near[0]['Variant_ID_b']}`)\n\n"
md+="Yakınlık; robust ölçeklenmiş sayısal farklar, kategorik uyuşmazlıklar ve eksiklik farklılıklarının birlikte ortalamasıdır. Bu tanısal eşik otomatik satır silme gerekçesi değildir.\n\n"
md+="## 2. Yalnız eksiklik maskesi modelleri\n\n|Model|Değişken maske|ROC-AUC|MCC|Proj. F1|Recall|Specificity|FP/100|\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
for r in miss_summary: md+=f"|{r['model']}|{r['nonconstant_missing_masks']}|{r['roc_auc']:.4f}|{r['mcc']:.4f}|{r['projected_f1']:.4f}|{r['recall']:.4f}|{r['specificity']:.4f}|{r['fp_per_100_benign']:.2f}|\n"
md+="\nYalnız eksiklik deseninin güçlü sonuç vermesi, biyolojik sinyal olabileceği gibi veri hazırlama kaynaklı sızıntı şüphesi de oluşturur; sonuç bu açıdan yorumlanmalıdır.\n\n"
md+="## 3. Etiket permütasyonu\n\n"
md+=f"- Gözlenen MCC: **{perm['observed_mcc']:.4f}**; permütasyon ortalaması: **{perm['mcc_null_mean']:.4f}**; p: **{perm['mcc_p_value']:.6f}**\n- Gözlenen projekte F1: **{perm['observed_projected_f1']:.4f}**; permütasyon ortalaması: **{perm['projected_f1_null_mean']:.4f}**; p: **{perm['projected_f1_p_value']:.6f}**\n\n"
md+="Bu test, dondurulmuş dış-fold tahminleriyle etiket ilişkisinin rastlantısal olup olmadığını sınar. Bütün eğitim hattının her permütasyonda yeniden eğitildiği daha ağır test değildir; bu sınırlılık açıkça korunmuştur.\n"
(OUT/"KALAN_TANISAL_TEKNIK_KONTROLLER.md").write_text(md,encoding="utf-8")
print(md)
