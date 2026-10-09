from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"; ART=ROOT/"final_package"/"artifacts"

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); total=0.; rows=[]
    for i in range(bins):
        mask=(p>=edges[i])&(p<(edges[i+1]) if i<bins-1 else p<=edges[i+1])
        if not mask.any(): continue
        conf=float(p[mask].mean()); obs=float(y[mask].mean()); weight=float(mask.mean()); total+=weight*abs(conf-obs)
        rows.append({"bin":i+1,"lower":edges[i],"upper":edges[i+1],"n":int(mask.sum()),"mean_probability":conf,"observed_positive_rate":obs,"absolute_gap":abs(conf-obs)})
    return total,rows

sets=[]
pooled=pd.read_csv(OUT/"randomforest_319_hybrid_predictions.csv")
pooled=pooled[pooled.recipe=="ID3_CAT_RF_equal"]
sets.append(("pooled_5x5",pooled.actual.to_numpy(int),pooled.score.to_numpy(float)))
bag=pd.read_csv(OUT/"cross_fitted_bagging_variant_predictions.csv")
sets.append(("cross_fitted_variant_average",bag.actual.to_numpy(int),bag.score_mean.to_numpy(float)))
summ=[]; bins=[]
for name,y,p in sets:
    val,rows=ece(y,p,10); frac,mean=calibration_curve(y,p,n_bins=10,strategy="uniform")
    summ.append({"evaluation":name,"n_predictions":len(y),"brier_score":brier_score_loss(y,p),"ece_10_equal_width":val,"probability_mean":p.mean(),"observed_positive_rate":y.mean()})
    bins.extend({"evaluation":name,**r} for r in rows)
pd.DataFrame(summ).to_csv(OUT/"calibration_metrics.csv",index=False)
pd.DataFrame(bins).to_csv(OUT/"calibration_curve_bins.csv",index=False)
fig,ax=plt.subplots(figsize=(7,6)); ax.plot([0,1],[0,1],"--",color="gray",label="İdeal kalibrasyon")
for name,y,p in sets:
    frac,mean=calibration_curve(y,p,n_bins=10,strategy="uniform")
    ax.plot(mean,frac,marker="o",label=name)
ax.set(xlabel="Ortalama tahmin olasılığı",ylabel="Gözlenen pozitif oranı",title="CFTR Ensemble Kalibrasyon Eğrisi",xlim=(0,1),ylim=(0,1)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"calibration_curve.png",dpi=180); plt.close(fig)

meta=json.loads((ART/"preprocessing.json").read_text(encoding="utf-8")); raw=pd.read_csv(ROOT/"YARISMA_TRAIN_CFTR_REDUCED.csv").drop(columns=["Variant_ID","Label"])
X=raw.drop(columns=meta["removed"]).copy()
for new,members in meta["groups"].items():
    vals=[]
    for c in members:
        q=meta["group_parameters"][new][c]; vals.append((pd.to_numeric(raw[c],errors="coerce")-q["median"])/q["scale"])
    X[new]=pd.concat(vals,axis=1).mean(axis=1,skipna=True)
for c in meta["categorical_features"]: X[c]=X[c].fillna("__MISSING__").astype(str)
X=X[meta["transformed_features"]]; cat_idx=[X.columns.get_loc(c) for c in meta["categorical_features"]]; pool=Pool(X,cat_features=cat_idx)
all_shap=[]
for i in range(1,6):
    model=CatBoostClassifier(); model.load_model(ART/f"catboost_repeat_{i}.cbm")
    all_shap.append(np.abs(model.get_feature_importance(pool,type="ShapValues")[:,:-1]))
importance=np.mean(np.stack(all_shap),axis=(0,1)); shap=pd.DataFrame({"feature":X.columns,"mean_absolute_shap":importance}).sort_values("mean_absolute_shap",ascending=False)
def family(c):
    if c.startswith("GRUP_"): return "GRUP"
    return c.split("_",1)[0]
shap["family"]=shap.feature.map(family); shap.to_csv(OUT/"catboost_shap_feature_importance.csv",index=False)
fam=shap.groupby("family",as_index=False).mean_absolute_shap.sum().sort_values("mean_absolute_shap",ascending=False); fam.to_csv(OUT/"catboost_shap_family_importance.csv",index=False)

def mdtab(frame):
    h=list(frame.columns); lines=["|"+"|".join(h)+"|","|"+"|".join(["---"]*len(h))+"|"]
    for _,r in frame.iterrows(): lines.append("|"+"|".join(f"{v:.6f}" if isinstance(v,(float,np.floating)) else str(v) for v in r)+"|")
    return "\n".join(lines)
md="# Kalibrasyon ve SHAP Tanısal Raporu\n\nBu analiz mevcut modeli veya `0.7224` eşiğini değiştirmez.\n\n## Kalibrasyon\n\n"+mdtab(pd.DataFrame(summ))+"\n\nBrier ve ECE için daha düşük değer daha iyidir. Pooled 5×5 satırlar aynı varyantın beş tahminini içerir; cross-fitted varyant ortalaması her varyantı bir kez sayar ve ana tanısal gösterimdir. Bu sonuçlardan yeni eşik seçilmemiştir.\n\n## CatBoost SHAP — ilk 20\n\n"+mdtab(shap.head(20))+"\n\n## Özellik ailesi toplamı\n\n"+mdtab(fam)+"\n\nSHAP yalnız dondurulmuş CatBoost bileşenini açıklar; ID3+CatBoost+RF ensemble'ının tamamının nedensel açıklaması değildir. Özellik silmek veya modeli yeniden seçmek için kullanılmamıştır.\n"
(OUT/"KALIBRASYON_VE_SHAP_TANISAL_RAPORU.md").write_text(md,encoding="utf-8")
print(pd.DataFrame(summ).to_string(index=False)); print(shap.head(10).to_string(index=False)); print(fam.to_string(index=False))
