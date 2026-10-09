from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"
FEATURE_SET=int(sys.argv[1]) if len(sys.argv)>1 else 319
if FEATURE_SET not in (319,320): raise ValueError("Özellik sayısı 319 veya 320 olmalıdır")
MISSING_INDICATORS=os.environ.get("MISSING_INDICATORS")=="1"
OUTPUT_TAG=f"{FEATURE_SET}_missing_indicators" if MISSING_INDICATORS else str(FEATURE_SET)
df=pd.read_csv(ROOT/"YARISMA_TRAIN_CFTR_REDUCED.csv"); y=df.Label.astype(int).reset_index(drop=True); X0=df.drop(columns=["Variant_ID","Label"]).reset_index(drop=True)
GROUPS={"GRUP_AL6_AL251":["AL_6","AL_251"],"GRUP_AL1_AL211":["AL_1","AL_211"]}; REMOVE=(["CAT_6"] if FEATURE_SET==319 else [])+["AL_6","AL_251","AL_1","AL_211"]
GRID=[("depth3_leaf3",3,3,"sqrt"),("depth5_leaf3",5,3,"sqrt"),("depth7_leaf3",7,3,"sqrt"),("depth5_leaf5",5,5,"sqrt"),("depth5_mf03",5,3,.3)]
def gp(frame):
 r={}
 for new,members in GROUPS.items():
  r[new]={}
  for c in members:
   s=pd.to_numeric(frame[c],errors="coerce"); med=float(s.median()); scale=float(s.quantile(.75)-s.quantile(.25))
   if not np.isfinite(scale) or scale==0: scale=float(s.std(ddof=0))
   if not np.isfinite(scale) or scale==0: scale=1.
   r[new][c]=(med,scale)
 return r
def transform(frame,p):
 z=frame.drop(columns=REMOVE).copy()
 for new,members in GROUPS.items(): z[new]=pd.concat([(pd.to_numeric(frame[c],errors="coerce")-p[new][c][0])/p[new][c][1] for c in members],axis=1).mean(axis=1,skipna=True)
 return z
def fit_prob(tr,yt,te,cfg,seed):
 if MISSING_INDICATORS:
  tr=tr.copy(); te=te.copy(); seen=set()
  for col in list(tr.columns):
   mask=tr[col].isna(); n=int(mask.sum())
   if n<5 or len(tr)-n<5: continue
   signature=tuple(mask.to_numpy(dtype=np.uint8))
   if signature in seen: continue
   seen.add(signature); name=f"MISS_{col}"; tr[name]=mask.astype(np.uint8); te[name]=te[col].isna().astype(np.uint8)
 cats=[c for c in tr.columns if c.startswith("CAT_") or c.startswith("AA_")]; nums=[c for c in tr.columns if c not in cats]
 prep=ColumnTransformer([("num",SimpleImputer(strategy="median",keep_empty_features=True),nums),("cat",Pipeline([("imp",SimpleImputer(strategy="constant",fill_value="__MISSING__",keep_empty_features=True)),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=True))]),cats)])
 _,depth,leaf,mf=cfg; model=RandomForestClassifier(n_estimators=150,max_depth=depth,min_samples_leaf=leaf,max_features=mf,class_weight="balanced_subsample",random_state=seed,n_jobs=-1); pipe=Pipeline([("prep",prep),("model",model)]); pipe.fit(tr,yt); return pipe.predict_proba(te)[:,1]
def met(a,p):
 a=np.asarray(a,int); p=np.asarray(p,int); tn=int(np.sum((a==0)&(p==0))); fp=int(np.sum((a==0)&(p==1))); fn=int(np.sum((a==1)&(p==0))); tp=int(np.sum((a==1)&(p==1))); sp=tn/(tn+fp) if tn+fp else 0; re=tp/(tp+fn) if tp+fn else 0; p1=tp/(tp+fp) if tp+fp else 0; p0=tn/(tn+fn) if tn+fn else 0; f1=2*p1*re/(p1+re) if p1+re else 0; f10=2*p0*sp/(p0+sp) if p0+sp else 0; pfp=100*(1-sp); ptp=20*re; pp=ptp/(ptp+pfp) if ptp+pfp else 0; pf1=2*pp*re/(pp+re) if pp+re else 0; den=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5; mcc=(tp*tn-fp*fn)/den if den else 0
 return {"tn":tn,"fp":fp,"fn":fn,"tp":tp,"accuracy":(tn+tp)/len(a),"specificity":sp,"recall":re,"macro_f1":(f1+f10)/2,"mcc":mcc,"projected_f1":pf1,"fp_per_100_benign":pfp}
def threshold(a,s):
 rows=[]
 for t in np.round(np.arange(.05,.951,.01),2):
  m=met(a,(s>=t).astype(int)); rows.append((m["projected_f1"],-m["fp_per_100_benign"],m["mcc"],m["recall"],t))
 return max(rows)[-1]
preds=[]; folds=[]; inner_rows=[]; choices=[]
for rep in range(5):
 outer=StratifiedKFold(5,shuffle=True,random_state=73000+rep)
 for fold,(tri,tei) in enumerate(outer.split(X0,y),1):
  xo=X0.iloc[tri].reset_index(drop=True); yo=y.iloc[tri].reset_index(drop=True); probs={g[0]:np.zeros(len(xo)) for g in GRID}; inner=StratifiedKFold(4,shuffle=True,random_state=83000+rep*10+fold)
  for inf,(fi,vi) in enumerate(inner.split(xo,yo),1):
   p=gp(xo.iloc[fi]); xf=transform(xo.iloc[fi],p); xv=transform(xo.iloc[vi],p)
   for gi,g in enumerate(GRID): probs[g[0]][vi]=fit_prob(xf,yo.iloc[fi],xv,g,3000000+rep*10000+fold*100+inf*10+gi)
  ranked=[]
  for g in GRID:
   pr=probs[g[0]]; th=threshold(yo,pr); m=met(yo,(pr>=th).astype(int)); ranked.append((m["projected_f1"],-m["fp_per_100_benign"],m["mcc"],m["recall"],g[0],th))
  *_,name,th=max(ranked); cfg=next(g for g in GRID if g[0]==name); p=gp(xo); score=fit_prob(transform(xo,p),yo,transform(X0.iloc[tei],p),cfg,3200000+rep*100+fold); actual=y.iloc[tei].to_numpy(); pred=(score>=th).astype(int); m=met(actual,pred)
  choices.append({"repeat":rep+1,"fold":fold,"config":name,"threshold":th}); folds.append({"repeat":rep+1,"fold":fold,"config":name,"threshold":th,**m}); preds.extend({"repeat":rep+1,"fold":fold,"row":int(row),"actual":int(a),"probability":float(s),"threshold":th,"predicted":int(q),"config":name} for row,a,s,q in zip(tei,actual,score,pred)); inner_rows.extend({"repeat":rep+1,"fold":fold,"row":int(tri[i]),"actual":int(yo.iloc[i]),"probability":float(probs[name][i]),"config":name} for i in range(len(tri)))
pred=pd.DataFrame(preds); f=pd.DataFrame(folds); c=pd.DataFrame(choices); summary=met(pred.actual,pred.predicted); counts=c.config.value_counts(); summary.update({"threshold_mean":c.threshold.mean(),"threshold_sd":c.threshold.std(ddof=1),**{f"selected_{k}":int(v) for k,v in counts.items()}})
pd.DataFrame([summary]).to_csv(OUT/f"randomforest_{OUTPUT_TAG}_nested_score.csv",index=False); f.to_csv(OUT/f"randomforest_{OUTPUT_TAG}_nested_folds.csv",index=False); pred.to_csv(OUT/f"randomforest_{OUTPUT_TAG}_nested_predictions.csv",index=False); pd.DataFrame(inner_rows).to_csv(OUT/f"randomforest_{OUTPUT_TAG}_nested_inner_predictions.csv",index=False); c.to_csv(OUT/f"randomforest_{OUTPUT_TAG}_nested_choices.csv",index=False)
print(pd.DataFrame([summary]).to_string(index=False)); print("\nConfigs:\n",counts.to_string())
