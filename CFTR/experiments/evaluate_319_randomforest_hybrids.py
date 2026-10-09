from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent/"results"; KEYS=["repeat","fold","row"]
RECIPES={"baseline_ID3_CAT_ADA":("ID3","CAT","ADA",[1,1,1]),"ID3_CAT_RF_equal":("ID3","CAT","RF",[1,1,1]),"ID3_CAT_RF_442":("ID3","CAT","RF",[.4,.4,.2]),"ID3_CAT_ADA_RF_4321":("ID3","CAT","ADA","RF",[.4,.3,.2,.1]),"ID3_CAT_ADA_RF_equal":("ID3","CAT","ADA","RF",[1,1,1,1]),"ID3_RF_equal":("ID3","RF",[1,1])}
def load(inner):
 id3=pd.read_csv(OUT/("id3_benign_weighted_322_vs_320_inner_predictions.csv" if inner else "id3_benign_weighted_322_vs_320_predictions.csv")); id3=id3[id3.key=="id3_scenario1d_319"]
 cat=pd.read_csv(OUT/("catboost_benign_weighted_finalists_inner_predictions.csv" if inner else "catboost_benign_weighted_finalists_predictions.csv")); cat=cat[cat.key=="scenario1_robust_groups_no_cat6"]
 sk=pd.read_csv(OUT/("sklearn_scenario1_319_nested_inner_predictions.csv" if inner else "sklearn_scenario1_319_nested_predictions.csv")); ada=sk[sk.model=="AdaBoost"]
 rf=pd.read_csv(OUT/("randomforest_319_nested_inner_predictions.csv" if inner else "randomforest_319_nested_predictions.csv"))
 r=None
 for name,z in [("ID3",id3),("CAT",cat),("ADA",ada),("RF",rf)]:
  q=z[KEYS+["actual","probability"]].copy().rename(columns={"actual":f"actual_{name}","probability":name}); r=q if r is None else r.merge(q,on=KEYS)
 r["actual"]=r.actual_ID3.astype(int); return r
def met(a,p):
 a=np.asarray(a,int); p=np.asarray(p,int); tn=int(np.sum((a==0)&(p==0))); fp=int(np.sum((a==0)&(p==1))); fn=int(np.sum((a==1)&(p==0))); tp=int(np.sum((a==1)&(p==1))); sp=tn/(tn+fp) if tn+fp else 0; re=tp/(tp+fn) if tp+fn else 0; p1=tp/(tp+fp) if tp+fp else 0; p0=tn/(tn+fn) if tn+fn else 0; f1=2*p1*re/(p1+re) if p1+re else 0; f10=2*p0*sp/(p0+sp) if p0+sp else 0; pfp=100*(1-sp); ptp=20*re; pp=ptp/(ptp+pfp) if ptp+pfp else 0; pf1=2*pp*re/(pp+re) if pp+re else 0; den=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5; mcc=(tp*tn-fp*fn)/den if den else 0
 return {"tn":tn,"fp":fp,"fn":fn,"tp":tp,"accuracy":(tn+tp)/len(a),"specificity":sp,"recall":re,"macro_f1":(f1+f10)/2,"mcc":mcc,"projected_f1":pf1,"fp_per_100_benign":pfp}
def threshold(a,s):
 rows=[]
 for t in np.round(np.arange(.05,.951,.01),2):
  m=met(a,(s>=t).astype(int)); rows.append((m["projected_f1"],-m["fp_per_100_benign"],m["mcc"],m["recall"],t))
 return max(rows)[-1]
inn,out=load(True),load(False); rows=[]; folds=[]
for rep in range(1,6):
 for fold in range(1,6):
  i=inn[(inn.repeat==rep)&(inn.fold==fold)]; o=out[(out.repeat==rep)&(out.fold==fold)]
  for name,spec in RECIPES.items():
   cols=list(spec[:-1]); weights=spec[-1]; si=np.average(i[cols],axis=1,weights=weights); so=np.average(o[cols],axis=1,weights=weights); th=threshold(i.actual,si); pred=(so>=th).astype(int); m=met(o.actual,pred); folds.append({"recipe":name,"repeat":rep,"fold":fold,"threshold":th,**m}); rows.extend({"recipe":name,"repeat":rep,"fold":fold,"row":int(row),"actual":int(a),"score":float(s),"threshold":th,"predicted":int(p)} for row,a,s,p in zip(o.row,o.actual,so,pred))
pred=pd.DataFrame(rows); f=pd.DataFrame(folds); summary=[]
for name,z in pred.groupby("recipe"):
 q=f[f.recipe==name]; summary.append({"recipe":name,**met(z.actual,z.predicted),"threshold_mean":q.threshold.mean(),"threshold_sd":q.threshold.std(ddof=1)})
summary=pd.DataFrame(summary).sort_values(["projected_f1","mcc"],ascending=False); summary.to_csv(OUT/"randomforest_319_hybrid_scores.csv",index=False); f.to_csv(OUT/"randomforest_319_hybrid_folds.csv",index=False); pred.to_csv(OUT/"randomforest_319_hybrid_predictions.csv",index=False); print(summary.to_string(index=False))
