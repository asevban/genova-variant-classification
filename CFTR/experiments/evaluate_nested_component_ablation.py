from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"; KEYS=["repeat","fold","row"]
def load(inner=False):
    suffix="_inner_predictions.csv" if inner else "_predictions.csv"
    id3=pd.read_csv(OUT/f"id3_benign_weighted_322_vs_320{suffix}"); id3=id3[id3.key=="id3_scenario1d_319"]
    cat=pd.read_csv(OUT/f"catboost_benign_weighted_finalists{suffix}"); cat=cat[cat.key=="scenario1_robust_groups_no_cat6"]
    rf=pd.read_csv(OUT/f"randomforest_319_nested{suffix}")
    merged=None
    for name,frame in [("ID3",id3),("CAT",cat),("RF",rf)]:
        q=frame[KEYS+["actual","probability"]].rename(columns={"actual":f"actual_{name}","probability":name})
        merged=q if merged is None else merged.merge(q,on=KEYS)
    merged["actual"]=merged.actual_ID3.astype(int); return merged
def met(a,p):
    a=np.asarray(a,int);p=np.asarray(p,int);tn=int(np.sum((a==0)&(p==0)));fp=int(np.sum((a==0)&(p==1)));fn=int(np.sum((a==1)&(p==0)));tp=int(np.sum((a==1)&(p==1)))
    sp=tn/(tn+fp) if tn+fp else 0;re=tp/(tp+fn) if tp+fn else 0;pfp=100*(1-sp);ptp=20*re;pp=ptp/(ptp+pfp) if ptp+pfp else 0;pf1=2*pp*re/(pp+re) if pp+re else 0;den=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,specificity=sp,recall=re,mcc=(tp*tn-fp*fn)/den if den else 0,projected_f1=pf1,fp_per_100_benign=pfp)
def threshold(a,s):
    z=[]
    for t in np.round(np.arange(.05,.951,.01),2):
        m=met(a,s>=t);z.append((m["projected_f1"],-m["fp_per_100_benign"],m["mcc"],m["recall"],t))
    return max(z)[-1]
recipes={"ID3":["ID3"],"CAT":["CAT"],"RF":["RF"],"ID3+CAT":["ID3","CAT"],"ID3+RF":["ID3","RF"],"CAT+RF":["CAT","RF"],"ID3+CAT+RF":["ID3","CAT","RF"]}
inn,out=load(True),load(False);pred=[]
for rep,fold in itertools.product(range(1,6),range(1,6)):
    i=inn[(inn.repeat==rep)&(inn.fold==fold)];o=out[(out.repeat==rep)&(out.fold==fold)]
    for recipe,cols in recipes.items():
        si=i[cols].mean(axis=1).to_numpy();so=o[cols].mean(axis=1).to_numpy();th=threshold(i.actual,si);p=(so>=th).astype(int)
        pred.extend(dict(recipe=recipe,repeat=rep,fold=fold,row=int(row),actual=int(a),score=float(s),threshold=th,predicted=int(y)) for row,a,s,y in zip(o.row,o.actual,so,p))
pred=pd.DataFrame(pred);pred.to_csv(OUT/"nested_component_ablation_predictions.csv",index=False)
rows=[]
for recipe,z in pred.groupby("recipe"):
    pooled=met(z.actual,z.predicted);rows.append({"recipe":recipe,"evaluation":"pooled_5x5",**pooled})
    b=z.groupby("row",as_index=False).agg(actual=("actual","first"),score=("score","mean"),threshold=("threshold","mean"));b["predicted"]=(b.score>=b.threshold).astype(int);m=met(b.actual,b.predicted);rows.append({"recipe":recipe,"evaluation":"cross_fitted_bagged",**m})
scores=pd.DataFrame(rows).sort_values(["evaluation","projected_f1","mcc"],ascending=[True,False,False]);scores.to_csv(OUT/"nested_component_ablation_scores.csv",index=False)
md="# Nested Model Bileşeni Ablation Karşılaştırması\n\nHer tekil, ikili ve üçlü yapı eşit ağırlıkla birleştirilmiş; eşik yalnız ilgili dış fold'un iç-CV tahminlerinden öğrenilmiştir. Modeller yeniden eğitilmemiştir.\n\n|Yapı|Değerlendirme|Proj. F1|MCC|Recall|Specificity|FP/100|TP|FP|FN|TN|\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
for _,r in scores.iterrows():md+=f"|{r.recipe}|{r.evaluation}|{r.projected_f1:.4f}|{r.mcc:.4f}|{r.recall:.4f}|{r.specificity:.4f}|{r.fp_per_100_benign:.2f}|{int(r.tp)}|{int(r.fp)}|{int(r.fn)}|{int(r.tn)}|\n"
md+="\nKarar yalnız en yüksek tek skora göre verilmez. Üçlü yapının ikililere göre F1/FP dengesi ve modeller arası tamamlayıcılığı birlikte değerlendirilir.\n"
(OUT/"NESTED_MODEL_BILESENI_ABLATION_RAPORU.md").write_text(md,encoding="utf-8");print(md)
