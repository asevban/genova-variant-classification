from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
KEYS = ["repeat", "fold", "row"]

def load(inner=False):
    suffix = "_inner_predictions.csv" if inner else "_predictions.csv"
    id3 = pd.read_csv(OUT / f"id3_benign_weighted_322_vs_320{suffix}")
    id3 = id3[id3.key == "id3_scenario1d_319"]
    cat = pd.read_csv(OUT / f"catboost_benign_weighted_finalists{suffix}")
    cat = cat[cat.key == "scenario1_robust_groups_no_cat6"]
    rf = pd.read_csv(OUT / f"randomforest_319_nested{suffix}")
    merged = None
    for name, frame in [("ID3", id3), ("CAT", cat), ("RF", rf)]:
        q = frame[KEYS + ["actual", "probability"]].copy()
        q = q.rename(columns={"actual": f"actual_{name}", "probability": name})
        merged = q if merged is None else merged.merge(q, on=KEYS)
    merged["actual"] = merged.actual_ID3.astype(int)
    return merged

def metrics(actual, predicted):
    a=np.asarray(actual,int); p=np.asarray(predicted,int)
    tn=int(np.sum((a==0)&(p==0))); fp=int(np.sum((a==0)&(p==1)))
    fn=int(np.sum((a==1)&(p==0))); tp=int(np.sum((a==1)&(p==1)))
    sp=tn/(tn+fp) if tn+fp else 0; re=tp/(tp+fn) if tp+fn else 0
    precision=tp/(tp+fp) if tp+fp else 0
    projected_fp=100*(1-sp); projected_tp=20*re
    projected_precision=projected_tp/(projected_tp+projected_fp) if projected_tp+projected_fp else 0
    projected_f1=2*projected_precision*re/(projected_precision+re) if projected_precision+re else 0
    den=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,specificity=sp,recall=re,
                mcc=(tp*tn-fp*fn)/den if den else 0,projected_f1=projected_f1,
                fp_per_100_benign=projected_fp)

def best_threshold(actual, score):
    candidates=[]
    for threshold in np.round(np.arange(.05,.951,.01),2):
        m=metrics(actual,score>=threshold)
        candidates.append((m["projected_f1"],-m["fp_per_100_benign"],m["mcc"],m["recall"],threshold))
    return max(candidates)[-1]

# Her bileşen en az %10 katkı verir. Böylece bu deney ağırlık kararlılığıdır;
# bileşen çıkarma ayrı ablation testinde değerlendirilir.
weights=[(i/10,j/10,(10-i-j)/10) for i in range(1,9) for j in range(1,10-i) if 10-i-j>=1]
inner,outer=load(True),load(False)
fold_rows=[]; predictions=[]
for repeat,fold in itertools.product(range(1,6),range(1,6)):
    inn=inner[(inner.repeat==repeat)&(inner.fold==fold)]
    out=outer[(outer.repeat==repeat)&(outer.fold==fold)]
    candidates=[]
    for w_id3,w_cat,w_rf in weights:
        score=w_id3*inn.ID3.to_numpy()+w_cat*inn.CAT.to_numpy()+w_rf*inn.RF.to_numpy()
        threshold=best_threshold(inn.actual.to_numpy(),score)
        m=metrics(inn.actual.to_numpy(),score>=threshold)
        # Son bağlayıcı: eşit ağırlığa yakınlık. Tek bir uç ağırlığı tesadüfen seçmeyi azaltır.
        distance=sum((x-1/3)**2 for x in (w_id3,w_cat,w_rf))
        candidates.append((m["projected_f1"],-m["fp_per_100_benign"],m["mcc"],m["recall"],-distance,
                           w_id3,w_cat,w_rf,threshold))
    chosen=max(candidates)
    _,_,_,_,_,w_id3,w_cat,w_rf,threshold=chosen
    score=w_id3*out.ID3.to_numpy()+w_cat*out.CAT.to_numpy()+w_rf*out.RF.to_numpy()
    pred=(score>=threshold).astype(int); m=metrics(out.actual.to_numpy(),pred)
    fold_rows.append(dict(repeat=repeat,fold=fold,w_id3=w_id3,w_cat=w_cat,w_rf=w_rf,threshold=threshold,**m))
    for row,actual,s,p in zip(out.row,out.actual,score,pred):
        predictions.append(dict(repeat=repeat,fold=fold,row=int(row),actual=int(actual),score=float(s),threshold=threshold,predicted=int(p)))

folds=pd.DataFrame(fold_rows); preds=pd.DataFrame(predictions)
folds.to_csv(OUT/"nested_weight_selection_folds.csv",index=False)
preds.to_csv(OUT/"nested_weight_selection_predictions.csv",index=False)
pooled=metrics(preds.actual,preds.predicted)

# Her varyant için beş dış-CV olasılığını ve eşiğini ortala.
bagged=preds.groupby("row",as_index=False).agg(actual=("actual","first"),score=("score","mean"),threshold=("threshold","mean"))
bagged["predicted"]=(bagged.score>=bagged.threshold).astype(int)
bagged.to_csv(OUT/"nested_weight_selection_bagged_predictions.csv",index=False)
bagged_metrics=metrics(bagged.actual,bagged.predicted)

summary=pd.DataFrame([{"evaluation":"pooled_5x5",**pooled},{"evaluation":"cross_fitted_bagged",**bagged_metrics}])
summary.to_csv(OUT/"nested_weight_selection_scores.csv",index=False)
counts=folds.groupby(["w_id3","w_cat","w_rf"]).size().reset_index(name="fold_count").sort_values("fold_count",ascending=False)

md="# Nested Ensemble Ağırlık Kararlılığı\n\n"
md+="Ağırlıklar dış test fold'una bakılmadan, her dış fold'un yalnız iç-CV tahminlerinden seçilmiştir. ID3, CatBoost ve Random Forest'ın her birine en az %10 ağırlık verilmiş; eşit ağırlığa yakınlık yalnız son bağlayıcı olarak kullanılmıştır. Böylece dış-fold skoruna bakarak tek bir sabit ağırlık seçilmemiştir.\n\n"
md+="|Değerlendirme|Proj. F1|MCC|Recall|Specificity|FP/100|TP|FP|FN|TN|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
for _,r in summary.iterrows():
    md+=f"|{r.evaluation}|{r.projected_f1:.4f}|{r.mcc:.4f}|{r.recall:.4f}|{r.specificity:.4f}|{r.fp_per_100_benign:.2f}|{int(r.tp)}|{int(r.fp)}|{int(r.fn)}|{int(r.tn)}|\n"
md+="\n## En sık seçilen ağırlıklar\n\n|ID3|CatBoost|Random Forest|Fold sayısı|\n|---:|---:|---:|---:|\n"
for _,r in counts.head(10).iterrows(): md+=f"|{r.w_id3:.1f}|{r.w_cat:.1f}|{r.w_rf:.1f}|{int(r.fold_count)}|\n"
md+="\nReferans eşit ağırlıklı beş-model cross-fitted sonuç: projeksiyon F1 `0.7580`, MCC `0.5717`, FP/100 benign `4.76`. Nested ağırlık seçimi bu referansı açık ve kararlı biçimde geçmiyorsa eşit ağırlık korunmalıdır.\n"
(OUT/"NESTED_ENSEMBLE_AGIRLIK_KARARLILIGI_RAPORU.md").write_text(md,encoding="utf-8")
print(md)
