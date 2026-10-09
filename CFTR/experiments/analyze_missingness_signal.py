from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"results"
df=pd.read_csv(ROOT/"YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv"); y=df.Label.astype(int); X=df.drop(columns=["Variant_ID","Label"])
rows=[]
def entropy(a):
    z=np.bincount(np.asarray(a,int),minlength=2); p=z[z>0]/z.sum(); return float(-(p*np.log2(p)).sum())
base=entropy(y)
for c in X:
    m=X[c].isna();
    if m.nunique()<2: continue
    cond=sum((idx.sum()/len(y))*entropy(y[idx]) for idx in (m,~m) if idx.sum())
    rows.append({"feature":c,"missing_benign":float(m[y==0].mean()),"missing_pathogenic":float(m[y==1].mean()),"absolute_rate_gap":float(abs(m[y==0].mean()-m[y==1].mean())),"missingness_information_gain":base-cond})
rank=pd.DataFrame(rows).sort_values(["missingness_information_gain","absolute_rate_gap"],ascending=False)
rank.to_csv(OUT/"missingness_signal_by_feature.csv",index=False)
counts=X.isna().sum(axis=1)
summary=pd.DataFrame({"class":["benign","pathogenic"],"n":[int((y==0).sum()),int((y==1).sum())],"missing_count_mean":[counts[y==0].mean(),counts[y==1].mean()],"missing_count_median":[counts[y==0].median(),counts[y==1].median()],"missing_count_min":[counts[y==0].min(),counts[y==1].min()],"missing_count_max":[counts[y==0].max(),counts[y==1].max()]})
summary.to_csv(OUT/"missing_count_by_class.csv",index=False)
def md_table(frame):
    h=list(frame.columns); out=["|"+"|".join(h)+"|","|"+"|".join(["---"]*len(h))+"|"]
    for _,r in frame.iterrows(): out.append("|"+"|".join(f"{v:.4f}" if isinstance(v,(float,np.floating)) else str(v) for v in r)+"|")
    return "\n".join(out)
lines=["# Eksiklik Sinyalinin Kaynağı","","Yalnız eksiklik maskesi modelinin güçlü çıkmasının hangi sütunlardan kaynaklandığını tanısal olarak inceler. Bu analiz model veya veri setini değiştirmez.","","## Sınıfa göre toplam eksik özellik sayısı","",md_table(summary),"","## Eksiklik durumu en fazla bilgi taşıyan 20 özellik","",md_table(rank.head(20)),"","## Yorum","","Eksiklik sinyali tek başına etiket sızıntısını kanıtlamaz. Ölçüm/annotation kapsamı varyant türüne göre biyolojik olarak farklı olabilir. Ancak final verisinin farklı bir hazırlama hattından gelmesi halinde bu sinyal genellenmeyebilir; gerçek testte drift raporundaki tamamen boş sütunlar ve eksiklik oranları özellikle kontrol edilmelidir."]
(OUT/"EKSIKLIK_SINYALI_KAYNAK_DENETIMI.md").write_text("\n".join(lines),encoding="utf-8")
print(summary.to_string(index=False)); print(rank.head(10).to_string(index=False))
