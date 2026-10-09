from pathlib import Path
import argparse
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument("--test",required=True); p.add_argument("--submission",required=True); a=p.parse_args()
test=pd.read_csv(a.test); sub=pd.read_csv(a.submission)
errors=[]
if list(sub.columns)!=["Variant_ID","Label"]: errors.append("Çıktı sütunları tam olarak Variant_ID,Label değil.")
if len(test)!=len(sub): errors.append(f"Satır sayısı farklı: test={len(test)}, çıktı={len(sub)}")
if "Variant_ID" in test and "Variant_ID" in sub and not test["Variant_ID"].astype(str).equals(sub["Variant_ID"].astype(str)): errors.append("Variant_ID sırası test dosyasıyla aynı değil.")
if "Label" in sub and not set(sub["Label"].dropna().unique()).issubset({0,1}): errors.append("Label yalnızca 0/1 içermiyor.")
if sub.isna().any().any(): errors.append("Çıktıda eksik değer var.")
if errors: raise SystemExit("DOĞRULAMA BAŞARISIZ:\n- "+"\n- ".join(errors))
print(f"DOĞRULAMA BAŞARILI: {len(sub)} satır, sıra ve çıktı biçimi doğru.")
