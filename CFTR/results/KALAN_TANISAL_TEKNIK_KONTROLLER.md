# Kalan Tanısal Teknik Kontroller

Bu kontroller modeli, özellikleri veya eşiği değiştirmek için kullanılmamıştır.

## 1. Aynı/yakın satır denetimi

- Tam aynı özellik satırı grubu: **0**
- Benzerliği ≥0.98 olan çift: **0**
- Bunların içinde farklı etiketli çift: **0**
- En yakın çift benzerliği: **0.6735** (`VAR_001901`–`VAR_001912`)

Yakınlık; robust ölçeklenmiş sayısal farklar, kategorik uyuşmazlıklar ve eksiklik farklılıklarının birlikte ortalamasıdır. Bu tanısal eşik otomatik satır silme gerekçesi değildir.

## 2. Yalnız eksiklik maskesi modelleri

|Model|Değişken maske|ROC-AUC|MCC|Proj. F1|Recall|Specificity|FP/100|
|---|---:|---:|---:|---:|---:|---:|---:|
|LogisticRegression|306|0.8592|0.5805|0.5711|0.8756|0.7619|23.81|
|RandomForest|306|0.8166|0.5509|0.5157|0.8933|0.6857|31.43|

Yalnız eksiklik deseninin güçlü sonuç vermesi, biyolojik sinyal olabileceği gibi veri hazırlama kaynaklı sızıntı şüphesi de oluşturur; sonuç bu açıdan yorumlanmalıdır.

## 3. Etiket permütasyonu

- Gözlenen MCC: **0.5225**; permütasyon ortalaması: **-0.0000**; p: **0.000010**
- Gözlenen projekte F1: **0.6798**; permütasyon ortalaması: **0.2627**; p: **0.000010**

Bu test, dondurulmuş dış-fold tahminleriyle etiket ilişkisinin rastlantısal olup olmadığını sınar. Bütün eğitim hattının her permütasyonda yeniden eğitildiği daha ağır test değildir; bu sınırlılık açıkça korunmuştur.
