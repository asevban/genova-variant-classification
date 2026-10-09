# Nihai Model Seçimi – 319 Ortak Özellik

## Karar

Bütün modeller aynı kolon sayısını kullanacaktır. `CAT_6` çıkarılmış 319 özellikli Senaryo 1D üzerinde model seçimi yeniden yapılmıştır.

Yeni ana model:

```text
ID3 + CatBoost + AdaBoost
Eşit ağırlıklı Soft Voting
```

## Veri hattı

```text
351 ham özellik
− 29 IG/gereksizlik özelliği
= 322 özellik

AL_6 + AL_251 → GRUP_AL6_AL251
AL_1 + AL_211 → GRUP_AL1_AL211
322 − 4 + 2 = 320 özellik

CAT_6 çıkarıldı
320 − 1 = 319 özellik
```

`CAT_6` çıkarma gerekçesi:

- 111 satırın 109'unda boş (`%98.2`)
- Yalnız iki dolu satır
- İki dolu değer de aynı kategori
- Information Gain: `0.00550697`
- Label ile Cramér's V: `0.06543`

## Model seçimi yöntemi

- Bütün modeller aynı 319 özellikli giriş şemasını kullandı.
- Dış değerlendirme: 5-fold stratified CV × 5 tekrar
- İç değerlendirme: 4-fold stratified CV
- İç OOF tahminleriyle eşik seçimi
- Yaklaşık hedef dağılım: 100 benign + 20 patojenik
- 33 ikili, üçlü ve dörtlü ensemble yapısı
- Hard, eşit soft ve ağırlıklı soft voting karşılaştırması

## Kazanan skorlar

|Ölçüt|Skor|
|---|---:|
|Accuracy|0.7477|
|Specificity|0.9143|
|Recall|0.7089|
|Macro-F1|0.6992|
|MCC|0.4964|
|Projeksiyon F1|0.6633|
|FP/100 benign|8.6|
|Ortalama eşik|0.6760 ± 0.0394|

Eşit soft voting olduğu için her model aynı olasılık ağırlığına sahiptir:

```text
ID3      = 1/3
CatBoost = 1/3
AdaBoost = 1/3
```

## Önceki 320 modelle karşılaştırma

|Ölçüt|Eski 320 aday|Yeni 319 aday|Değişim|
|---|---:|---:|---:|
|Accuracy|0.7387|0.7477|+0.0090|
|Specificity|0.8857|0.9143|+0.0286|
|Recall|0.7044|0.7089|+0.0045|
|Macro-F1|0.6879|0.6992|+0.0113|
|MCC|0.4705|0.4964|+0.0259|
|Projeksiyon F1|0.6191|0.6633|+0.0442|
|FP/100 benign|11.4|8.6|−2.8|

Eski 320 aday:

```text
ID3 + CatBoost + Extra Trees
Ağırlıklı Soft Voting
```

Yeni 319 aday bütün raporlanan ölçütlerde daha iyi sonuç vermiştir.

## Önemli metodolojik not

Bu fark yalnız CAT_6'nın çıkarılmasına ait değildir. Model seçimi yeniden açıldığı için:

- Extra Trees yerine AdaBoost seçilmiştir.
- Ağırlıklı soft voting yerine eşit soft voting seçilmiştir.

Dolayısıyla doğru ifade şudur:

> CAT_6 çıkarıldıktan sonra 319 özellik üzerinde model seçimi yeniden yapıldığında, ID3 + CatBoost + AdaBoost eşit soft voting en iyi aday olmuştur.

## Bundan sonraki çalışma kuralı

1. Bütün modeller aynı 319 kolonu kullanacaktır.
2. 319 ve başka özellik sayılarına sahip modeller aynı ensemble içinde karıştırılmayacaktır.
3. Nihai aday ID3 + CatBoost + AdaBoost eşit soft voting olacaktır.
4. Final eşik, bütün eğitim verisinde yalnız eğitim-içi CV ile yeniden seçilecektir.
5. Kesin karar bağımsız test verisinde doğrulanacaktır.

## Kanıt dosyaları

- `results/ensemble_all_319_nested_scores.csv`
- `results/ensemble_all_319_nested_folds.csv`
- `results/ensemble_all_319_nested_predictions.csv`
- `results/sklearn_scenario1_319_nested_scores.csv`
- `results/catboost_benign_weighted_finalists_scores.csv`
- `results/id3_benign_weighted_322_vs_320_scores.csv`
- `results/ENSEMBLE_TUM_MODELLER_319_RAPORU.md`
- `ensemble_all_322_nested_from_oof.py 319`
