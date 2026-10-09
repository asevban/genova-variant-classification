# Güncel Veri Seti ve Seçilen Ensemble Modeli

## 1. Güncel veri seti

Projenin güncel ve ortak model veri seti:

```text
YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv
```

|İçerik|Sayı|
|---|---:|
|Gözlem|111|
|Model özelliği|319|
|Kimlik sütunu|1 (`Variant_ID`)|
|Hedef sütunu|1 (`Label`)|
|Dosyadaki toplam sütun|321|
|Benign (`Label=0`)|21|
|Patojenik (`Label=1`)|90|

`Variant_ID` model girdisi değildir. Bütün temel modeller aynı 319 model özelliğini ve aynı kolon sırasını kullanacaktır.

## 2. Veri setinin oluşturulması

```text
351 ham model özelliği
− 29 IG/gereksizlik özelliği
= 322 özellik
```

Korelasyonlu çiftler robust standardizasyon sonrasında birleştirildi:

```text
AL_6 + AL_251 → GRUP_AL6_AL251
AL_1 + AL_211 → GRUP_AL1_AL211
```

```text
322 − 4 ham kolon + 2 birleşik kolon = 320 özellik
```

Son olarak `CAT_6` çıkarıldı:

```text
320 − CAT_6 = 319 özellik
```

`CAT_6` çıkarma gerekçeleri:

- 111 satırın 109'unda boş (`%98.2`)
- Yalnız iki dolu değer
- İki dolu değer aynı kategori
- Information Gain: `0.00550697`
- Label ile Cramér's V: `0.06543`

## 3. Birleşik özellik formülü

Her grup üyesi şu şekilde robust standardize edilmiştir:

\[
z_{robust}=\frac{x-medyan}{IQR}
\]

Ardından aynı gruptaki mevcut standartlaştırılmış değerlerin ortalaması alınmıştır. CV skorları hesaplanırken medyan ve IQR yalnız ilgili eğitim fold'undan öğrenilmiştir.

Kaydedilen tam eğitim veri setindeki preprocessing parametreleri:

```text
YARISMA_TRAIN_CFTR_SCENARIO1D_319.preprocessing.json
```

Bu JSON dosyasında:

- çıkarılan kolonlar,
- grup üyeleri,
- medyan ve ölçek değerleri,
- nihai 319 kolonun sırası

bulunmaktadır.

## 4. Seçilen ensemble modeli

319 özellik üzerinde 33 ensemble kombinasyonu yeniden değerlendirilmiştir. Seçilen ana model:

```text
ID3 + CatBoost + AdaBoost
Eşit ağırlıklı Soft Voting
```

Her model aynı `YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv` kolon şemasını kullanacaktır.

Soft voting olasılık hesabı:

\[
P_{ensemble}=\frac{P_{ID3}+P_{CatBoost}+P_{AdaBoost}}{3}
\]

Ağırlıklar:

```text
ID3      = 1/3
CatBoost = 1/3
AdaBoost = 1/3
```

## 5. Değerlendirme yöntemi

- Dış değerlendirme: 5-fold stratified CV × 5 tekrar
- İç değerlendirme: 4-fold stratified CV
- Eşik seçimi: yalnız iç OOF tahminleri
- Beklenen test dağılımı: yaklaşık 100 benign + 20 patojenik
- Eşik amacı: projeksiyon F1; eşitlikte daha az FP ve daha yüksek recall
- Dış test fold'u model veya eşik seçimine katılmamıştır.

## 6. Seçilen modelin skorları

|Ölçüt|Skor|
|---|---:|
|Accuracy|0.7477|
|Specificity|0.9143|
|Recall|0.7089|
|Macro-F1|0.6992|
|MCC|0.4964|
|Projeksiyon F1|0.6633|
|FP/100 benign|8.6|
|Ortalama karar eşiği|0.6760 ± 0.0394|

Skorların yorumu:

- 100 benign örneğin yaklaşık `91.4` tanesi doğru benign bulunur.
- Yaklaşık `8.6` benign örnek yanlışlıkla patojenik tahmin edilir.
- Patojenik örneklerin yaklaşık `%70.9`u yakalanır.
- MCC `0.4964`, sınıf dengesizliği altında orta düzey dengeli ayırt etme gücü göstermektedir.

Ortalama `0.676` değeri doğrudan sabit final eşik olarak kullanılmamalıdır. Nihai model bütün eğitim verisiyle kurulurken karar eşiği yalnız eğitim-içi CV ile yeniden seçilmelidir.

## 7. Önceki 320 özellikli adayla karşılaştırma

|Ölçüt|Eski 320 aday|Yeni 319 aday|Değişim|
|---|---:|---:|---:|
|Accuracy|0.7387|0.7477|+0.0090|
|Specificity|0.8857|0.9143|+0.0286|
|Recall|0.7044|0.7089|+0.0045|
|Macro-F1|0.6879|0.6992|+0.0113|
|MCC|0.4705|0.4964|+0.0259|
|Projeksiyon F1|0.6191|0.6633|+0.0442|
|FP/100 benign|11.4|8.6|−2.8|

Eski aday `ID3 + CatBoost + Extra Trees ağırlıklı soft voting`, yeni aday ise `ID3 + CatBoost + AdaBoost eşit soft voting`dir. Bu nedenle fark yalnız CAT_6 silmenin değil, 319 özellik üzerinde modelin yeniden seçilmesinin sonucudur.

## 8. Güncel çalışma kuralı

1. Bütün modeller aynı 319 özelliği kullanacaktır.
2. Farklı özellik sayısındaki modeller aynı ensemble içinde karıştırılmayacaktır.
3. Güncel ana ensemble: ID3 + CatBoost + AdaBoost eşit soft voting.
4. Train ve test kolon isimleri ile sıraları preprocessing JSON'una göre eşleştirilecektir.
5. Test verisinde `CAT_6` kullanılmayacaktır.
6. `AL_6`, `AL_251`, `AL_1` ve `AL_211` doğrudan modele verilmez; bunlardan iki birleşik özellik oluşturulur.
7. Nihai başarı bağımsız testte doğrulanacaktır.

## 9. Kanıt dosyaları

|Amaç|Dosya|
|---|---|
|Güncel 319 eğitim verisi|`YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv`|
|Preprocessing parametreleri|`YARISMA_TRAIN_CFTR_SCENARIO1D_319.preprocessing.json`|
|Veri setini üretme kodu|`build_scenario1d_319_dataset.js`|
|319 ensemble skorları|`results/ensemble_all_319_nested_scores.csv`|
|Fold ve eşik kanıtı|`results/ensemble_all_319_nested_folds.csv`|
|Dış test tahminleri|`results/ensemble_all_319_nested_predictions.csv`|
|Ayrıntılı 319 raporu|`results/ENSEMBLE_TUM_MODELLER_319_RAPORU.md`|
|Nihai seçim raporu|`results/NIHAI_319_OZELLIK_MODEL_SECIM_RAPORU.md`|
