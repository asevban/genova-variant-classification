# Deney 2 — Özellik Ailesi Uzmanları ve Late Fusion

## Amaç ve yöntem

319 özellik silinmeden üç teknik aileye ayrıldı:

- `AL`: 305 AL/birleşik-AL özelliği
- `CAT+AA`: 5 kategorik özellik
- `OTHER`: 9 EK özelliği

Her aile için ayrı bir CatBoost uzmanı eğitildi. Uzman olasılıkları önceden belirlenmiş sabit ağırlıklarla birleştirildi. Ayrıca family-fusion çıktısı mevcut ana modeldeki CatBoost'un yerine konularak `ID3 + Family Fusion + Random Forest` yapısı sınandı.

- Dış değerlendirme: 5-fold × 5 tekrar
- İç doğrulama: 4-fold
- Ön işleme, model/ağırlık ve eşik seçimi yalnız dış eğitim fold'unda yapıldı.
- Ana karşılaştırma hedefi: projeksiyon F1 yükselirken FP'nin düşük kalması.

## Referans ana model

| Model | Macro-F1 | MCC | Proj. F1 | Recall | Specificity | FP/100 |
|---|---:|---:|---:|---:|---:|---:|
| ID3 + CatBoost + Random Forest | **0.7190** | 0.5225 | **0.6798** | 0.7356 | **0.9143** | **8.57** |

## Öne çıkan late-fusion sonuçları

| Yapı | Macro-F1 | MCC | Proj. F1 | Recall | Specificity | FP/100 |
|---|---:|---:|---:|---:|---:|---:|
| Family fusion — fold-içi seçilen ağırlık | 0.6943 | 0.4902 | 0.6591 | 0.7022 | **0.9143** | **8.57** |
| ID3 + seçilen family fusion + RF | 0.7102 | 0.5036 | 0.6485 | 0.7311 | 0.8952 | 10.48 |
| ID3 + %40 AL/%40 CAT-AA/%20 Other + RF | 0.7129 | 0.5108 | 0.6624 | 0.7311 | 0.9048 | 9.52 |
| ID3 + %50 AL/%25 CAT-AA/%25 Other + RF | **0.7276** | **0.5238** | 0.6507 | **0.7578** | 0.8857 | 11.43 |

%50/%25/%25 birleşimi Macro-F1, MCC ve recall'u küçük miktarda artırdı; ancak FP/100 benign 8.57'den 11.43'e çıktı ve benign-ağırlıklı projeksiyon F1 0.6798'den 0.6507'ye düştü. En yüksek projeksiyon F1 veren late-fusion alternatifi 0.6624 ile yine baseline 0.6798'in altında kaldı.

## Uzmanların tek başına sonucu

| Uzman | MCC | Proj. F1 | FP/100 |
|---|---:|---:|---:|
| AL | 0.3955 | 0.5204 | 19.05 |
| OTHER/EK | 0.3775 | 0.5091 | 19.05 |
| CAT+AA | 0.1506 | 0.3292 | 40.00 |

Özellikle yalnız 5 kolon içeren CAT+AA ailesi tek başına yeterli sinyal üretmemiştir. Aileleri eşit düzeyde temsil etmek, çok daha geniş AL ailesindeki ortak ilişkileri parçalamıştır.

## Karar

Late fusion bazı geleneksel metriklerde küçük artış üretse de yarışmanın benign-ağırlıklı hedefinde FP'yi artırmış ve projeksiyon F1'i düşürmüştür. Bu nedenle yöntem **ana model için reddedilmiştir**. Güncel 319 özellikli `ID3 + CatBoost + Random Forest` eşit soft-voting modeli korunur.

## Kanıt dosyaları

- `feature_family_late_fusion_scores.csv`
- `feature_family_late_fusion_folds.csv`
- `feature_family_late_fusion_predictions.csv`
- `feature_family_late_fusion_choices.csv`
- `feature_family_all_recipe_scores.csv`
- `feature_family_all_recipe_folds.csv`
- `feature_family_expert_predictions.csv`
- `feature_family_expert_inner_predictions.csv`

