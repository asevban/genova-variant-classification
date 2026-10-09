# Veri Kartı — CFTR Yarışma Eğitim Verisi

## Kullanım amacı

CFTR varyantlarının benign/patojenik sınıflandırılması için yarışma kapsamında model geliştirme ve doğrulama.

## Kaynak ve kapsam

- Kaynak: yarışma komitesinin sağladığı `YARISMA_TRAIN_CFTR.csv`.
- Nihai eğitim temsili: `YARISMA_TRAIN_CFTR_SCENARIO1D_319.csv`.
- Satır: 111 varyant.
- Benign (`0`): 21.
- Patojenik (`1`): 90.
- Kimlik: `Variant_ID`; model özelliği olarak kullanılmaz.
- Nihai model özelliği: 319.

## Özellik aileleri

|Aile|Sayı|
|---|---:|
|AL|303|
|EK|9|
|CAT|3|
|AA|2|
|Birleşik GRUP|2|

Sütunların biyolojik anlamları yarışma veri sözlüğü/şartnamesi kapsamında değerlendirilmelidir; gizli sütunlara doğrulanmamış anlam yüklenmemiştir.

## Ön işleme geçmişi

1. 351 özellik Information Gain, sabitlik, eksiklik ve model doğrulama bulgularıyla 322 özelliğe indirildi.
2. `AL_6–AL_251` ve `AL_1–AL_211`, eğitim tarafında öğrenilen medyan/IQR robust standardizasyon sonrasında ikişer birleşik sütuna dönüştürüldü: `322 → 320`.
3. Düşük ve kararsız katkılı `CAT_6` çıkarıldı: `320 → 319`.
4. Sayısal eksikler model içinde eğitim medyanıyla; kategorik eksikler `__MISSING__` ile ele alınır.

## Eksiklik yapısı

- Değişen eksiklik desenine sahip 306 özellik vardır.
- Ortalama eksik özellik sayısı benign sınıfta `87.43`, patojenik sınıfta `94.78`.
- Medyan eksik özellik sayısı benign sınıfta `16`, patojenik sınıfta `42.5`.
- Bazı AL özelliklerinde eksiklik oranı benign ve patojenik sınıflar arasında belirgin farklıdır.

Eksiklik biyolojik veya hesaplama kapsamı kaynaklı gerçek sinyal olabilir; ancak veri hazırlama sürecine bağlı yapay sinyal olasılığı dışlanamaz.

## Kalite kontrolleri

- Tam aynı özellik satırı grubu: 0.
- Benzerliği ≥0.98 olan çift: 0.
- En yakın çiftin hesaplanan benzerliği: 0.6735.
- Satır veya etiket, yalnız model skorunu iyileştirmek amacıyla silinmedi/değiştirilmedi.

## Bölme ve değerlendirme

Temel değerlendirme stratified 5-fold × 5 tekrar nested-CV’dir. Aynı varyanta ait tekrar tahminleri bootstrap aşamasında `Variant_ID` kümesi olarak birlikte tutulmuştur. Kullanıcının son talebi doğrultusunda yeni GroupKFold/Variant_ID deneyi yapılmamıştır.

## Bilinen sınırlılıklar

- Örnek sayısı ve özellikle benign sınıfı küçüktür.
- Eğitim sınıf dağılımı beklenen final dağılımından farklıdır.
- Eksiklik yapısı sınıfla ilişkilidir ve dış veride değişebilir.
- İlk özellik azaltma kararı tümüyle nested olarak tekrarlanmamıştır.
- Harici klinik veri veya harici gerçek etiket kullanılmamıştır.

## Uygun kullanım

Yalnız yarışma kapsamında sağlanan aynı şemadaki varyant profillerine tahmin üretme ve araştırma amaçlı karar desteği.

## Uygun olmayan kullanım

Doğrudan klinik tanı, tedavi kararı, hasta risk değerlendirmesi veya yarışma dışı popülasyonlarda yeniden doğrulama yapılmadan kullanım.
