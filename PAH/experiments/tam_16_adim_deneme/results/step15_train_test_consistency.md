# Adım 15 — Train-Test Sütun Uyumu (İlke Dokümantasyonu)

**Bu deney kapsamında gerçek bir test seti yok** (yarışma test seti henüz
paylaşılmadı) — bu adım yalnızca **ilke olarak** dokümante ediliyor +
küçük bir doğrulama fonksiyonu (`validate_train_test_column_consistency()`,
bu script içinde tanımlı) ekleniyor.

## Gelecekte Test Seti Geldiğinde Uygulanacak Prosedür

1. **Train'de çıkarılan/dönüştürülen her özellik, test'ten de aynı şekilde
   çıkarılacak/dönüştürülecek.** Hiçbir istatistik (medyan, frekans, ölçek
   parametresi) test verisinden öğrenilmeyecek — yalnızca train'den `fit`
   edilip test'e `transform` uygulanacak. Bu, resmi
   `src/genova/pah/fold_features.py::build_fold_features`'ın **zaten**
   uyguladığı ilkeyle birebir aynı (bu deney o modülü değiştirmeden import
   ederek kullandı, aynı ilkeyi miras aldı).
2. **`Variant_ID` hiçbir zaman modele girdi olarak verilmeyecek** — yalnızca
   kimlik/eşleme amaçlı saklanacak (tahminleri `Variant_ID`'ye göre
   raporlamak için).
3. **İstatistikler yalnızca train'den öğrenilecek** — medyan doldurma,
   frekans kodlama, ölçekleme parametreleri (varsa) dahil.
4. **Kolon sırası/eksik-fazla sütun kontrolü yapılacak** — `X_test`,
   `X_train`'in kolon listesine (sırayla) `reindex` edilecek; train'de
   olmayan bir test-kolonu sessizce atılacak, train'de olan ama test'te
   eksik bir kolon `fill_value=0` ile doldurulacak (resmi
   `fold_features.py`'nin zaten yaptığı gibi — `X_test.reindex(columns=
   feature_cols, fill_value=0)`).

## Kod Doğrulaması (bu turda)

`validate_train_test_column_consistency(X_train, X_test, feature_cols,
id_col_excluded_check)` fonksiyonu yazıldı ve **resmi
`build_fold_features()` çıktısı üzerinde bağımsız olarak test edildi** —
yukarıdaki `PASS` sonucu, resmi pipeline'ın bu ilkeyi zaten doğru
uyguladığının bu deney tarafından bağımsız bir teyididir.
