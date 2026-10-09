# 06 — Threshold Belirleme Notu

Model skoru doğrudan sınıf değildir. `skor ≥ threshold` patojenik, altı benign kabul edilir. Threshold düşürülürse recall artabilir ama yanlış pozitifler; yükseltilirse specificity artabilir ama kaçırılan pozitifler çoğalabilir.

## Karşılaştırılacak yöntemler

- `0,50`: değişmeyen referans.
- Youden J: `sensitivity + specificity − 1` en yüksek eşik.
- F1-optimal: OOF F1'i en yüksek yapan eşik.
- PR/ROC eğrisi: dengesiz sınıflarda precision–recall ve hata dengesinin görsel kontrolü.

## Sızıntısız akış

```text
outer-train
→ inner-CV OOF olasılıkları
→ kalibrasyon/threshold seçimi
→ outer-train'de yeniden fit
→ kilitli threshold ile outer-validation
```

Outer-validation, test veya final etiketi threshold'u değiştirmek için kullanılmaz. Final pipeline seçilince tüm eğitim verisinde cross-fitted OOF olasılıkları üretilir, threshold bir kez kaydedilir ve etiketsiz final verisine aynı değer uygulanır.

Yalnızca en yüksek tek F1 noktası yerine yakın-F1 veren kararlı aralık; MCC, sensitivity, specificity ve benign F1 ile birlikte incelenir. Pozitif sınıf tanımı, CV manifesti, model/preprocessing sürümü, kalibrasyon ve seed threshold kaydına eklenir.
