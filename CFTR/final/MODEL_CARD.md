# CFTR Yarışma Tahmin Paketi

Bu paket, yeni özellik eklemeden hazırlanmış **319 özellikli nihai veri şemasını** kullanır.

## Modeller

- Ana model: ID3 + CatBoost + Random Forest eşit ağırlıklı soft voting; eşik `0.7224`.
- Yedek model: yalnız ID3 hard voting; eşik `0.780952` (`16.4/21`).
- Her bileşenin beş farklı tohumla eğitilmiş modeli ortalanır.

Ana modelin 5x5 nested-CV sonucu: projekte F1 `0.6798`, MCC `0.5225`, özgüllük `0.9143`, duyarlılık `0.7356`, 100 benign başına FP `8.57`.

Beş tekrarın cross-fitted ortalama aday sonucu: projekte F1 `0.7580`, MCC `0.5717`, 100 benign başına FP `4.76`. Bu değer üretim modelinin dış test garantisi değil, kararlılık deneyidir.

## Kurulum

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
```

## Tahmin

Ana model:

```powershell
.\.venv\Scripts\python.exe .\predict.py --input .\test.csv --output .\tahmin.csv --mode main --audit .\audit.csv --drift-report .\drift.json
```

Yedek model:

```powershell
.\.venv\Scripts\python.exe .\predict.py --input .\test.csv --output .\tahmin_yedek.csv --mode backup --audit .\audit_yedek.csv --drift-report .\drift_yedek.json
```

Girdi `Variant_ID` sütununu ve manifestte listelenen ham özellikleri içermelidir. `Label` bulunması gerekmez. Çıktı tam olarak `Variant_ID,Label` biçimindedir. Satır sırası korunur; fazla sütunlar yok sayılır ve drift raporunda bildirilir. Eksik zorunlu sütun varsa program durur.

## Yarışma günü kontrolü

1. Test dosyasını değiştirmeden yedekleyin.
2. Önce ana modelle tahmin üretin.
3. Konsolda satır sayısını, `drift.json` içinde eksik/tamamen boş sütunları kontrol edin.
4. `tahmin.csv` satır sayısının testle aynı, Variant_ID sırasının aynı ve Label değerlerinin yalnız 0/1 olduğunu doğrulayın.
5. Eşiği veya modeli test sonucuna bakarak değiştirmeyin. Teknik hata olursa yedek ID3 çıktısını kullanın.
