# KANSER Veri Ön İşleme Özeti

Ham dosya değiştirilmeden saklandı; `Variant_ID` ve `Label` model özelliklerinden ayrıldı. `./.` kategorik eksikliği gerçek eksiklik olarak tanındı ve çıktı CSV'lerinde `__MISSING__` ile standartlaştırıldı. Sayısal eksikler doldurulmadı; böylece model bazlı imputasyon daha sonra, yalnızca training fold'unda yapılabilir.

## Yapılan dört temel işlem

1. **Güvenli yerel sürüm (V1):** 351 özelliğin tamamı korunur. Bu dosya, bilgi kaybının fayda sağlayıp sağlamadığını ölçen kontrol grubudur.
2. **Kompakt sürüm (V2):** Training fold'unda sabit olan ve başka bir sütunla ham değer/eksik konumu birebir aynı olan sütunlar çıkarılır. Tam veri referansında 280 özellik kalır.
3. **Kapsama-dayanıklı sürüm (V3):** V2'ye ek olarak training fold'unda anlamsal eksikliği en az %85 olan sütunlar çıkarılır. 262 özellikli daha muhafazakâr adaydır.
4. **Kaynak-dayanıklılığı araştırması (A1–A4):** `CAT_1/2`, eksiklik özetleri ve aminoasit türetimleri tek değişiklikli kollarla ayrıştırılır. A4'te `CAT_1/2` hem ana özelliklerden hem eksiklik hesabından çıkarılır.

## Neden birden fazla CSV var?

Birden fazla CSV aynı anda tek modele yığılmak için değil, ön işleme kararlarını kontrollü karşılaştırmak içindir. V1–V3 aynı split ve sabit modelle karşılaştırılır; en dengeli F1/MCC, sınıf bazlı performans, düşük train–OOF farkı ve kararlılık gösteren politika seçilir. A1–A4 ise yüksek skorun biyolojik sinyalden mi, eksiklik/kaynak izinden mi geldiğini araştırır.

## Kritik bulgu

Satır eksiklik oranının tek değişkenli AUC değeri yaklaşık `0,768` bulundu. Ortalama eksiklik oranı Label 0'da `%33,0`, Label 1'de `%68,6` düzeyindedir. Bu bir model başarısı değildir; eksiklik örüntüsünün hedefle ilişkili olduğunu ve kaynak kestirmesi riski taşıdığını gösteren tanısal uyarıdır.

## Eğitimde önerilen sıra

- Aşama 1: V1, V2, V3 + sabit basit model + aynı 5×10 split bankası.
- Aşama 2: seçilen politika üzerinde Logistic Regression, Random Forest/ExtraTrees, CatBoost, LightGBM ve XGBoost.
- Aşama 3: A0–A4 dayanıklılık deneyleri, alt grup analizi ve gerekirse adversarial validation.

İlk çalıştırma V2 ile yapılabilir; ancak V1–V3 OOF karşılaştırması tamamlanmadan final ön işleme ilan edilmemelidir.
