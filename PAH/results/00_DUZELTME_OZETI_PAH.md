# PAH Paneli — Aşama D Düzeltme Turu Özeti

Aşama E'ye (modelleme) geçmeden önce yapılan 3 bloklayıcı + 3 küçük düzeltme/dokümantasyon görevinin sonuçları. Split bankası (`data/splits/pah/`) bu turda değiştirilmedi. Hiçbir sınıflandırıcı eğitilmedi (Görev 1'deki betimsel confound kontrolü hariç, o da model kurmadı).

| # | Görev | Sonuç |
|---|---|---|
| 1 | `al_all_missing` × `CAT_1`/`CAT_2` confound kontrolü | **Bulgu var — provenance-confound riski VAR.** `al_all_missing=1` olan 89 satırın **%100'ü** hem `CAT_1` hem `CAT_2` açısından da eksik (Cramér's V=0.755 / 0.451, p<10⁻¹³). Özellik havuzundan çıkarılmadı; **Aşama F adversarial validation'a öncelikli işaretli özellik olarak bağlandı.** → `reports/03b_MISSINGNESS_PROVENANCE_KONTROLU_PAH.md` |
| 2 | v3'teki ek ~100 kolonun dokümantasyonu | **Çözüldü.** Tam fark 105 kolon: 57 one-hot açılım (satır-bazlı, sızıntısız) + 48 korelasyon-kümesi grup-özeti (satır-bazlı hesaplama sızıntısız, ama küme **üyeliği** tüm veri setinden seçildiği için hafif yapısal sızıntı riski var — işaretlendi). Ayrıca 352 ortak-isimli kolondan `CAT_1`/`CAT_2`'nin v2→v3'te dtype/anlam değiştirdiği (category→frekans) not edildi. → `reports/02_ON_ISLEME_KARARLARI_PAH.md` §6 |
| 3 | `CAT_1` `&` anomalisi kök nedeni | **Kök neden: gerçek veri, EDA hatası değil.** Ham veride 5 satır gerçekten çok-değerli (`gnomADe_*`'nin 9 alt-popülasyonunun tamamının birleşimi). Figür/rapor doğru, yeniden üretilmedi. `FrequencyEncoder`'ın bunu tek atomik kategori olarak ele aldığı (semantik bilgi kaybı, %1.35 satırı etkiliyor) tespit edildi ve gerekli `MultiHotEncoder` düzeltmesi **belgelendi, uygulanmadı** (kapsam dışı). → `reports/01_EDA_RAPORU_PAH.md`, `reports/02_ON_ISLEME_KARARLARI_PAH.md` §7 |
| 4 | GBDT/permutation bağımsızlık notu | **Çözüldü.** İkisi aynı fold-içi LightGBM'den türediği için bağımsız değil, açıkça not edildi. Final 27 özellikten 22'si (%81) bağımsız bir üçüncü yöntemle (MI/elastic-net) de teyitli; 5'i (`AL_318,AL_7,AL_331,AL_12,AL_22`) yalnızca bağımlı ikiliyle stabil — daha temkinli değerlendirilmeli. → `reports/03_OZELLIK_SECIMI_PAH.md` |
| 5 | v1 kolon aritmetiği teyidi | **Çözüldü/doğrulandı.** `assert 'CAT_6' not in v1.columns` ve `assert 'group_id' in v1.columns` ikisi de geçti; 353=353 örtüşmesinin CAT_6 düşürme + group_id ekleme işlemlerinin birbirini götürmesinden kaynaklandığı teyit edildi. → `reports/02_ON_ISLEME_KARARLARI_PAH.md` §8 |
| 6 | Test sayısı tutarlılığı | **Çözüldü/açıklandı.** `pytest --collect-only -q tests/` → 27 test (`test_preprocessing_pah.py`: 24, `test_split_bank_pah.py`: 3). "21 test" (Aşama B) ve "27 test" (durdurma özeti) arasındaki fark, zaman içinde kümülatif test ekleme sürecinden kaynaklanıyor — sayım hatası değil. → `reports/02_ON_ISLEME_KARARLARI_PAH.md` §9 |

## Genel Değerlendirme

- **Kritik/engelleyici bir hata bulunmadı.** Görev 1'in confound bulgusu ciddi (Cramér's V=0.755) ama görev talimatına uygun şekilde ele alındı: özellik çıkarılmadı, ileriye dönük test noktası (Aşama F adversarial validation) olarak açıkça bağlandı — bu, Aşama E'nin başlamasını engelleyen bir durum değil, Aşama F'de mutlaka kontrol edilmesi gereken bir öncelik.
- Görev 2 ve 3, var olan çıktıların (v3, EDA figürü) **yanlış olmadığını**, yalnızca yeterince belgelenmemiş olduğunu doğruladı; ikisi de artık belgelendi.
- Görev 3'te tespit edilen `CAT_1` multi-hot ihtiyacı küçük ölçekli (%1.35 satır) bir iyileştirme fırsatı — bloklayıcı değil, kod değişikliği bu turda kapsam dışı bırakıldı.

## Aşama E'ye Geçiş Önerisi: **EVET**

Aşama E'ye (modelleme) geçilebilir. Tek koşul: Aşama F'ye ulaşıldığında `al_all_missing` (ve `CAT_1`/`CAT_2` eksiklik göstergeleri) adversarial validation'da **öncelikli test edilecek** — bu rapor o kontrolün gerekçesini ve referans sayılarını (Cramér's V, çapraz tablolar) zaten sağlıyor.

---

## Neden Alternatifi Seçmedim

**Neden Görev 1'in confound bulgusu üzerine `al_all_missing` özelliği doğrudan çıkarılmadı?**
Cramér's V=0.755 ciddi bir kaynak-provenance riski gösteriyor, ama bu **kesin bir sızıntı kanıtı** değil — özelliğin kendisi hâlâ gerçek bir biyolojik sinyal taşıyor olabilir (bkz. CLAUDE.md: "popülasyon veritabanında hiç gözlenmeme, potansiyel patojenite sinyali"). Özelliği körü körüne atmak, doğru olabilecek bir sinyali de kaybetme riski taşırdı. Bunun yerine daha temkinli bir orta yol seçildi: özellik kalır, ama Aşama F adversarial validation'da öncelikli test noktası olarak işaretlenir — kanıt netleşince (sızıntı ise çıkar, sinyal ise kalır) kesin karar orada verilecek.

**Neden `CAT_1`'in multi-hot düzeltmesi bu turda uygulanmadı?**
Yalnızca 5/369 satırı (%1.35) etkiliyor — mevcut model performansına etkisi muhtemelen küçük. Kapsam bu turda "kök nedeni bul ve belgele"ydi, kod değişikliği değil; küçük ölçekli bir düzeltmeyi onay almadan pipeline'a sokmak yerine, gerekçesiyle birlikte belgelenip onay sonrasına bırakıldı.
