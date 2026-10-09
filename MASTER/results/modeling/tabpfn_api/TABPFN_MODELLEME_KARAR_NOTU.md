# TabPFN Modelleme Karar Notu

Bu not, MASTER paneli icin TabPFN/Prior Labs API ile yaptigimiz ek modelleme denemesini ve karar gerekcesini ozetler. TabPFN calismasi ana Hybrid-E kararini degistirmek icin degil, yeni bir tabular foundation model sinyalinin mevcut benign-heavy final hedefimize katki verip vermedigini test etmek icin yapildi.

## 1. Neden TabPFN Denendi?

MASTER paneli icin mevcut en guclu governance adayi **Hybrid-E_bestMCC_nested** idi. Buna ragmen TabPFN denemesi su nedenle anlamliydi:

- TabPFN, tabular veri icin gelistirilmis farkli bir model ailesidir.
- Mevcut RF, XGBoost, CatBoost ve LightGBM tabanli ensemble'lardan farkli bir skor siralamasi uretebilir.
- Eger TabPFN benign false positive yukunu artirmadan sensitivity/F1 katkisi verseydi, Hybrid-E icine dusuk agirlikli bir ek bilesen olarak degerlendirilebilirdi.
- Amac final skorunu yapay olarak artirmak degil, yeni aday sinyalin governance'a deger olup olmadigini kontrollu bicimde anlamakti.

## 2. Kurulum ve Guvenlik

Proje sanal ortaminda TabPFN API istemcisi kuruldu:

```bash
.venv/bin/pip install --upgrade tabpfn-client
```

`requirements.txt` dosyasina su bagimlilik eklendi:

```text
tabpfn-client>=0.5,<0.6
```

API key guvenligi icin asagidaki kararlar uygulandi:

- API key chat'e yazilmadi.
- API key rapor, CSV veya JSON ciktilarina yazilmadi.
- `.env` dosyasindan otomatik okuma varsayilan olarak kapatildi.
- Gercek API kosusu terminalde sessiz giris yapan `scripts/run_tabpfn_api_secure.sh` ile baslatildi.
- Calisma metadata'sinda `token_value_logged=false` ve `env_file_loaded=false` olarak kaydedildi.

Kullanilan scriptler:

| Script | Amac |
|---|---|
| `scripts/master_tabpfn_api_experiments.py` | TabPFN API ile fold-safe OOF skor uretimi |
| `scripts/run_tabpfn_api_secure.sh` | API key'i terminalde sessiz alip modeli calistirma |
| `scripts/master_tabpfn_posthoc_analysis.py` | TabPFN threshold ve Hybrid-E blend taramasini lokal OOF skorlarla yapma |

## 3. Deney Protokolu

TabPFN ilk asamada tam 25 fold governance'a sokulmadi. Bunun yerine kota/maliyet ve karar guvenligi nedeniyle once smoke test yapildi.

| Baslik | Deger |
|---|---|
| Model | TabPFN API |
| Veri seti | `M3_missing_aware_compact` |
| Fold kapsami | `seed=42`, 5 fold |
| Fit edilen satir sayisi | Her fold icin yaklasik 1875-1877 |
| Holdout satir sayisi | Her fold icin yaklasik 468-470 |
| Preprocessing | Her fold icinde yeniden fit edildi |
| Final/test etiketi | Kullanilmadi |
| Dış veri | Eklenmedi |
| API upload | Sadece bu 5 fold smoke test icin yapildi |

Bu nedenle sonuclar resmi final governance kaniti degildir. Resmi adaylik icin TabPFN'in inner selection iceren nested governance akisi icinde tekrar sinanmasi gerekir.

## 4. TabPFN Tekil Sonuclari

TabPFN'in ilk kosuda verdigi ham default threshold sonucu yaniltici derecede yuksek F1 uretmektedir. Bunun sebebi modelin cok fazla ornegi pozitif/pathogenic sinifa itmesidir.

| Durum | Threshold | Final F1 | Final MCC | Specificity | Sensitivity | Precision | FP / 3500 | FN / 3500 | Karar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Default threshold | 0.500 | 0.3863 | 0.3217 | 0.5032 | 0.9529 | 0.2422 | 1490.4 | 23.6 | FP cok yuksek |
| Best MCC | 0.815 | 0.5280 | 0.4456 | 0.8450 | 0.6923 | 0.4268 | 464.9 | 153.9 | MCC iyi, FP siniri ustu |
| FP <= 400 zorlamali | 0.844 | 0.5251 | 0.4393 | 0.8674 | 0.6393 | 0.4456 | 397.8 | 180.3 | FP uygun, F1 strict altinda |
| Strict gate | - | - | - | - | - | - | - | - | Gecen threshold yok |

Yorum:

- TabPFN tek basina guclu sensitivity uretti.
- Ancak benign-heavy final senaryoda false positive yuku yuksek kaldi.
- FP `<= 400` sinirina zorlaninca F1 `0.530` strict esiginin altina dustu.
- Bu nedenle TabPFN tekil model olarak final kilit adayi yapilmadi.

## 5. Seed-42 Uzerinde Mevcut Adaylarla Karsilastirma

TabPFN sadece `seed=42` uzerinde calistirildigi icin karsilastirma ayni 5 fold ile sinirli tutuldu.

| Aday | Fold | F1 | MCC | Specificity | Sensitivity | FP / 3500 | Final weighted AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabPFN API best MCC | 5 | 0.5280 | 0.4456 | 0.8450 | 0.6923 | 464.9 | 0.4741 |
| TabPFN API FP <= 400 | 5 | 0.5251 | 0.4393 | 0.8674 | 0.6393 | 397.8 | 0.4741 |
| Hybrid-E_bestMCC_nested | 5 | 0.5241 | 0.4385 | 0.8562 | 0.6567 | 431.4 | 0.5014 |
| Hybrid-E_AUPRC_guard_nested | 5 | 0.5232 | 0.4378 | 0.8546 | 0.6591 | 436.2 | 0.5098 |
| Hybrid-E_lowFP_nested | 5 | 0.5243 | 0.4388 | 0.8642 | 0.6433 | 407.4 | 0.5027 |
| CATOPT-A_nested_reference | 5 | 0.5270 | 0.4427 | 0.8674 | 0.6370 | 397.8 | 0.5083 |

Yorum:

- TabPFN best-MCC noktasi F1/MCC olarak guclu gorundu, fakat FP `464.9` ile yuksek kaldi.
- FP siniri altina cekilmis TabPFN, CATOPT-A ile FP/specificity olarak benzer seviyeye geldi; ancak final weighted AUPRC belirgin dusuk kaldi.
- Hybrid-E ailesi final-prior weighted AUPRC tarafinda daha guvenli gorundu.

## 6. Hybrid-E + TabPFN Blend Taramasi

TabPFN skorlarinin Hybrid-E icine kucuk agirlikla eklenmesi de denendi. Bu tarama yeni API cagrisi yapmadan, kaydedilmis OOF skorlar uzerinde yapildi.

Taranan TabPFN agirliklari:

```text
0.000, 0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.250, 0.300
```

FP `<= 400` ve sensitivity `>= 0.62` altinda en iyi post-hoc karisim:

| Base strateji | TabPFN agirligi | Threshold | Final F1 | Final MCC | Specificity | Sensitivity | FP / 3500 | Final weighted AUPRC | Statu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hybrid-E_AUPRC_guard_nested | 0.300 | 0.836 | 0.5208 | 0.4337 | 0.8706 | 0.6254 | 388.2 | 0.4883 | fp_forced |

Blend yorumu:

- Strict kosulu gecen Hybrid-E + TabPFN karisimi bulunmadi.
- En iyi constrained karisim FP'yi dusurdu, fakat F1 ve MCC'yi ana Hybrid-E adayini destekleyecek kadar iyilestirmedi.
- Final weighted AUPRC de Hybrid-E/CATOPT referanslarinin altinda kaldi.
- Bu nedenle TabPFN blend'i resmi final modeline eklenmedi.

## 7. Neden Final Model Olarak Secilmedi?

TabPFN'in final model olarak secilmemesinin ana nedenleri:

| Neden | Aciklama |
|---|---|
| Strict threshold bulunmadi | F1 koruma, FP siniri ve sensitivity kosullari ayni anda saglanamadi |
| FP yuku yuksek | Best-MCC threshold'da FP/3500 `464.9` oldu |
| FP zorlamasi performansi dusurdu | FP `397.8` seviyesine inince F1 `0.5251` ve MCC `0.4393` oldu |
| Final weighted AUPRC zayif kaldi | TabPFN `0.4741`, Hybrid-E/CATOPT seed-42 referanslari yaklasik `0.50+` |
| Kapsam smoke test ile sinirli | Sadece seed=42, 5 fold kosuldu; 25 fold nested governance degil |
| Blend post-hoc idi | Ağırlık/threshold outer OOF uzerinde bakildi; resmi secim icin inner-CV gerekir |
| Ana aday daha guvenli | Hybrid-E_bestMCC_nested 25 outer fold governance ile daha guclu kanita sahip |

## 8. Son Karar

TabPFN kurulumu ve modelleme hattı basariyla tamamlandi. Model MASTER panelinde sinyal tasiyor; ozellikle sensitivity tarafinda guclu bir egilim var. Ancak benign-heavy final hedefinde false positive yuku ve final weighted AUPRC dengesi yeterince iyi degil.

Bu nedenle mevcut kanitla:

| Rol | Karar |
|---|---|
| Tekil final model | Secilmedi |
| Hybrid-E icinde kucuk agirlikli bilesen | Secilmedi |
| Raporlanabilir ek deneme | Evet |
| Gelecekte nested governance'a alinabilecek aday | Evet |

Final icin ana karar degismemistir: **Hybrid-E_bestMCC_nested** ana governance adayi olarak korunmalidir. TabPFN ise bu asamada "calisti, sinyal verdi, fakat final icin yeterli governance ve metric dengesi saglamadi" seklinde raporlanmalidir.

## 9. Uretilen Ciktilar

TabPFN API ve post-hoc analiz ciktisi:

- `results/modeling/tabpfn_api/tabpfn_api_fold_plan.csv`
- `results/modeling/tabpfn_api/tabpfn_api_fold_results.csv`
- `results/modeling/tabpfn_api/tabpfn_api_oof_predictions.csv`
- `results/modeling/tabpfn_api/tabpfn_api_threshold_table.csv`
- `results/modeling/tabpfn_api/tabpfn_api_summary.csv`
- `results/modeling/tabpfn_api/tabpfn_api_fp_forced_threshold_summary.csv`
- `results/modeling/tabpfn_api/tabpfn_seed42_governance_reference.csv`
- `results/modeling/tabpfn_api/tabpfn_hybrid_e_blend_screening.csv`
- `results/modeling/tabpfn_api/TABPFN_API_SCREENING_REPORT.md`
- `results/modeling/tabpfn_api/tabpfn_api_metadata.json`
- `results/modeling/tabpfn_api/tabpfn_posthoc_metadata.json`

Bu karar notu:

- `results/modeling/tabpfn_api/TABPFN_MODELLEME_KARAR_NOTU.md`

## 10. Resmi Rapor Cümlesi

TabPFN API ile yapılan seed=42, 5-fold smoke test modelin MASTER panelinde ek sinyal taşıdığını göstermiştir. Ancak benign-heavy final varsayımında TabPFN tek başına strict koşulları sağlayamamış, FP `<= 400` zorlandığında F1/MCC kaybı oluşmuş ve Hybrid-E ile post-hoc küçük ağırlıklı karışımlar da resmi final adayını güçlendirecek düzeyde iyileşme üretmemiştir. Bu nedenle TabPFN final model olarak seçilmemiş; yalnızca raporlanabilir ek aday ve ileride nested governance içine alınabilecek araştırma yönü olarak bırakılmıştır.
