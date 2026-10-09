# Adım 9 — Kısmi Korelasyon / Koşullu IG Genellemesi (Faz 2'nin Yöntemi, Yeni Çiftler)

Faz 2'nin `partial_spearman`/`conditional_mutual_info` fonksiyonları
**doğrudan import edildi** (yeniden yazılmadı), Adım 6-7'de öne çıkan yeni
çiftlere uygulandı. Küçük ortak-n uyarısı: her satırda `ortak_n` açıkça
raporlanıyor.

    X      Y      Z_kontrol                                                                      aciklama  r_XY_kosulsuz  r_XY_kismi  ortak_n  NaN_nedeniyle_dusurulen_satir  IG_X_vs_Label_kosulsuz Y_kontrol_degiskeni  IG_X_vs_Label_Y_medyan_ile_kosullu
 EK_7   EK_9 al_all_missing  En güçlü EK_-EK_ çifti (Spearman=0.774), ikisi de resmi 25-özellik havuzunda       0.774410    0.772745      361                              8                0.067294                EK_9                            0.066286
AL_88 AL_121 al_all_missing GENOVA'nın resmi EDA'sının bildirdiği en güçlü AL_-AL_ çifti (Spearman=0.777)       0.677082    0.615544      369                              0                0.001331              AL_121                            0.006580

**Not — v3 (sıfır-doldurulmuş) vs ham veri farkı:** `AL_88`↔`AL_121` burada
r=0.677 çıkıyor, Adım 6'nın ham-veri (yalnızca gerçek ortak gözlemler,
common_n=123) hesaplamasındaki 0.777'den **farklı** — çünkü bu script v3'ü
(zero-fill uygulanmış, common_n=369) kullanıyor; sıfır-doldurma iki kolonun
"gerçek gözlemlerdeki" korelasyon yapısını seyreltiyor. Bu, ölçüm hatası
değil, hangi veri versiyonunun sorulduğuna bağlı gerçek bir fark — ikisi de
geçerli, farklı soruları cevaplıyor (v3=model-girdisi-olarak-korelasyon,
ham=gerçek-gözlemler-arası-korelasyon).

**Yalnızca güçlü kısmi korelasyon gösterdiği için özellik silme kararı
verilmedi** — bu sonuçlar yalnızca Adım 13'ün nested karşılaştırmasına
girdi sağlayan gözlemler.
