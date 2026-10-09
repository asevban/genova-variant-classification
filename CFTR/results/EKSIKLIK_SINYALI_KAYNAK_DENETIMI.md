# Eksiklik Sinyalinin Kaynağı

Yalnız eksiklik maskesi modelinin güçlü çıkmasının hangi sütunlardan kaynaklandığını tanısal olarak inceler. Bu analiz model veya veri setini değiştirmez.

## Sınıfa göre toplam eksik özellik sayısı

|class|n|missing_count_mean|missing_count_median|missing_count_min|missing_count_max|
|---|---|---|---|---|---|
|benign|21|87.4286|16.0000|0|265|
|pathogenic|90|94.7778|42.5000|0|306|

## Eksiklik durumu en fazla bilgi taşıyan 20 özellik

|feature|missing_benign|missing_pathogenic|absolute_rate_gap|missingness_information_gain|
|---|---|---|---|---|
|AL_2|0.5714|0.9333|0.3619|0.0985|
|AL_3|0.5714|0.9333|0.3619|0.0985|
|AL_4|0.5714|0.9333|0.3619|0.0985|
|AL_5|0.5714|0.9333|0.3619|0.0985|
|AL_27|0.5714|0.9333|0.3619|0.0985|
|AL_28|0.5714|0.9333|0.3619|0.0985|
|AL_29|0.5714|0.9333|0.3619|0.0985|
|AL_30|0.5714|0.9333|0.3619|0.0985|
|AL_31|0.5714|0.9333|0.3619|0.0985|
|AL_32|0.5714|0.9333|0.3619|0.0985|
|AL_33|0.5714|0.9333|0.3619|0.0985|
|AL_34|0.5714|0.9333|0.3619|0.0985|
|AL_35|0.5714|0.9333|0.3619|0.0985|
|AL_36|0.5714|0.9333|0.3619|0.0985|
|AL_37|0.5714|0.9333|0.3619|0.0985|
|AL_38|0.5714|0.9333|0.3619|0.0985|
|EK_3|0.1905|0.5000|0.3095|0.0464|
|AL_295|0.0000|0.1667|0.1667|0.0443|
|AL_296|0.0000|0.1667|0.1667|0.0443|
|AL_297|0.0000|0.1667|0.1667|0.0443|

## Yorum

Eksiklik sinyali tek başına etiket sızıntısını kanıtlamaz. Ölçüm/annotation kapsamı varyant türüne göre biyolojik olarak farklı olabilir. Ancak final verisinin farklı bir hazırlama hattından gelmesi halinde bu sinyal genellenmeyebilir; gerçek testte drift raporundaki tamamen boş sütunlar ve eksiklik oranları özellikle kontrol edilmelidir.