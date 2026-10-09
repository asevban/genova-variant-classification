# Veri kullanılabilirliği

CFTR, KANSER, MASTER ve PAH panellerinin test CSV dosyaları TÜSEP tarafından sağlanmıştır ve bu public repoda paylaşılmaz. Repo hiçbir panelin ham test tablosunu içermez.

Public kopyada kaynak kod, yöntem açıklamaları, analiz metinleri ve grafikler bulunabilir. Test verisi üzerinden üretilmiş panel sonuç/tahmin JSON'ları ve birleşik teslim JSON'u da paylaşılmaz.

Paneli yeniden çalıştırmak için yetkili veri sahibi kendi test CSV'sini ilgili panelin `input/` klasörüne koymalıdır. Bu dosyalar `.gitignore` ile engellenir. Public yayın öncesinde `python scripts/check_public_release.py` kontrolü çalıştırılmalıdır.
