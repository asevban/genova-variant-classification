"""P2 madde 19: EK/madde script'lerinin ortak idempotent CSV yazma
yardımcısı. Eski desen (`to_csv(mode="a")`) her çalıştırmada satır
EKLİYORDU -- script iki kez çalıştırılırsa çift satır oluşuyordu. Bu
modül, `key_cols`'a göre AYNI anahtarlı eski satırları çıkarıp yeni
satırlarla birleştirip TEK SEFERDE (append değil, yeniden) yazar --
script kaç kez çalıştırılırsa çalıştırılsın, aynı `key_cols` kombinasyonu
için dosyada tam olarak bir satır kalır.

Yalnızca YAZMA MEKANİZMASI değişiyor -- hiçbir script'in ürettiği
bilimsel sonuç/sayı değişmiyor.
"""
from pathlib import Path

import pandas as pd


def upsert_csv(new_df, path, key_cols):
    """`path`'teki CSV'yi idempotent şekilde günceller: dosya varsa,
    `key_cols`'a göre `new_df` ile çakışan eski satırlar çıkarılır,
    kalanlar `new_df` ile birleştirilip TEK SEFERDE yeniden yazılır.
    Dosya yoksa `new_df` doğrudan yazılır (ilk çalıştırma).

    Kolon uyumsuzluğunda (ör. eski dosyada olup `new_df`'te olmayan bir
    kolon) hata FIRLATMAZ -- `pd.concat`'in doğal birleşimini kullanır,
    eksik hücreler NaN kalır (mevcut satırların içeriği bozulmaz).

    >>> import tempfile, os
    >>> path = Path(tempfile.mktemp(suffix=".csv"))
    >>> df1 = pd.DataFrame({"k": [1, 2], "v": ["a", "b"]})
    >>> _ = upsert_csv(df1, path, ["k"])
    >>> df2 = pd.DataFrame({"k": [2, 3], "v": ["B", "c"]})
    >>> result = upsert_csv(df2, path, ["k"])
    >>> result["v"].tolist()
    ['a', 'B', 'c']
    >>> os.remove(path)
    """
    path = Path(path)
    new_df = new_df.reset_index(drop=True)

    if path.exists():
        existing = pd.read_csv(path)
        key_df = new_df[key_cols].drop_duplicates()
        merge_check = existing.merge(key_df, on=key_cols, how="left", indicator=True)
        existing_keep = existing[merge_check["_merge"].values == "left_only"].reset_index(drop=True)
        combined = pd.concat([existing_keep, new_df], ignore_index=True)
        ordered_cols = list(existing.columns) + [c for c in new_df.columns if c not in existing.columns]
        combined = combined[ordered_cols]
    else:
        combined = new_df

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined
