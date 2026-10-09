# GENOVA JSON Merger Input

Finalde panel JSON dosyalarını bu klasöre kopyalayın.

Beklenen dosyalar:

- `TEAM_885171_CFTR_FINAL.json`
- `TEAM_885171_KANSER_FINAL.json`
- `TEAM_885171_MASTER_FINAL.json`
- `TEAM_885171_PAH_FINAL.json`

Notlar:

- Dört panelin tamamı yoksa varsayılan merge işlemi FAIL verir.
- Duplicate `Variant_ID` varsa merge işlemi durur.
- Takım bilgileri panel JSON dosyaları arasında uyuşmazsa merge işlemi durur.
- Bu klasöre model dosyası veya CSV koymak gerekmez; sadece panel JSON çıktıları konur.
