# GENOVA MASTER PANEL

Bu klasor MASTER panelinin bagimsiz final calistirma paketidir.

## Final gunu

1. MASTER test CSV dosyasini `input/MASTER.csv` olarak koyun.
2. Bu klasorde terminal acin.
3. Calistirin:

```powershell
.\RUN_PANEL.ps1
```

Basarili sonuc:

- `STATUS: PASS`
- `GENOVA MASTER PANEL JSON READY`

Resmi panel JSON ciktisi:

```text
output/TEAM_885171_MASTER_FINAL.json
```

`final/` resmi frozen model ve inference katmanidir. `experiments/`, `results/`, `jury/` ve `archive/` juri kaniti icin durur; panel runner bu klasorlerden model secmez.
