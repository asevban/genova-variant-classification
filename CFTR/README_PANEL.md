# GENOVA CFTR PANEL

Bu klasor CFTR panelinin bagimsiz final calistirma paketidir.

## Final gunu

1. CFTR test CSV dosyasini `input/CFTR.csv` olarak koyun.
2. Bu klasorde terminal acin.
3. Calistirin:

```powershell
.\RUN_PANEL.ps1
```

Basarili sonuc:

- `STATUS: PASS`
- `GENOVA CFTR PANEL JSON READY`

Resmi panel JSON ciktisi:

```text
output/TEAM_885171_CFTR_FINAL.json
```

`final/` resmi frozen model ve inference katmanidir. `experiments/`, `results/`, `jury/` ve `archive/` juri kaniti icin durur; panel runner bu klasorlerden model secmez.
