# GENOVA PAH PANEL

Bu klasor PAH panelinin bagimsiz final calistirma paketidir.

## Final gunu

1. PAH test CSV dosyasini `input/PAH.csv` olarak koyun.
2. Bu klasorde terminal acin.
3. Calistirin:

```powershell
.\RUN_PANEL.ps1
```

Basarili sonuc:

- `STATUS: PASS`
- `GENOVA PAH PANEL JSON READY`

Resmi panel JSON ciktisi:

```text
output/TEAM_885171_PAH_FINAL.json
```

`final/` resmi frozen model ve inference katmanidir. `experiments/`, `results/`, `jury/` ve `archive/` juri kaniti icin durur; panel runner bu klasorlerden model secmez.
