# GENOVA JSON Merger

Bu klasör dört bağımsız panel JSON çıktısını tek final JSON dosyasında birleştirmek için hazırlanmıştır.

Model tarafında birleştirme yapmaz. CFTR, KANSER, MASTER ve PAH panelleri kendi klasörlerinde bağımsız çalışmaya devam eder. Bu araç yalnızca panel çıktılarını teslim için tek dosyada toplar.

## Final Günü Akış

1. Her panel kendi klasöründe çalıştırılır.
2. Her panelin `output` klasöründen JSON çıktısı alınır.
3. Dört JSON dosyası bu klasördeki `input` klasörüne kopyalanır.
4. Bu klasörde `RUN_MERGE.ps1` çalıştırılır.
5. Birleşik dosya `output\GENOVA_FINAL_SUBMISSION.json` olarak oluşur.

Beklenen panel JSON dosyaları:

- `TEAM_885171_CFTR_FINAL.json`
- `TEAM_885171_KANSER_FINAL.json`
- `TEAM_885171_MASTER_FINAL.json`
- `TEAM_885171_PAH_FINAL.json`

## Tek Komut

PowerShell:

```powershell
.\RUN_MERGE.ps1
```

BAT yedeği:

```bat
RUN_MERGE.bat
```

Doğrudan Python:

```powershell
python merge_panel_jsons.py --input-dir input --output output\GENOVA_FINAL_SUBMISSION.json --format internal
```

## Kontroller

Merge işlemi şu durumlarda FAIL verir:

- Dört panelden biri eksikse.
- `Variant_ID` / `id` eksikse.
- Duplicate `Variant_ID` varsa.
- `predicted_class` / `Label` değeri `0` veya `1` değilse.
- `predicted_prob` varsa 0-1 aralığı dışında ise.
- Panel JSON dosyalarındaki takım bilgileri birbiriyle uyuşmuyorsa.

## Çıktı Formatları

### internal

Varsayılan ve önerilen formattır. İzlenebilirliği korur:

```json
{
  "team_name": "GENOVA",
  "team_id": "885171",
  "application_id": "4807248",
  "competition_level": "UNIVERSITE_VE_UZERI",
  "submission_format": "GENOVA_INTERNAL_MERGED_JSON_V1",
  "predictions": [
    {
      "id": "VARIANT_ID",
      "panel": "PAH",
      "predicted_class": "1",
      "predicted_prob": 0.8123
    }
  ]
}
```

### official

Resmî teslim şeması sadece `Variant_ID` ve `Label` isterse kullanılabilecek minimal adaydır:

```powershell
python merge_panel_jsons.py --input-dir input --output output\GENOVA_FINAL_SUBMISSION_OFFICIAL.json --format official
```

Önemli: Resmî şema final günü farklı ilan edilirse sadece bu merger katmanı güncellenir; panel modellerine dokunulmaz.

