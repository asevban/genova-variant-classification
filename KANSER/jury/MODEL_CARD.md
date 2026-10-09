# KANSER MODEL_CARD

Panel: KANSER

Final artefakt: `final/models/KANSER_FINAL_BENIGN_SHIFT_MODEL.joblib`

Inference entrypoint: `final/predict.py`

Python: 3.11.3

Output kolonlari: `Variant_ID`, `prediction`, `pathogenic_score`

Kapsam: Final joblib artefakti icindeki sabit pipeline ve model bilesenleriyle test CSV girdisinden kanser panel tahmini uretir.

Kisit: Final model yeniden egitilmez; secim, agirlik ve esik degerleri korunur.
