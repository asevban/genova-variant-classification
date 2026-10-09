# CFTR MODEL_CARD

Panel: CFTR

Final artefaktlar: `final/models/artifacts`

Inference entrypoint: `final/predict.py`

Python: 3.12.10

Output kolonlari: `Variant_ID`, `Label`

Kapsam: 15 base model artefakti ve `preprocessing.json` ile ana ensemble tahmini uretir. Ana mod `--mode main` olarak `run_all.py` tarafindan cagrilir.

Kisit: Final ensemble bilesenleri, esikler ve preprocessing dosyasi yeniden olusturulmaz.
