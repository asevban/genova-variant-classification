# PAH MODEL_CARD

Panel: PAH

Final artefakt: `final/models/final_model_bundle_v2.pkl`

Inference entrypoint: `final/predict.py`

Python: 3.8.10

Output kolonlari: `Variant_ID`, `predicted_label`, `predicted_probability`, `panel`

Kapsam: Ham panel CSV girdisinden isim bazli sema dogrulama, dondurulmus on isleme, 26 ozellikli final havuz, model, kalibrator, prior correction ve sabit esik ile tahmin uretir.

Kisit: Model yeniden egitilmez; `Label` kolonu tahmin icin gerekli degildir.
