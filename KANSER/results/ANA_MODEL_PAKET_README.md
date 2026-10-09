# GENOVA KANSER — Ana Model

Bu paket yarışma finalinde kullanılacak ana modeli içerir.

- Model: `%75 CatBoost + %25 Balanced Random Forest` soft-voting
- Eşik: `0,486465580331`
- Nested-CV hedef F1: `0,621995 ± 0,028428`
- Nested-CV hedef MCC: `0,556257 ± 0,034524`
- Benign özgüllüğü: `%91,92`
- Patojenik duyarlılığı: `%66,87`

Tahmin üretmek için paket kökünde:

```powershell
python predict_model.py --model KANSER_FINAL_BENIGN_SHIFT_MODEL.joblib --input final_test.csv --output final_predictions.csv
```

Skorlar 3.000 benign / 500 patojenik final dağılımına yansıtılmış nested-CV tahminleridir; gerçek final skoru değildir.
