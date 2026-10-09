# KANSER MODEL_SELECTION

Final model: `final/models/KANSER_FINAL_BENIGN_SHIFT_MODEL.joblib`

Secim gerekcesi: Resmi final model, benign-shift odakli 75% CatBoost + 25% Balanced RandomForest soft voting hattidir. Mevcut sonuclar nested target F1 0.621995 +/- 0.028428, MCC 0.556257 +/- 0.034524, specificity 0.9192 ve sensitivity 0.6687 olarak raporlanmistir.

Denenen aileler/hatlar: `RESULTS.json` kaynaklarinda genis aday, transfer ve ensemble taramalari yer alir. Finalist secimden sonra tek joblib final artefakti korunmustur.

Arsivlenen modeller: Pakette final disi serialized model bulunmadigi icin arsivlenen eski model sayisi 0'dur.
