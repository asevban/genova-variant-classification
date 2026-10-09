# PAH MODEL_SELECTION

Final model: `final/models/final_model_bundle_v2.pkl`

Secim gerekcesi: PAH icin resmi teslimde CatBoost tabanli v2 bundle korunmustur. Kaynak raporlarda v2 icin F1 0.8177, MCC 0.3921, specificity 0.7869, sensitivity 0.7208 ve AUROC 0.8309 degerleri yer almaktadir.

Denenen aileler/hatlar: CatBoost, XGBoost, LightGBM, RF/ID3 varyantlari, ensemble ve TabPFN odakli deney dosyalari `experiments/` ve `results/` altina ayrildi.

Arsivlenen modeller: 6 eski veya challenger bundle `archive/` altina tasindi. Final inference bunlari kullanmaz.
