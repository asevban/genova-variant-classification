"""Asama E3: kalibrator sarmalayicilari. Ucu de ayni fit(scores, y) /
transform(scores) arayuzune sahip -- yalnizca 1D olasilik skorlarini ve
etiketleri gorur, hicbir zaman ozellik matrisine dokunmaz. Cagiran taraf
(`e3_calibration_run.py`) bu skorlari HER ZAMAN capraz-fit edilmis
(taban modelin kendi egitim verisiyle uretilmemis) olasiliklarla besler --
bu modul kendisi sizinti-guvenligini garanti etmez, yalnizca cagiran
tarafin sagladigi skorlari fit/transform eder.
"""
import numpy as np
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

try:
    from betacal import BetaCalibration
    BETACAL_AVAILABLE = True
except ImportError:
    BETACAL_AVAILABLE = False


class PlattCalibrator:
    """sklearn'in kanonik Platt/sigmoid kalibrasyonu -- `CalibratedClassifierCV
    (method="sigmoid")`'in ic. kullandigi ayni `_SigmoidCalibration` sinifi."""

    def fit(self, scores, y):
        self.model_ = _SigmoidCalibration()
        self.model_.fit(np.asarray(scores, dtype=float), np.asarray(y))
        return self

    def transform(self, scores):
        return self.model_.predict(np.asarray(scores, dtype=float))


class BetaCalibrator:
    def fit(self, scores, y):
        if not BETACAL_AVAILABLE:
            raise RuntimeError("betacal kurulu degil")
        self.model_ = BetaCalibration(parameters="abm")
        self.model_.fit(np.asarray(scores, dtype=float), np.asarray(y))
        return self

    def transform(self, scores):
        return np.asarray(self.model_.predict(np.asarray(scores, dtype=float)))


class IsotonicCalibrator:
    def fit(self, scores, y):
        self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model_.fit(np.asarray(scores, dtype=float), np.asarray(y))
        return self

    def transform(self, scores):
        return self.model_.predict(np.asarray(scores, dtype=float))


CALIBRATORS = {"platt": PlattCalibrator, "beta": BetaCalibrator, "isotonic": IsotonicCalibrator}


def evaluate_probabilities(y_true, proba, eps=1e-6):
    """Brier + log-loss; log_loss'un 0/1 uc degerlerde patlamamasi icin
    olasiliklar guvenli araliga kirpilir (yalnizca metrik hesaplamasi
    icin -- kaydedilen olasiliklar kirpilmiyor)."""
    proba_safe = np.clip(np.asarray(proba, dtype=float), eps, 1 - eps)
    return {
        "brier": brier_score_loss(y_true, proba_safe),
        "logloss": log_loss(y_true, proba_safe, labels=[0, 1]),
    }
