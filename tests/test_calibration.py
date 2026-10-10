"""OOF separation, rare-fold behavior, calibration maps and AdaSTEM integration."""

import pickle

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from stemflow.calibration import (AffineCalibrator, BetaCalibrator, SpatioTemporalKFold,
    CrossValidatedCalibratedClassifier, CrossValidatedCalibratedRegressor,
    calibration_bins, calibration_scores, stixel_calibration_diagnostics,
    plot_calibration_diagnostics)
from stemflow.calibration.cross_validation import InsufficientSplitError
from stemflow.tuning import SelfTuningClassifier, SelfTuningRegressor, local_beta_xgboost
from .test_execution_rewrite import make_model, sample


class BiasedClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.training_ids_ = X.index.to_numpy()
        return self

    def predict_proba(self, X):
        p = expit(X["signal"].to_numpy() - 1.5)
        return np.column_stack([1-p, p])


def test_beta_known_bias_improves_independent_probabilities():
    rng = np.random.default_rng(5)
    signal = rng.normal(size=12000)
    truth = expit(signal)
    y = rng.binomial(1, truth)
    raw = expit(signal - 1.5)
    beta = BetaCalibrator(ridge=10).fit(raw[:6000], y[:6000])
    assert log_loss(y[6000:], beta.predict(raw[6000:])) < log_loss(y[6000:], raw[6000:]) - .1
    assert np.all(np.diff(beta.predict(np.linspace(0, 1, 1000))) >= 0)
    identity = BetaCalibrator(ridge=1e10).fit(raw[:6000], y[:6000])
    assert_allclose(identity.theta_, [1, 1, 0], atol=1e-6)
    assert_allclose(pickle.loads(pickle.dumps(beta)).predict(raw), beta.predict(raw))


def test_oof_models_are_independent_and_no_training_data_retained():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"signal": rng.normal(size=600)})
    y = rng.binomial(1, expit(X.signal))
    est = CrossValidatedCalibratedClassifier(BiasedClassifier(), BetaCalibrator(10),
        KFold(3, shuffle=True, random_state=42)).fit(X, y)
    assert len(est.fold_models_) == 3
    assert sum(v["n_validation"] for v in est.fold_support_) == len(X)
    for fold, (_, va) in zip(est.fold_models_, est.cv_.split(X, y)):
        assert not np.intersect1d(fold.training_ids_, X.index[va]).size
    assert not any(name in est.__dict__ for name in ["X_train_", "y_train_", "oof_"])
    assert clone(est).get_params()["calibrator__ridge"] == 10


def test_spatial_folds_keep_circular_copies_together_and_audit_rare_support():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"longitude": rng.uniform(-100, -80, 200),
                      "latitude": rng.uniform(20, 40, 200), "DOY": rng.uniform(1, 50, 200)})
    copied = X.copy()
    copied.DOY += 366
    X = pd.concat([X, copied], ignore_index=True)
    y = np.zeros(len(X), int)
    y[[0, 200]] = 1
    cv = SpatioTemporalKFold(min_pos=5, max_trials=2)
    folds = list(cv.split(X, y))
    assert not cv.support_sufficient_
    for tr, va in folds:
        assert set(va[va < 200]) == set(va[va >= 200] - 200)
    with pytest.raises(InsufficientSplitError):
        list(SpatioTemporalKFold(min_pos=5, max_trials=2, rare_policy="raise").split(X, y))


def test_no_independent_geometry_uses_audited_constant():
    X = pd.DataFrame({"longitude": np.ones(20), "latitude": np.ones(20), "DOY": np.ones(20)})
    est = CrossValidatedCalibratedClassifier(BiasedClassifier(), cv=SpatioTemporalKFold()).fit(X, np.arange(20) % 2)
    assert est.status_ == "no_nonempty_st_split_constant"
    assert est.calibrator_ is None
    assert_allclose(est.predict_proba(X)[:, 1], .5)


def test_invalid_fold_overlap_is_rejected():
    X = pd.DataFrame({"signal": np.arange(10)})
    with pytest.raises(ValueError, match="disjoint"):
        CrossValidatedCalibratedClassifier(BiasedClassifier(), cv=[(np.arange(10), np.arange(10))]).fit(X, np.arange(10) % 2)


@pytest.mark.parametrize("backend", ["native", "memmap"])
def test_local_calibration_inside_adastem_and_diagnostics(sample, tmp_path, backend):
    X, y = sample
    base = local_beta_xgboost(tuning=False, n_estimators=5, ridge=100)
    model = make_model(n_jobs=2, lazy_loading=True)
    model.base_model = base
    train, test = X.iloc[:400], X.iloc[400:]
    model.fit(train, y.iloc[:400], training_data_backend=backend, temporal_window_prequery=True)
    p = model.predict_proba(test, temporal_window_prequery=True)[:, 1]
    fitted = [v for v in model.model_dict.values() if hasattr(v, "fold_support_")]
    assert fitted
    diagnostics = stixel_calibration_diagnostics(model, test, y.iloc[400:], p)
    assert set(diagnostics["scores"].level) == {"overall", "spatial", "temporal", "spatiotemporal", "stixel"}
    assert "expected_detections" in diagnostics["bins"]
    assert "common" in set(diagnostics["bins"].strategy)
    plot_calibration_diagnostics(diagnostics, tmp_path, max_curves=10)
    assert (tmp_path / "stixel_calibration_overlay.pdf").is_file()
    model.calculate_feature_importances()
    assert len(model.feature_importances_) > 0


def test_regression_point_calibration_and_standalone_tuning():
    X = pd.DataFrame({"signal": np.linspace(-1, 1, 200)})
    y = 2*X.signal.to_numpy() + 3
    affine = AffineCalibrator().fit(X.signal, y)
    assert_allclose(affine.theta_, [2, 3], atol=1e-12)
    tuned = SelfTuningRegressor(LinearRegression(), {"fit_intercept": [True, False]}, n_trials=2)
    calibrated = CrossValidatedCalibratedRegressor(tuned).fit(X, y)
    assert_allclose(calibrated.predict(X), y, atol=1e-10)
    assert np.isfinite(SelfTuningClassifier(RandomForestClassifier(n_estimators=3, random_state=3),
        {"max_depth": [2, 3]}, n_trials=2).fit(X, np.arange(len(X)) % 2).predict_proba(X)).all()


def test_quantile_minima_zero_events_and_expected_detection_counts():
    y = np.array([0, 0, 0, 1])
    p = np.array([0., 0., 0., .4])
    b = calibration_bins(y, p)
    assert b.n.sum() == len(y)
    assert_allclose(b.expected_detections.sum(), p.sum())
    score, _ = calibration_scores(y, p)
    assert score["quantile_zero_event_mass"] == .75
    assert score["coverage"] == 1
    with pytest.raises(ValueError):
        calibration_bins(y, [0., 0., 0., 2.])


def test_optional_optuna_nested_search_is_reproducible():
    pytest.importorskip("optuna")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(dict(signal=rng.normal(size=600), DOY=rng.integers(1, 367, 600),
        longitude=rng.uniform(-100, -70, 600), latitude=rng.uniform(20, 50, 600)))
    y = rng.binomial(1, expit(X.signal.to_numpy()-1))
    template = local_beta_xgboost(n_trials=2, n_estimators=5, tuning_n_estimators=3)
    a, b = clone(template).fit(X, y), clone(template).fit(X, y)
    assert [m.best_params_ for m in a.fold_models_] == [m.best_params_ for m in b.fold_models_]
    assert all(len(m.trials_) == 2 for m in a.fold_models_)
    assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-14)


@pytest.mark.parametrize("tuning", [False, True])
@pytest.mark.parametrize("calibration", [False, True])
def test_four_independent_estimator_choices(tuning, calibration):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(dict(signal=rng.normal(size=600), DOY=rng.integers(1, 367, 600),
        longitude=rng.uniform(-100, -70, 600), latitude=rng.uniform(20, 50, 600)))
    y = rng.binomial(1, expit(X.signal.to_numpy()-1))
    base = RandomForestClassifier(n_estimators=5, max_depth=4, n_jobs=1, random_state=42)
    if tuning:
        base = SelfTuningClassifier(base, {"max_depth": [2, 4]}, n_trials=2,
            cv=SpatioTemporalKFold(n_splits=2, random_state=59))
    if calibration:
        base = CrossValidatedCalibratedClassifier(base, BetaCalibrator(ridge=100),
            cv=SpatioTemporalKFold(n_splits=3, random_state=42))
    fitted = clone(base).fit(X, y)
    p = fitted.predict_proba(X)[:, 1]
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    models = fitted.fold_models_ if calibration else [fitted]
    assert len(models) == (3 if calibration else 1)
    for model in models:
        assert isinstance(model, SelfTuningClassifier) == tuning
        if tuning:
            assert model.status_ == "tuned" and len(model.trials_) == 2
        elif calibration:
            assert model.max_depth == 4
    if calibration:
        assert fitted.calibrator_.ridge == 100
        assert fitted.cv_.support_sufficient_
        assert "longitude" not in fitted.features_ or tuning
