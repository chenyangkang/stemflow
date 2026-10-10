"""Composable OOF calibration estimators, usable inside or outside AdaSTEM."""

import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from .cross_validation import InsufficientSplitError
from .maps import AffineCalibrator, BetaCalibrator, _weights


class ConstantBinaryClassifier(ClassifierMixin, BaseEstimator):
    """Smoothed fold fallback, distinct from AdaSTEM's intentional dummy zeros."""

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y).reshape(-1)
        w = _weights(sample_weight, len(y))
        self.probability_ = float((w @ y + .5) / (w.sum() + 1))
        self.classes_ = np.array([0, 1])
        self.feature_importances_ = np.zeros(X.shape[1])
        return self

    def predict_proba(self, X):
        return np.tile([1 - self.probability_, self.probability_], (len(X), 1))

    def predict(self, X):
        return np.full(len(X), int(self.probability_ >= .5))


def resolve_cv(cv, classification):
    if cv is None or isinstance(cv, int):
        count = 3 if cv is None else cv
        return (StratifiedKFold(count, shuffle=True, random_state=42) if classification
                else KFold(count, shuffle=True, random_state=42))
    return copy.deepcopy(cv)


def validated_folds(cv, X, y):
    folds = list(cv.split(X, y)) if hasattr(cv, "split") else list(cv)
    visits = np.zeros(len(y), dtype=int)
    output = []
    for a, b in folds:
        a, b = np.asarray(a, int), np.asarray(b, int)
        if not len(a) or not len(b) or np.any(a < 0) or np.any(b < 0) or np.any(a >= len(y)) or np.any(b >= len(y)):
            raise ValueError("CV folds must contain nonempty valid positional indices.")
        if len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b) or np.intersect1d(a, b).size:
            raise ValueError("Training and calibration indices must be unique and disjoint.")
        visits[b] += 1
        output.append((a, b))
    if not np.all(visits == 1):
        raise ValueError("Each row must receive exactly one out-of-fold prediction.")
    return output


class _CrossValidatedCalibration:
    @property
    def required_metadata_columns(self):
        return tuple(dict.fromkeys([*getattr(self.cv, "metadata_columns", ()),
                                   *getattr(self.estimator, "required_metadata_columns", ())]))

    def fit(self, X, y, sample_weight=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = np.asarray(y).reshape(-1)
        if not len(y) or len(X) != len(y) or not np.isfinite(y).all():
            raise ValueError("Expected matching nonempty finite targets.")
        if self._classification and not np.isin(y, [0, 1]).all():
            raise ValueError("Probability calibration currently supports binary 0/1 targets.")
        weights = None if sample_weight is None else _weights(sample_weight, len(y))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_, self.n_train_, self.events_ = X.shape[1], len(y), int(np.sum(y == 1))
        excluded = (getattr(self.cv, "feature_exclusions", ()) if self.exclude_features is None
                    else self.exclude_features)
        metadata = getattr(self.estimator, "required_metadata_columns", ())
        self.features_ = [c for c in X if c not in excluded or c in metadata]
        if not self.features_:
            raise ValueError("No estimator features remain after excluding metadata.")
        self.cv_ = resolve_cv(self.cv, self._classification)
        self.fold_models_, self.fold_support_, self.calibrator_ = [], [], None
        if self._classification:
            self.classes_ = np.array([0, 1])
        try:
            folds = validated_folds(self.cv_, X, y)
        except InsufficientSplitError:
            if not self._classification or getattr(self.cv_, "rare_policy", "allow") == "raise":
                raise
            self.fold_models_ = [ConstantBinaryClassifier().fit(X[self.features_], y, weights)]
            self.status_ = "no_nonempty_st_split_constant"
            return self
        self.status_ = ("rare_fold_fallback" if not getattr(self.cv_, "support_sufficient_", True)
                        else "calibrated")
        oof = np.empty(len(y), dtype=float)
        for tr, va in folds:
            est = clone(self.estimator)
            if self._classification and len(np.unique(y[tr])) == 1:
                est = ConstantBinaryClassifier()
            fit_kw = {} if weights is None else {"sample_weight": weights[tr]}
            est.fit(X.iloc[tr][self.features_], y[tr], **fit_kw)
            oof[va] = self._raw_prediction(est, X.iloc[va][self.features_])
            self.fold_models_.append(est)
            self.fold_support_.append(dict(n_train=len(tr), n_validation=len(va),
                training_events=int(np.sum(y[tr] == 1)), validation_events=int(np.sum(y[va] == 1)),
                estimator_status=getattr(est, "status_", "fitted")))
        if not np.isfinite(oof).all():
            raise ValueError("Base estimator produced nonfinite out-of-fold predictions.")
        default = BetaCalibrator() if self._classification else AffineCalibrator()
        self.calibrator_ = clone(default if self.calibrator is None else self.calibrator)
        fit_kw = {} if weights is None else {"sample_weight": weights}
        self.calibrator_.fit(oof, y, **fit_kw)
        # Keep fitted models and audit counts, not training rows or OOF scores.
        return self

    def _raw_prediction(self, estimator, X):
        return (np.asarray(estimator.predict_proba(X))[:, 1] if self._classification
                else np.asarray(estimator.predict(X)).reshape(-1))

    def _predict_mean(self, X):
        check_is_fitted(self, "fold_models_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        mean = np.zeros(len(X))
        xm = X[self.features_]
        for est in self.fold_models_:
            p = self._raw_prediction(est, xm)
            mean += p if self.calibrator_ is None else self.calibrator_.predict(p)
        return mean / len(self.fold_models_)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "fold_models_")
        values = sum(np.asarray(est.feature_importances_) for est in self.fold_models_) / len(self.fold_models_)
        lookup = dict(zip(self.features_, values))
        return np.asarray([lookup.get(c, 0.) for c in self.feature_names_in_])


class CrossValidatedCalibratedClassifier(ClassifierMixin, _CrossValidatedCalibration, BaseEstimator):
    """Fit retained fold models and one map on pooled OOF probabilities.

    Each fold probability is calibrated before averaging, matching the supplied
    local-beta strategy. Inside AdaSTEM this is a separate map per stixel.
    Calibrators follow ``fit(probability, y)`` / ``predict(probability)``.
    Use an explicit SpatioTemporalKFold for within-stixel blocked calibration.
    A fixed estimator requires no tuning; wrap it in SelfTuningClassifier only
    when nested hyperparameter selection is wanted.
    """
    _classification = True

    def __init__(self, estimator, calibrator=None, cv=None, exclude_features=None):
        self.estimator, self.calibrator, self.cv = estimator, calibrator, cv
        self.exclude_features = exclude_features

    def predict_proba(self, X):
        p = self._predict_mean(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self._predict_mean(X) >= .5).astype(int)


class CrossValidatedCalibratedRegressor(RegressorMixin, _CrossValidatedCalibration, BaseEstimator):
    """OOF calibration of point means; supplies no prediction-interval guarantee."""
    _classification = False

    def __init__(self, estimator, calibrator=None, cv=None, exclude_features=None):
        self.estimator, self.calibrator, self.cv = estimator, calibrator, cv
        self.exclude_features = exclude_features

    def predict(self, X):
        return self._predict_mean(X)
