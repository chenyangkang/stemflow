"""Small sequential searches, composable with OOF calibration and AdaSTEM."""

from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import ParameterSampler
from sklearn.utils.validation import check_is_fitted

from .calibration.cross_validation import InsufficientSplitError, SpatioTemporalKFold
from .calibration.estimators import ConstantBinaryClassifier, resolve_cv, validated_folds
from .calibration.maps import _weights


@contextmanager
def _optuna_logging(verbose):
    """Scope Optuna's process-local level, including when a search fails."""
    import optuna
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    try:
        yield optuna
    finally:
        optuna.logging.set_verbosity(previous)


class _SelfTuning:
    @property
    def required_metadata_columns(self):
        return tuple(getattr(self.cv, "metadata_columns", ()))

    def _base_estimator(self):
        if self.estimator is None:
            raise ValueError("Provide an estimator to tune.")
        return clone(self.estimator)

    def fit(self, X, y, sample_weight=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = np.asarray(y).reshape(-1)
        if not len(y) or len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError("Expected matching nonempty finite training targets.")
        if self._classification and not np.isin(y, [0, 1]).all():
            raise ValueError("This tuner supports binary 0/1 classification.")
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive.")
        if self.search not in {"random", "optuna"}:
            raise ValueError("search must be 'random' or 'optuna'.")
        weights = None if sample_weight is None else _weights(sample_weight, len(y))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        excluded = (getattr(self.cv, "feature_exclusions", ()) if self.exclude_features is None
                    else self.exclude_features)
        self.features_ = [c for c in X if c not in excluded]
        if not self.features_:
            raise ValueError("No estimator features remain after excluding metadata.")
        xm = X[self.features_]
        self.cv_ = resolve_cv(self.cv, self._classification)
        self.best_params_, self.trials_ = {}, []
        try:
            folds = validated_folds(self.cv_, X, y)
        except InsufficientSplitError:
            if getattr(self.cv_, "rare_policy", "allow") == "raise":
                raise
            folds = []
        supported = bool(folds) and getattr(self.cv_, "support_sufficient_", True)
        base = self._base_estimator()
        if supported:
            def evaluate(params):
                loss, mass = 0., 0.
                for tr, va in folds:
                    est = clone(base).set_params(**{**params, **(self.search_estimator_params or {})})
                    if self._classification and len(np.unique(y[tr])) == 1:
                        est = ConstantBinaryClassifier()
                    kw = {} if weights is None else {"sample_weight": weights[tr]}
                    est.fit(xm.iloc[tr], y[tr], **kw)
                    pred = est.predict_proba(xm.iloc[va])[:, 1] if self._classification else est.predict(xm.iloc[va])
                    w = None if weights is None else weights[va]
                    amount = len(va) if w is None else w.sum()
                    if amount == 0:
                        continue
                    value = (log_loss(y[va], pred, labels=[0, 1], sample_weight=w)
                             if self._classification else mean_squared_error(y[va], pred, sample_weight=w))
                    loss, mass = loss + amount*value, mass + amount
                if not mass:
                    raise ValueError("Validation weights have zero total mass.")
                score = loss / mass
                self.trials_.append(dict(params=params.copy(), loss=float(score)))
                return score
            space = self.param_distributions or {}
            if self.search == "optuna":
                with _optuna_logging(getattr(self, "verbose", False)) as optuna:
                    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(
                        seed=self.random_state, n_startup_trials=3))
                    default = base.get_params()
                    initial = {k: default[k] for k, values in space.items() if default.get(k) in values}
                    if len(initial) == len(space):
                        study.enqueue_trial(initial)
                    study.optimize(lambda trial: evaluate({k: trial.suggest_categorical(k, list(values))
                        for k, values in space.items()}), n_trials=self.n_trials, n_jobs=1,
                        show_progress_bar=False)
                    self.best_params_, self.best_score_ = study.best_params, float(study.best_value)
            elif self.search == "random":
                candidates = ParameterSampler(space, n_iter=self.n_trials, random_state=self.random_state)
                results = [(evaluate(p), p) for p in candidates]
                self.best_score_, self.best_params_ = min(results, key=lambda item: item[0])
            else:
                raise ValueError("search must be 'random' or 'optuna'.")
            self.status_ = "tuned"
        else:
            self.status_ = "insufficient_nested_class_support"
        self.estimator_ = base.set_params(**self.best_params_)
        if self._classification:
            self.classes_ = np.array([0, 1])
            if len(np.unique(y)) == 1:
                self.estimator_ = ConstantBinaryClassifier()
        kw = {} if weights is None else {"sample_weight": weights}
        self.estimator_.fit(xm, y, **kw)
        return self

    def _input(self, X):
        check_is_fitted(self, "estimator_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        return X[self.features_]

    @property
    def feature_importances_(self):
        check_is_fitted(self, "estimator_")
        lookup = dict(zip(self.features_, self.estimator_.feature_importances_))
        return np.asarray([lookup.get(c, 0.) for c in self.feature_names_in_])


class SelfTuningClassifier(ClassifierMixin, _SelfTuning, BaseEstimator):
    """Tune a classifier without calibration, or nest it inside a calibrator.

    Search is sequential; parallelism belongs to the outer framework.
    ``verbose=False`` hides Optuna study/trial INFO messages while retaining
    warnings and errors. ``True`` enables INFO messages. The previous logging
    level is restored afterward; estimator verbosity is configured separately.
    """
    _classification = True

    def __init__(self, estimator, param_distributions, cv=None, n_trials=4,
                 random_state=42, search="random", search_estimator_params=None,
                 exclude_features=None, verbose=False):
        self.estimator, self.param_distributions, self.cv = estimator, param_distributions, cv
        self.n_trials, self.random_state, self.search = n_trials, random_state, search
        self.search_estimator_params, self.exclude_features = search_estimator_params, exclude_features
        self.verbose = verbose

    def predict_proba(self, X):
        X = self._input(X)
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        X = self._input(X)
        return self.estimator_.predict(X)


class SelfTuningRegressor(RegressorMixin, _SelfTuning, BaseEstimator):
    """Sequential MSE search; ``verbose`` controls Optuna INFO messages only."""
    _classification = False

    def __init__(self, estimator, param_distributions, cv=None, n_trials=4,
                 random_state=42, search="random", search_estimator_params=None,
                 exclude_features=None, verbose=False):
        self.estimator, self.param_distributions, self.cv = estimator, param_distributions, cv
        self.n_trials, self.random_state, self.search = n_trials, random_state, search
        self.search_estimator_params, self.exclude_features = search_estimator_params, exclude_features
        self.verbose = verbose

    def predict(self, X):
        X = self._input(X)
        return self.estimator_.predict(X)


def local_beta_xgboost(*, tuning=True, ridge=100., n_trials=4, n_estimators=100,
                       tuning_n_estimators=40, random_state=42, max_trials=10,
                       Spatio1="longitude", Spatio2="latitude", Temporal1="DOY",
                       search="optuna", Spatio_blocks_count=5, Temporal_blocks_count=5,
                       n_splits=3, tuning_n_splits=2, tuning_verbose=False):
    """Compose the supplied pilot's XGBoost search and stixel-local beta map.

    XGBoost is imported only when this factory is called; Optuna only during
    tuning. No classifier balancing or final global probability map is added.
    Block counts configure both calibration and nested tuning splits. Compose
    the wrappers directly when those two stages need different granularities.
    ``n_splits`` controls calibration folds; ``tuning_n_splits`` controls the
    nested search folds and is used only when tuning is enabled.
    ``tuning_verbose`` controls Optuna study/trial INFO messages inside each
    worker. It is independent of AdaSTEM's progress and monitoring logs.
    """
    for name, count in (("n_splits", n_splits), ("tuning_n_splits", tuning_n_splits)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 2:
            raise ValueError(f"{name} must be an integer >= 2.")
    from xgboost import XGBClassifier
    from .calibration import BetaCalibrator, CrossValidatedCalibratedClassifier
    base = XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=.3,
        min_child_weight=1, reg_lambda=1., n_jobs=1, tree_method="hist", verbosity=0,
        random_state=random_state, eval_metric="logloss")
    outer = SpatioTemporalKFold(int(n_splits), Spatio1=Spatio1, Spatio2=Spatio2,
        Temporal1=Temporal1, random_state=random_state, max_trials=max_trials,
        Spatio_blocks_count=Spatio_blocks_count, Temporal_blocks_count=Temporal_blocks_count)
    if tuning:
        inner = SpatioTemporalKFold(int(tuning_n_splits), Spatio1=Spatio1, Spatio2=Spatio2,
            Temporal1=Temporal1, random_state=random_state+17, max_trials=max_trials,
            Spatio_blocks_count=Spatio_blocks_count, Temporal_blocks_count=Temporal_blocks_count)
        base = SelfTuningClassifier(base, dict(max_depth=[3, 4, 6], learning_rate=[.1, .3],
            min_child_weight=[1, 5, 10], reg_lambda=[1., 5., 10.]), cv=inner,
            n_trials=n_trials, random_state=random_state, search=search,
            search_estimator_params={"n_estimators": tuning_n_estimators}, verbose=tuning_verbose)
    return CrossValidatedCalibratedClassifier(base, BetaCalibrator(ridge), cv=outer)
