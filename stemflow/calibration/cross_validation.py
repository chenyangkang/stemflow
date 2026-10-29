"""Audited spatial-temporal folds for local calibration and nested tuning."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..model_selection import ST_KFold


class InsufficientSplitError(ValueError):
    """There are no nonempty independent spatial-temporal folds."""


class SpatioTemporalKFold(BaseEstimator):
    """Retry blocked folds, recording event support rather than hiding scarcity.

    ``rare_policy='allow'`` retains the first nonempty blocked split when the
    requested class support cannot be met. No random row split is substituted.
    ``period`` folds circular-year copies onto their original seasonal day.
    Spatial and temporal block counts are configurable, independently of
    ``n_splits``. They retain ST_KFold's linspace-cutpoint semantics and use
    the observed range of the data passed to each split, without a buffer.
    """

    def __init__(self, n_splits=3, Spatio1="longitude", Spatio2="latitude",
                 Temporal1="DOY", Spatio_blocks_count=5, Temporal_blocks_count=5,
                 random_state=42, max_trials=10, min_pos=1, min_neg=1,
                 rare_policy="allow", period=366):
        self.n_splits = n_splits
        self.Spatio1, self.Spatio2, self.Temporal1 = Spatio1, Spatio2, Temporal1
        self.Spatio_blocks_count = Spatio_blocks_count
        self.Temporal_blocks_count = Temporal_blocks_count
        self.random_state, self.max_trials = random_state, max_trials
        self.min_pos, self.min_neg = min_pos, min_neg
        self.rare_policy, self.period = rare_policy, period

    @property
    def metadata_columns(self):
        return (self.Spatio1, self.Spatio2, self.Temporal1)

    @property
    def feature_exclusions(self):
        return (self.Spatio1, self.Spatio2)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Spatial-temporal cross-validation requires a DataFrame.")
        if self.n_splits < 2 or self.max_trials < 1 or self.min_pos < 0 or self.min_neg < 0:
            raise ValueError("Invalid fold count, trial count or event threshold.")
        if self.rare_policy not in {"allow", "raise"}:
            raise ValueError("rare_policy must be 'allow' or 'raise'.")
        geometry = X[list(self.metadata_columns)].copy()
        if self.period is not None:
            if self.period <= 0:
                raise ValueError("period must be positive or None.")
            geometry[self.Temporal1] = (geometry[self.Temporal1] - 1) % self.period + 1
        if not np.isfinite(geometry.to_numpy()).all():
            raise ValueError("Spatial-temporal coordinates must be finite.")
        labels = None if y is None else np.asarray(y).reshape(-1)
        if labels is not None and len(labels) != len(X):
            raise ValueError("X and y lengths differ.")
        binary = labels is not None and np.isin(labels, [0, 1]).all()
        first, selected, sufficient = None, None, False
        for trial in range(self.max_trials):
            seed = None if self.random_state is None else self.random_state + trial
            splitter = ST_KFold(Spatio1=self.Spatio1, Spatio2=self.Spatio2,
                Temporal1=self.Temporal1, Spatio_blocks_count=self.Spatio_blocks_count,
                Temporal_blocks_count=self.Temporal_blocks_count,
                n_splits=self.n_splits, random_state=seed)
            folds = [(np.asarray(a, int), np.asarray(b, int)) for a, b in splitter.split(geometry)]
            if not all(len(a) and len(b) for a, b in folds):
                continue
            if first is None:
                first = folds
            sufficient = not binary or all(
                np.sum(labels[v] == 1) >= self.min_pos and
                np.sum(labels[v] == 0) >= self.min_neg for fold in folds for v in fold)
            if sufficient:
                selected = folds
                break
        self.support_sufficient_ = sufficient
        self.trials_ = trial + 1
        selected = selected if selected is not None else first
        if selected is None or (not sufficient and self.rare_policy == "raise"):
            raise InsufficientSplitError("No blocked split meets the requested support.")
        self.status_ = "supported" if sufficient else "rare_fold_fallback"
        self.fold_support_ = [dict(n_train=len(a), n_validation=len(b),
            training_events=int(np.sum(labels[a] == 1)) if binary else None,
            validation_events=int(np.sum(labels[b] == 1)) if binary else None)
            for a, b in selected]
        yield from selected
