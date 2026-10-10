"""Bounded prediction reduction and exact rectangular row selection."""

import numpy as np
import pandas as pd


class PredictionMoments:
    """Online per-row moments; missing ensembles do not count toward coverage."""

    def __init__(self, index, with_std=False, logit=False):
        self.index = pd.Index(index)
        self.count = np.zeros(len(index), dtype=np.int32)
        self.mean = np.zeros(len(index), dtype=float)
        self.m2 = np.zeros(len(index), dtype=float) if with_std else None
        self.logit = logit
        self.seen = False

    def update(self, prediction):
        if not len(prediction):
            return
        self.seen = True
        positions = self.index.get_indexer(prediction.index)
        keep = positions >= 0
        values = prediction.iloc[:, 0].to_numpy(dtype=float)[keep]
        positions = positions[keep]
        keep = ~np.isnan(values)
        values, positions = values[keep], positions[keep]
        if self.logit:
            values = np.clip(values, 1e-16, 1 - 1e-16)
            values = np.log(values / (1 - values))
        count = self.count[positions] + 1
        delta = values - self.mean[positions]
        self.mean[positions] += delta / count
        if self.m2 is not None:
            self.m2[positions] += delta * (values - self.mean[positions])
        self.count[positions] = count

    def finish(self, min_ensembles):
        mean = self.mean.copy()
        if self.logit:
            from scipy.special import expit
            mean = expit(mean)
            mean[mean <= 1e-16] = 0
        mean[self.count < min_ensembles] = np.nan
        std = None
        if self.m2 is not None:
            std = np.full(len(mean), np.nan)
            valid = (self.count >= min_ensembles) & (self.count > 1)
            std[valid] = np.sqrt(np.maximum(0, self.m2[valid]) / (self.count[valid] - 1))
        return mean, std


class SpatialWindowIndex:
    """Sort one coordinate once, then filter only rectangle candidates.

    Returned positions retain source order and half-open boundaries. Small
    windows use direct masks to avoid sorting overhead.
    """

    def __init__(self, x, y, n_stixels):
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.order = (np.argsort(self.x, kind="stable")
                      if len(x) >= 1000 and n_stixels >= 16 else None)
        self.sorted_x = self.x[self.order] if self.order is not None else None

    def select(self, grid):
        left = grid["stixel_calibration_point_transformed_left_bound"]
        right = grid["stixel_calibration_point_transformed_right_bound"]
        lower = grid["stixel_calibration_point_transformed_lower_bound"]
        upper = grid["stixel_calibration_point_transformed_upper_bound"]
        if self.order is None:
            return np.flatnonzero((self.x >= left) & (self.x < right) &
                                  (self.y >= lower) & (self.y < upper))
        lo, hi = np.searchsorted(self.sorted_x, [left, right], side="left")
        candidates = self.order[lo:hi]
        return np.sort(candidates[(self.y[candidates] >= lower) &
                                   (self.y[candidates] < upper)])
