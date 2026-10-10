"""Standalone probability and point-prediction calibration maps."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class BetaCalibrator(BaseEstimator):
    """Increasing beta map, regularized toward the identity probability map.

    Minimizes SUM weighted binary log loss plus
    ``ridge / 2 * ((a - 1)**2 + (b - 1)**2 + c**2)`` for
    ``sigmoid(a*log(p) - b*log(1-p) + c)``. Ridge strength therefore depends
    on sample size and weight scale. The default matches the supplied pilot.
    """

    def __init__(self, ridge=100., eps=1e-10):
        self.ridge, self.eps = ridge, eps

    def _design(self, p):
        p = np.clip(np.asarray(p, float).reshape(-1), self.eps, 1 - self.eps)
        return np.column_stack([np.log(p), -np.log1p(-p), np.ones(len(p))])

    def fit(self, p, y, sample_weight=None):
        p, y = np.asarray(p, float).reshape(-1), np.asarray(y, float).reshape(-1)
        if not len(y) or len(p) != len(y) or not np.isfinite(p).all():
            raise ValueError("Expected matching nonempty finite probabilities and labels.")
        if not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Expected binary labels and probabilities in [0,1].")
        if not np.isfinite(self.ridge) or self.ridge < 0 or not 0 < self.eps < .5:
            raise ValueError("ridge must be nonnegative and eps between 0 and 0.5.")
        w = _weights(sample_weight, len(y))
        x, center = self._design(p), np.array([1., 1., 0.])
        def objective(theta):
            z, delta = x @ theta, theta - center
            value = np.sum(w * (np.logaddexp(0, z) - y*z)) + self.ridge/2 * (delta @ delta)
            gradient = x.T @ (w * (expit(z) - y)) + self.ridge * delta
            return value, gradient
        bounds = [(0.02, 20.), (0.02, 20.), (-30., 30.)]
        result = minimize(objective, center, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-7, "maxls": 50})
        if not result.success:
            result = minimize(objective, result.x, jac=True, method="SLSQP", bounds=bounds,
                              options={"maxiter": 300, "ftol": 1e-10})
            gradient = objective(result.x)[1].copy()
            for j, (lo, hi) in enumerate(bounds):
                if result.x[j] <= lo + 1e-7:
                    gradient[j] = min(gradient[j], 0.)
                if result.x[j] >= hi - 1e-7:
                    gradient[j] = max(gradient[j], 0.)
            if not result.success and np.max(np.abs(gradient)) > 1e-5:
                raise RuntimeError(f"Beta optimizer failed: {result.message}")
        if not np.isfinite(result.fun) or result.fun > objective(center)[0] + 1e-6:
            raise RuntimeError("Beta optimizer failed its objective check.")
        self.theta_, self.n_samples_ = result.x, len(y)
        self.penalty_ = float(self.ridge/2 * np.sum((self.theta_ - center)**2))
        self.objective_ = float(result.fun)
        return self

    def predict(self, p):
        check_is_fitted(self, "theta_")
        return expit(self._design(p) @ self.theta_)


def _weights(sample_weight, n):
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float).reshape(-1)
    if len(w) != n or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("Weights must be finite, nonnegative and have positive total mass.")
    return w


class AffineCalibrator(BaseEstimator):
    """Ridge correction of a regression point mean; not interval calibration."""

    def __init__(self, ridge=0.):
        self.ridge = ridge

    def fit(self, p, y, sample_weight=None):
        p, y = np.asarray(p, float).reshape(-1), np.asarray(y, float).reshape(-1)
        if not len(y) or len(p) != len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError("Expected matching finite regression predictions and targets.")
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative.")
        x, w = np.column_stack([p, np.ones(len(p))]), _weights(sample_weight, len(p))
        center = np.array([1., 0.])
        self.theta_ = np.linalg.lstsq(x.T @ (w[:, None] * x) + self.ridge*np.eye(2),
            x.T @ (w*y) + self.ridge*center, rcond=None)[0]
        return self

    def predict(self, p):
        check_is_fitted(self, "theta_")
        return self.theta_[0] * np.asarray(p, float).reshape(-1) + self.theta_[1]
