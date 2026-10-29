"""Binary calibration metrics with explicit rare-bin support diagnostics."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score


def calibration_bins(y, p, strategy="quantile", n_bins=8, bin_edges=None):
    y, p = np.asarray(y, float).reshape(-1), np.asarray(p, float).reshape(-1)
    if len(y) != len(p) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected matching finite probabilities and binary labels.")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be positive.")
    if not len(p):
        return pd.DataFrame()
    if bin_edges is not None:
        edges = np.asarray(bin_edges, float)
        if len(edges) < 2 or not np.all(np.diff(edges) > 0) or edges[0] > 0 or edges[-1] < 1:
            raise ValueError("Common bin edges must increase and cover [0,1].")
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins+1)))
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins+1)
    elif strategy == "log":
        edges = np.r_[0., np.geomspace(1e-6, 1, 17)]
    else:
        raise ValueError(strategy)
    bid = np.searchsorted(edges[1:-1], p, side="left")
    n = np.bincount(bid)
    keep = n > 0
    events = np.bincount(bid, weights=y, minlength=len(n))[keep]
    expected = np.bincount(bid, weights=p, minlength=len(n))[keep]
    n = n[keep]
    obs = events/n
    z = 1.96
    denominator = 1+z*z/n
    center = (obs+z*z/(2*n))/denominator
    half = z*np.sqrt(obs*(1-obs)/n+z*z/(4*n*n))/denominator
    with np.errstate(divide="ignore"):
        log_survival = np.log1p(-p)
    log_p0 = np.bincount(bid, weights=log_survival, minlength=int(bid.max())+1)[keep]
    return pd.DataFrame(dict(bin=np.flatnonzero(keep), n=n, detections=events.astype(int),
                             observed=obs, predicted=expected/n, weight=n/len(p),
                             expected_detections=expected,
                             probability_no_detections_independent=np.exp(log_p0),
                             observed_wilson_low=np.maximum(0, center-half),
                             observed_wilson_high=np.minimum(1, center+half)))


def calibration_scores(y, p, n_bins=8):
    y, p = np.asarray(y, float).reshape(-1), np.asarray(p, float).reshape(-1)
    if len(y) != len(p):
        raise ValueError("Prediction/outcome lengths differ.")
    n_total = len(y)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    score = dict(n_total=n_total, n_valid=len(y), coverage=len(y)/n_total if n_total else np.nan,
                 detections=int(y.sum()), non_detections=int(len(y)-y.sum()))
    names = ["observed_rate", "predicted_rate", "bias_pp", "rate_ratio", "Brier", "log_loss", "AUC", "average_precision",
             "ECE_pp", "quantile_ECE_pp", "ECE_relative", "quantile_ECE_relative", "quantile_relative_floor_1e8",
             "uniform_zero_event_mass", "quantile_zero_event_mass"]
    score.update(dict.fromkeys(names, np.nan))
    if not len(y):
        return score, pd.DataFrame()
    if not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected binary labels and probabilities in [0,1].")
    pc = np.clip(p, 1e-12, 1-1e-12)
    score.update(observed_rate=y.mean(), predicted_rate=p.mean(), bias_pp=100*(p.mean()-y.mean()),
                 rate_ratio=p.mean()/y.mean() if y.mean() > 0 else np.nan,
                 Brier=np.mean((p-y)**2), log_loss=-np.mean(y*np.log(pc)+(1-y)*np.log1p(-pc)),
                 AUC=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
                 average_precision=average_precision_score(y, p) if y.sum() else np.nan)
    tables = []
    for strategy, prefix in [("uniform", ""), ("quantile", "quantile_")]:
        b = calibration_bins(y, p, strategy, n_bins)
        discrepancy = np.abs(np.log(np.maximum(b.predicted, 1e-6)/np.maximum(b.observed, 1e-6)))
        score[prefix+"ECE_pp"] = 100*np.sum(b.weight*np.abs(b.predicted-b.observed))
        score[prefix+"ECE_relative"] = np.exp(np.sum(b.weight*discrepancy))
        score[strategy+"_zero_event_mass"] = b.loc[b.detections == 0, "weight"].sum()
        if strategy == "quantile":
            score["quantile_relative_floor_1e8"] = np.exp(np.sum(b.weight*np.abs(
                np.log(np.maximum(b.predicted, 1e-8)/np.maximum(b.observed, 1e-8)))))
        b["strategy"] = strategy
        b["weighted_abs_log_ratio"] = b.weight*discrepancy
        tables.append(b)
    return score, pd.concat(tables, ignore_index=True)
