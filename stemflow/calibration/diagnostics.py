"""Held-out diagnostics for final probabilities inside actual fitted stixels."""

import numpy as np
import pandas as pd

from .metrics import calibration_bins, calibration_scores


def calibration_by_group(y, p, groups, n_bins=8, include_bins=True, common_bin_edges=None):
    """Score overlapping positional groups; units are not assumed independent."""
    y, p = np.asarray(y).reshape(-1), np.asarray(p).reshape(-1)
    if len(y) != len(p):
        raise ValueError("Prediction/outcome lengths differ.")
    scores, tables = [], []
    for metadata, positions in groups:
        values, bins = calibration_scores(y[positions], p[positions], n_bins)
        scores.append({**metadata, **values})
        if include_bins and len(bins):
            tables.append(bins.assign(**metadata))
            if common_bin_edges is not None:
                good = np.isfinite(y[positions]) & np.isfinite(p[positions])
                common = calibration_bins(y[positions][good], p[positions][good],
                    bin_edges=common_bin_edges).assign(strategy="common", **metadata)
                tables.append(common)
    return pd.DataFrame(scores), pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def stixel_calibration_diagnostics(model, X, y, pred=None, n_bins=8,
                                   spatial_bin_width=5., temporal_bin_width=40.,
                                   include_bins=True, support_min_events=20,
                                   support_min_samples=200):
    """Evaluate on caller-supplied held-out rows, using the fitted partition.

    Returns overall, spatial, temporal, space-time and overlapping stixel scores.
    Group accuracy uses final ensemble probabilities, not in-sample fold scores.
    Relative ECE is floor-sensitive and accompanied by zero-event bin mass.
    """
    from ..lazyloading.prediction import SpatialWindowIndex
    from ..model.static_func_AdaSTEM import transform_pred_set_to_STEM_quad
    if not isinstance(X, pd.DataFrame) or not X.index.is_unique:
        raise TypeError("Diagnostics require a DataFrame with unique observation IDs.")
    if isinstance(y, (pd.Series, pd.DataFrame)) and not y.index.equals(X.index):
        raise ValueError("Held-out X and y indices must match in order.")
    if spatial_bin_width <= 0 or temporal_bin_width <= 0:
        raise ValueError("Diagnostic bin widths must be positive.")
    if support_min_events < 0 or support_min_samples < 1:
        raise ValueError("Diagnostic support thresholds must be nonnegative/positive.")
    if "stixel_calibration_point_transformed_left_bound" not in model.ensemble_df:
        raise NotImplementedError("Actual-stixel diagnostics currently require planar AdaSTEM geometry.")
    coordinates = X[[model.Spatio1, model.Spatio2, model.Temporal1]]
    if not np.isfinite(coordinates.to_numpy()).all():
        raise ValueError("Diagnostic coordinates must be finite.")
    y = np.asarray(y).reshape(-1)
    if pred is None:
        pred = model.predict_proba(X, temporal_window_prequery=True)[:, 1]
    pred = np.asarray(pred).reshape(-1)
    if len(X) != len(y) or len(X) != len(pred):
        raise ValueError("Held-out X, y and predictions must have matching lengths.")
    spatial = list(zip(np.floor(X[model.Spatio1].to_numpy()/spatial_bin_width).astype(int),
                       np.floor(X[model.Spatio2].to_numpy()/spatial_bin_width).astype(int)))
    temporal = np.floor((X[model.Temporal1].to_numpy()-1)/temporal_bin_width).astype(int)
    def location(positions):
        return dict(longitude=X[model.Spatio1].iloc[positions].mean(),
                    latitude=X[model.Spatio2].iloc[positions].mean(),
                    DOY=X[model.Temporal1].iloc[positions].mean())
    def groups():
        yield dict(level="overall", group="all"), np.arange(len(X))
        keys = pd.DataFrame(dict(spatial=pd.Series(spatial), temporal=temporal))
        for level, cols in [("spatial", ["spatial"]), ("temporal", ["temporal"]),
                            ("spatiotemporal", ["spatial", "temporal"])]:
            for name, positions in keys.groupby(cols, sort=True).indices.items():
                yield dict(level=level, group=str(name), **location(positions)), positions
        columns = [model.Temporal1, model.Spatio1, model.Spatio2]
        for ensemble, grid in model.ensemble_df.groupby("ensemble_index", sort=True):
            for start, window_grid in grid.groupby(f"{model.Temporal1}_start", sort=True):
                positions = np.flatnonzero((X[model.Temporal1].to_numpy() >= start) &
                    (X[model.Temporal1].to_numpy() < start + model.temporal_bin_interval))
                coords = transform_pred_set_to_STEM_quad(model.Spatio1, model.Spatio2,
                    X.iloc[positions][columns], grid)
                lookup = SpatialWindowIndex(coords[f"{model.Spatio1}_new"].to_numpy(),
                                            coords[f"{model.Spatio2}_new"].to_numpy(), len(window_grid))
                for _, row in window_grid.iterrows():
                    name = str(row.unique_stixel_id)
                    if f"{name}_model" not in model.model_dict:
                        continue
                    estimator = model.model_dict[f"{name}_model"]
                    support = vars(estimator).get("_stemflow_training_support_", {})
                    selected = lookup.select(row)
                    yield dict(level="stixel", group=name, ensemble_index=ensemble,
                        DOY_start=start, partition_n=row.stixel_checklist_count,
                        training_n=support.get("n", np.nan),
                        training_detections=support.get("positives", np.nan),
                        **location(positions[selected])), positions[selected]
    scores, bins = calibration_by_group(y, pred, groups(), n_bins, include_bins,
                                      common_bin_edges=np.r_[0., np.geomspace(1e-8, 1, 65)])
    for name in ["training_n", "training_detections"]:
        if name not in scores:
            scores[name] = np.nan
    scores["supported"] = ((scores.n_valid >= support_min_samples) &
        (scores.detections >= support_min_events) & (scores.non_detections >= support_min_events))
    scores["training_supported"] = ((scores.level == "stixel") &
        (scores.training_detections >= support_min_events) &
        (scores.training_n-scores.training_detections >= support_min_events))
    metrics = ["ECE_pp", "quantile_ECE_pp", "ECE_relative", "quantile_ECE_relative",
               "quantile_relative_floor_1e8", "quantile_zero_event_mass", "bias_pp",
               "rate_ratio", "AUC", "average_precision", "Brier", "log_loss", "coverage"]
    records = []
    for scope, selected in [("all", scores), ("supported", scores[scores.supported]),
                            ("training_supported", scores[scores.training_supported])]:
        for level, rows in selected.groupby("level", sort=True):
            for metric in metrics:
                values = rows[metric].replace([np.inf, -np.inf], np.nan).dropna()
                records.append(dict(scope=scope, level=level, metric=metric, n_groups=len(values),
                    **dict(zip(["q25", "median", "q75", "p90"], values.quantile([.25, .5, .75, .9])))))
    return dict(scores=scores, bins=bins, summary=pd.DataFrame(records))


def plot_calibration_diagnostics(diagnostics, output_path, max_curves=200):
    """Save overall/stixel diagnostics and transparent log-scale overlays."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    scores, bins = diagnostics["scores"], diagnostics["bins"]
    if bins.empty:
        raise ValueError("Plotting requires diagnostics computed with include_bins=True.")
    stixels = scores[(scores.level == "stixel") & (scores.n_valid > 0)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for strategy in ["uniform", "quantile"]:
        b = bins[(bins.level == "overall") & (bins.strategy == strategy)]
        axes[0, 0].plot(b.predicted, b.observed, "o-", label=strategy)
    axes[0, 0].plot([0, 1], [0, 1], "k--")
    axes[0, 0].set(xlabel="Predicted probability", ylabel="Observed frequency", title="Overall reliability")
    axes[0, 0].legend()
    axes[0, 1].scatter(stixels.predicted_rate, stixels.observed_rate, s=5, alpha=.25, rasterized=True)
    axes[0, 1].plot([0, 1], [0, 1], "k--")
    axes[0, 1].set(xlabel="Stixel mean prediction", ylabel="Stixel observed frequency", title="Actual fitted stixels")
    for axis, metric in zip([axes[0, 2], axes[1, 0]], ["quantile_ECE_pp", "quantile_ECE_relative"]):
        for label, group in [("All", stixels), ("Training-supported", stixels[stixels.training_supported])]:
            v = np.sort(group[metric].dropna())
            if len(v):
                axis.plot(v, np.arange(1, len(v)+1)/len(v), label=f"{label}: {len(v)}")
        axis.set(xlabel=metric, ylabel="Fraction of stixels", ylim=(0, 1))
        axis.legend(fontsize=8)
        if metric.endswith("relative"):
            axis.set_xscale("log")
    spatial = scores[scores.level == "spatial"]
    half = max(float(spatial.bias_pp.abs().max()), 1e-9)
    artist = axes[1, 1].scatter(spatial.longitude, spatial.latitude, c=spatial.bias_pp,
        cmap="coolwarm", vmin=-half, vmax=half, s=20)
    axes[1, 1].set(xlabel="Longitude", ylabel="Latitude", title="Spatial-bin bias")
    fig.colorbar(artist, ax=axes[1, 1], label="Predicted − observed (pp)")
    temporal = scores[scores.level == "temporal"].sort_values("DOY")
    axes[1, 2].plot(temporal.DOY, temporal.observed_rate, "o-", label="Observed")
    axes[1, 2].plot(temporal.DOY, temporal.predicted_rate, "o-", label="Predicted")
    axes[1, 2].set(xlabel="DOY", ylabel="Occurrence probability", title="Temporal-bin means")
    axes[1, 2].legend()
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output/f"calibration_diagnostics.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = stixels.group.to_numpy()
    if len(names) > max_curves:
        names = np.random.default_rng(42).choice(names, max_curves, replace=False)
    for axis, strategy in zip(axes, ["uniform", "quantile"]):
        b = bins[(bins.level == "stixel") & (bins.strategy == strategy)]
        for _, curve in b[b.group.isin(names)].groupby("group", sort=False):
            good = (curve.predicted > 0) & (curve.observed > 0)
            axis.plot(curve.predicted.where(good), curve.observed.where(good), color="C0", alpha=.08)
        # Common probability intervals avoid averaging unrelated quantile-bin numbers.
        common = bins[(bins.level == "stixel") & (bins.strategy == "common")]
        mean = common.groupby("bin")[["predicted", "observed"]].mean()
        good = (mean.predicted > 0) & (mean.observed > 0)
        axis.plot(mean.predicted.where(good), mean.observed.where(good), color="crimson", lw=2,
                  label="Equal-stixel mean in common probability bins")
        axis.plot([1e-8, 1], [1e-8, 1], "k--", label="Perfect calibration")
        axis.set(xscale="log", yscale="log", xlim=(1e-8, 1), ylim=(1e-8, 1),
                 xlabel="Mean predicted probability", ylabel="Observed frequency", title=strategy)
        formatter = FuncFormatter(lambda value, _: f"{value:.8f}".rstrip("0").rstrip("."))
        axis.xaxis.set_major_formatter(formatter)
        axis.yaxis.set_major_formatter(formatter)
        axis.tick_params(axis="x", rotation=45)
        axis.legend(fontsize=8)
    fig.suptitle("Held-out final probabilities within stixels; zeros break log lines")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output/f"stixel_calibration_overlay.{ext}", dpi=180)
    plt.close(fig)
