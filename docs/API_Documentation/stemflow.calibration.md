# stemflow.calibration

Composable calibration maps, OOF estimators and spatial-temporal diagnostics. See [local calibration and tuning](../Tips/Local_calibration_and_tuning.md).

## Public choices

| Component | Purpose |
| --- | --- |
| `BetaCalibrator` | Fit an increasing binary probability map with an identity-centered ridge penalty. |
| `AffineCalibrator` | Calibrate regression point means. |
| `CrossValidatedCalibratedClassifier` / `Regressor` | Fit the map from out-of-fold scores and retain fold estimators. |
| `SpatioTemporalKFold` | Block by space/time and audit fold support, retries and fallback status. |
| `calibration_scores`, `calibration_bins` | Summarize held-out scores and bin counts/uncertainty references. |
| `calibration_by_group`, `stixel_calibration_diagnostics` | Diagnose supplied groups or actual fitted planar stixel geometry. |
| `plot_calibration_diagnostics` | Save multi-level diagnostics and stixel overlays from those tables. |

For `SpatioTemporalKFold`, `n_splits` controls folds independently of spatial/temporal block counts. `min_pos`/`min_neg` are feasibility thresholds for each training and validation fold, not statistical reliability guarantees. `rare_policy="allow"` reports a fallback; `"raise"` rejects insufficient support. Inspect `fold_support_`, `support_sufficient_`, `status_` and `trials_`.

Generic wrappers use shuffled folds when `cv` is omitted. Supply the spatial-temporal splitter explicitly for blocked calibration. Intentional all-zero/all-one AdaSTEM stixels keep their dummy behavior. Local calibration and relative ECE do not establish exact conditional probabilities or copula accuracy; examine support, coverage, proper losses and floor sensitivity together.

::: stemflow.calibration.maps

::: stemflow.calibration.estimators

::: stemflow.calibration.cross_validation

::: stemflow.calibration.metrics

::: stemflow.calibration.diagnostics
