# Local calibration and tuning

Calibration is an optional base-estimator composition. Keep using `AdaSTEMClassifier`: each fitted stixel receives its own calibrated estimator, and AdaSTEM combines those local probabilities. Existing classifiers and regressors retain their behavior. No new task string or final global probability map is required.

## Choose tuning and calibration independently

| Choice | AdaSTEM base model | What happens within each stixel |
| --- | --- | --- |
| Fixed hyperparameters | Your classifier | Fit once; no search or probability map |
| Tuning only | `SelfTuningClassifier(classifier, ...)` | Select hyperparameters by CV, then refit |
| Calibration only | `CrossValidatedCalibratedClassifier(classifier, ...)` | Fixed hyperparameters; blocked OOF predictions fit the local map |
| Tuning and calibration | `CrossValidatedCalibratedClassifier(SelfTuningClassifier(...), ...)` | Tune within each calibration training fold; fit the map from OOF predictions |

```python
from sklearn.ensemble import RandomForestClassifier
from stemflow.model.AdaSTEM import AdaSTEMClassifier
from stemflow.calibration import (
    BetaCalibrator, CrossValidatedCalibratedClassifier, SpatioTemporalKFold,
)
from stemflow.tuning import SelfTuningClassifier

fixed = RandomForestClassifier(
    n_estimators=40, max_depth=8, min_samples_leaf=5,
    n_jobs=1, random_state=42,
)
tuned = SelfTuningClassifier(
    fixed, {"max_depth": [4, 8], "min_samples_leaf": [5, 20]},
    cv=SpatioTemporalKFold(n_splits=2, Spatio_blocks_count=5,
                         Temporal_blocks_count=5, random_state=59), n_trials=4,
)
calibrated_fixed = CrossValidatedCalibratedClassifier(
    fixed, calibrator=BetaCalibrator(ridge=100),
    cv=SpatioTemporalKFold(n_splits=3, Spatio_blocks_count=5,
                         Temporal_blocks_count=5, random_state=42),
)
calibrated_tuned = CrossValidatedCalibratedClassifier(
    tuned, calibrator=BetaCalibrator(ridge=100),
    cv=SpatioTemporalKFold(n_splits=3, Spatio_blocks_count=5,
                         Temporal_blocks_count=5, random_state=42),
)

# Choose one of these four objects.
base = calibrated_fixed
model = AdaSTEMClassifier(base_model=base, sample_weights_for_classifier=False)
```

Each wrapper clones the supplied estimator before fitting, so these templates can be reused. Every choice also works independently with `.fit(X, y)` and `.predict_proba(X)`. Fixed hyperparameters do not prevent blocked calibration: the same parameters are used to fit a different model in each calibration fold.

The calibration class does not enable tuning by itself. Its `cv` controls calibration folds; the tuner's own `cv` controls hyperparameter selection inside those training folds. This nesting keeps calibration labels out of tuning. Supply `SpatioTemporalKFold` explicitly for spatial-temporal folds; the generic wrappers use ordinary shuffled folds when `cv` is omitted. Separate classes keep these choices inspectable without adding a new AdaSTEM task type.

For the XGBoost recipe below, `local_beta_xgboost(tuning=False)` is the short form for calibration only, and `tuning=True` adds the nested search. Use the generic classes when choosing another estimator, search space or map. The analogous regression classes are `SelfTuningRegressor` and `CrossValidatedCalibratedRegressor`, with the narrower point-mean interpretation described below.

## The local beta strategy

```python
from stemflow.model.AdaSTEM import AdaSTEMClassifier
from stemflow.tuning import local_beta_xgboost

base = local_beta_xgboost(
    tuning=True, ridge=100, n_trials=4,
    n_splits=3, tuning_n_splits=2,
    Spatio_blocks_count=5, Temporal_blocks_count=5,
)
model = AdaSTEMClassifier(
    base_model=base, ensemble_fold=50, min_ensemble_required=45,
    n_jobs=4, lazy_loading=True, sample_weights_for_classifier=False,
    random_state=42,
)
model.fit(X_path, y_path, temporal_window_prequery=True,
          training_data_backend="memmap", inner_max_num_threads=1,
          persist_prediction_state=False)
p = model.predict_proba(X_test, temporal_window_prequery=True)[:, 1]
```

Use your established gridding, temporal windows and sample thresholds in the constructor. This example illustrates estimator composition, not a replacement for species-specific grid settings. `memmap` training is optional; `native` retains DuckDB window queries.

The factory follows the supplied pilot strategy:

1. Within each stixel, make `n_splits` spatial-temporal calibration folds (default 3). Longitude and latitude are splitting metadata, excluded from XGBoost predictors; DOY remains a predictor.
2. Within each training fold, tune a small XGBoost using `tuning_n_splits` additional spatial-temporal folds (default 2). Search sequentially, so AdaSTEM controls ensemble parallelism.
3. Pool out-of-fold probabilities and fit one increasing beta map for that stixel.
4. Retain the `n_splits` fold models. Apply the local map to each fold prediction, then average. AdaSTEM subsequently averages eligible ensemble probabilities.

The search space is depth `{3,4,6}`, learning rate `{0.1,0.3}`, minimum child weight `{1,5,10}` and L2 penalty `{1,5,10}`. Defaults use four trials, 40 trees during search and 100 trees for retained fold models. These match the supplied small search; they are not established optima for every species. `tuning=False` retains local beta calibration without hyperparameter search.

`Spatio_blocks_count` and `Temporal_blocks_count` are configurable inputs, separate from `n_splits`. The factory forwards them to both calibration and nested tuning. Compose the wrappers explicitly to give those two stages different block counts. Bounds are calculated from the data passed to each splitter, so widths vary between stixels and nested training subsets. The existing `ST_KFold` uses that many `linspace` cutpoints: a count of 5 gives four equal-width interior intervals plus the maximum-boundary group on each axis. It does not specify five degrees or five days, and no exclusion buffer is added between adjacent blocks. The default granularity reproduces the earlier workflow; it has not been optimized for every species.

Both fold counts are configurable factory inputs and must be integers of at least 2. `n_splits` controls calibration; `tuning_n_splits` is used only with `tuning=True`. More calibration folds retain more models and generally cost more fitting time, storage and prediction work. More nested tuning folds increase search work. These settings do not change spatial or temporal block widths.

The factory defaults to Optuna/TPE. Install optional dependencies in your own environment with `pip install -e '.[tuning]'`. `search="random"` explicitly selects sklearn's parameter sampler. Local validation exercised both routes. The four-trial Optuna probe produced identical selected parameters, beta coefficients and probabilities to the supplied standalone helper on the same fixture. Different candidate sequences mean random search is not identical to the TPE experiment.

If your runner retains `longitude_tmp` and `latitude_tmp`, pass `Spatio1="longitude_tmp", Spatio2="latitude_tmp"` to the factory, as in the reference helper. Otherwise those copies would be ordinary estimator features. New workflows can use original coordinate names without creating copies.

## What ridge penalizes

For raw probability `p`, the beta map is `sigmoid(a log(p) − b log(1−p) + c)`. `BetaCalibrator` minimizes **summed**, optionally weighted, binary log loss plus:

```text
ridge / 2 × [(a−1)² + (b−1)² + c²]
```

The identity map has `(a,b,c)=(1,1,0)`. Ridge 100 pulls the local map toward that identity; it does not mean 100 observations or an error threshold. Its effect depends on sample size and weight scale. Slope bounds keep the map increasing. Monotonicity preserves ranking for a single map, but averaging fold models and stixel maps can change AUC. Check AUC as well as calibration.

Local calibration does not guarantee exact conditional probabilities at every location, date or environmental combination, and does not establish accurate copula correlation by itself.

## Reuse the components independently

```python
from sklearn.ensemble import RandomForestClassifier
from stemflow.calibration import (
    BetaCalibrator, CrossValidatedCalibratedClassifier, SpatioTemporalKFold,
)
from stemflow.tuning import SelfTuningClassifier

forest = SelfTuningClassifier(
    RandomForestClassifier(n_estimators=40, n_jobs=1, random_state=42),
    {"max_depth": [4, 8], "min_samples_leaf": [5, 20]},
    cv=SpatioTemporalKFold(n_splits=2, Spatio_blocks_count=5,
                         Temporal_blocks_count=5), n_trials=4,
)
calibrated = CrossValidatedCalibratedClassifier(
    forest, calibrator=BetaCalibrator(ridge=100),
    cv=SpatioTemporalKFold(n_splits=3, Spatio_blocks_count=5,
                         Temporal_blocks_count=5),
)
calibrated.fit(X_train, y_train)
p = calibrated.predict_proba(X_test)[:, 1]
```

The same estimator can be an AdaSTEM base model. Both wrappers are sklearn-cloneable. Alternate maps implement `fit(prediction, y, sample_weight=...)` and `predict(prediction)`; weights are only passed when supplied. `IsotonicRegression` can serve as a map. Generic wrappers do not require XGBoost or Optuna.

`CrossValidatedCalibratedRegressor` uses an OOF `AffineCalibrator` for regression point means. `SelfTuningRegressor` searches using validation mean squared error. This does not calibrate uncertainty intervals or predictive distributions.

## Sparse folds and intentional zeros

`SpatioTemporalKFold` retries blocked folds up to `max_trials`, exposing `fold_support_`, `support_sufficient_`, `status_` and `trials_`. Circular-year copies at identical coordinates share a seasonal block.

With `rare_policy="allow"`, the first nonempty blocked split is retained if requested event counts cannot be met. A single-class training fold uses an audited smoothed constant; no independent split at all produces an audited constant estimator. The tuner skips its search when nested support is insufficient. These fallbacks do not establish reliable calibration. `rare_policy="raise"` rejects insufficient support. One positive per fold is a feasibility threshold, not a reliability threshold.

AdaSTEM's intentional all-zero/all-one stixels retain dummy predictions and bypass the calibration wrapper. Missing or untrained coverage remains missing, subject to the minimum-ensemble requirement.

## Diagnose downstream probabilities

```python
from pathlib import Path
from stemflow.calibration import plot_calibration_diagnostics

# X_test/y_test must be held out and have matching indexes.
diagnostics = model.calibration_diagnostics(X_test, y_test, pred=p)
output = Path("data/calibration")
output.mkdir(parents=True, exist_ok=True)
for name, table in diagnostics.items():
    table.to_csv(output / f"calibration_{name}.csv", index=False)
plot_calibration_diagnostics(diagnostics, output)
```

Scores cover overall, spatial, temporal, space-time and actual fitted stixel groups. Stixel metrics evaluate **final ensemble probabilities** inside each fitted cell. Overlapping ensemble stixels are not independent replicates. Actual-stixel geometry currently supports planar AdaSTEM/STEM; `calibration_by_group` accepts arbitrary supplied groups.

Metrics include uniform and quantile ECE in percentage points, relative ECE, relative ECE at a second floor, zero-event bin mass, bias, rate ratio, AUC, average precision, Brier, log loss and coverage. Relative ECE is `exp(sum(weight × abs(log(max(pred,1e-6)/max(obs,1e-6)))))`; ideal is 1. Its dependence on probability floors and zero-event bins is explicit.

Bins retain counts, observed and expected detections, and Wilson intervals. The product of `(1-p)` is labeled an independent-Bernoulli reference, and Wilson intervals assume independent binomial observations; neither accounts for spatial dependence. Few events can make relative ECE enormous even under a correct rare-event model.

Support flags are descriptive and configurable:

- `supported`: at least 200 valid held-out rows and 20 of each outcome by default. This uses held-out labels and should not be the sole comparison population.
- `training_supported`: at least 20 training positives and 20 training negatives after filtering and bootstrapping. It does not use held-out labels. Old estimators without training counts are marked unsupported rather than having support guessed from partition size.

Plots show overall reliability, stixel means, stixel ECE distributions, spatial bias, temporal means, and uniform/quantile stixel overlays. Zero observations break log lines. The red overlay uses equal-stixel means in common probability intervals, including zeros in the arithmetic mean; it does not average unrelated quantile-bin numbers. A deterministic sample of thin curves is drawn when there are many stixels, while the red mean uses all stixels.

For a copula partition different from the occurrence partition, call `stixel_calibration_diagnostics(copula_model, X_test, y_test, pred=p)`: the model supplies geometry, while `p` supplies held-out marginal probabilities. This diagnoses marginals at the inference unit; it does not measure unknown real-data copula truth.

The small real Acadian audit is in `scripts/02_calibration/compare_local_calibration.py`. Sparse held-out data demonstrate execution and diagnostics, not a definitive method ranking. Compare local losses and relative ECE alongside overall metrics on a common held-out population.
