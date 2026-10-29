# stemflow.tuning

Optional sequential tuning for generic estimators, with an XGBoost/local-beta factory.

## XGBoost recipe parameters

Install `stemflow[tuning]` for XGBoost and Optuna. Generic random search with a supplied estimator does not require those optional packages.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `tuning` | `True` | Enable the nested hyperparameter search; `False` keeps local beta calibration with a fixed XGBoost. |
| `ridge` | `100.0` | Identity-centered beta penalty applied to summed loss; not a universal optimum. |
| `n_trials` | `4` | Sequential search trials inside each calibration training fold. |
| `n_estimators` | `100` | Trees in each retained XGBoost fold model. |
| `tuning_n_estimators` | `40` | Trees per search candidate before fitting the retained model. |
| `n_splits` | `3` | Calibration folds and number of retained fold models per fitted stixel. |
| `tuning_n_splits` | `2` | Nested search folds, used only when tuning is enabled. |
| `Spatio_blocks_count`, `Temporal_blocks_count` | `5`, `5` | Block cutpoint counts, forwarded to both calibration and tuning; not degrees/days or fold counts. |
| `Spatio1`, `Spatio2`, `Temporal1` | `longitude`, `latitude`, `DOY` | Columns for CV metadata; the spatial columns are excluded from predictors. |
| `max_trials` | `10` | Maximum blocked-split retries when checking fold support; distinct from tuning trials. |
| `search` | `"optuna"` | Search backend: Optuna/TPE or explicit `"random"`. |
| `tuning_verbose` | `False` | Control Optuna study/trial INFO messages independently of estimator/AdaSTEM logs. |
| `random_state` | `42` | Reproducible estimator/search/split seeds. |

For other search spaces, classifiers, regressors, maps, CV granularities or rare-fold policies, compose the generic wrappers directly. See [local calibration and tuning](../Tips/Local_calibration_and_tuning.md) and the [executed example](../Examples/11.Local_calibration_and_profiling.ipynb). Regression calibration concerns point means, not predictive intervals.

::: stemflow.tuning
