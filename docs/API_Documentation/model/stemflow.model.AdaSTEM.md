# stemflow.model.AdaSTEM

The reference below is generated from the current class signatures and docstrings. The same planar execution controls are available through STEM. See the [execution notebook](../../Examples/10.Execution_and_memory.ipynb) for executed examples and [local calibration](../../Examples/11.Local_calibration_and_profiling.ipynb) for estimator composition.

## Execution controls in 2.0.0

### Fitting

| Argument | Default | Meaning |
| --- | --- | --- |
| `training_data_backend` | `"native"` | Use Pandas/DuckDB/Parquet queries, or opt into `"memmap"` for shared numeric input maps. |
| `temporal_window_prequery` | `False` | Cache one temporal window on the native route; remains compatible with memmap selection. |
| `inner_max_num_threads` | `None` | Optional native-library thread limit for loky training workers; configure estimator threads separately. |
| `max_concurrent_ensembles` | `None` | Cap partitioning/training concurrency without changing the ensemble count. |
| `persist_prediction_state` | `True` | Preserve lazy-estimator prediction-time state. Use `False` only when that state need not be written back. |
| `ensemble_df` | `None` | Reuse supplied partition metadata instead of generating partitions during fitting. |
| `stixel_filter_func` | `None` | Filter each stixel's training DataFrame before fitting. |

### Prediction

| Argument | Default | Meaning |
| --- | --- | --- |
| `n_jobs` | `None` | Set prediction workers independently of fitting; `None` uses the model setting. |
| `prediction_backend` | `"native"` | Opt into `"memmap"` for shared numeric inputs and bounded temporary result buffers. |
| `streaming` | `True` | Accumulate mean/std without retaining all ensemble columns; median/separate output uses the matrix route. |
| `return_by_separate_ensembles` | `False` | Return the row-by-ensemble matrix instead of an aggregate. |
| `prediction_output_path` | `None` | With memmap separate-ensemble output, create a new `.npy` file and return a writable NumPy memmap. The caller owns the file. |

`AdaSTEMRegressor.predict` and `AdaSTEMClassifier.predict` forward the prediction execution options to `predict_proba`. For occurrence probabilities use the classifier's `predict_proba`; `predict` applies a classification threshold. Missing coverage remains explicit in the outputs. Sample standard deviations use `ddof=1`; `logit_agg=True` reports std on the logit scale. Earlier `shared`/`bounded` backend names remain aliases.

Memmap execution supports built-in numeric planar workflows. Use native execution for SphereAdaSTEM or unsupported custom prediction/training overrides. `max_mem` is a DuckDB buffer budget, not a total RAM limit; mapping does not cap base-estimator workspace.

### Monitoring and diagnostics

`model.monitor_execution(...)` records inclusive stages and sampled process-tree memory. The finalized report remains on `model.execution_monitor_`. `execution_info_` and `prediction_execution_info_` record structural execution information independently of the live monitor. See the [monitor API](../utils/stemflow.utils.execution_monitor.md) and [execution guide](../../Tips/Tips_for_parallel_execution.md).

`model.eval_calibration(...)` provides binary probability scores. `model.calibration_diagnostics(...)` evaluates held-out probabilities at overall, spatial, temporal, space-time and actual-stixel levels; its keyword options are documented by [stixel_calibration_diagnostics](../stemflow.calibration.md).

## Generated reference

::: stemflow.model.AdaSTEM
