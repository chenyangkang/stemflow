# Migrating from 1.1.x

Start with the [2.0.0 comparison](../Releases/2.0.0.md). Existing classes, native inputs, lazy loading, custom estimators and temporal prequeries remain supported. Back up scientific outputs before refitting: corrected bootstrap, aggregation and gridding can change predictions.

## Keep the existing execution route

Native execution remains the default. You can continue passing DataFrames, DuckDB paths or Parquet paths and keep `temporal_window_prequery=True`. Set the base estimator's own thread count deliberately; ensemble workers and native estimator threads are different controls.

Use the existing classes for classification, regression, hurdle and spherical workflows. Memmap is optional and currently supports built-in numeric planar execution; use native execution for SphereAdaSTEM and unsupported custom overrides.

## Opt into memory-mapped execution

```python
model.fit(
    X_train, y_train,
    temporal_window_prequery=True,
    training_data_backend="memmap",
    inner_max_num_threads=1,
    max_concurrent_ensembles=4,
)
probability, std = model.predict_proba(
    X_test, return_std=True,
    temporal_window_prequery=True,
    prediction_backend="memmap",
    n_jobs=2,
)
```

These worker counts illustrate independent controls, not a recommended allocation for every job. `max_mem` budgets DuckDB buffers, not total RAM. Base estimators can dominate peak memory. `shared`/`bounded` remain accepted aliases; use the clearer `memmap` spelling in new scripts.

Mean/std prediction streams ensemble results by default. Median and separate-ensemble output need matrix storage. With memmap separate output, `prediction_output_path` creates a new `.npy` file owned by the caller; process it in slices rather than copying the whole matrix into RAM. See the executed [memory example](../Examples/10.Execution_and_memory.ipynb).

## Preserve scientific meaning

- For occurrence probabilities call `predict_proba`; classifier `predict` applies a threshold.
- Arithmetic averaging and logit averaging are distinct. Choose explicitly when reproducing an analysis. `logit_agg=True` returns mean probabilities with std measured on logits; sample std uses `ddof=1`.
- Use `completely_random_rotation=False` when you want evenly spaced angles. This does not restore the old canceled-jitter bug. Reusing saved geometry is a separate option through `ensemble_df`.
- Do not treat missing coverage as a zero. Intentional all-zero/all-one fitted stixels keep their constant behavior.
- `persist_prediction_state=True` remains the compatibility default. Set it to `False` only when prediction-time changes to a lazy estimator do not need to be written back.
- Saved models can depend on Python and estimator-library versions. Preserve those environments when loading old models; validate new fits independently.

## Use the new optional components

Database helpers create aligned inputs from familiar DataFrames. Calibration and tuning can be used independently, combined as base estimators or omitted. There is no automatic final global beta map. See [local calibration and tuning](Local_calibration_and_tuning.md); ridge strength is a model choice, not a universal constant.

```python
with model.monitor_execution(log_interval=30.0) as monitor:
    model.fit(X_train, y_train, temporal_window_prequery=True)
    with monitor.phase("held_out_prediction"):
        probability = model.predict_proba(X_test, temporal_window_prequery=True)

report = model.execution_monitor_
```

Install `stemflow[monitoring]` to measure process memory. Final reports persist on the model, but are sampled local process-tree measurements, not a RAM limit or exact scheduler accounting. The [monitor API](../API_Documentation/utils/stemflow.utils.execution_monitor.md) explains shared-page counting and unsupported measurements.
