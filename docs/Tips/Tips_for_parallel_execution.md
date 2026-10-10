# Parallel execution and memory

AdaSTEM can keep all ensembles while limiting how many run at once. Start with DuckDB input, temporal-window prequerying, and lazy loading. Measure the peak of the complete job, including prediction, before increasing the worker count. A large base estimator can dominate memory even when input data are shared.

## Prepare databases from Pandas

The preparation helpers create the `tmp` tables and observation ID columns that AdaSTEM expects. The DataFrame index supplies observation IDs. Pandas labels align by ID, while NumPy or list labels align by position. Inputs must have unique, non-null numeric indexes and numeric predictors without missing values. Encode categorical predictors and handle missing values first.

```python
from stemflow.lazyloading import make_training_databases, make_duckdb_database

X_path, y_path = make_training_databases(X_train, y_train, "data/training")
prediction_path = make_duckdb_database(X_predict, "data/prediction.duckdb")

# Release these only when the rest of your workflow no longer needs them.
del X_train, y_train, X_predict
```

These functions return absolute filenames. Existing files are protected unless `overwrite=True`. They leave the caller's DataFrames unchanged and close the databases before returning. Files remain available for later jobs. Preparing a database also takes time and memory; include that step when comparing complete workflows.

For temporary training input, use the context manager instead:

```python
import os
from stemflow.lazyloading import training_databases

scratch = os.environ.get("SLURM_TMPDIR")
with training_databases(X_train, y_train, directory=scratch) as (X_path, y_path):
    model.fit(X_path, y_path, temporal_window_prequery=True)
```

Keep the context open until every read of these files has finished. Each context owns a separate directory and removes it on success or failure. The helper releases its references to the source DataFrames before yielding; references retained by your own code still consume memory.

## Choose the training backend

| Backend | Behavior | Trade-off |
| --- | --- | --- |
| `native` (default) | Uses Pandas, or read-only DuckDB/Parquet queries; prequerying caches one temporal window per worker | Keeps existing input behavior; wide window caches may be repeated across workers |
| `memmap` (opt-in) | Streams numeric inputs once into temporary NumPy files; workers map them read-only and select stixel rows | Adds preparation time and scratch files; can reduce repeated wide window copies |

```python
model.fit(
    X_path,
    y_path,
    temporal_window_prequery=True,
    training_data_backend="memmap",
    inner_max_num_threads=1,
    max_concurrent_ensembles=4,
)
print(model.execution_info_)
```

The shared backend retains original numeric dtypes, feature order, observation IDs and label alignment. Each estimator receives an ordinary Pandas DataFrame containing its stixel rows. It does not change model complexity, calibration, or the number of ensembles. Estimator preprocessing can still copy these rows. Temporary shared files are removed after training workers finish; they can be larger than the compressed source database. Their location is under the model's temporary working directory, derived from `lazy_loading_dir`.

Shared training currently supports built-in planar AdaSTEM and STEM execution. SphereAdaSTEM and subclasses overriding ensemble training should use `native`; an unsupported shared backend raises an explicit error. Prediction continues to use the original Pandas/DuckDB/Parquet input route for both training backends.

## Control cores without changing the model

`ensemble_fold` controls the statistical ensemble size. `n_jobs` controls concurrent ensemble tasks. `max_concurrent_ensembles`, supplied to `fit`, caps partitioning and training concurrency independently of ensemble size. Prediction has its own `n_jobs` argument:

```python
model.fit(X_path, y_path, n_jobs=8, max_concurrent_ensembles=4,
          inner_max_num_threads=1, temporal_window_prequery=True)
pred = model.predict_proba(prediction_path, n_jobs=2,
                           temporal_window_prequery=True)
```

Set a base estimator's own `n_jobs=1` when parallelizing ensembles. For loky training workers, `inner_max_num_threads=1` also limits supported native math libraries. It cannot control every estimator's custom thread pool. Set appropriate environment limits before starting Python when prediction or custom estimators also use native threads. Joblib describes these controls in its [parallel execution guide](https://joblib.readthedocs.io/en/stable/user_guide/parallel.html).

Training and prediction bound queued ensemble tasks to the worker count, consume completed tasks without waiting for submission order, and restore deterministic output order afterward. Built-in planar prediction sends each worker only its assigned ensemble's fitted models. Custom ensemble or stixel prediction methods retain the full model context for compatibility.

## Account for the entire job

Peak memory can occur during input preparation, partitioning, training or prediction. Training workspace is only one component:

| Component | Practical control |
| --- | --- |
| Source DataFrames retained in the parent | Release unneeded references after database creation |
| Database buffers and temporal windows | Use read-only databases, retain prequerying, profile worker count and window size |
| Stixel copies and estimator workspaces | Limit concurrent fits; keep estimator thread counts explicit |
| Completed fitted models | Use `lazy_loading=True` with suitable scratch space |
| Prediction results | Consider prediction worker count and the size of separate-ensemble output |

`max_mem` configures DuckDB memory; it is **not a RAM limit for an ensemble or the job**. DuckDB's [memory limit](https://duckdb.org/docs/current/operations_manual/limits) does not cover every allocation, and it does not constrain Pandas, XGBoost, Random Forests, calibration wrappers, or joblib. Changing `max_mem` to `50MB` cannot guarantee a 50 MB worker and may increase spilling or query failures.

Lazy loading bounds retained model payload differently from input sharing. A worker still needs enough memory to fit a large stixel, and fitting one ensemble can retain its fitted models until the ensemble is returned. There is no automatic memory admission controller or estimator workspace cap in this implementation.

## Apple Silicon and Linux HPC

Keep `native` as the starting point on both platforms and compare the optional shared backend on the target workload. Memory mapping uses the operating system's file cache; the benefit depends on file layout, storage speed, page sharing and estimator copies. Node-local scratch is usually the first location to test for temporary databases, maps and lazy models. Use a persistent location when the fitted model must survive the job.

The benchmark harness reports summed process RSS and a second platform-specific measure: macOS physical footprint or Linux proportional set size (PSS). Summed RSS counts shared pages repeatedly. PSS apportions shared pages; macOS footprint uses different accounting. Neither is the same as Slurm's cgroup memory limit, which may include additional file-cache and kernel charges. Validate the complete job under the actual scheduler allocation before choosing a production memory request.

An Apple Silicon CPU run does not establish performance on Linux CPUs or CUDA GPUs. GPU estimators have separate device data and workspace allocations; [XGBoost's GPU documentation](https://xgboost.readthedocs.io/en/stable/gpu/index.html) describes those components. Start by profiling one fit per GPU and measure device memory separately before allowing competing fits. Shared host arrays do not automatically share estimator allocations on the GPU. The rewrite does not choose devices or distribute fits across GPUs.

The validated process backend is single-node loky. Existing joblib backend selection remains available, but this change does not introduce or validate multi-node scheduling. Distributed workers would need accessible input and model paths, and shared-array scratch paths would need the same visibility. A local temporary path cannot be assumed to exist on another node.

## Reproduce the comparisons

From the repository root:

```bash
python scripts/00_benchmarks/run_matrix.py
python scripts/00_benchmarks/summarize_results.py
```

Real-data cases expect `X_train.duckdb` and `COMMON%20NAME=Acadian%20Flycatcher/data_0.parquet` in the repository's parent directory. A single comparison supports `--x-database` and `--species-parquet` to select other paths. Synthetic cases require no external data:

```bash
python scripts/00_benchmarks/run_matrix.py --cases synthetic_200000_e4 forest_50000_e8
```

Run with the same dependencies and hardware when comparing variants. `--repeat 3` provides repeated runs; the checked-in results contain one run per case and should be treated as exploratory timings. Each comparison removes temporary input files, maps, fitted models and telemetry, retaining a compact JSON report. The executed [parallel execution example](../Examples/10.Execution_and_memory.ipynb) demonstrates the public API with bundled data.
