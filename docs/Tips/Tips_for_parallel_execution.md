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

### Stream ensemble means and standard deviations

Built-in mean aggregation now consumes each completed ensemble and updates per-row running moments. It retains counts and means, plus a second-moment array only when `return_std=True`. For `N` rows this uses `12N` accumulator bytes for means or `20N` for means and standard deviations, rather than an `N × ensembles` float64 matrix. These sizes exclude indexes, current worker results, outputs and estimator memory.

```python
mean, std = model.predict_proba(prediction_path, return_std=True,
    temporal_window_prequery=True, streaming=True)
print(model.prediction_execution_info_)
```

The existing sample standard deviation (`ddof=1`), missing-ensemble coverage and logit-space standard deviation convention are retained. Float64 accumulation can differ by rounding from the previous float32 reduction. Parallel completion order can also produce tiny floating-point differences.

Median aggregation and `return_by_separate_ensembles=True` retain the full matrix. Custom `SAC_predict` overrides retain their own route. `streaming=False` explicitly selects the matrix reduction for comparisons.

For estimators with read-only prediction, `fit(..., persist_prediction_state=False)` lets lazy models be loaded and released without rewriting their files. The default is True, preserving custom estimators whose prediction-time mutations must persist. This choice is independent of streaming aggregation. The standalone `LazyLoadingEstimator` offers the corresponding `dump_on_read` option and also defaults to True. Performance comparisons explicitly disable repeated writes for the tested forests/XGBoost models.

### Bound predictor batches within temporal queries

Streaming ensemble aggregation bounds the parent's reduction arrays. It does not
bound a worker's temporal-window covariates: a wide prediction window is otherwise
materialized once in each concurrent worker. For pointwise estimators such as
XGBoost, Random Forests and their local beta wrappers, opt into bounded batches:

```python
pred = model.predict_proba(prediction_path, n_jobs=30,
    temporal_window_prequery=True, streaming=True,
    prediction_batch_size=100_000)
```

Each worker still issues one query per temporal window. DuckDB yields Arrow
record batches and only the current batch becomes a Pandas covariate table;
stixel geometry, probabilities, overlapping-window means and coverage rules are
retained. Pandas and Parquet inputs are also supported. No prediction copy of the
full covariate dataset is saved to scratch. The option is independent of the
training backend, lazy loading and total number of ensembles.

The row limit is per worker, not a total RAM cap. DuckDB caches, fitted models,
ensemble results, indexes and parent output arrays still consume memory. Smaller
batches reduce input memory but can increase model loads and prediction calls;
compare 25,000, 50,000 and 100,000 on the target workload. Keep `None` (the default)
for custom batch-dependent or prediction-stateful estimators, because splitting
their prediction calls can change their behavior. Custom ensemble prediction
overrides, including SphereAdaSTEM, reject this option explicitly rather than
silently ignoring it. Median and separate-ensemble returns still require their
full result matrix even when input batches are bounded.

An isolated, repeatable comparison is available with
`python scripts/00_benchmarks/compare_prediction_batches.py --calibrated`.
It checks predictions, standard deviations and missing-value masks against the
whole-window route, reports sampled prediction memory and time, and removes its
temporary databases, model files and prediction arrays afterward.

### Monitor a running workflow

Install the optional monitoring dependency in the environment that runs the model:

```bash
pip install "stemflow[monitoring]"
# From a checkout of this branch: pip install -e ".[monitoring]"
```

Wrap fitting and prediction in one context. This records the largest observed process-tree total across the complete context, rather than adding separate peaks from workers or steps:

```python
with model.monitor_execution(
    sample_interval=1.0,
    log_interval=30.0,
    output_path="data/execution_monitor.json",
) as monitor:
    model.fit(X_path, y_path, temporal_window_prequery=True)
    with monitor.phase("testing"):
        test_pred = model.predict_proba(X_test_path, temporal_window_prequery=True)
        scores = model.eval_STEM_res("classification", y_test, test_pred[:, 1])
    pred = model.predict_proba(prediction_path, temporal_window_prequery=True)

report = model.execution_monitor_
peak = report["peak_tree_rss_bytes"]
print("Job peak RSS (GiB):", "unavailable" if peak is None else peak / 2**30)
for stage in report["stages"]:
    peak = stage["peak_tree_rss_bytes"]
    print(stage["name"], stage["wall_seconds"],
          "unavailable" if peak is None else peak / 2**30)

# Save after the context to retain the complete report, including prediction.
model.save("data/model.tar.gz", remove_temporary_file=False)
```

The example paths stand for your own training, held-out evaluation and final prediction inputs. Include your own preparation, evaluation or plotting in named `monitor.phase(...)` stages. For independent data preparation or other estimators, use the same context directly with `from stemflow.utils import ExecutionMonitor`.

`model.execution_monitor_` is a plain dictionary containing workflow and stage timings, sampled peaks, parent/child memory, measurement coverage and any errors. It updates after each outer AdaSTEM method finishes and is finalized when the context exits, including on failure. It survives pickling and `model.save`/`AdaSTEM.load`. Save after exiting the context to capture the complete run; a save inside it contains the latest available snapshot. The report is created only when monitoring is enabled. Unmonitored calls leave it intact; a new monitored context replaces it with that context's report. To keep fit and prediction together, use one context as above. You can still call `monitor.summary()` during or after execution.

Live stdout updates are flushed into ordinary Slurm logs. They show elapsed wall time, the current stage, parent RSS, descendants' RSS, and the simultaneous process-tree peak. Built-in stages include input validation, partitioning, training data preparation, training, prediction setup, prediction/aggregation, and cleanup. Feature importance calculation, model saving, `score`, and calibration diagnostics also have stage hooks. Streaming prediction and aggregation overlap, so their main loop is reported as one stage. The matrix route reports ensemble prediction and aggregation separately.

The JSON report updates alongside live logs and on context exit, replacing one file. If the process is killed abruptly, the most recent update can remain with `running=True`; no Python monitor can guarantee a final update after SIGKILL. `log_interval=None` disables logging and writes only the final report. The default retains no raw memory history. Set `max_samples=600` to retain at most the most recent 600 samples; lifetime and stage peaks remain available after old samples are discarded.

The portable reader uses [psutil](https://psutil.io/api/#psutil.Process.memory_info) on macOS (including Apple Silicon), Linux, Windows and other supported Unix systems. A macOS sandbox fallback queries the job's descendants through libproc when global process enumeration is blocked. Read failures retain unavailable values and sample coverage counts rather than inventing zero usage. `rss_complete` and `partial_rss_samples` help identify incomplete readings.

| Measurement | Meaning and limits |
| --- | --- |
| RSS | Resident process memory; Windows working set. Summing processes counts shared pages repeatedly. |
| USS, optional | Private memory on supported systems. Excludes shared pages and is not total job memory. |
| PSS, optional | Linux proportional accounting, which apportions shared pages among processes. |
| Physical footprint, optional | macOS accounting, including compressed memory. Different from RSS, USS and Linux PSS. |

Set `detailed_memory=True` to request USS/PSS and macOS physical footprint when available. Detailed queries can be slower or require permissions; missing readings remain `None`. Each sample includes the number of processes successfully read for each metric and a completeness flag. No extra measurements are substituted for each other. Peaks from different metrics or stages must not be added together. Stage timings are inclusive, so nested stages also overlap.

This is sampled host memory for the calling process and currently discoverable descendants on one node. Readings within each pass are sequential, so totals approximate a snapshot. Brief spikes between samples, orphaned workers and GPU device memory may be missed. It does not measure Slurm/cgroup charges or remote distributed workers. It also cannot separate an estimator's workspace from data copies inside the same process: parent/child memory is a process breakdown, not allocation ownership. Retained arrays/models contribute to every stage where they are resident. Use the detailed profiler below to investigate allocations, and scheduler accounting to choose HPC requests.

Monitoring is off by default and leaves temporal-window prequerying, concurrency, lazy loading and aggregation choices intact. The report includes sampling time and maximum sample gap so that you can assess observation cost. Increase `sample_interval` for long production runs; run expensive allocation tracing separately. A monitor context belongs to the calling thread. Only its plain report is stored on the model; the live sampler, locks and thread are never passed to workers or saved with the model.

To check the sampler with two known native memory allocations on any target machine:

```bash
python scripts/00_benchmarks/check_execution_monitor.py
python -m pytest tests/test_execution_monitor.py -q
```

The quick check retains one small report under `data/00_benchmarks/execution_monitor/` and closes its workers. The dedicated CI workflow runs the checks on macOS, Linux and Windows; configuring this matrix is not evidence of a successful run until CI has executed it.

### Attribute expensive steps

Profiling is opt-in and records compact JSON per parent fit/predict and per ensemble worker:

```bash
export STEMFLOW_PROFILE_DIR="$PWD/data/execution_profiles"
export STEMFLOW_PROFILE_LINES=1
export STEMFLOW_PROFILE_ALLOCATIONS=1
python run_model.py
```

`STEMFLOW_PROFILE_DIR` alone enables function timing with file/line attribution. `STEMFLOW_PROFILE_LINES=1` adds line timings within stemflow, including called work. `STEMFLOW_PROFILE_ALLOCATIONS=1` adds net retained Python allocations by source line and a traced Python peak. Native XGBoost/forest allocations, mapped pages and GPU device memory are not measured by `tracemalloc`. Use the execution monitor or benchmark process-tree sampler for sampled host-memory peaks, and scheduler accounting for HPC charges. Inclusive function/line timings overlap and must not be summed as independent components.

Profiling adds overhead, particularly with line and allocation tracing. Run detailed profiles separately from comparative timing runs. It is intended for serial or process workers; concurrent threads share profiler state. Unset these variables for normal execution. A failed profile write emits a warning while preserving the model call's result or exception.

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
