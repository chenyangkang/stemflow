# Memory-mapped prediction results

Measured locally on 2026-10-11, Apple Silicon, 36 GiB RAM, on `codex-rewrite`
based on commit `271017a`. This report covers the memory-mapped prediction
implementation introduced after that baseline. The earlier
`PREDICTION_STRESS_RESULTS.txt` describes the preceding prototype stage.

## What changed

The public option is now named `prediction_backend="memmap"`. Historical result
folders and strategy IDs retain their original `bounded` spelling to preserve
the recorded measurements and source hashes. The earlier local spellings
`"shared"` and `"bounded"` are accepted as aliases. The implementation lives in
`stemflow/lazyloading/shared_prediction.py`.

Training uses the same public value: `training_data_backend="memmap"`. The local
training value `"shared"` remains an alias. These names select the existing
implementations; the naming change does not change the measured algorithms.

After unifying the names, 72 training, data-preparation and prediction checks
passed. They include alias compatibility, bootstrap row and label alignment,
separate ensemble output, cleanup and saved monitoring reports. A small
four-strategy launcher check also passed for native, memmap processes, memmap
threads and memmap file output; it is a correctness check, not a timing benchmark.

`predict_proba(..., prediction_backend="memmap")` selects a new execution
path. The default remains `native`. Training, hyperparameter tuning,
calibration, partition construction, coverage thresholds and ensemble counts
are unchanged.

1. Project required numeric columns and stream them once from Pandas, DuckDB
   or Parquet into temporary read-only NumPy maps. This avoids repeated wide
   temporal-window DataFrames in each ensemble process. DuckDB remains the
   source database.
2. Select the same temporal windows, rotations and half-open stixel bounds
   using source-row positions. Estimators receive ordinary DataFrames with
   their full stixel rows in one call. There is no estimator row batching.
3. Accumulate stixel outputs directly into numeric sums and counts, avoiding
   per-stixel index lists, prediction DataFrames, concatenation and groupby.
4. Give each worker one reusable result file. Return a small completion
   record, consume the complete result in the parent, then reuse that slot.
   At most `n_jobs` tasks/results are outstanding. This prevents a backlog of
   large completed ensemble results in process queues.
5. Stream means and sample standard deviations, or fill a single preallocated
   observation-by-ensemble matrix. An optional `.npy` output returns a writable
   NumPy memmap when every ensemble must be retained.
6. Gather exact scikit-learn Random Forest classes directly into float32,
   matching their own prediction input conversion and avoiding an additional
   full stixel copy. Other estimators retain original numeric dtypes.

Each task owns its map handles, including with threads. Lazy models retain
`persist_prediction_state` semantics. Temporary input maps and result slots
are removed after workers stop, including on exceptions. Existing requested
output files cannot be overwritten. Failed calls remove partial output files.

This is a single-node numeric planar backend, currently supporting `loky` and
`threading`. SphereAdaSTEM and custom ensemble/stixel overrides retain the
native route. It is not a cap on estimator workspace or operating-system RAM.

## Measured comparison

All rows below use four workers, identical fitted models and input rows, lazy
loading, and `temporal_window_prequery=True`. Times include fresh input-cache
preparation and aggregation, but exclude training, interpreter startup and
writing the harness's verification files. Two repetitions use reversed
strategy order; the other rows are exploratory single runs.

| Workload | Repetitions | Native time | Shared process time | Native summed RSS | Bounded summed RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3M rows, 50 ensembles, separate output | 1 | 23.71 s | 5.88 s | 7.64 GiB | 5.64 GiB |
| 1M rows, 50 ensembles, tuned XGBoost + local beta | 2 | 28.27 s | 19.87 s | 7.93 GiB | 5.66 GiB |
| 1M rows, 20 ensembles, large forests | 1 | 62.11 s | 49.01 s | 6.96 GiB | 5.19 GiB |
| 300k rows, 12 ensembles, 6,796 fitted stixels | 2 | 6.10 s | 5.23 s | 2.71 GiB | 1.80 GiB |

The process backend reduced sampled summed RSS by about 25–33% in these
comparisons, while prediction time fell about 14–75%. macOS physical footprint
also fell, but measures different memory from summed RSS. Both are reported
in the accompanying CSV and plot.

Explicit threading is useful for some workloads, with a clear trade-off:

| Workload | Shared thread time | Shared thread summed RSS | Trade-off against native |
| --- | ---: | ---: | --- |
| Tuned XGBoost + local beta | 19.26 s | 3.38 GiB | 57% less RSS; 32% less time |
| Large forests | 48.74 s | 2.61 GiB | 63% less RSS; 22% less time |
| 6,796 stixels | 11.02 s | 0.63 GiB | 77% less RSS; 81% more time |

Processes remain the default. Small stixels make Python scheduling and
DataFrame work important, so threads must not be selected universally.

The forest fixture has 40 fitted forests, 6,000 trees and 480,487,336 serialized
model bytes (458 MiB). Gathering directly into float32 reduced shared-process
RSS from 6.93 to 5.19 GiB and time from 54.37 to 49.01 s in a paired exploratory
comparison. This avoids the original mixed-dtype frame followed by the
forest's float32 conversion.

The tuned fixture has 100 fitted stixels, 300 final XGBoost estimators,
18,000 retained trees and 900 trial metadata records. Its models total
28,996,844 serialized bytes. Calibration intentionally retains three fold
models per stixel; the tuner does not retain all trial estimators. Its largest
temporal window contains 750,045 prediction rows.

For 3M rows and 50 ensembles, explicit file output took 6.33 s versus 5.88 s
for the shared in-memory matrix. Summed RSS was similar, 5.63 versus
5.64 GiB: memory mapping does not make resident file pages free. The output
remains usable after the call without loading the whole array, and parent
array allocation is avoided.

Shared-backend matrices are float64. The raw XGBoost native separate result here is
float32: its final matrix is about 0.56 GiB, versus 1.12 GiB for shared output.
Peak memory still decreased because intermediate frames, serialization and
result retention decreased. Complete separate output necessarily needs
storage proportional to rows times ensembles; mean/std output does not.

## Correctness and scope of evidence

Every successful comparison checks the entire output in chunked verification
blocks, including identical missing-value masks and standard deviations.
Tuned local-beta and forest outputs match exactly, apart from floating-point
reduction noise in some standard deviations. Raw XGBoost differences in these
trials are below 2e-8, caused by float64 accumulation versus native float32
within-ensemble reductions. The stress harness records strict 1e-12 and
relaxed 1e-6 checks separately; a relaxed pass is not presented as bit equality.

Targeted compatibility tests cover Pandas/DuckDB/Parquet, shuffled observation
IDs, overlapping temporal windows, missing coverage, mean/std, median, logit
aggregation, regression, custom base prediction methods, dummy zero models,
separate output, writable file ownership, failure cleanup, lazy prediction
state, thread map ownership, monitoring and optional process-worker profiling.

The local sampler observes the simultaneous process tree, not the sum of
separate per-process peaks. RSS counts shared pages in multiple processes;
macOS footprint excludes much shared file cache. Neither is Linux PSS nor a
Slurm/cgroup charge. These measurements do not establish a safe HPC memory
request. Native estimator, GPU and file-cache allocations are not an additive
partition of the array/model byte counts. Component maxima can occur at
different times and must not be added together.

Large individual stixels and base-model workspaces can still dominate memory.
This implementation reduces repeated input and output overhead; it does not
add automatic memory admission control or change base-model complexity.
Linux, Windows, GPU and distributed performance have not been tested here.

## Use and reproduce

```python
mean, std = model.predict_proba(
    prediction_path,
    n_jobs=4,
    temporal_window_prequery=True,
    prediction_backend="memmap",
    return_std=True,
)
print(model.prediction_execution_info_)
```

Keep the existing `return_by_separate_ensembles=True` parameter when needed.
Add `prediction_output_path="data/ensemble_predictions.npy"` to keep that
matrix in a file. Omit the path to receive an ordinary ndarray. To test
threads explicitly, set `model.joblib_backend="threading"` before prediction.

The detailed API, memory costs and monitor example are in
`docs/Tips/Tips_for_parallel_execution.md`. Temporary files live under the
model's scratch location. The structural report remains on
`model.prediction_execution_info_`; a monitor context also records the shared
input preparation and prediction/aggregation stages on the model.

From the repository root:

```bash
python scripts/06_prediction_stress/run_stress.py \
  --cases combined_tuned,many_stixels \
  --strategies baseline,memmap,memmap_threads --jobs 4 --repeat 2 \
  --memory-gib 6.5 --timeout 1200 \
  --output data/06_prediction_stress/bounded_repeated_v2

python scripts/06_prediction_stress/summarize_memory.py
python -m pytest tests/test_shared_prediction.py \
  tests/test_prediction_streaming.py tests/test_prediction_stress_prototypes.py -q
```

The sampler needs `psutil`; tuned tests also need Optuna. Runtime versions here
were Python 3.13.15, NumPy 2.5.1, pandas 3.0.5, DuckDB 1.5.5,
scikit-learn 1.9.0 and XGBoost 3.3.0. Raw JSON includes source hashes in the
later trials. Final edits add independent task map ownership and an optional
profiling decorator; both leave unprofiled prediction arithmetic unchanged.
Results retain compact metadata, not models, full predictions or databases.

The targeted suite passed 47 tests, followed by a clean 15-test shared-backend
rerun after the failure-shutdown fix. One additional incomplete-grid check
also passed, for 48 distinct targeted tests. There are 36 additional successful
prediction stress trials in the shared-backend experiment folders, beyond
the earlier prototype tests. No local memory guard was triggered in these
36 trials. The test suite also verifies that monitored reports survive model
pickling and that optional profiling produces reports from process workers.
Forced worker exceptions now use graceful shutdown before deleting mapped
files. This avoids a sandbox failure in forced process-tree termination and
is verified to leave no additional multiprocessing children. An incomplete
ensemble grid is rejected for matrix output rather than returning unfilled
columns. Existing mean aggregation can still use the available ensembles.

A separate fit-and-predict-in-one-job check used 3M prediction rows and 50
ensemble columns. Training took about 3 s in both runs; prediction took
27.77 versus 6.69 s, and complete child wall time was 34.81 versus 14.16 s.
The complete-job summed RSS peak fell from 6.47 to 5.76 GiB (11%), while
prediction footprint fell from 4.25 to 2.16 GiB. Both jobs peaked during
prediction. This single run overlapped small compatibility tests, so it is
supporting evidence rather than a replacement for the isolated comparisons.
The smaller RSS reduction also shows why phase and whole-job accounting must
be measured, rather than extrapolated from a file-map size or one trial.

The supplied HPC launcher preserves the user's 30-core, 80-GiB, combined
stdout/stderr species-style logging structure. Copy this checkout's changed
files to the server first; the remote branch does not yet include them.
Submit from the checkout root:

```bash
sbatch scripts/06_prediction_stress/02.Prediction_memory.slurm \
  result_pressure 4,8,16,30 3000000 separate 50 2

sbatch scripts/06_prediction_stress/02.Prediction_memory.slurm \
  combined_tuned 4,8,16,30 1000000 mean 50 2
```

The arguments are case, worker counts, prediction rows, output mode, ensemble
count and repetitions. The launcher uses the previously supplied absolute
`calc_asso` Python path. Logs go to `log_files/prediction_memory_CASE_JOBID.log`;
compact results go to `data/06_prediction_stress/hpc_memory_CASE_JOBID`.
Its sampled guard is not a hard allocation limit. No HPC job was submitted.

## References for execution behavior

- [Joblib generator memory behavior](https://joblib.readthedocs.io/en/latest/auto_examples/parallel_generator.html)
- [Joblib shared memory and thread controls](https://joblib.readthedocs.io/en/latest/user_guide/parallel.html)
- [Loky executor API](https://loky.readthedocs.io/en/stable/API.html)
- [NumPy file-backed arrays](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html)
