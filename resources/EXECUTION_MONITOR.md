# Execution monitor validation

The optional `ExecutionMonitor` records inclusive wall timings and sampled host memory for the parent process and its descendants. It spans fitting, evaluation and prediction without changing prequerying, worker counts, lazy loading or aggregation. Normal modeling does not require psutil; monitoring uses the optional `monitoring` dependency extra.

Each monitored AdaSTEM model retains the latest context's plain report as `execution_monitor_`. It is refreshed after outer model calls and on context exit, survives pickle and lazy model save/load, and includes failed stages. The live sampler is held outside the model. Logging can be disabled with `log_interval=None`; the report remains available.

## Checks run locally

Platform: Apple Silicon, macOS 15.7.7; psutil 7.2.2.

- Python 3.13: 166 checks passed across execution monitoring, execution rewrite, streaming prediction, profiling, calibration, data preparation, gridding and benchmark memory sampling.
- Python 3.13, after adding persistent model reports: 32 monitoring/profiling checks passed, including pickle, lazy save/load, report replacement and failed-fit persistence.
- Python 3.14: the same 32 monitoring/profiling checks passed, including persistent reports and denied enumeration.
- Parallel prediction comparisons cover Pandas, DuckDB and Parquet inputs, native and shared training, lazy models, bootstrapping, mean/std streaming and matrix aggregation. Enabled and disabled monitoring produced matching outputs and ensemble metadata.
- A native-allocation test detected both a child and grandchild holding 80 MiB between them. The portable validation script separately detected two workers each holding a 32 MiB anonymous mapping, including touched resident pages.
- Failure checks preserve original exceptions, stop the sampler, handle inaccessible/exited workers, retain missing readings, and tolerate report write failures. Trace retention is bounded.
- After removing the new monitoring hooks, the AdaSTEM and SphereAdaSTEM abstract syntax trees match the committed model logic. The hooks do not change model operations or their ordering.

The macOS sandbox blocks psutil's global process listing and USS reads. Tests exercised the libproc descendant-query fallback and obtained RSS and macOS physical footprint; inaccessible USS and unsupported Linux PSS remained unavailable.

## Quick sampling measurement

Run `python scripts/00_benchmarks/check_execution_monitor.py` after installing the monitoring extra. It retains one report at [data/00_benchmarks/execution_monitor/validation.json](../data/00_benchmarks/execution_monitor/validation.json).

In the saved local run, two workers held 64 MiB of deliberate native allocations. The monitor observed 82.16 MiB of child RSS, including interpreter overhead, and retained the peak after both workers exited. All RSS samples were complete.

| Reader | Median time per reading, three processes | Maximum |
| --- | ---: | ---: |
| RSS only | 0.100 ms | 0.221 ms |
| Detailed | 0.156 ms | 0.276 ms |

These are 20-read microbenchmarks on this Mac. They are not measured whole-model slowdowns or estimates for 30 HPC workers. Detailed queries were affected by this sandbox's permissions and can be more expensive on other systems.

## Native platforms still to validate

Linux and Windows were not available locally. The dedicated [CI matrix](../.github/workflows/execution_monitor.yml) runs portable sampling, known native allocations and parallel AdaSTEM comparisons on Ubuntu, macOS and Windows with Python 3.11 and 3.13, and uploads the small validation reports. It has been configured locally but has not run remotely. Capability tests exercise both older and newer psutil detail APIs; the installed psutil tested here is 7.2.2.

RSS, USS, PSS and physical footprint have different meanings. Readings are sequential snapshots and may miss brief peaks. Process memory does not establish scheduler/cgroup charges, GPU memory, remote-node memory or the division between data copies and estimator workspace. See the [usage and interpretation guide](../docs/Tips/Tips_for_parallel_execution.md#monitor-a-running-workflow).
