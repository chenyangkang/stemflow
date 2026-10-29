# stemflow.utils.execution_monitor

Install `stemflow[monitoring]` for psutil-backed process-memory measurements. The usual entry point is `model.monitor_execution(...)`; `ExecutionMonitor` can also wrap a standalone workflow.

```python
with model.monitor_execution(
    sample_interval=1.0,
    log_interval=30.0,
    output_path="data/execution_monitor.json",
    detailed_memory=False,
    max_samples=0,
) as monitor:
    model.fit(X_train, y_train, temporal_window_prequery=True)
    with monitor.phase("held_out_prediction"):
        probability = model.predict_proba(X_test, n_jobs=2)

report = model.execution_monitor_
```

Set `log_interval=None` to disable live logging. A new context replaces the previous model report. Save the model after leaving the context to retain the finalized report; the live thread and locks are not serialized with it. JSON output replaces one report file rather than accumulating snapshots. `max_samples=0` retains summaries only; a positive value bounds the recent raw samples retained.

## Report interpretation

| Field | Meaning |
| --- | --- |
| `wall_seconds` | Elapsed time for the monitored context. |
| `stages` | Inclusive stage timings and sampled stage peaks; nested durations are not additive. |
| `peak_tree_rss_bytes` | Maximum sampled simultaneous sum of parent and descendant RSS on this node. |
| `peak_process_count` | Maximum sampled process count in the monitored tree. |
| `sampling_error`, `output_error` | Measurement/reporting failures, reported independently of model errors. |
| `latest_sample`, `recent_samples` | Latest reading and optionally bounded raw history. |

RSS can count shared pages in multiple processes. With `detailed_memory=True`, supported USS, Linux PSS and macOS physical-footprint readings remain separate fields. Inaccessible/disappearing workers and incomplete coverage are explicit. Sampling can miss short peaks. These values are not GPU memory, remote-node measurements, Slurm/cgroup accounting, an exact allocator breakdown, or a memory limit.

See the [executed monitoring example](../../Examples/10.Execution_and_memory.ipynb) and [measurement guide](../../Tips/Tips_for_parallel_execution.md).

::: stemflow.utils.execution_monitor.ExecutionMonitor
