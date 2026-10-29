stemflow 2.0.0 combines the community-project features with an execution rewrite, modular local calibration/tuning and persistent resource monitoring. The immediate code baseline is 1.1.9; the cumulative upgrade guide also distinguishes changes already present since the previous 1.1.6 release.

### Added and improved

- Optional numeric memmap training and prediction, bounded ensemble work and streaming mean/std aggregation. Native Pandas/DuckDB/Parquet execution remains the default.
- Independent prediction workers, smaller ensemble model payloads and optional file-backed separate-ensemble output.
- DataFrame-friendly DuckDB preparation, target alignment and temporary-input cleanup.
- Composable blocked out-of-fold calibration and tuning, a local beta/XGBoost recipe, and overall/spatial/temporal/space-time/stixel diagnostics.
- Optional live timing/memory logs and reports retained on `model.execution_monitor_`.
- Supplied ensemble geometry, additional quadtree criteria and per-stixel training filters from the community branch.
- Tested DuckDB 1.x/2.x API adapters; installation extras for monitoring and tuning.

### Correctness and migration

Bootstrap observations retain their actual sampling multiplicity. Logit aggregation, named-target alignment, temporal caches and planar spatial jitter have corrected paths. AdaSTEM/STEM now default to fully random rotation; pass `completely_random_rotation=False` for evenly spaced angles. Existing exported geometry retains its schema and tested reuse behavior.

These corrections can intentionally change numerical outputs when refitting. Constant zero/one stixels, missing coverage, lazy loading and temporal prequeries remain supported. SphereAdaSTEM/custom overrides retain native execution where the planar memmap route is unsupported. Earlier backend spellings remain aliases.

### Performance and validation

Local resident-forest whole-job measurements against the merged main/community baseline changed from 43.47 to 25.72 seconds and 2,476 to 1,465 MiB sampled macOS footprint. A later identical-model prediction probe with 3M rows and 50 ensemble columns changed from 23.71 to 5.88 seconds and 7.64 to 5.64 GiB summed RSS. Other workloads improved less. These are identified local baselines, not universal speedups or HPC memory guarantees.

The release workflow builds the source archive and wheel in isolation, checks metadata and contents, and tests core/extras plus source installation on Python 3.9–3.15 before publication. Full pytest covers Python 3.11–3.15; monitoring additionally runs across Linux, macOS and Windows. See [PR #110](https://github.com/chenyangkang/stemflow/pull/110) for revision-specific evidence, resolved issues and limitations.

### Install and read more

```bash
python -m pip install --upgrade 'stemflow==2.0.0'
python -m pip install --upgrade 'stemflow[monitoring,tuning]==2.0.0'
```

Conda-forge updates separately. Read the [version comparison](https://chenyangkang.github.io/stemflow/Releases/2.0.0.html), [migration guide](https://chenyangkang.github.io/stemflow/Tips/Migrating_to_2.0.html), [execution notebook](https://chenyangkang.github.io/stemflow/Examples/10.Execution_and_memory.html) and [calibration notebook](https://chenyangkang.github.io/stemflow/Examples/11.Local_calibration_and_profiling.html). The two current notebooks retain executed outputs; older notebooks retain historical results requiring external data.
