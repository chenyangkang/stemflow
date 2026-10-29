-------
stemflow version 2.0.0
-------
**Prepared Oct 28, 2026 — `codex-rewrite`**

This major release combines the `community_proj` additions with an execution rewrite, local calibration and tuning, diagnostics, and optional execution monitoring. The merge target is `main` version 1.1.9.

Features:

1. Optional `training_data_backend="memmap"` and `prediction_backend="memmap"` share numeric inputs and bound queued ensemble work. Native Pandas/DuckDB/Parquet execution remains the default. Mean/std aggregation streams completed results; separate-ensemble and median output remain available, including optional file-backed ensemble output.
2. Pandas-friendly `make_duckdb_database`, `make_training_databases`, and `training_databases` helpers preserve numeric dtypes and align targets without manual SQL.
3. Independent calibration and tuning estimators support fixed, tuned, calibrated, and tuned-and-calibrated base models. The local-beta XGBoost recipe exposes spatial/temporal block counts, fold counts, trial counts, ridge strength, and tuning verbosity. Calibration is optional and no final global map is imposed.
4. Held-out calibration diagnostics cover overall, spatial, temporal, space-time, and actual stixel partitions, with support/count information, relative-error floor sensitivity, proper losses, and overlay plots.
5. `model.monitor_execution(...)` retains timing and sampled process-tree memory reports on `model.execution_monitor_`, with optional live logging. Lazy predictions can explicitly skip rewriting unchanged models through `persist_prediction_state=False`; the compatibility default remains `True`.
6. Community-project features include reuse of a supplied `ensemble_df`, additional quadtree criteria, and per-stixel training filters. DuckDB relation access supports the tested 1.x and 2.x APIs (`duckdb>=1.1.3,<3`).
7. Release tooling builds isolated source/wheel distributions, excludes compiled caches and the top-level test package, validates metadata, and checks installed core/extras/source workflows across Python 3.9–3.15 before tag-triggered publication. A conda-forge recipe generator uses the exact published PyPI source checksum.

Correctness and migration:

1. Bootstrap observations retain their actual draw multiplicity; duplicate IDs are not selected repeatedly a second time. Logit aggregation preserves positive probabilities, and named Series targets keep their label alignment.
2. Planar spatial jitter now shifts grid boundaries relative to the observations instead of canceling at the grid origin. Regenerating geometry can change stixel counts and predictions even with the same seed. Existing exported geometry retains its schema and transform meanings; old-geometry reuse is tested. AdaSTEM and STEM rotation defaults now use fully random angles; pass `completely_random_rotation=False` explicitly for the evenly spaced schedule.
3. Existing classification, regression, hurdle, lazy-loading, custom-model, and temporal-window prequery workflows remain supported. SphereAdaSTEM and custom overrides retain native execution where the new planar memmap backend is unsupported. Earlier `shared`/`bounded` backend spellings remain aliases.
4. Floating-point accumulation and corrected bugs can intentionally change numerical output. Memmap ensemble matrices use float64. Revalidate fitted results when regenerating grids or rerunning a previously affected workflow; bitwise agreement with buggy behavior is not promised.

Validation and measured performance:

- At the pre-release implementation head, 212 targeted checks passed in each of six Linux Python/DuckDB combinations. Monitoring and native-allocation checks passed on Ubuntu, macOS, and Windows with Python 3.11/3.13. The release PR runs the full existing test workflow separately.
- Local Apple Silicon whole-job benchmarks against the merged main/community baseline reduced resident-forest wall time from 43.47 to 25.72 seconds and sampled footprint from 2,476 to 1,465 MiB. Already lazy-loaded small forests showed little whole-job change.
- Later native-versus-memmap prediction probes used identical fitted models and four workers. Three million rows with 50 separate ensemble columns took 23.71 versus 5.88 seconds, with sampled summed RSS of 7.64 versus 5.64 GiB. These baselines are identified in the reports; they are not direct timings against untouched `main`.
- These are workload-specific local measurements. Memmap does not cap estimator workspace, eliminate output storage, or guarantee a Slurm allocation. RSS, macOS footprint, and scheduler accounting are different measurements. Monitoring samples can miss brief peaks.

The earlier rare-species experiments support a local-beta regularization trade-off, not a universal ridge or demonstrated copula-correlation accuracy. The maintainer reports the DuckDB workflow incident resolved; the access/cache fixes and compatibility checks do not establish a proven upstream ALPRD decoder repair.

See [execution benchmarks](resources/REWRITE.md), [calibration and profiling](resources/PIPELINE_OPTIMIZATION.md), [prediction stress results](resources/PREDICTION_MEMORY_RESULTS.md), and the added executed examples. Completion evidence for issues #97–#109 is recorded in their issue comments; those issues close through the merge PR.

See the [2.0.0 comparison](docs/Releases/2.0.0.md), [migration guide](docs/Tips/Migrating_to_2.0.md), and [release procedure](resources/RELEASING.md). GitHub, PyPI and conda-forge publication are separate steps.


-------
stemflow version 1.1.6
-------
**Oct, 2025**

Fixed several issues. Fix prediction bug, lazyloading bug; update plotting function; update docs. #82. Also: A previous bug: after getting an attribute of a LazyLoadingEstimator object, the model was not auto-dumped. This is now fixed.


-------
stemflow version 1.1.5
-------
**Oct, 2025**

This is a large update

Features:
1. The major changes are that the `AdaSTEM` class now supports `duckdb` and `parquet` file path as input, this allow the user to pass in large dataset without duplicating the pandas dataframe cross the processors when working with n_jobs>1 parallel computing. See the new Jupyter notebooks for details. #76 
2. The lazy loading is no longer realized by the`LazyLoadingEnsemble` class. Instead, it is realized by `LazyLoadingEstimator`. This allow the model to be dumped once its training/prediction is finished, and we don't need to accumulate the models (hence, memory) until the training is finished for the whole ensemble. This will largely reduce the memory use. See the new Jupyter notebooks for details. #77 
3. n_jobs > ensemble_folds are no longer supported for user-end clarity. Those jobs are paralleled by ensemble folds so n_jobs > ensemble_folds  is meaning less. We do not want to mislead users to think that a 10-ensemble model will be trained faster using n_jobs=20 compared to n_jobs=10.
4. These features will not be available in `SphereAdaSTEM` due to the negligible user market and the negligible advantages. #75 

Major bugs fixed:
1. Previously the models are stored in `self.model_dict` dynamically during the parallel ensemble training process, which means the dictionary is being altered during this process. However, we ask for a `self` as input argument for the ensemble-level training function serialization. This is not ideal since the object being serialized should not be changing. This is fixed by assigning the `model_dict` to `self` after all trainings are finished.
2. Also fixed #74



-------
stemflow version 1.1.3
-------
**May 14, 2025:**

1. ensemble_bootstrap argument: Defaults to False. if True, the data will be bootstrapped once for each ensemble. In this case users can generate ensemble-level uncertainty, accounting for variance in data.
2. joblib_backend argument: Defaults 'loky'. Other available arguments include 'threading' ('multiprocessing' will not work with `generator` as the return). Sometimes only threading may work on certain systems.
3. base_model_method argument: defaults to None. If None, `predict` or `predict_proba` will be used depending on the tasks. This argument is handy if you have a custom base model class that has a special prediction function. Notice that dummy model will still predict 0, so the ensemble-aggregated result is still an average of zeros and your special prediction function output. Therefore, it may only make sense if your special prediction function predicts 0 as the absense/control value. Defaults to None.

Only updated for AdaSTEM and STEM, not for SphereAdaSTEM.

<br>
<br>

**Nov 20, 2024:**

Added support for:

1.  min_class_sample. 

This allows the user to specify the threshold of "not training this base model", for the classification and hurdle tasks. In the past, this is hard coded as 1, meaning that the base model is only trained if there is at least 1 sample from a different class. Now users can set it to, e.g., 3, so that a stixel with 100 data points -- 98 0s and two 1s, will not be trained (instead, a dummy model that always predict zero will be used here), and a stixel will 100 data points -- 97 0s and three 1s will be trained.

This feature can be useful if you need to do cross-validation at base model level.

2. `n_jobs` in the `split` method.

The `split` method now use the user defined n_jobs. It was previously set to 1 since the performance on multi-core seems to be off. However, with large number of ensembles it seems to be doing a good job.

3. Passing arguments to the prediction method of base model.

This can now be realized by passing base_model_prediction_param parameters when you are calling `model.predict` or `model.predict_proba`, as long as the `predict` or `predict_proba` methods of your base model accept this argument.

4. The `logit_agg` parameter.

The `logit_agg` argument in the prediction method will allows "real" probability averaging. Meaning whether to use logit aggregation for the classification task. If True, the model is averaging the probability prediction estimated by all ensembles in logit scale, and then back-tranforms it to probability scale. It's recommended to be jointly used with the CalibratedClassifierCV class in sklearn as a wrapper of the classifier to estimate the calibrated probability. If False, the output is essentially the proportion of "1s" across the related ensembles; e.g., if 100 stixels covers this spatiotemporal points, and 90% of them predict that it is a "1", then the output probability is 0.9; Therefore it would be a probability estimated by the spatiotemporal neighborhood. Default is False, but can be set to truth for "real" probability averaging.
 
Minor changes:
1. The self.rng is now set at call of `fit`, instead of initiation stage.
2. The lazy-loading dir is created upon calling `fit`, instead of initiation stage.
3. Add probability clipping to the prediction output if using `predict_proba` in classification mode. clipping to `1e-6, 1 - 1e-6`.
4. The averaging of the probability for classification task is now on logit scale, and the `mean` prediction in the output is back-transformed to probability scale. However, the std in the output will still be in logit scale!
6. The roc_auc score is now calculated with probability and y_true. Previously a 0.5 threshold was applied to obtain a binary prediction results before calculating auc.
7. Removing "try-except" in the base model training process. If you failed in the base model training, that's a problem.

<br>
<br>


-------
stemflow version 1.1.2
-------
**Oct 25, 2024:**

Related: #59; #69

1. Add a option for completely randomized grids generation (compared to equal division of the 90 degree angle).
2. Implement Lazy-loading model dictionary for saving memory; Save ensmebles of models to disk when finish training, and loaded it when used for prediction
3. Update init parameters in AdaSTEM classes, STEM classes, and SphereAdaSTEM classes. 
4. Update lazy loading documentation & example notebooks.
5. Add related pytests.
