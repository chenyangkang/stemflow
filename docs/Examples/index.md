# Examples

The notebook pages retain code, tables and figures. Download a notebook or clone the repository to run it locally. The 2.0.0 examples below use a few ensembles and at most two workers; their timings are illustrative laptop measurements.

## Executed with 2.0.0

| Notebook | Demonstrates | Inputs |
| --- | --- | --- |
| [10. Execution and memory](10.Execution_and_memory.ipynb) | Database helpers, native/memmap training and prediction, mean/std equivalence, separate file-backed ensemble output, and persistent timing/memory reports. | Bundled `docs/mini_data/mini_data.pkl`; no external dataset. |
| [11. Local calibration and profiling](11.Local_calibration_and_profiling.ipynb) | Within-stixel blocked calibration, fixed/tuned recipe choices, held-out diagnostics, and native/memmap prediction equivalence. | Small seeded synthetic observations with known marginal probabilities. |

These notebooks retain fresh executed outputs, the stemflow/Python versions, and execution provenance. The first checks execution equivalence; the second illustrates calibration APIs and sparse-bin diagnostics. Neither establishes universally better calibration or accurate copula inference.

To reproduce from the repository root after installing the package and notebook tools:

```bash
python -m pip install -e ".[monitoring,tuning]" nbformat nbclient ipykernel matplotlib-inline
python scripts/01_documentation/execute_notebook.py docs/Examples/10.Execution_and_memory.ipynb --output docs/Examples/10.Execution_and_memory.ipynb --in-process
python scripts/01_documentation/execute_notebook.py docs/Examples/11.Local_calibration_and_profiling.ipynb --output docs/Examples/11.Local_calibration_and_profiling.ipynb --in-process
```

Omit `--in-process` to use a registered Jupyter kernel. Temporary databases/maps/models are cleaned up; tables and figures are embedded in the saved notebooks. Arial is used when installed; a message explicitly flags its absence on other systems.

## Retained historical examples

Notebooks 01–09 keep their original code and saved outputs. They have not been rerun for 2.0.0. Several contain unexecuted cells; their displayed outputs are not evidence of current-version results. Most require `Sample_data_Mallard.csv`; the main demo also uses `Predset_2020.csv`, and some comparisons require additional estimators or previously generated metric files. Those external inputs are not bundled here.

| Notebook | Topic |
| --- | --- |
| [01](01.AdaSTEM_demo.ipynb) | AdaSTEM demo |
| [02](02.AdaSTEM_learning_curve_analysis.ipynb) | Learning-curve analysis |
| [03](03.Binding_with_Maxent.ipynb) | Maxent base model |
| [04](04.SphereAdaSTEM_demo.ipynb) | Spherical indexing |
| [05](05.Hurdle_in_ada_or_ada_in_hurdle.ipynb) | Hurdle-model composition |
| [06](06.Base_model_choices.ipynb) | Base-model comparisons |
| [07](07.Optimizing_stixel_size.ipynb) | Stixel-size optimization |
| [08](08.Lazy_loading.ipynb) | Lazy loading |
| [09](09.Database_query.ipynb) | Original manual DuckDB/Parquet preparation |

When rerunning, consult the [2.0.0 execution options](../API_Documentation/model/stemflow.model.AdaSTEM.md). Corrected bootstrap, logit aggregation, spatial jitter and rotation defaults can change results. Existing saved geometry retains its documented compatibility; new fits should be evaluated again rather than compared to old figures as if only runtime changed.
