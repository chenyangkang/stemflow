"""Compare the merged baseline and rewrite in isolated Python processes.

Example: python scripts/00_benchmarks/compare_execution.py --source real --rows 30000
Temporary data, baseline sources, model files, and predictions are cleaned up.
"""

import argparse
from contextlib import contextmanager
import ctypes
import io
import importlib.util
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]


def process_memory(root_pid, details=False):
    """Return simultaneous tree RSS and footprint/PSS, without optional packages."""
    if sys.platform == "darwin":
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        def children(pid):
            buffer = (ctypes.c_int * 4096)()
            n = lib.proc_listchildpids(pid, buffer, ctypes.sizeof(buffer))
            # Unlike proc_listpids, proc_listchildpids already returns a PID count.
            return list(buffer)[:max(0, n)]
        def memory(pid):
            # rusage_info_v2: 16-byte UUID followed by uint64 fields.
            buffer = (ctypes.c_uint64 * 20)()
            ok = lib.proc_pid_rusage(pid, 2, ctypes.byref(buffer))
            return (int(buffer[8]), int(buffer[9])) if ok == 0 else (0, 0)
    else:
        def children(pid):
            try:
                return list({child for path in Path(f"/proc/{pid}/task").glob("*/children")
                             for child in map(int, path.read_text().split())})
            except OSError:
                return []
        def memory(pid):
            try:
                values = {}
                for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                    parts = line.split()
                    if parts[0] in {"Rss:", "Pss:"}:
                        values[parts[0]] = int(parts[1]) * 1024
                return values.get("Rss:", 0), values.get("Pss:", 0)
            except OSError:
                return 0, 0
    pending, visited, rss, accounted, processes = [root_pid], set(), 0, 0, {}
    while pending:
        pid = pending.pop()
        if pid in visited or pid <= 0:
            continue
        visited.add(pid)
        r, a = memory(pid)
        processes[pid] = (r, a)
        rss += r
        accounted += a
        pending.extend(children(pid))
    return (rss, accounted, processes) if details else (rss, accounted)


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


class PhaseRecorder:
    """Small phase markers; the independent parent samples memory concurrently."""

    def __init__(self, directory):
        self.directory = directory
        self.name = None
        self.started = None
        self.seconds = {}

    def begin(self, name):
        now = time.perf_counter()
        if self.name is not None:
            self.seconds[self.name] = self.seconds.get(self.name, 0) + now - self.started
        self.name, self.started = name, now
        write_json(self.directory / "phase.json", {"phase": name})

    @contextmanager
    def measure(self, name):
        previous = self.name
        self.begin(name)
        try:
            yield
        finally:
            self.begin(previous or "idle")


def build_fixture(args, target):
    import duckdb
    import numpy as np
    import pandas as pd
    if args.source == "real":
        config = {"threads": "1", "memory_limit": "500MB", "temp_directory": str(target / "spill")}
        with duckdb.connect(config=config) as con:
            source = str(args.x_database or args.data_dir / "X_train.duckdb").replace("'", "''")
            species = str(args.species_parquet or args.data_dir / f"COMMON%20NAME={args.species}" / "data_0.parquet")
            con.execute(f"ATTACH '{source}' AS input (READ_ONLY)")
            names = [row[0] for row in con.sql("DESCRIBE input.tmp").fetchall()]
            types = dict((r[0], r[1]) for r in con.sql("DESCRIBE input.tmp").fetchall())
            features = [c for c in names if types[c] not in {"VARCHAR", "BLOB"}
                        and c != "__index_level_0__"]
            selected = ', '.join('a."' + c.replace('"', '""') + '"' for c in features)
            presences = con.read_parquet(species).project('"SAMPLING EVENT IDENTIFIER"').distinct()
            con.register("presences", presences)
            frame = con.sql(f"""
                SELECT {selected}, CASE WHEN b."SAMPLING EVENT IDENTIFIER" IS NULL THEN 0 ELSE 1 END AS occurrence
                FROM input.tmp a LEFT JOIN presences b USING ("SAMPLING EVENT IDENTIFIER")
                WHERE hash(a.__index_level_0__) % 1000 < {args.sample_per_mille}
                  AND a.DOY >= 121 AND a.DOY < 201
                ORDER BY hash(a.__index_level_0__) LIMIT {args.rows}
            """).df()
        if len(frame) < args.rows:
            raise ValueError(f"Only {len(frame)} rows selected; request a smaller benchmark.")
    else:
        rng = np.random.default_rng(args.seed)
        frame = pd.DataFrame(rng.normal(size=(args.rows, args.features)),
                             columns=[f"feature_{i}" for i in range(args.features)])
        frame["longitude"] = rng.uniform(-120, -60, args.rows)
        frame["latitude"] = rng.uniform(10, 55, args.rows)
        frame["DOY"] = rng.uniform(1, 81, args.rows)
        p = 1 / (1 + np.exp(-(-3 + frame.feature_0 - 0.5 * frame.feature_1)))
        frame["occurrence"] = rng.binomial(1, p)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    frame.to_parquet(target / "fixture.parquet", index=False)
    return {"rows": len(frame), "features": len(frame.columns) - 1,
            "detections": int(frame.occurrence.sum())}


def worker(args):
    phases = PhaseRecorder(args.work)
    phases.begin("imports")
    worker_started = time.perf_counter()
    import duckdb
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    from xgboost import XGBClassifier
    from stemflow.model.AdaSTEM import AdaSTEMClassifier
    from sklearn.ensemble import RandomForestClassifier
    import gc
    phases.begin("input_preparation")
    telemetry = args.work / f"{args.variant}_telemetry"
    telemetry.mkdir()

    class ProfiledAdaSTEMClassifier(AdaSTEMClassifier):
        def stixel_fitting(self, stixel, stixel_filter_func=None):
            if not args.detail:
                return super().stixel_fitting(stixel, stixel_filter_func)
            key = f"{os.getpid()}_{int(stixel.ensemble_index.iloc[0])}"
            stats = getattr(self, "_benchmark_fits", {})
            record = stats.setdefault(key, {"fit_calls": 0, "fit_seconds_sum": 0,
                                           "max_stixel_dataframe_bytes": 0,
                                           "max_fit_rss_increase_bytes": 0,
                                           "max_fit_accounted_increase_bytes": 0})
            self._benchmark_fits = stats
            payload = int(stixel.memory_usage(index=True, deep=True).sum())
            before = process_memory(os.getpid())
            state = telemetry / f"active_{os.getpid()}.json"
            write_json(state, {"pid": os.getpid(), "active": True, "rss_before": before[0],
                               "accounted_before": before[1], "stixel_bytes": payload})
            started = time.perf_counter()
            try:
                result = super().stixel_fitting(stixel, stixel_filter_func)
                return result
            finally:
                elapsed = time.perf_counter() - started
                after = process_memory(os.getpid())
                record["fit_calls"] += 1
                record["fit_seconds_sum"] += elapsed
                record["max_stixel_dataframe_bytes"] = max(record["max_stixel_dataframe_bytes"], payload)
                record["max_fit_rss_increase_bytes"] = max(record["max_fit_rss_increase_bytes"], after[0] - before[0])
                record["max_fit_accounted_increase_bytes"] = max(record["max_fit_accounted_increase_bytes"], after[1] - before[1])
                write_json(telemetry / f"fits_{key}.json", record)
                write_json(state, {"pid": os.getpid(), "active": False})

        def SAC_training(self, *positional, **keywords):
            phases.begin("training")
            result = super().SAC_training(*positional, **keywords)
            phases.begin("fit_finalize")
            return result

    if args.variant != "baseline":
        from stemflow.lazyloading.training_data import SharedTrainingData
        original_build = SharedTrainingData.build.__func__
        def measured_build(cls, *positional, **keywords):
            with phases.measure("shared_input_preparation"):
                return original_build(cls, *positional, **keywords)
        SharedTrainingData.build = classmethod(measured_build)

    frame = pd.read_parquet(args.work / "fixture.parquet")
    rng = np.random.default_rng(args.seed)
    held = rng.random(len(frame)) < .2
    X = frame.drop(columns="occurrence")
    if args.base_model in {"local_beta", "tuned_local_beta"}:
        X["longitude_tmp"], X["latitude_tmp"] = X.longitude, X.latitude
    X.index = np.arange(len(X)) * 3 + 10
    y = pd.DataFrame({"occ": frame.occurrence.to_numpy()}, index=X.index)
    heldout_X = X.loc[held].copy()
    heldout_y = y.loc[held, "occ"].to_numpy().copy()
    input_bytes = int(X.memory_usage(index=True).sum() + y.memory_usage(index=True).sum())
    with tempfile.TemporaryDirectory(dir=args.work) as temporary:
        temporary = Path(temporary)
        train = X.loc[~held].copy()
        labels = y.loc[~held].copy()
        train_times = train.DOY.to_numpy().copy()
        bytes_per_row = sum(dtype.itemsize for dtype in train.dtypes) + 16
        temporal_start, temporal_end = float(X.DOY.min()), float(X.DOY.max()) + 1
        use_helper = args.variant != "baseline" and args.input == "duckdb"
        if use_helper:
            from stemflow.lazyloading import make_training_databases, make_duckdb_database
            make_training_databases(train, labels, temporary)
            make_duckdb_database(X, temporary / "prediction.duckdb")
        elif args.input != "pandas":
            for name, data in [("X", train), ("y", labels), ("prediction", X)]:
                stored = data.copy()
                stored["__index_level_0__"] = stored.index
                if args.input == "duckdb":
                    with duckdb.connect(str(temporary / f"{name}.duckdb"),
                                        config={"threads": "1", "memory_limit": "500MB"}) as con:
                        con.execute("CREATE TABLE tmp AS SELECT * FROM stored")
                else:
                    stored.to_parquet(temporary / f"{name}.parquet", index=False)
            del stored, data
        xs, ys = ((train, labels) if args.input == "pandas" else
                  (str(temporary / f"X.{args.input}"), str(temporary / f"y.{args.input}")))
        prediction_source = X if args.input == "pandas" else str(temporary / f"prediction.{args.input}")
        estimator = (XGBClassifier(n_estimators=args.trees, max_depth=args.depth,
            n_jobs=args.threads, tree_method="hist", random_state=args.seed) if args.base_model == "xgboost"
            else RandomForestClassifier(n_estimators=args.trees, max_depth=args.depth,
                                        n_jobs=args.threads, random_state=args.seed))
        if args.base_model in {"local_beta", "tuned_local_beta"}:
            options = dict(tuning=args.base_model == "tuned_local_beta", ridge=100.,
                n_trials=args.tuning_trials, n_estimators=args.trees,
                tuning_n_estimators=args.tuning_trees, random_state=args.seed)
            if args.variant == "baseline":
                if args.calibration_helper is None:
                    raise ValueError("A supplied --calibration-helper is required for the baseline beta strategy.")
                import importlib
                sys.path.insert(0, str(args.calibration_helper.parent))
                module = importlib.import_module("reference_calibration")
                estimator = module.TunedLocalBetaXGBClassifier(**options)
            else:
                from stemflow.tuning import local_beta_xgboost
                estimator = local_beta_xgboost(**options, Spatio1="longitude_tmp", Spatio2="latitude_tmp")
        model = ProfiledAdaSTEMClassifier(base_model=estimator,
            ensemble_fold=args.ensembles, min_ensemble_required=1,
            grid_len_upper_threshold=25, grid_len_lower_threshold=5,
            points_lower_threshold=100, temporal_start=temporal_start, temporal_end=temporal_end,
            temporal_step=40, temporal_bin_interval=40, temporal_bin_start_jitter=0,
            n_jobs=args.jobs, sample_weights_for_classifier=False,
            save_gridding_plot=False, random_state=args.seed, max_mem="500MB",
            lazy_loading=not args.resident_models, lazy_loading_dir=str(temporary / "models"), verbosity=0)
        model.subset_x_names = args.subset_x_names
        del frame, y
        if args.input != "pandas":
            del train, labels, X
        gc.collect()
        extra = {} if args.variant == "baseline" else {"training_data_backend": args.backend,
                                                       "inner_max_num_threads": args.threads,
                                                       "persist_prediction_state": False}
        phases.begin("partitioning")
        model.split(xs)
        grid = model.ensemble_df
        largest_window_payload = max(int(((train_times >= start) & (train_times < start + 40)).sum())
            * bytes_per_row for start in grid.DOY_start.unique())
        phases.begin("fit_preparation")
        model.fit(xs, ys, ensemble_df=grid, temporal_window_prequery=True, **extra)
        model_files = [p.stat().st_size for p in (temporary / "models").rglob("*.pkl")]
        phases.begin("heldout_prediction")
        prediction = model.predict_proba(heldout_X, temporal_window_prequery=True)[:, 1]
        phases.begin("evaluation")
        valid = np.isfinite(prediction)
        true = heldout_y[valid]
        p = prediction[valid]
        scores = dict(coverage=float(valid.mean()), auc=float(roc_auc_score(true, p)),
                      brier=float(brier_score_loss(true, p)), log_loss=float(log_loss(true, p)))
        phases.begin("prediction")
        full_prediction = model.predict_proba(prediction_source, temporal_window_prequery=True)[:, 1]
        phases.begin("finalize")
        metrics = dict(variant=args.variant, backend=args.backend, jobs=args.jobs,
            threads=args.threads, input=args.input, base_model=args.base_model,
            detailed_profiling=args.detail, trained_stixels=len(model.model_dict),
            input_preparation_method="helper" if use_helper else "manual",
            input_payload_bytes=input_bytes, estimated_largest_window_payload_bytes=largest_window_payload,
            saved_models_bytes=sum(model_files), largest_saved_model_bytes=max(model_files, default=0),
            lazy_loading=not args.resident_models, prediction_rows=len(full_prediction),
            prediction_output_bytes=int(full_prediction.nbytes), **scores,
            versions=dict(numpy=np.__version__, pandas=pd.__version__, duckdb=duckdb.__version__,
                          sklearn=sklearn.__version__, xgboost=xgboost.__version__),
            execution_info=getattr(model, "execution_info_", None))
        metrics["prediction_execution_info"] = getattr(model, "prediction_execution_info_", None)
        records = [json.loads(p.read_text()) for p in telemetry.glob("fits_*.json")]
        if records:
            metrics["stixel_measurements"] = {key: (sum(r[key] for r in records) if key in {"fit_calls", "fit_seconds_sum"}
                else max(r[key] for r in records)) for key in records[0]}
        np.save(args.work / f"{args.variant}_pred.npy", prediction)
        np.save(args.work / f"{args.variant}_full_pred.npy", full_prediction)
        phases.begin("done")
        metrics["phase_seconds"] = phases.seconds
        metrics["processing_seconds"] = time.perf_counter() - worker_started
        write_json(args.work / f"{args.variant}.json", metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent)
    parser.add_argument("--x-database", type=Path, help="Optional source checklist database in another directory.")
    parser.add_argument("--species-parquet", type=Path, help="Optional presence file in another directory.")
    parser.add_argument("--species", default="Acadian%20Flycatcher")
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--features", type=int, default=80)
    parser.add_argument("--ensembles", type=int, default=4)
    parser.add_argument("--trees", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--input", choices=["pandas", "duckdb", "parquet"], default="duckdb")
    parser.add_argument("--base-model", choices=["xgboost", "random_forest", "local_beta", "tuned_local_beta"], default="xgboost")
    parser.add_argument("--subset-x-names", action="store_true")
    parser.add_argument("--calibration-helper", type=Path)
    parser.add_argument("--tuning-trials", type=int, default=4)
    parser.add_argument("--tuning-trees", type=int, default=40)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--sample-per-mille", type=int, default=200)
    parser.add_argument("--resident-models", action="store_true")
    parser.add_argument("--no-detail", dest="detail", action="store_false")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", default="368acf6")
    parser.add_argument("--profile-dir", type=Path, help="Opt-in detailed profiles; use separate runs from timing comparisons.")
    parser.add_argument("--extra-site-packages", type=Path,
        help="Append optional validation dependencies without replacing runtime packages.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/00_benchmarks/comparison.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--work", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--variant", default="baseline", help=argparse.SUPPRESS)
    parser.add_argument("--backend", default="native", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.extra_site_packages is not None:
        sys.path.append(str(args.extra_site_packages.resolve()))
    if args.worker:
        worker(args)
        return
    with tempfile.TemporaryDirectory(prefix="stemflow_benchmark_") as temporary:
        work = Path(temporary)
        baseline = work / "baseline"
        baseline.mkdir()
        archive = subprocess.check_output(["git", "archive", args.baseline, "stemflow"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
            handle.extractall(baseline, filter="data")
        fixture = build_fixture(args, work)
        # Warm the common font cache before timing any variant.
        subprocess.run([sys.executable, "-c", "import matplotlib.font_manager"], check=True,
            env={**os.environ, "MPLCONFIGDIR": str(work / "matplotlib"), "MPLBACKEND": "Agg"},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        reference_helper = None
        if args.calibration_helper is not None:
            reference_helper = work / "reference_calibration.py"
            reference_helper.write_text(args.calibration_helper.read_text())
        results = []
        for variant, backend, package in [("baseline", "native", baseline),
                                          ("rewrite_native", "native", ROOT),
                                          ("rewrite_memmap", "memmap", ROOT)]:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--work", str(work),
                       "--variant", variant, "--backend", backend, "--jobs", str(args.jobs),
                       "--threads", str(args.threads), "--ensembles", str(args.ensembles),
                       "--trees", str(args.trees), "--seed", str(args.seed), "--input", args.input,
                       "--base-model", args.base_model, "--depth", str(args.depth)]
            command += ["--tuning-trials", str(args.tuning_trials), "--tuning-trees", str(args.tuning_trees)]
            if reference_helper is not None:
                command += ["--calibration-helper", str(reference_helper)]
            if args.subset_x_names:
                command.append("--subset-x-names")
            if args.extra_site_packages is not None:
                command += ["--extra-site-packages", str(args.extra_site_packages.resolve())]
            if not args.detail:
                command.append("--no-detail")
            if args.resident_models:
                command.append("--resident-models")
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(package), str(work)]), "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg",
                   "MPLCONFIGDIR": str(work / "matplotlib")}
            if args.profile_dir is not None:
                env["STEMFLOW_PROFILE_DIR"] = str(args.profile_dir.resolve() / variant)
            with (work / f"{variant}.log").open("w") as log:
                wall_started = time.perf_counter()
                process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
                peak_rss, peak_accounted, by_phase = 0, 0, {}
                while process.poll() is None:
                    rss, accounted, processes = process_memory(process.pid, details=True)
                    peak_rss, peak_accounted = max(peak_rss, rss), max(peak_accounted, accounted)
                    try:
                        phase = json.loads((work / "phase.json").read_text())["phase"]
                    except (OSError, ValueError):
                        phase = "startup"
                    record = by_phase.setdefault(phase, dict(samples=0, peak_rss_mb=0,
                        peak_accounted_mb=0, peak_parent_rss_mb=0, peak_workers_rss_mb=0,
                        peak_active_stixel_payload_mb=0, peak_active_fit_rss_increase_mb=0,
                        peak_process_count=0))
                    record["samples"] += 1
                    record["peak_process_count"] = max(record["peak_process_count"], len(processes))
                    parent_rss = processes.get(process.pid, (0, 0))[0]
                    for key, value in [("peak_rss_mb", rss), ("peak_accounted_mb", accounted),
                                       ("peak_parent_rss_mb", parent_rss),
                                       ("peak_workers_rss_mb", rss - parent_rss)]:
                        record[key] = max(record[key], value / 1024**2)
                    active_payload, active_delta = 0, 0
                    for state in (work / f"{variant}_telemetry").glob("active_*.json"):
                        try:
                            value = json.loads(state.read_text())
                            if value["active"] and value["pid"] in processes:
                                active_payload += value["stixel_bytes"]
                                active_delta += max(0, processes[value["pid"]][0] - value["rss_before"])
                        except (OSError, ValueError):
                            pass
                    record["peak_active_stixel_payload_mb"] = max(record["peak_active_stixel_payload_mb"], active_payload / 1024**2)
                    record["peak_active_fit_rss_increase_mb"] = max(record["peak_active_fit_rss_increase_mb"], active_delta / 1024**2)
                    time.sleep(.1)
            if process.returncode:
                raise RuntimeError((work / f"{variant}.log").read_text())
            result = json.loads((work / f"{variant}.json").read_text())
            expected_workers = min(args.jobs, args.ensembles)
            if expected_workers > 1 and by_phase.get("training", {}).get("peak_process_count", 0) < expected_workers + 1:
                raise RuntimeError("The memory sampler did not observe all training workers; do not report this run.")
            for phase in result["phase_seconds"]:
                by_phase.setdefault(phase, None)
            result.update(process_tree_peak_rss_mb=peak_rss / 1024**2,
                          process_tree_peak_accounted_mb=peak_accounted / 1024**2,
                          phase_memory=by_phase, wall_seconds=time.perf_counter() - wall_started)
            result["prepared_input_pipeline_peak_accounted_mb"] = max(
                value["peak_accounted_mb"] for phase, value in by_phase.items() if value and phase in {
                    "partitioning", "fit_preparation", "shared_input_preparation", "training",
                    "fit_finalize", "heldout_prediction", "evaluation", "prediction", "finalize"})
            results.append(result)
            print(json.dumps({key: result[key] for key in ["variant", "phase_seconds", "auc", "brier",
                "process_tree_peak_rss_mb", "process_tree_peak_accounted_mb"]}), flush=True)
        import numpy as np
        reference = np.load(work / "baseline_pred.npy")
        for result in results[1:]:
            p = np.load(work / f"{result['variant']}_pred.npy")
            result["same_nan_mask"] = bool(np.array_equal(np.isnan(reference), np.isnan(p)))
            result["max_prediction_difference"] = float(np.nanmax(np.abs(reference - p)))
            full_reference = np.load(work / "baseline_full_pred.npy")
            full = np.load(work / f"{result['variant']}_full_pred.npy")
            result["same_full_nan_mask"] = bool(np.array_equal(np.isnan(full_reference), np.isnan(full)))
            result["max_full_prediction_difference"] = float(np.nanmax(np.abs(full_reference - full)))
            result["equivalent_within_1e6"] = bool(
                np.allclose(reference, p, atol=1e-6, rtol=1e-6, equal_nan=True) and
                np.allclose(full_reference, full, atol=1e-6, rtol=1e-6, equal_nan=True))
            if not result["equivalent_within_1e6"]:
                raise RuntimeError(f"Prediction equivalence check failed: {result['variant']}")
        report = dict(fixture=fixture, baseline=args.baseline, results=results,
            sampler_version=2,
            host=dict(platform=platform.platform(), machine=platform.machine(),
                      python=platform.python_version(), logical_cpus=os.cpu_count()),
            configuration=dict(source=args.source, input=args.input, ensembles=args.ensembles,
                trees=args.trees, depth=args.depth, jobs=args.jobs, threads=args.threads,
                base_model=args.base_model, seed=args.seed, resident_models=args.resident_models,
                subset_x_names=args.subset_x_names, tuning_trials=args.tuning_trials,
                tuning_trees=args.tuning_trees, font_cache_warmed=True),
            memory_metric=("macOS physical footprint" if sys.platform == "darwin" else "Linux PSS"),
            limitations="Sampled every 100 ms; RSS sums double-count shared pages. "
                        "Footprint/PSS excludes some job-wide memory. Validate Slurm cgroup peak on HPC. "
                        "Saved model bytes and shared file bytes measure disk payload, not resident memory. "
                        "Fit RSS increases also include estimator preprocessing and allocator retention. "
                        "Logical dataframe bytes can share underlying memory and must not be added to RSS. "
                        "Phases shorter than the sampling interval may have no memory sample (null). "
                        "Detailed profiling adds boundary measurements and small temporary JSON writes. "
                        "Times are measured on this fixture, not extrapolations to full species data.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
