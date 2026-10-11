"""Compare prediction batches in isolated processes with identical fitted models.

Temporary databases, models and prediction arrays are removed after comparison.
Retain only a compact JSON containing timings, sampled memory and output checks.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"


def prepare(args):
    import duckdb
    import joblib
    import numpy as np
    import pandas as pd
    from stemflow.model.AdaSTEM import AdaSTEMClassifier
    from stemflow.tuning import local_beta_xgboost
    from xgboost import XGBClassifier
    rng = np.random.default_rng(231)
    n = args.train_rows
    train = pd.DataFrame(rng.normal(size=(n, args.features)),
        columns=[f"feature_{i}" for i in range(args.features)])
    train["longitude"] = rng.uniform(-100, -80, n)
    train["latitude"] = rng.uniform(30, 45, n)
    train["DOY"] = rng.uniform(1, 81, n)
    p = 1 / (1 + np.exp(-(-3 + train.feature_0.to_numpy())))
    labels = rng.binomial(1, p)
    if args.calibrated:
        base = local_beta_xgboost(tuning=False, n_estimators=args.trees, n_splits=3,
            Spatio_blocks_count=3, Temporal_blocks_count=3, ridge=100)
    else:
        base = XGBClassifier(n_estimators=args.trees, max_depth=4, n_jobs=1, random_state=42)
    model = AdaSTEMClassifier(base_model=base, ensemble_fold=args.ensembles,
        min_ensemble_required=1, grid_len_upper_threshold=12, grid_len_lower_threshold=4,
        points_lower_threshold=150, temporal_start=1, temporal_end=81,
        temporal_step=40, temporal_bin_interval=40, temporal_bin_start_jitter=0,
        sample_weights_for_classifier=False, n_jobs=args.jobs, random_state=32,
        save_gridding_plot=False, lazy_loading=True, max_mem=args.max_mem,
        lazy_loading_dir=str(args.work / "models"))
    model.fit(train, labels, temporal_window_prequery=True, persist_prediction_state=False,
        inner_max_num_threads=1)
    joblib.dump(model, args.work / "model.pkl")
    # The comparison's TemporaryDirectory owns these files across subprocesses.
    model._finalizer.detach()
    with duckdb.connect(str(args.work / "prediction.duckdb"),
            config={"threads": "1", "memory_limit": "500MB"}) as con:
        for start in range(0, args.rows, 50000):
            size = min(50000, args.rows - start)
            frame = pd.DataFrame(rng.normal(size=(size, args.features)), columns=train.columns[:args.features])
            frame["longitude"] = rng.uniform(-100, -80, size)
            frame["latitude"] = rng.uniform(30, 45, size)
            frame["DOY"] = rng.uniform(1, 81, size)
            frame["__index_level_0__"] = (np.arange(start, start + size) * 7 + 100)[::-1]
            con.register("frame", frame)
            con.execute("CREATE TABLE tmp AS SELECT * FROM frame" if start == 0 else
                        "INSERT INTO tmp SELECT * FROM frame")


def predict(args):
    import joblib
    import numpy as np
    model = joblib.load(args.work / "model.pkl")
    (args.work / "active.json").write_text(json.dumps({"phase": "prediction", "pid": os.getpid()}))
    started = time.perf_counter()
    p, sd = model.predict_proba(str(args.work / "prediction.duckdb"), return_std=True,
        n_jobs=args.jobs, temporal_window_prequery=True,
        prediction_batch_size=args.batch_size or None)
    elapsed = time.perf_counter() - started
    (args.work / "active.json").write_text(json.dumps({"phase": "finished"}))
    np.save(args.work / f"prediction_{args.batch_size}.npy", np.column_stack([p[:, 1], sd]))
    (args.work / f"result_{args.batch_size}.json").write_text(json.dumps({
        "seconds": elapsed, "coverage": float(np.isfinite(p[:, 1]).mean()),
        "execution": model.prediction_execution_info_}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=400000)
    parser.add_argument("--train-rows", type=int, default=10000)
    parser.add_argument("--features", type=int, default=96)
    parser.add_argument("--ensembles", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--trees", type=int, default=10)
    parser.add_argument("--calibrated", action="store_true")
    parser.add_argument("--max-mem", default="500MB")
    parser.add_argument("--batches", type=int, nargs="+", default=[0, 25000, 100000])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path,
        default=ROOT / "data/00_benchmarks/prediction_batches/comparison.json")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--mode", choices=["prepare", "predict"])
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()
    if args.mode:
        (prepare if args.mode == "prepare" else predict)(args)
        return
    spec = importlib.util.spec_from_file_location("execution_benchmark",
        ROOT / "scripts/00_benchmarks/compare_execution.py")
    sampler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    report = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "platform": sys.platform, "accounted_metric": "physical_footprint" if sys.platform == "darwin" else "pss",
        "runs": []}
    import numpy as np
    with tempfile.TemporaryDirectory(prefix="stemflow_prediction_") as work:
        work = Path(work)
        common = [sys.executable, str(Path(__file__).resolve()), "--work", str(work),
            "--rows", str(args.rows), "--train-rows", str(args.train_rows),
            "--features", str(args.features), "--ensembles", str(args.ensembles),
            "--jobs", str(args.jobs), "--trees", str(args.trees), "--max-mem", args.max_mem]
        if args.calibrated:
            common.append("--calibrated")
        subprocess.run(common + ["--mode", "prepare"], check=True, cwd=ROOT)
        reference = None
        for repeat in range(args.repeat):
            # Alternate order to expose cache/order effects.
            for batch_size in (args.batches if repeat % 2 == 0 else args.batches[::-1]):
                active = work / "active.json"
                active.unlink(missing_ok=True)
                child = subprocess.Popen(common + ["--mode", "predict", "--batch-size", str(batch_size)], cwd=ROOT)
                peak_rss, peak_accounted = 0, 0
                while child.poll() is None:
                    try:
                        predicting = json.loads(active.read_text())["phase"] == "prediction"
                    except (OSError, ValueError):
                        predicting = False
                    if predicting:
                        rss, accounted = sampler.process_memory(child.pid)
                        peak_rss, peak_accounted = max(peak_rss, rss), max(peak_accounted, accounted)
                    time.sleep(.05)
                if child.returncode:
                    raise RuntimeError(f"Prediction subprocess failed: {child.returncode}")
                result = json.loads((work / f"result_{batch_size}.json").read_text())
                values = np.load(work / f"prediction_{batch_size}.npy")
                if reference is None:
                    reference = values.copy()
                same_missing = bool(np.array_equal(np.isnan(values), np.isnan(reference)))
                maximum = float(np.nanmax(np.abs(values - reference)))
                np.testing.assert_allclose(values, reference, equal_nan=True, atol=3e-7, rtol=3e-7)
                result.update(repeat=repeat + 1, batch_size=batch_size or None,
                    peak_tree_rss_bytes=peak_rss, peak_tree_accounted_bytes=peak_accounted,
                    same_nan_mask=same_missing, max_absolute_difference=maximum)
                report["runs"].append(result)
                print(json.dumps(result), flush=True)
                (work / f"prediction_{batch_size}.npy").unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Comparison: {args.output}", flush=True)


if __name__ == "__main__":
    main()
