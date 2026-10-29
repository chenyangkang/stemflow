"""Small real-data audit of composable calibration and tuning components."""

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/00_benchmarks"))
from compare_execution import build_fixture
from stemflow.calibration import (BetaCalibrator, CrossValidatedCalibratedClassifier,
    SpatioTemporalKFold, calibration_scores, plot_calibration_diagnostics)
from stemflow.lazyloading import make_training_databases
from stemflow.model.AdaSTEM import AdaSTEMClassifier
from stemflow.model_selection import ST_train_test_split
from stemflow.tuning import SelfTuningClassifier, local_beta_xgboost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent)
    parser.add_argument("--x-database", type=Path)
    parser.add_argument("--species-parquet", type=Path)
    parser.add_argument("--species", default="Acadian%20Flycatcher")
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--ensembles", type=int, default=2)
    parser.add_argument("--search", choices=["random", "optuna"], default="random")
    parser.add_argument("--output", type=Path, default=ROOT/"data/02_calibration/acadian_small")
    parser.add_argument("--figures", type=Path, default=ROOT/"figures/02_calibration/acadian_small")
    args = parser.parse_args()
    args.source, args.sample_per_mille, args.seed = "real", 200, 42
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stemflow_calibration_") as temporary:
        temporary = Path(temporary)
        fixture = build_fixture(args, temporary)
        frame = pd.read_parquet(temporary/"fixture.parquet")
        X, y = frame.drop(columns="occurrence"), frame[["occurrence"]]
        X_train, X_test, y_train, y_test = ST_train_test_split(X, y,
            Spatio_blocks_count=10, Temporal_blocks_count=5, random_state=42, test_size=.2)
        xs, ys = make_training_databases(X_train, y_train, temporary/"input")
        cv = SpatioTemporalKFold()
        rf = SelfTuningClassifier(RandomForestClassifier(n_estimators=20, n_jobs=1, random_state=42),
            dict(max_depth=[4, 8], min_samples_leaf=[5, 20]), cv=SpatioTemporalKFold(2),
            n_trials=2, search=args.search)
        methods = dict(
            raw_xgboost=XGBClassifier(n_estimators=30, max_depth=6, learning_rate=.3,
                n_jobs=1, tree_method="hist", random_state=42),
            local_beta=local_beta_xgboost(tuning=False, n_estimators=30),
            tuned_xgboost_local_beta=local_beta_xgboost(tuning=True, n_trials=2,
                n_estimators=30, tuning_n_estimators=10, search=args.search),
            tuned_forest_local_beta=CrossValidatedCalibratedClassifier(rf, BetaCalibrator(100), cv))
        rows, grid = [], None
        for name, base in methods.items():
            print(f"Running {name}", flush=True)
            model = AdaSTEMClassifier(base_model=base, ensemble_fold=args.ensembles,
                min_ensemble_required=1, grid_len_upper_threshold=25, grid_len_lower_threshold=5,
                points_lower_threshold=200, temporal_start=121, temporal_end=201,
                temporal_step=40, temporal_bin_interval=40, temporal_bin_start_jitter=0,
                ensemble_bootstrap=True, n_jobs=args.jobs, random_state=42,
                save_gridding_plot=False, sample_weights_for_classifier=False,
                lazy_loading=True, lazy_loading_dir=str(temporary/name), verbosity=0)
            started = time.perf_counter()
            model.fit(xs, ys, temporal_window_prequery=True, training_data_backend="memmap",
                      ensemble_df=grid, inner_max_num_threads=1, persist_prediction_state=False)
            fit_seconds = time.perf_counter()-started
            if grid is None:
                grid = model.ensemble_df.copy()
            started = time.perf_counter()
            p = model.predict_proba(X_test, temporal_window_prequery=True)[:, 1]
            prediction_seconds = time.perf_counter()-started
            scores, _ = calibration_scores(y_test.to_numpy(), p)
            started = time.perf_counter()
            diagnostics = model.calibration_diagnostics(X_test, y_test, p)
            target = args.output/name
            target.mkdir(exist_ok=True)
            for key, table in diagnostics.items():
                table.to_csv(target/f"calibration_{key}.csv", index=False)
            plot_calibration_diagnostics(diagnostics, args.figures/name)
            local = diagnostics["scores"]
            local = local[(local.level == "stixel") & (local.n_valid > 0)]
            rows.append(dict(method=name, fit_seconds=fit_seconds,
                prediction_seconds=prediction_seconds, diagnostics_seconds=time.perf_counter()-started,
                stixels=len(model.model_dict), stixel_median_quantile_ECE_pp=local.quantile_ECE_pp.median(),
                stixel_median_quantile_ECE_relative=local.quantile_ECE_relative.median(), **scores))
            del model
            gc.collect()
        pd.DataFrame(rows).to_csv(args.output/"comparison.csv", index=False)
        (args.output/"configuration.json").write_text(json.dumps(dict(fixture=fixture,
            training_rows=len(X_train), heldout_rows=len(X_test),
            training_events=int(y_train.to_numpy().sum()), heldout_events=int(y_test.to_numpy().sum()),
            ensembles=args.ensembles, jobs=args.jobs, bootstrap=True, search=args.search,
            note="Small blocked held-out execution/calibration audit, not a full-species model ranking. "
                 "Final probabilities evaluated in overlapping actual fitted stixels; ridge=100 uses summed loss."), indent=2))
        print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
