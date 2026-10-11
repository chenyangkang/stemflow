"""Occurrence runner for stemflow's codex-rewrite branch.

Place next to the original runner. All results and figures go into the species
folder under 040.AdaSTEM_results_one_sp_predict_occ_test.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

# Limit native threads; AdaSTEM supplies the outer parallelism.
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.calibration import CalibrationDisplay

import stemflow
from stemflow.calibration import calibration_scores, plot_calibration_diagnostics
from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier
from stemflow.model_selection import ST_train_test_split
from stemflow.tuning import local_beta_xgboost

SCRIPT_DIR = Path(__file__).resolve().parent
INDEX = "__index_level_0__"
COORDS = ["longitude", "latitude", "DOY"]
pd.set_option("display.max_columns", None)


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def db_config(directory, memory_limit="10GB"):
    return {"threads": "1", "memory_limit": memory_limit,
            "temp_directory": str(Path(directory) / "duckdb")}


def save_json(value, filename):
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [clean(v) for v in obj]
        if isinstance(obj, np.generic):
            return clean(obj.item())
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        if isinstance(obj, Path):
            return str(obj)
        return obj
    Path(filename).write_text(json.dumps(clean(value), indent=2, allow_nan=False))


def make_circular_year(X, pad=40, period=366):
    # Retain the original padding and row-ID convention.
    tail = X[X.DOY > period - pad].copy()
    head = X[X.DOY <= pad].copy()
    tail["DOY"] -= period
    head["DOY"] += period
    result = pd.concat([tail, X, head], ignore_index=True)
    result[INDEX] = np.arange(len(result))
    return result


def get_training_data(species, directory, args):
    checklist = args.data_dir / "030.Annotated_ebird_data_and_prediction_set/checklist/X_train.duckdb"
    species_file = args.species_dir / f"COMMON%20NAME={species}" / "data_0.parquet"
    sp_data = pd.read_parquet(species_file, columns=["SAMPLING EVENT IDENTIFIER"])
    sp_data = sp_data.drop_duplicates("SAMPLING EVENT IDENTIFIER")
    sp_data["occ"] = 1
    with duckdb.connect(config=db_config(directory)) as con:
        con.execute(f"ATTACH {sql_string(checklist)} AS X_train (READ_ONLY)")
        con.register("sp_data", sp_data)
        merged = con.sql('''
            SELECT a.* EXCLUDE (a."SAMPLING EVENT IDENTIFIER"),
                   COALESCE(b.occ, 0) AS occ
            FROM X_train.tmp AS a LEFT JOIN sp_data AS b
            ON a."SAMPLING EVENT IDENTIFIER" = b."SAMPLING EVENT IDENTIFIER"
            ORDER BY a."__index_level_0__"
        ''').df()
    merged = make_circular_year(merged)
    final_X = merged.drop(columns="occ")
    final_y = merged[[INDEX, "occ"]].copy()
    paths = [str(Path(directory) / name) for name in ("X_train.duckdb", "y_train.duckdb")]
    for path, frame in zip(paths, (final_X, final_y)):
        with duckdb.connect(path, config=db_config(directory)) as con:
            con.register("frame", frame)
            con.execute("CREATE TABLE tmp AS SELECT * FROM frame")
    return paths


def make_fold_databases(directory, X, y):
    with duckdb.connect(X, read_only=True, config=db_config(directory)) as con:
        meta = con.sql('SELECT "__index_level_0__", longitude, latitude, DOY FROM tmp').df()
    with duckdb.connect(y, read_only=True, config=db_config(directory)) as con:
        labels = con.sql("SELECT * FROM tmp").df().set_index(INDEX)
    meta = meta.set_index(INDEX).sort_index()
    labels = labels.reindex(meta.index)
    if not meta.index.is_unique or labels.occ.isna().any():
        raise ValueError("Training observation IDs are duplicated or X/y do not align.")
    train_X, test_X, train_y, test_y = ST_train_test_split(
        meta, labels, Spatio1="longitude", Spatio2="latitude", Temporal1="DOY",
        Spatio_blocks_count=50, Temporal_blocks_count=50,
        random_state=42, test_size=0.2,
    )
    paths = {name: str(Path(directory) / f"{name}_0.duckdb")
             for name in ("X_train", "y_train", "X_test", "y_test")}
    with duckdb.connect(X, config=db_config(directory)) as con:
        con.execute(f"ATTACH {sql_string(y)} AS labels (READ_ONLY)")
        for name, path in paths.items():
            con.execute(f"ATTACH {sql_string(path)} AS fold_{name}")
        for split, subset in (("train", train_X), ("test", test_X)):
            con.register("selected_ids", subset.reset_index()[[INDEX]])
            con.execute(f'''
                CREATE TABLE fold_X_{split}.tmp AS
                SELECT a.*
                FROM main.tmp AS a RIGHT JOIN selected_ids AS b
                ON a."__index_level_0__" = b."__index_level_0__"
                ORDER BY b."__index_level_0__"
            ''')
            con.execute(f'''
                CREATE TABLE fold_y_{split}.tmp AS
                SELECT a.* FROM labels.tmp AS a RIGHT JOIN selected_ids AS b
                ON a."__index_level_0__" = b."__index_level_0__"
                ORDER BY b."__index_level_0__"
            ''')
    print(f"Train: {len(train_X):,} rows, {int(train_y.occ.sum()):,} positives; "
          f"test: {len(test_X):,} rows, {int(test_y.occ.sum()):,} positives", flush=True)
    # These full preparation databases are no longer needed.
    Path(X).unlink()
    Path(y).unlink()
    return paths


def make_model(args, lazy_directory):
    base = local_beta_xgboost(
        tuning=args.tuning, ridge=args.ridge, n_trials=args.n_trials,
        n_estimators=args.n_estimators, tuning_n_estimators=args.tuning_n_estimators,
        n_splits=args.calibration_folds, tuning_n_splits=args.tuning_folds,
        Spatio_blocks_count=args.spatial_blocks, Temporal_blocks_count=args.temporal_blocks,
        # The package passes coordinates to local CV without using them as predictors.
        Spatio1="longitude", Spatio2="latitude", Temporal1="DOY",
        random_state=42, max_trials=10, search="optuna",
    )
    return AdaSTEMClassifier(
        base_model=base, ensemble_fold=args.ensemble_fold,
        min_ensemble_required=args.min_ensemble_required,
        grid_len_upper_threshold=25, grid_len_lower_threshold=3,
        temporal_start=1, temporal_end=366, temporal_step=40, temporal_bin_interval=40,
        points_lower_threshold=args.points_lower_threshold,
        Spatio1="longitude", Spatio2="latitude", Temporal1="DOY",
        use_temporal_to_train=True, n_jobs=args.n_jobs, random_state=42,
        save_gridding_plot=True, lazy_loading=True, lazy_loading_dir=str(lazy_directory),
        min_class_sample=1, sample_weights_for_classifier=False,
        ensemble_bootstrap=True, completely_random_rotation=True,
        joblib_backend="loky", max_mem=args.max_mem, verbosity=1,
    )


def read_metadata(path, directory):
    # Match the table scan used by prediction; align labels by ID afterward.
    with duckdb.connect(str(path), read_only=True, config=db_config(directory)) as con:
        return con.sql('SELECT "__index_level_0__", longitude, latitude, DOY FROM tmp').df().set_index(INDEX)


def save_metrics(y, pred, doy, output, prefix):
    score, _ = calibration_scores(y, pred, n_bins=8)
    valid = np.isfinite(pred) & np.isfinite(y)
    if not valid.any():
        raise RuntimeError("No valid held-out predictions; check trained coverage.")
    p0 = np.clip(np.asarray(y)[valid].mean(), 1e-12, 1 - 1e-12)
    null_loss = -np.mean(np.asarray(y)[valid] * np.log(p0) +
                         (1 - np.asarray(y)[valid]) * np.log1p(-p0))
    metrics = (AdaSTEM.eval_STEM_res("classification", np.asarray(y)[valid], pred[valid])
               if np.unique(np.asarray(y)[valid]).size == 2 else {})
    months = (pd.Timestamp("2022-01-01") + pd.to_timedelta(np.asarray(doy) - 1, unit="D")).month
    monthly = {}
    for month in np.unique(months):
        selected = months == month
        month_score, _ = calibration_scores(np.asarray(y)[selected], pred[selected], n_bins=8)
        for name, value in month_score.items():
            monthly.setdefault(name + "_monthly", {})[int(month)] = value
    metrics.update(ll_test=-score["log_loss"], ll_null=-null_loss,
                   pseudo_r2=1 - score["log_loss"] / null_loss if null_loss > 0 else np.nan,
                   calibration_metrics={**score, "monthly": monthly})
    save_json(metrics, output / f"{prefix}_calibration_metrics.json")
    for strategy, suffix in (("uniform", ""), ("quantile", "_quantile")):
        fig, ax = plt.subplots()
        CalibrationDisplay.from_predictions(np.asarray(y)[valid], pred[valid],
                                            n_bins=8, strategy=strategy, ax=ax)
        fig.tight_layout()
        fig.savefig(output / f"{prefix}{suffix}.png", dpi=300)
        plt.close(fig)


def save_diagnostics(model, meta, y, pred, output, args):
    diagnostics = model.calibration_diagnostics(
        meta, y, pred=pred, n_bins=8, spatial_bin_width=5, temporal_bin_width=40,
        support_min_samples=args.support_min_samples, support_min_events=args.support_min_events,
        include_bins=True,
    )
    for key, filename in (("scores", "calibration_by_group.csv"),
                          ("bins", "calibration_bins.csv"), ("summary", "calibration_summary.csv")):
        diagnostics[key].to_csv(output / filename, index=False)
    plot_calibration_diagnostics(diagnostics, output, max_curves=200)
    scores = diagnostics["scores"]
    print("Overall diagnostics:", flush=True)
    print(scores.loc[scores.level == "overall"].to_string(index=False), flush=True)
    print("Stixel diagnostics, equal weight per stixel:", flush=True)
    print(diagnostics["summary"].query("level == 'stixel'").to_string(index=False), flush=True)
    # Retain the inspection table/plot, now using the fitted occurrence stixels.
    cells = scores[(scores.level == "stixel") & (scores.n_valid > 0)].copy()
    cells = cells.merge(model.ensemble_df[["unique_stixel_id", "stixel_width"]],
                        left_on="group", right_on="unique_stixel_id", how="left", validate="one_to_one")
    cells = cells.rename(columns={"predicted_rate": "pred_mean", "observed_rate": "y_mean",
                                  "quantile_ECE_relative": "ece_relative"})
    cells.to_csv(output / f"{args.species}.stixel_inspection_res.csv", index=False)
    fig, ax = plt.subplots()
    artist = ax.scatter(np.sqrt(cells.pred_mean), np.sqrt(cells.y_mean),
                        s=1, c=cells.stixel_width, rasterized=True)
    ax.plot([0, 1], [0, 1], color="red", alpha=.7)
    ax.set(xlabel="pred_mean **(1/2)", ylabel="y_mean **(1/2)")
    fig.colorbar(artist, ax=ax, label="stixel_width")
    fig.tight_layout()
    fig.savefig(output / f"{args.species}.stixel_inspection_res.pdf")
    plt.close(fig)


def prediction_grid():
    import h3
    sys.path.append(str(SCRIPT_DIR.parent / "050.stemflow_utils_scripts"))
    from other_functions import filter_ocean
    bbox = {"type": "Polygon", "coordinates": [[[-180, -30], [-30, -30],
             [-30, 90], [-180, 90], [-180, -30]]]}
    cells = list(h3.geo_to_cells(bbox, 4))
    centers = np.array([h3.cell_to_latlng(cell) for cell in cells])
    spatial = filter_ocean(pd.DataFrame({"latitude": centers[:, 0], "longitude": centers[:, 1]}))
    grid = spatial.merge(pd.DataFrame({"DOY": np.arange(1, 367, 2)}), how="cross")
    grid["year"] = 2022
    return grid.sort_values(["latitude", "longitude", "DOY"]).reset_index(drop=True)


def save_annual_prediction(model, directory, output, args, monitor):
    pred_file = args.prediction_db or (args.data_dir /
        "030.Annotated_ebird_data_and_prediction_set/prediction_set/2022_combined_special.duckdb")
    with monitor.phase("annual_prediction"):
        meta = read_metadata(pred_file, directory)
        pred = model.predict_proba(str(pred_file), temporal_window_prequery=True,
                                   logit_agg=False, streaming=True,
                                   prediction_batch_size=args.prediction_batch_size or None)[:, 1]
        raw = meta.reset_index(drop=True).astype("float32")
        raw["DOY"] = raw.DOY.astype(int)
        raw["pred_mean"] = pred.astype("float32")
        raw = raw.sort_values(["latitude", "longitude", "DOY"])
        raw.to_parquet(output / f"{args.species}.pred_df.whole_year.raw.parquet", index=False)
        del meta, pred
    with monitor.phase("grid_snapping"):
        grid = prediction_grid()
        centers = grid.loc[grid.DOY == 1, ["latitude", "longitude"]]
        _, positions = cKDTree(centers.to_numpy()).query(raw[["latitude", "longitude"]].to_numpy())
        raw[["latitude", "longitude"]] = centers.to_numpy()[positions]
        aggregated = raw.groupby(["latitude", "longitude", "DOY"], as_index=False).pred_mean.mean()
        final = grid.merge(aggregated, on=["latitude", "longitude", "DOY"], how="left")
        final = final.sort_values(["latitude", "longitude", "DOY"])
        final.to_parquet(output / f"{args.species}.pred_df.whole_year.parquet", index=False)
        del raw, grid, aggregated, positions
        gc.collect()
    if not args.skip_animation:
        from stemflow.utils.plot_gif import make_sample_gif
        with monitor.phase("animation"):
            finite = final.pred_mean[np.isfinite(final.pred_mean)]
            if finite.empty:
                print("No finite grid predictions; skipping animation.", flush=True)
            else:
                make_sample_gif(final.sort_values(["DOY", "latitude", "longitude"]),
                    str(output / f"{args.species}_occ_animation.gif"),
                    continental_boundary=True, show_major_lakes=True, political_boundary=None,
                    lightgrey_under=False, col="pred_mean", log_scale=False,
                    Spatio1="longitude", Spatio2="latitude", Temporal1="DOY",
                    vmin=0, vmax=max(float(finite.quantile(.95)), 1e-8), cmap="viridis",
                    figsize=(7.5, 5.5), xlims=(-180, -30), ylims=(-20, 90), grid=True,
                    xtick_interval=20, ytick_interval=20, lng_size=150, lat_size=120, dpi=150, fps=30)


def main(args):
    output = args.output_root / args.species
    output.mkdir(parents=True, exist_ok=True)
    print(f"stemflow path: {stemflow.__file__}\nOutput: {output}", flush=True)
    save_json(vars(args), output / "run_settings.json")
    with tempfile.TemporaryDirectory(prefix="stemflow_", dir=os.environ.get("SLURM_TMPDIR")) as directory:
        print(f"database_dir: {directory}", flush=True)
        model = make_model(args, Path(directory) / "models")
        with model.monitor_execution(sample_interval=1, log_interval=args.log_interval or None,
                output_path=output / "execution_monitor.json", max_samples=0) as monitor:
            with monitor.phase("data_preparation"):
                X, y = get_training_data(args.species, directory, args)
                paths = make_fold_databases(directory, X, y)
                gc.collect()
            model.fit(paths["X_train"], paths["y_train"], overwrite=True, verbosity=1,
                      temporal_window_prequery=True, training_data_backend=args.training_data_backend,
                      inner_max_num_threads=1, persist_prediction_state=False)
            with monitor.phase("feature_importances"):
                model.calculate_feature_importances()
                grid = model.ensemble_df.copy()
                grid[["stixel_calibration_point_left", "stixel_calibration_point_right"]] = pd.DataFrame(
                    grid["stixel_calibration_point(transformed)"].tolist(), index=grid.index)
                grid.merge(model.feature_importances_, left_on="unique_stixel_id", right_on="stixel_index",
                           how="left").to_csv(output / f"{args.species}_fold1__feature_importances.csv", index=False)
                model.gridding_plot.savefig(output / "ensemble_gridding.pdf")
                del grid
            print("Predicting on X_test...", flush=True)
            pred = model.predict_proba(paths["X_test"], temporal_window_prequery=True,
                                       logit_agg=False, streaming=True,
                                       prediction_batch_size=args.prediction_batch_size or None)[:, 1]
            with monitor.phase("heldout_evaluation"):
                meta = read_metadata(paths["X_test"], directory)
                with duckdb.connect(paths["y_test"], read_only=True, config=db_config(directory)) as con:
                    labels = con.sql("SELECT * FROM tmp").df().set_index(INDEX).occ.reindex(meta.index)
                if not meta.index.is_unique or labels.isna().any():
                    raise ValueError("Held-out X/y observation IDs do not align.")
                save_metrics(labels.to_numpy(), pred, meta.DOY, output,
                             f"{args.species}_fold1_overall_calibration_")
                save_diagnostics(model, meta, labels, pred, output, args)
                del pred, meta, labels
                gc.collect()
            if not args.skip_annual_prediction:
                print("Start making the real prediction...", flush=True)
                save_annual_prediction(model, directory, output, args, monitor)
        save_json({"training": model.execution_info_,
                   "prediction": model.prediction_execution_info_}, output / "execution_info.json")
        # Save after monitoring ends so the archive includes execution_monitor_.
        if args.save_model:
            model.save(str(output / f"{args.species}.model.tar.gz"), remove_temporary_file=False)
        del model
    print(f"Finished. Results and figures: {output}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("species", help="Encoded name, e.g. American%%20Redstart")
    parser.add_argument("tuning", nargs="?", default="false", choices=["true", "false", "1", "0", "yes", "no"])
    parser.add_argument("--ridge", type=float, default=100)
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "25")))
    parser.add_argument("--ensemble-fold", type=int, default=50)
    parser.add_argument("--min-ensemble-required", type=int, default=45)
    parser.add_argument("--points-lower-threshold", type=int, default=1000)
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--tuning-n-estimators", type=int, default=40)
    parser.add_argument("--calibration-folds", type=int, default=3)
    parser.add_argument("--tuning-folds", type=int, default=2)
    parser.add_argument("--spatial-blocks", type=int, default=5)
    parser.add_argument("--temporal-blocks", type=int, default=5)
    parser.add_argument("--max-mem", default="500MB", help="DuckDB limit per connection, not a job RAM limit")
    parser.add_argument("--prediction-batch-size", type=int, default=100000,
                        help="Rows per temporal-query batch per worker; 0 restores whole windows")
    parser.add_argument("--training-data-backend", choices=["native", "memmap"], default="native")
    parser.add_argument("--log-interval", type=float, default=30, help="Seconds between monitor logs; 0 silences logs")
    parser.add_argument("--support-min-samples", type=int, default=200)
    parser.add_argument("--support-min-events", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "../../data")
    parser.add_argument("--species-dir", type=Path, default=SCRIPT_DIR / "../../../ebird/partitioned_species_data")
    parser.add_argument("--prediction-db", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--skip-annual-prediction", action="store_true")
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument("--save-model", action="store_true", help="Optional archive; may be large")
    args = parser.parse_args(argv)
    args.tuning = args.tuning in {"true", "1", "yes"}
    args.data_dir, args.species_dir = args.data_dir.resolve(), args.species_dir.resolve()
    args.output_root = (args.output_root or args.data_dir /
                       "040.AdaSTEM_results_one_sp_predict_occ_test").resolve()
    if args.n_jobs < 1 or not 1 <= args.min_ensemble_required <= args.ensemble_fold:
        parser.error("Use n_jobs >= 1 and 1 <= min_ensemble_required <= ensemble_fold.")
    if args.prediction_batch_size < 0:
        parser.error("Use prediction_batch_size >= 0.")
    return args


if __name__ == "__main__":
    main(parse_args())
