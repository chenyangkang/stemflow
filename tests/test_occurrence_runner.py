"""Small end-to-end checks for the project occurrence runner."""

import importlib.util
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit


@pytest.fixture
def runner():
    filename = Path(__file__).resolve().parents[1] / "scripts/04_occurrence/run_model_test.py"
    spec = importlib.util.spec_from_file_location("occurrence_runner", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(731)
    n = 2400
    frame = pd.DataFrame({
        "__index_level_0__": rng.permutation(np.arange(n) * 3 + 17),
        "SAMPLING EVENT IDENTIFIER": [f"event_{i}" for i in range(n)],
        "longitude": rng.uniform(-91, -87, n),
        "latitude": rng.uniform(30, 34, n),
        "DOY": rng.integers(1, 367, n),
        "signal": rng.normal(size=n).astype("float32"),
        "year": np.full(n, 2022),
    })
    y = rng.binomial(1, expit(-2 + frame.signal.to_numpy()))
    data = tmp_path / "data"
    checklist = data / "030.Annotated_ebird_data_and_prediction_set/checklist/X_train.duckdb"
    checklist.parent.mkdir(parents=True)
    with duckdb.connect(str(checklist)) as con:
        con.register("frame", frame)
        con.execute("CREATE TABLE tmp AS SELECT * FROM frame")
    species = tmp_path / "species/COMMON%20NAME=Test%20Species"
    species.mkdir(parents=True)
    positives = frame.loc[y == 1, ["SAMPLING EVENT IDENTIFIER"]]
    pd.concat([positives, positives.iloc[:2]]).to_parquet(species / "data_0.parquet", index=False)
    prediction = tmp_path / "prediction.duckdb"
    prediction_frame = frame.drop(columns="SAMPLING EVENT IDENTIFIER").iloc[:50].copy()
    # Prediction scans must retain this deliberately unsorted ID order.
    with duckdb.connect(str(prediction)) as con:
        con.register("frame", prediction_frame)
        con.execute("CREATE TABLE tmp AS SELECT * FROM frame")
    return frame, y, data, species.parent, prediction


def test_occurrence_preparation_retains_original_join_padding_and_ids(runner, source, tmp_path):
    frame, y, data, species, _ = source
    args = runner.parse_args(["Test%20Species", "--data-dir", str(data), "--species-dir", str(species)])
    directory = tmp_path / "prepared"
    directory.mkdir()
    X, target = runner.get_training_data(args.species, directory, args)
    expected = frame.drop(columns="SAMPLING EVENT IDENTIFIER").assign(occ=y)
    expected = runner.make_circular_year(expected.sort_values(runner.INDEX))
    with duckdb.connect(X, read_only=True) as con:
        actual_X = con.sql("SELECT * FROM tmp").df()
    with duckdb.connect(target, read_only=True) as con:
        actual_y = con.sql("SELECT * FROM tmp").df()
    pd.testing.assert_frame_equal(actual_X, expected.drop(columns="occ"), check_dtype=False)
    pd.testing.assert_frame_equal(actual_y, expected[[runner.INDEX, "occ"]], check_dtype=False)
    assert actual_X[runner.INDEX].is_unique
    assert (actual_X.DOY < 1).any() and (actual_X.DOY > 366).any()
    folds = runner.make_fold_databases(directory, X, target)
    ids = []
    for split in ("train", "test"):
        with duckdb.connect(folds[f"X_{split}"], read_only=True) as con:
            predictors = con.sql("SELECT * FROM tmp").df()
        with duckdb.connect(folds[f"y_{split}"], read_only=True) as con:
            labels = con.sql("SELECT * FROM tmp").df()
        np.testing.assert_array_equal(predictors[runner.INDEX], labels[runner.INDEX])
        assert "longitude_tmp" not in predictors and "latitude_tmp" not in predictors
        np.testing.assert_array_equal(labels.occ, expected.set_index(runner.INDEX).loc[labels[runner.INDEX], "occ"])
        ids.append(set(predictors[runner.INDEX]))
    assert not ids[0] & ids[1]
    assert ids[0] | ids[1] == set(expected[runner.INDEX])
    assert not Path(X).exists() and not Path(target).exists()


@pytest.mark.parametrize("backend,tuning", [("native", "false"), ("memmap", "true")])
def test_occurrence_runner_parallel_fit_diagnostics_prediction_and_monitor(
        runner, source, tmp_path, monkeypatch, backend, tuning):
    pytest.importorskip("psutil")
    if tuning == "true":
        pytest.importorskip("optuna")
    _, _, data, species, prediction = source
    output = tmp_path / backend
    args = runner.parse_args([
        "Test%20Species", tuning, "--data-dir", str(data), "--species-dir", str(species),
        "--prediction-db", str(prediction), "--output-root", str(output),
        "--n-jobs", "2", "--ensemble-fold", "2", "--min-ensemble-required", "1",
        "--points-lower-threshold", "40", "--n-estimators", "5",
        "--tuning-n-estimators", "3", "--n-trials", "1", "--skip-animation",
        "--log-interval", "0", "--training-data-backend", backend,
        "--prediction-batch-size", "17",
        "--save-model",
    ])
    grid = pd.DataFrame({"latitude": [31., 33.], "longitude": [-90., -88.]}).merge(
        pd.DataFrame({"DOY": np.arange(1, 367, 2)}), how="cross")
    grid["year"] = 2022
    monkeypatch.setattr(runner, "prediction_grid", lambda: grid.copy())
    observed = {}
    original_save_annual = runner.save_annual_prediction

    def check_prediction_order(model, directory, destination, options, monitor):
        meta = runner.read_metadata(prediction, directory)
        with duckdb.connect(str(prediction), read_only=True) as con:
            X = con.sql("SELECT * FROM tmp").df().set_index(runner.INDEX)
        np.testing.assert_array_equal(meta.index, X.index)
        p = model.predict_proba(X, temporal_window_prequery=True)[:, 1]
        expected = meta.reset_index(drop=True).astype("float32")
        expected["DOY"] = expected.DOY.astype(int)
        expected["pred_mean"] = p.astype("float32")
        observed["expected"] = expected.sort_values(["latitude", "longitude", "DOY"]).reset_index(drop=True)
        original_save_annual(model, directory, destination, options, monitor)

    monkeypatch.setattr(runner, "save_annual_prediction", check_prediction_order)
    runner.main(args)
    destination = output / args.species
    actual = pd.read_parquet(destination / f"{args.species}.pred_df.whole_year.raw.parquet")
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), observed["expected"], atol=1e-7, rtol=1e-7)
    final = pd.read_parquet(destination / f"{args.species}.pred_df.whole_year.parquet")
    assert len(final) == len(grid)
    groups = pd.read_csv(destination / "calibration_by_group.csv")
    assert set(groups.level) == {"overall", "stixel", "spatial", "temporal", "spatiotemporal"}
    assert groups.loc[groups.level == "overall", "n_valid"].item() > 0
    for name in ("calibration_diagnostics", "stixel_calibration_overlay"):
        for extension in ("png", "pdf"):
            assert (destination / f"{name}.{extension}").stat().st_size > 1000
    report = json.loads((destination / "execution_monitor.json").read_text())
    assert not report["running"] and report["error"] is None
    assert report["wall_seconds"] > 0 and report["peak_tree_rss_bytes"] > 0
    archive = destination / f"{args.species}.model.tar.gz"
    assert archive.stat().st_size > 1000
    loaded = runner.AdaSTEM.load(str(archive), new_lazy_loading_path=str(tmp_path / "reloaded"))
    assert loaded.execution_monitor_["wall_seconds"] == report["wall_seconds"]
    assert loaded.execution_monitor_["error"] is None
    metrics = json.loads(next(destination.glob("*_calibration_metrics.json")).read_text())
    assert "quantile_ECE_relative" in metrics["calibration_metrics"]
    assert "monthly" in metrics["calibration_metrics"]
