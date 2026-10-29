"""Offline compatibility checks for shared training and bootstrap occurrences."""

import os
import pickle
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from stemflow.lazyloading.training_data import SharedTrainingData, TrainingWindow
from stemflow.model.AdaSTEM import AdaSTEMClassifier
from stemflow.model.static_func_AdaSTEM import transform_pred_set_to_STEM_quad


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.rows_ = X["row_code"].to_numpy().copy()
        self.labels_ = np.asarray(y).copy()
        self.p_ = float(np.mean(y))
        self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        return self

    def predict_proba(self, X):
        return np.tile([1 - self.p_, self.p_], (len(X), 1))

    def predict(self, X):
        return np.full(len(X), self.p_ >= 0.5)


@pytest.fixture
def sample():
    rng = np.random.default_rng(73)
    n = 600
    X = pd.DataFrame(dict(longitude=rng.uniform(-100, -80, n),
                          latitude=rng.uniform(30, 45, n), DOY=rng.uniform(1, 80, n),
                          row_code=np.arange(n), env=rng.normal(size=n)),
                     index=np.arange(n) * 7 + 100)
    X["env"] = X["env"].astype("float32")
    y = pd.DataFrame({"occ": (X.env + rng.normal(size=n) > 0).astype("int8")}, index=X.index)
    return X, y


def make_model(**kwargs):
    return AdaSTEMClassifier(base_model=RecordingClassifier(), ensemble_fold=2,
        min_ensemble_required=1, grid_len_upper_threshold=40, grid_len_lower_threshold=5,
        points_lower_threshold=15, temporal_start=1, temporal_end=81, temporal_step=40,
        temporal_bin_interval=40, temporal_bin_start_jitter=0, random_state=32,
        save_gridding_plot=False, sample_weights_for_classifier=False, verbosity=0,
        **kwargs)


def sources(X, y, kind, path):
    if kind == "pandas":
        return X, y
    paths = []
    for name, frame in [("X", X), ("y", y)]:
        frame = frame.copy()
        frame["__index_level_0__"] = frame.index
        target = path / f"{name}.{kind}"
        if kind == "duckdb":
            with duckdb.connect(str(target)) as con:
                con.execute("CREATE TABLE tmp AS SELECT * FROM frame")
        else:
            frame.to_parquet(target, index=False)
        paths.append(str(target))
    return paths


@pytest.mark.parametrize("kind", ["pandas", "duckdb", "parquet"])
def test_memmap_name_matches_shared_alias(sample, tmp_path, kind):
    X, y = sample
    xs, ys = sources(X, y, kind, tmp_path)
    legacy = make_model(ensemble_bootstrap=True).fit(
        xs, ys, training_data_backend="shared", temporal_window_prequery=True)
    shared = make_model(ensemble_bootstrap=True).fit(
        xs, ys, training_data_backend="memmap", temporal_window_prequery=True)
    pd.testing.assert_frame_equal(shared.ensemble_df, legacy.ensemble_df)
    for key, estimator in shared.model_dict.items():
        reference = legacy.model_dict[key]
        if hasattr(estimator, "rows_"):
            assert_array_equal(estimator.rows_, reference.rows_)
            assert_array_equal(estimator.labels_, reference.labels_)
    expected = legacy.predict_proba(xs, return_by_separate_ensembles=True)
    actual = shared.predict_proba(xs, prediction_backend="memmap",
                                  return_by_separate_ensembles=True)
    assert_allclose(actual, expected, equal_nan=True, atol=1e-13, rtol=0)
    for model in [shared, legacy]:
        assert model.execution_info_["training_data_backend"] == "memmap"
        assert model.execution_info_["shared_input_bytes"] > 0
        assert not os.path.exists(model.joblib_tmp_dir)


@pytest.mark.parametrize("kind", ["pandas", "duckdb", "parquet"])
@pytest.mark.parametrize("backend", ["native", "memmap"])
@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize("prequery", [False, True])
def test_bootstrap_rows_match_direct_oracle(sample, tmp_path, kind, backend, bootstrap, prequery):
    X, y = sample
    model = make_model(ensemble_bootstrap=bootstrap)
    xs, ys = sources(X, y, kind, tmp_path)
    model.fit(xs, ys, temporal_window_prequery=prequery, training_data_backend=backend)
    recorded = 0
    for _, grid in model.ensemble_df.iterrows():
        est = model.model_dict.get(f"{grid.unique_stixel_id}_model")
        if est is None or not hasattr(est, "rows_"):
            continue
        recorded += 1
        info = model.ensemble_df[model.ensemble_df.ensemble_index == grid.ensemble_index]
        coordinates = transform_pred_set_to_STEM_quad("longitude", "latitude", X, info)
        mask = ((X.DOY >= grid.DOY_start) & (X.DOY < grid.DOY_start + 40) &
                (coordinates.longitude_new >= grid.stixel_calibration_point_transformed_left_bound) &
                (coordinates.longitude_new < grid.stixel_calibration_point_transformed_right_bound) &
                (coordinates.latitude_new >= grid.stixel_calibration_point_transformed_lower_bound) &
                (coordinates.latitude_new < grid.stixel_calibration_point_transformed_upper_bound))
        ids = X.index.to_numpy()
        if bootstrap:
            ids = np.random.default_rng(grid.bootstrap_random_state).choice(ids, len(ids), replace=True)
        expected = X.loc[ids[np.isin(ids, X.index[mask])]].sort_index()
        assert_array_equal(est.rows_, expected.row_code)
        assert_array_equal(est.labels_, y.loc[expected.index, "occ"])
    assert recorded > 0
    assert_array_equal(model.classes_, [0, 1])
    assert not hasattr(model, "_shared_training_data")
    assert not os.path.exists(model.joblib_tmp_dir)


@pytest.mark.parametrize("jobs,lazy", [(1, False), (2, False), (2, True)])
def test_parallel_shared_predictions_and_features(sample, tmp_path, jobs, lazy):
    X, y = sample
    reference = make_model()
    reference.base_model = DecisionTreeClassifier(max_depth=3, random_state=8)
    reference.fit(X, y, temporal_window_prequery=True)
    rewritten = make_model(n_jobs=jobs, lazy_loading=lazy,
                           lazy_loading_dir=str(tmp_path / "models"))
    rewritten.base_model = DecisionTreeClassifier(max_depth=3, random_state=8)
    rewritten.fit(X, y, temporal_window_prequery=True, training_data_backend="memmap",
                  inner_max_num_threads=1)
    for aggregation in ["mean", "median"]:
        expected = reference.predict_proba(X, aggregation=aggregation, temporal_window_prequery=True)
        actual = rewritten.predict_proba(X, aggregation=aggregation, temporal_window_prequery=True)
        assert_allclose(actual, expected, equal_nan=True)
    assert_allclose(rewritten.predict_proba(X, return_by_separate_ensembles=True),
                    reference.predict_proba(X, return_by_separate_ensembles=True), equal_nan=True)
    rewritten.calculate_feature_importances()
    assert len(rewritten.feature_importances_) > 0


def test_window_repeats_each_bootstrap_occurrence_once():
    coordinates = pd.DataFrame({"longitude": [0, 1, 2]}, index=[10, 20, 30])
    window = TrainingWindow(coordinates, lambda positions: coordinates.iloc[positions],
                            np.array([10, 10, 30, 20, 10, 99]))
    assert_array_equal(window.select([True, False, True]).index, [10, 10, 30, 10])


def test_shared_data_is_readonly_and_small_when_serialized(sample, tmp_path):
    X, y = sample
    shared = SharedTrainingData.build(X, y.iloc[::-1], {"threads": "1"}, tmp_path)
    assert shared.array("env").dtype == np.float32
    assert not shared.array("env").flags.writeable
    assert_array_equal(shared.labels, y.occ)
    assert len(pickle.dumps(shared)) < 10000


def test_shared_blocks_preserve_order_dtypes_and_do_not_mutate_source(sample, tmp_path):
    X, y = sample
    X["second_float"] = X.longitude * 2
    X["third_float"] = X.longitude * 3
    X["__label__"] = X.env * 2
    shared = SharedTrainingData.build(X, y, {"threads": "1"}, tmp_path, batch_size=113)
    positions = np.array([27, 5, 27, 100])
    actual = shared.frame(positions)
    expected = X.iloc[positions].copy()
    expected.index.name = "__index_level_0__"
    pd.testing.assert_frame_equal(actual, expected)
    assert shared.paths["second_float"][0] == shared.paths["third_float"][0]
    assert_array_equal(shared.labels, y.occ)
    actual.iloc[0, actual.columns.get_loc("third_float")] = -900
    assert shared.array("third_float")[27] == X.third_float.iloc[27]


def test_logit_aggregation_preserves_positive_probabilities(sample):
    X, y = sample
    model = make_model().fit(X, y)
    separate = model.predict_proba(X, return_by_separate_ensembles=True)
    clipped = np.clip(separate, 1e-16, 1 - 1e-16)
    with np.errstate(invalid="ignore"):
        expected = 1 / (1 + np.exp(-np.nanmean(np.log(clipped / (1 - clipped)), axis=1)))
    actual = model.predict_proba(X, logit_agg=True)[:, 1]
    assert np.nanmax(actual) > 0.1
    assert_allclose(actual, expected, equal_nan=True)


def test_import_does_not_override_thread_configuration():
    env = {**os.environ, "OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3"}
    result = subprocess.run([sys.executable, "-c", "import os; import stemflow.model.AdaSTEM; "
                             "assert os.environ['OMP_NUM_THREADS'] == '3'; "
                             "assert os.environ['MKL_NUM_THREADS'] == '3'"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["native", "memmap"])
@pytest.mark.parametrize("value", [0, 1])
def test_single_class_stixels_keep_intended_constant(sample, backend, value):
    X, y = sample
    y = y * 0 + value
    model = make_model().fit(X, y, training_data_backend=backend)
    p = model.predict_proba(X, temporal_window_prequery=True)[:, 1]
    assert np.isfinite(p).any()
    assert_allclose(p[np.isfinite(p)], value)
    outside = X.copy()
    outside["longitude"] = 1000
    with pytest.raises(ValueError, match="not predictable"):
        model.predict_proba(outside)


@pytest.mark.parametrize("backend", ["native", "memmap"])
def test_shared_filter_receives_pandas_and_can_mutate(sample, backend):
    X, y = sample
    original = X.copy(deep=True)
    def keep_rows(frame):
        assert isinstance(frame, pd.DataFrame)
        frame = frame[frame.row_code % 2 == 0].copy()
        frame["env"] = frame.env + 1
        return frame
    model = make_model().fit(X, y, training_data_backend=backend, stixel_filter_func=keep_rows)
    recorded = [m for m in model.model_dict.values() if hasattr(m, "rows_")]
    assert recorded
    assert all(np.all(m.rows_ % 2 == 0) for m in recorded)
    for estimator in recorded:
        assert_array_equal(estimator.labels_.ravel(), y.occ.to_numpy()[estimator.rows_.astype(int)])
    pd.testing.assert_frame_equal(X, original)


def test_corrected_quadtree_spelling_keeps_old_alias(sample):
    X, _ = sample
    criterion = lambda frame: len(frame) > 10
    old, new = make_model(), make_model()
    old.split(X, quadtree_arg_dict={"additional_features": ["env"],
                                  "addiitonal_quadtree_criteria": criterion})
    new.split(X, quadtree_arg_dict={"additional_features": ["env"],
                                  "additional_quadtree_criteria": criterion})
    pd.testing.assert_frame_equal(old.ensemble_df, new.ensemble_df)


def test_execution_worker_cap_and_invalid_parameters(sample):
    X, y = sample
    model = make_model(n_jobs=-1).fit(X, y, max_concurrent_ensembles=1)
    assert model.execution_info_["workers"] == 1
    for keyword in [dict(training_data_backend="unknown"), dict(inner_max_num_threads=0),
                    dict(max_concurrent_ensembles=False)]:
        with pytest.raises(ValueError):
            make_model().fit(X, y, **keyword)
    with pytest.raises(ValueError, match="n_jobs"):
        make_model(n_jobs=0).fit(X, y)


def test_duckdb_source_files_remain_readonly(sample, tmp_path):
    import hashlib
    from pathlib import Path
    xs, ys = sources(*sample, "duckdb", tmp_path)
    before = [hashlib.sha256(Path(p).read_bytes()).digest() for p in [xs, ys]]
    make_model(n_jobs=2).fit(xs, ys, training_data_backend="memmap", temporal_window_prequery=True)
    after = [hashlib.sha256(Path(p).read_bytes()).digest() for p in [xs, ys]]
    assert before == after
    assert not list(tmp_path.glob("*.wal"))


@pytest.mark.parametrize("jobs", [1, 2])
def test_failed_fit_cleans_shared_files_and_adds_context(sample, jobs, monkeypatch):
    if jobs > 1:
        from joblib._parallel_backends import LokyBackend
        def graceful_abort(backend, ensure_ready=True):
            # Controlled failures finish quickly; avoid OS signals denied by sandboxes.
            backend._workers.terminate(kill_workers=False)
            backend._workers = None
            if ensure_ready:
                backend.configure(n_jobs=backend.parallel.n_jobs, parallel=backend.parallel)
        monkeypatch.setattr(LokyBackend, "abort_everything", graceful_abort)
    X, y = sample
    model = make_model(n_jobs=jobs)
    def fail(frame):
        raise RuntimeError("intentional estimator failure")
    with pytest.raises(RuntimeError, match="intentional") as caught:
        model.fit(X, y, training_data_backend="memmap", stixel_filter_func=fail)
    if sys.version_info >= (3, 11):
        assert any("ensemble=" in note and "stixel=" in note for note in caught.value.__notes__)
    assert not hasattr(model, "_shared_training_data")
    assert not os.path.exists(model.joblib_tmp_dir)
    if jobs > 1:
        import multiprocessing as mp
        assert not [p for p in mp.active_children() if p.name.startswith("LokyProcess")]


def test_worker_thread_limit_is_applied(sample):
    X, y = sample
    class ThreadRecorder(RecordingClassifier):
        def fit(self, X, y, sample_weight=None):
            self.worker_omp_ = os.environ.get("OMP_NUM_THREADS")
            return super().fit(X, y, sample_weight)
    model = make_model(n_jobs=2)
    model.base_model = ThreadRecorder()
    model.fit(X, y, inner_max_num_threads=2)
    fitted = [m for m in model.model_dict.values() if hasattr(m, "worker_omp_")]
    assert fitted
    assert all(m.worker_omp_ == "2" for m in fitted)


def test_named_series_target_keeps_values_and_validates_index(sample):
    X, y = sample
    model = make_model().fit(X, y.occ.rename("occurrence"))
    assert_array_equal(model.classes_, [0, 1])
    with pytest.raises(ValueError, match="Indexes"):
        make_model().fit(X, y.occ.iloc[::-1])


def test_prediction_payload_contains_only_assigned_ensemble(sample):
    X, y = sample
    model = make_model().fit(X, y)
    ensemble = model.ensemble_df[model.ensemble_df.ensemble_index == 0]
    worker = model._prediction_worker(ensemble)
    expected = set(ensemble.unique_stixel_id)
    assert worker is not model
    assert set(worker.model_dict) == {f"{name}_model" for name in expected
                                     if f"{name}_model" in model.model_dict}
    assert len(worker.model_dict) < len(model.model_dict)
    assert len(pickle.dumps(worker)) < len(pickle.dumps(model))
    assert "_finalizer" in model.__dict__
    assert "_finalizer" not in worker.__dict__


def test_custom_prediction_subclass_keeps_original_state(sample):
    class CustomPredictor(AdaSTEMClassifier):
        def SAC_ensemble_predict(self, *args, **kwargs):
            return super().SAC_ensemble_predict(*args, **kwargs)
    model = CustomPredictor(base_model=RecordingClassifier())
    assert model._prediction_worker(pd.DataFrame()) is model
