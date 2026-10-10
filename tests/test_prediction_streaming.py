"""Prediction memory scaling, output compatibility and read-only model loading."""

import pickle

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from stemflow.lazyloading.lazyloading import LazyLoadingEstimator
from stemflow.lazyloading.prediction import PredictionMoments, SpatialWindowIndex
from stemflow.lazyloading.training_data import SharedTrainingData, TrainingWindow
from .test_execution_rewrite import make_model, sample, sources


class StatefulClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.calls_, self.classes_ = 0, np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.calls_ += 1
        p = np.full(len(X), self.calls_/10)
        return np.column_stack([1-p, p])


@pytest.mark.parametrize("kind", ["pandas", "duckdb", "parquet"])
@pytest.mark.parametrize("logit", [False, True])
@pytest.mark.parametrize("jobs", [1, 2])
def test_streaming_matches_matrix(sample, tmp_path, kind, logit, jobs):
    X, y = sample
    xs, ys = sources(X, y, kind, tmp_path)
    model = make_model(n_jobs=jobs, ensemble_bootstrap=True, lazy_loading=True).fit(xs, ys)
    a, sa = model.predict_proba(xs, return_std=True, logit_agg=logit,
                               temporal_window_prequery=True)
    assert model.prediction_execution_info_["aggregation_backend"] == "streaming"
    b, sb = model.predict_proba(xs, return_std=True, logit_agg=logit,
                               temporal_window_prequery=True, streaming=False)
    assert_allclose(a, b, atol=2e-15, rtol=2e-14, equal_nan=True)
    assert_allclose(sa, sb, atol=2e-15, rtol=2e-14, equal_nan=True)
    separate = model.predict_proba(xs, return_by_separate_ensembles=True)
    assert separate.shape == (len(X), model.ensemble_fold)
    assert model.prediction_execution_info_["aggregation_backend"] == "matrix"
    median = model.predict_proba(xs, aggregation="median")
    assert_allclose(median, model.predict_proba(xs, aggregation="median", streaming=False), equal_nan=True)


def test_moments_coverage_reordered_ids_and_memory():
    ids = pd.Index([42, 4, 100, 8])
    values = np.array([[.1, .2, .3], [np.nan, np.nan, .5], [.5, .5, .5], [np.nan]*3])
    reducer = PredictionMoments(ids, with_std=True)
    size = reducer.count.nbytes + reducer.mean.nbytes + reducer.m2.nbytes
    for col in range(3):
        reducer.update(pd.DataFrame({"pred": values[::-1, col]}, index=ids[::-1]))
    mean, sd = reducer.finish(2)
    expected = pd.DataFrame(values).mean(axis=1).to_numpy(copy=True)
    expected[np.sum(~np.isnan(values), axis=1) < 2] = np.nan
    assert_allclose(mean, expected, equal_nan=True)
    assert_allclose(sd, pd.DataFrame(values).std(axis=1).to_numpy(), equal_nan=True)
    assert size == len(ids) * 20


def test_lazy_reads_do_not_rewrite_and_legacy_behavior_remains(tmp_path):
    X = pd.DataFrame({"x": np.arange(10)})
    y = np.arange(10) % 2
    for read_dump in [False, True]:
        est = LazyLoadingEstimator(DecisionTreeClassifier(), tmp_path,
            filename=f"model_{read_dump}", auto_dump=True, dump_on_read=read_dump).fit(X, y)
        path = est._resolve_path()
        before = path.stat().st_mtime_ns
        restored = pickle.loads(pickle.dumps(est))
        prediction = restored.predict_proba(X)
        assert restored.estimator is None
        assert_allclose(prediction[:, 1], y)
        after = path.stat().st_mtime_ns
        assert (after == before) if not read_dump else (after > before)


def test_spatial_lookup_matches_masks_and_bootstrap_occurrences():
    rng = np.random.default_rng(25)
    x, y = rng.normal(size=(2, 2000))
    x[:4], y[:4] = [0, 1, 0, 1], [0, 0, 1, 1]
    x[-1] = np.nan
    lookup = SpatialWindowIndex(x, y, 20)
    grid = dict(stixel_calibration_point_transformed_left_bound=0,
        stixel_calibration_point_transformed_right_bound=1,
        stixel_calibration_point_transformed_lower_bound=0,
        stixel_calibration_point_transformed_upper_bound=1)
    expected = np.flatnonzero((x >= 0) & (x < 1) & (y >= 0) & (y < 1))
    assert_array_equal(lookup.select(grid), expected)
    frame = pd.DataFrame({"x": x, "y": y}, index=np.arange(len(x))*3)
    bootstrap = rng.choice(frame.index, len(frame)*2, replace=True)
    window = TrainingWindow(frame, lambda positions: frame.iloc[positions], bootstrap)
    mask = np.zeros(len(frame), bool)
    mask[expected] = True
    pd.testing.assert_frame_equal(window.select_positions(expected), window.select(mask))


def test_wide_shared_stixels_keep_dtype_blocks_and_private_storage(tmp_path):
    X = pd.DataFrame(np.arange(200*130, dtype=float).reshape(200, 130))
    X.columns = [f"feature_{i}" for i in range(130)]
    X["DOY"], X["longitude"], X["latitude"] = 1, 2., 3.
    y = pd.DataFrame({"y": np.arange(len(X)) % 2})
    shared = SharedTrainingData.build(X, y, {"threads": "1"}, tmp_path)
    frame = shared.training_rows(np.array([20, 4, 20]))
    # One pandas block per column previously triggered fragmentation on inserts.
    assert frame._mgr.nblocks < 10
    expected = X.iloc[[20, 4, 20]].copy()
    expected["true_y"] = y.iloc[[20, 4, 20]].y.to_numpy()
    expected.index.name = "__index_level_0__"
    pd.testing.assert_frame_equal(frame, expected)
    frame.iloc[0, 0] = -100
    assert shared.array("feature_0")[20] == X.iloc[20, 0]


def test_constant_feature_filter_computes_std_once(monkeypatch):
    from stemflow.model.static_func_AdaSTEM import train_one_stixel
    X = pd.DataFrame({"signal": np.arange(30), "constant": np.ones(30),
                      "metadata": np.ones(30), "true_y": np.arange(30) % 2})
    base = DecisionTreeClassifier(random_state=42)
    base.required_metadata_columns = ("metadata",)
    count, original = [], pd.DataFrame.std
    def counting(self, *args, **kwargs):
        count.append(len(self.columns))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "std", counting)
    fitted, names, status = train_one_stixel(1, ["signal", "constant", "metadata"],
        "classification", base, False, True, X, 1, None)
    assert status == "Success"
    assert names == ["signal", "metadata"]
    assert count == [3]
    assert fitted._stemflow_training_support_ == {"n": 30, "positives": 15}


def test_float32_xgboost_streaming_has_only_roundoff_difference(sample):
    from xgboost import XGBClassifier
    X, y = sample
    model = make_model()
    model.base_model = XGBClassifier(n_estimators=5, n_jobs=1, random_state=42)
    model.fit(X, y)
    a, sa = model.predict_proba(X, return_std=True)
    b, sb = model.predict_proba(X, return_std=True, streaming=False)
    assert_allclose(a, b, atol=3e-7, rtol=3e-7, equal_nan=True)
    assert_allclose(sa, sb, atol=3e-7, rtol=3e-7, equal_nan=True)


@pytest.mark.parametrize("persist", [True, False])
def test_adastem_preserves_stateful_prediction_by_default(sample, persist):
    X, y = sample
    model = make_model(lazy_loading=True)
    model.base_model = StatefulClassifier()
    model.fit(X, y, persist_prediction_state=persist)
    a = model.predict_proba(X)[:, 1]
    b = model.predict_proba(X)[:, 1]
    valid = (a > 0) & (a < 1)
    assert valid.any()
    assert_allclose(b[valid]-a[valid], .1 if persist else 0, atol=1e-14)
