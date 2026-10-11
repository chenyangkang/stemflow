"""Bounded temporal queries preserve pointwise predictions and coverage."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stemflow.lazyloading.open_db_connection import open_db_connection
from stemflow.lazyloading.training_data import iter_prediction_windows
from stemflow.model.AdaSTEM import AdaSTEMClassifier, AdaSTEMRegressor
from .test_execution_rewrite import make_model, sample, sources


@pytest.mark.parametrize("kind", ["pandas", "duckdb", "parquet"])
def test_temporal_reader_bounds_rows_and_preserves_values(sample, tmp_path, kind):
    X, y = sample
    X = X.iloc[::-1].copy()
    xs, _ = sources(X, y, kind, tmp_path)
    columns = ["DOY", "longitude", "latitude"]
    with open_db_connection(xs, {"threads": "1"}) as (source, con):
        class CountingConnection:
            def __init__(self):
                self.queries = []
            def register(self, *args):
                return con.register(*args)
            def sql(self, query):
                self.queries.append(query)
                return con.sql(query)
        counting = CountingConnection()
        actual = []
        for window in iter_prediction_windows(source, counting, 20, 60, columns, True, 37):
            assert len(window.coordinates) <= 37
            frame = window.select_positions(np.arange(len(window.coordinates)))
            assert not frame.empty
            actual.append(frame)
            # A base estimator can mutate its workspace without changing the source.
            private = window.select_positions(np.array([0]))
            original = window.select_positions(np.array([0])).env.iloc[0]
            private.iloc[0, private.columns.get_loc("env")] = -999
            assert window.select_positions(np.array([0])).env.iloc[0] == original
        assert len(counting.queries) == (0 if kind == "pandas" else 1)
        assert list(iter_prediction_windows(source, con, 999, 1000, columns, True, 37)) == []
    actual = pd.concat(actual)
    expected = X.loc[(X.DOY >= 20) & (X.DOY < 60)].copy()
    pd.testing.assert_frame_equal(actual, expected, check_names=False)


@pytest.mark.parametrize("kind", ["pandas", "duckdb", "parquet"])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("jobs", [1, 2])
def test_batched_prediction_matches_whole_window(sample, tmp_path, kind, overlap, jobs):
    X, y = sample
    X = X.iloc[::-1].copy()
    y = y.reindex(X.index)
    xs, ys = sources(X, y, kind, tmp_path)
    model = make_model(n_jobs=jobs, lazy_loading=True, lazy_loading_dir=str(tmp_path / "models"))
    model.base_model = DecisionTreeClassifier(max_depth=3, random_state=4)
    if overlap:
        model.temporal_step = 20
    model.fit(xs, ys, temporal_window_prequery=True, persist_prediction_state=False)
    for logit in [False, True]:
        expected, expected_sd = model.predict_proba(xs, temporal_window_prequery=True,
            return_std=True, logit_agg=logit)
        actual, actual_sd = model.predict_proba(xs, temporal_window_prequery=True,
            return_std=True, logit_agg=logit, prediction_batch_size=73)
        assert model.prediction_execution_info_["prediction_batch_size"] == 73
        assert_allclose(actual, expected, equal_nan=True, atol=2e-14)
        assert_allclose(actual_sd, expected_sd, equal_nan=True, atol=2e-14)
    expected = model.predict_proba(xs, temporal_window_prequery=True,
        return_by_separate_ensembles=True)
    actual = model.predict_proba(xs, temporal_window_prequery=True,
        return_by_separate_ensembles=True, prediction_batch_size=73)
    assert_allclose(actual, expected, equal_nan=True)
    # An ordinary call restores the original whole-window route.
    model.predict_proba(xs, temporal_window_prequery=True)
    assert model.prediction_execution_info_["prediction_batch_size"] is None


def test_regression_and_dummy_predictions_keep_nan_coverage(sample, tmp_path):
    X, y = sample
    xs, ys = sources(X, y, "duckdb", tmp_path)
    classification = make_model()
    classification.fit(X, np.zeros(len(X)), temporal_window_prequery=True)
    expected = classification.predict_proba(xs, temporal_window_prequery=True)
    actual = classification.predict_proba(xs, temporal_window_prequery=True, prediction_batch_size=41)
    assert_allclose(actual, expected, equal_nan=True)
    regression = AdaSTEMRegressor(base_model=DecisionTreeRegressor(max_depth=2, random_state=9),
        ensemble_fold=2, min_ensemble_required=2, points_lower_threshold=20,
        grid_len_upper_threshold=40, grid_len_lower_threshold=5, temporal_start=1,
        temporal_end=81, temporal_step=20, temporal_bin_interval=40,
        save_gridding_plot=False, random_state=7, n_jobs=2)
    regression.fit(xs, ys, temporal_window_prequery=True)
    a, sa = regression.predict(xs, temporal_window_prequery=True, return_std=True)
    b, sb = regression.predict(xs, temporal_window_prequery=True, return_std=True,
        prediction_batch_size=41)
    assert_allclose(a, b, equal_nan=True, atol=2e-14)
    assert_allclose(sa, sb, equal_nan=True, atol=2e-14)


@pytest.mark.parametrize("batch_size", [True, 0, -1, 2.5, "100"])
def test_invalid_batch_size_rejected(sample, batch_size):
    X, y = sample
    model = make_model().fit(X, y)
    with pytest.raises(ValueError, match="positive integer"):
        model.predict_proba(X, temporal_window_prequery=True, prediction_batch_size=batch_size)


def test_batches_require_prequery_and_builtin_ensemble(sample):
    X, y = sample
    model = make_model().fit(X, y)
    with pytest.raises(ValueError, match="temporal_window_prequery"):
        model.predict_proba(X, prediction_batch_size=100)
    class CustomEnsemble(AdaSTEMClassifier):
        def SAC_ensemble_predict(self, *args, **kwargs):
            return super().SAC_ensemble_predict(*args, **kwargs)
    custom = CustomEnsemble(**model.get_params(deep=False))
    with pytest.raises(NotImplementedError, match="built-in planar"):
        custom.predict_proba(X, temporal_window_prequery=True, prediction_batch_size=100)


def test_beta_xgboost_batches_match_whole_window(sample, tmp_path):
    from stemflow.tuning import local_beta_xgboost
    X, y = sample
    xs, ys = sources(X, y, "duckdb", tmp_path)
    model = make_model(n_jobs=2, lazy_loading=True, lazy_loading_dir=str(tmp_path / "models"))
    model.base_model = local_beta_xgboost(tuning=False, n_estimators=5, n_splits=2,
        Spatio_blocks_count=2, Temporal_blocks_count=2, ridge=100)
    model.fit(xs, ys, temporal_window_prequery=True, persist_prediction_state=False)
    a, sa = model.predict_proba(xs, return_std=True, temporal_window_prequery=True)
    b, sb = model.predict_proba(xs, return_std=True, temporal_window_prequery=True,
        prediction_batch_size=53)
    assert_allclose(a, b, equal_nan=True, atol=3e-7, rtol=3e-7)
    assert_allclose(sa, sb, equal_nan=True, atol=3e-7, rtol=3e-7)


def test_legacy_arrow_reader_is_supported_and_closed(sample, tmp_path):
    X, y = sample
    xs, _ = sources(X, y, "duckdb", tmp_path)
    columns = ["DOY", "longitude", "latitude"]
    with open_db_connection(xs, {"threads": "1"}) as (source, con):
        class LegacyRelation:
            def __init__(self, relation):
                self.fetch_record_batch = relation.fetch_record_batch
        class LegacyConnection:
            def register(self, *args):
                return con.register(*args)
            def sql(self, query):
                return LegacyRelation(con.sql(query))
        frames = [window.select_positions(np.arange(len(window.coordinates)))
                  for window in iter_prediction_windows(source, LegacyConnection(),
                      20, 60, columns, True, 37)]
    expected = X.loc[(X.DOY >= 20) & (X.DOY < 60)]
    pd.testing.assert_frame_equal(pd.concat(frames), expected, check_names=False)


def test_arrow_reader_closes_if_consumer_stops(sample):
    X, _ = sample
    class Batch:
        def to_pandas(self):
            return X.iloc[:5].assign(__index_level_0__=X.index[:5])
    class Reader:
        closed = False
        def __iter__(self):
            yield Batch()
        def close(self):
            self.closed = True
    reader = Reader()
    class Connection:
        def register(self, *args):
            pass
        def sql(self, query):
            return self
        def to_arrow_reader(self, batch_size):
            return reader
    windows = iter_prediction_windows(object(), Connection(), 1, 80,
        ["DOY", "longitude", "latitude"], True, 5)
    assert len(next(windows).coordinates) == 5
    windows.close()
    assert reader.closed
