"""Compatibility checks for isolated prediction experiments, not package changes."""

import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/06_prediction_stress'))
from strategies import build_shared_prediction, configure, rotate_arrays
from stemflow.model.static_func_AdaSTEM import transform_pred_set_to_STEM_quad
from .test_execution_rewrite import make_model, sample, sources


@pytest.mark.parametrize('kind', ['pandas', 'duckdb', 'parquet'])
def test_prototypes_preserve_overlap_coverage_and_std(sample, tmp_path, kind):
    X, y = sample
    xs, ys = sources(X, y, kind, tmp_path)
    model = make_model(n_jobs=1, lazy_loading=True)
    model.temporal_bin_interval = 60
    model.fit(xs, ys, temporal_window_prequery=True, persist_prediction_state=False)
    encoded = pickle.dumps(model)
    shared = build_shared_prediction(xs, model.duckdb_config, tmp_path / 'shared', validate_index=True)
    for logit in [False, True]:
        expected, expected_std = model.predict_proba(xs, temporal_window_prequery=True,
                                                     return_std=True, logit_agg=logit)
        for strategy in ['rotation', 'lean', 'arrow', 'shared', 'shared_positions', 'shared_threads', 'lean_threads']:
            candidate = configure(pickle.loads(encoded), strategy, shared)
            actual, std = candidate.predict_proba(xs, temporal_window_prequery=True,
                                                  return_std=True, logit_agg=logit, n_jobs=1)
            np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-13, rtol=0)
            np.testing.assert_allclose(std, expected_std, equal_nan=True, atol=1e-13, rtol=0)
    before = X.copy(deep=True)
    selected = shared.frame(np.array([4, 1, 7]))
    selected.iloc[0, 0] = -999
    pd.testing.assert_frame_equal(X, before)
    assert shared.frame(np.array([4])).iloc[0, 0] != -999


def test_array_rotation_is_identical_to_original(sample, tmp_path):
    X, y = sample
    model = make_model().fit(X, y)
    for _, grid in model.ensemble_df.groupby('ensemble_index'):
        transformed = transform_pred_set_to_STEM_quad('longitude', 'latitude', X, grid)
        x, y = rotate_arrays(X, grid, 'longitude', 'latitude')
        np.testing.assert_array_equal(x, transformed.longitude_new)
        np.testing.assert_array_equal(y, transformed.latitude_new)


def test_parallel_shared_prediction_and_intentional_zeros(sample, tmp_path):
    X, y = sample
    y.iloc[:] = 0
    xs, ys = sources(X, y, 'duckdb', tmp_path)
    model = make_model(n_jobs=2, lazy_loading=True).fit(
        xs, ys, temporal_window_prequery=True, persist_prediction_state=False)
    encoded = pickle.dumps(model)
    expected, expected_std = model.predict_proba(xs, temporal_window_prequery=True, return_std=True)
    shared = build_shared_prediction(xs, model.duckdb_config, tmp_path / 'shared')
    for strategy in ['lean', 'arrow', 'shared', 'shared_positions', 'shared_threads', 'lean_threads']:
        candidate = configure(pickle.loads(encoded), strategy, shared)
        actual, std = candidate.predict_proba(xs, temporal_window_prequery=True, return_std=True, n_jobs=2)
        np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-13, rtol=0)
        np.testing.assert_allclose(std, expected_std, equal_nan=True, atol=1e-13, rtol=0)


@pytest.mark.parametrize('logit', [False, True])
def test_float32_xgboost_overlap_and_input_order(sample, tmp_path, logit):
    from xgboost import XGBClassifier
    X, y = sample
    model = make_model(n_jobs=2, lazy_loading=True)
    model.base_model = XGBClassifier(n_estimators=7, max_depth=3, n_jobs=1, random_state=9)
    model.temporal_step = 10
    model.temporal_bin_interval = 60
    xs, ys = sources(X, y, 'duckdb', tmp_path)
    model.fit(xs, ys, temporal_window_prequery=True, persist_prediction_state=False)
    encoded = pickle.dumps(model)
    prediction_frame = X.sample(frac=1, random_state=81).copy()
    prediction_frame.loc[prediction_frame.index[:4], 'longitude'] = 1000
    prediction_frame['DOY'] = prediction_frame.DOY.astype('float64')
    expected, expected_std = model.predict_proba(prediction_frame, temporal_window_prequery=True,
                                                return_std=True, logit_agg=logit)
    shared = build_shared_prediction(prediction_frame, model.duckdb_config, tmp_path / 'shared')
    assert np.isnan(expected[:4, 1]).all()
    for strategy in ['lean', 'arrow', 'shared', 'shared_positions', 'shared_threads']:
        candidate = configure(pickle.loads(encoded), strategy, shared)
        actual, std = candidate.predict_proba(prediction_frame, temporal_window_prequery=True,
                                              return_std=True, logit_agg=logit, n_jobs=2)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_array_equal(np.isnan(std), np.isnan(expected_std))
        # Original pandas means use float32; online prototype sums use float64.
        np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-6, rtol=0)
        np.testing.assert_allclose(std, expected_std, equal_nan=True, atol=1e-6, rtol=0)


def test_forest_inventory_counts_fitted_trees_not_template(sample):
    from sklearn.ensemble import RandomForestClassifier
    from run_stress import estimator_inventory
    X, y = sample
    forest = RandomForestClassifier(n_estimators=3, max_depth=2, n_jobs=1).fit(X, y.values.ravel())
    assert estimator_inventory(forest) == dict(fitted_estimators=1, retained_trees=3, trial_records=0)


def test_shared_positions_reject_duplicate_ids(sample, tmp_path):
    X, _ = sample
    X.index = np.zeros(len(X), dtype=int)
    with pytest.raises(ValueError, match='unique and non-null'):
        build_shared_prediction(X, {'threads': '1'}, tmp_path / 'shared', validate_index=True)


@pytest.mark.parametrize('kind,jobs', [('pandas', 1), ('duckdb', 2), ('parquet', 2)])
def test_separate_ensemble_columns_preserved(sample, tmp_path, kind, jobs):
    X, y = sample
    model = make_model(n_jobs=jobs, lazy_loading=True,
                       lazy_loading_dir=str(tmp_path / 'models'))
    model.temporal_step = 10
    model.temporal_bin_interval = 60
    model.fit(X, y, temporal_window_prequery=True, persist_prediction_state=False)
    encoded = pickle.dumps(model)
    prediction = X.sample(frac=1, random_state=81).copy()
    prediction.loc[prediction.index[:4], 'longitude'] = 1000
    prediction_dir = tmp_path / 'prediction'
    prediction_dir.mkdir()
    source, _ = sources(prediction, y.reindex(prediction.index), kind, prediction_dir)
    expected = model.predict_proba(source, temporal_window_prequery=True,
                                   return_by_separate_ensembles=True)
    assert expected.shape == (len(prediction), model.ensemble_fold)
    assert np.isnan(expected[:4]).all()
    common = np.isfinite(expected).all(axis=1)
    assert np.any(np.abs(expected[common, 0] - expected[common, 1]) > 1e-6)
    shared = build_shared_prediction(source, model.duckdb_config,
                                     tmp_path / 'shared', validate_index=True)
    for strategy in ['lean', 'arrow', 'shared', 'shared_positions']:
        candidate = configure(pickle.loads(encoded), strategy, shared)
        actual = candidate.predict_proba(source, temporal_window_prequery=True,
                                         return_by_separate_ensembles=True)
        assert candidate.prediction_execution_info_['aggregation_backend'] == 'matrix'
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-13, rtol=0)
