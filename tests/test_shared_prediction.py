"""Compatibility, ownership, and failure cleanup for shared prediction."""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from stemflow.model.AdaSTEM import AdaSTEMRegressor
from stemflow.lazyloading.shared_prediction import _DenseMoments
from .test_execution_rewrite import RecordingClassifier, make_model, sample, sources


class SpecialClassifier(RecordingClassifier):
    def special(self, X, offset=0):
        return np.full(len(X), self.p_ + offset)


class ExplodingClassifier(RecordingClassifier):
    def predict_proba(self, X):
        raise RuntimeError('Intentional prediction failure')


@pytest.mark.parametrize('kind,jobs', [('pandas', 1), ('duckdb', 2), ('parquet', 2)])
def test_shared_matches_native_modes(sample, tmp_path, kind, jobs):
    X, y = sample
    model = make_model(n_jobs=jobs, lazy_loading=True,
                       lazy_loading_dir=str(tmp_path / 'models'))
    model.temporal_step = 10
    model.temporal_bin_interval = 60
    model.fit(X, y, temporal_window_prequery=True, persist_prediction_state=False)
    encoded = pickle.dumps(model)
    prediction = X.sample(frac=1, random_state=81).copy()
    prediction.loc[prediction.index[:4], 'longitude'] = 1000
    directory = tmp_path / 'prediction'
    directory.mkdir()
    source, _ = sources(prediction, y.reindex(prediction.index), kind, directory)
    for options in [dict(return_std=True), dict(return_std=True, logit_agg=True),
                    dict(return_std=True, aggregation='median', logit_agg=True),
                    dict(return_by_separate_ensembles=True), dict(streaming=False, return_std=True)]:
        expected = model.predict_proba(source, temporal_window_prequery=True, **options)
        candidate = pickle.loads(encoded)
        actual = candidate.predict_proba(source, temporal_window_prequery=True,
                                         prediction_backend='memmap', **options)
        if isinstance(expected, tuple):
            for left, right in zip(actual, expected):
                np.testing.assert_allclose(left, right, equal_nan=True, atol=1e-13, rtol=0)
        else:
            assert actual.shape == (len(prediction), model.ensemble_fold)
            np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-13, rtol=0)
        info = candidate.prediction_execution_info_
        assert info['max_pending_tasks'] <= jobs
        assert info['result_transport'] == 'completion_records'
        assert not Path(candidate.joblib_tmp_dir).exists()
    output = tmp_path / 'ensemble_predictions.npy'
    actual = model.predict_proba(source, temporal_window_prequery=True,
        prediction_backend='memmap', return_by_separate_ensembles=True, prediction_output_path=output)
    expected = model.predict_proba(source, temporal_window_prequery=True, return_by_separate_ensembles=True)
    assert isinstance(actual, np.memmap)
    np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-13, rtol=0)
    actual[4, 0] = 0.123
    actual.flush()
    assert np.load(output, mmap_mode='r')[4, 0] == 0.123
    assert not Path(model.joblib_tmp_dir).exists()
    with pytest.raises(FileExistsError):
        model.predict_proba(source, prediction_backend='memmap',
            return_by_separate_ensembles=True, prediction_output_path=output)


def test_shared_regression_custom_method_and_dummy_zeros(sample, tmp_path):
    X, y = sample
    params = make_model().get_params(deep=False)
    params.pop('task', None)
    params['base_model'] = DecisionTreeRegressor(max_depth=2, random_state=9)
    model = AdaSTEMRegressor(**params).fit(X, X.env.to_numpy(), temporal_window_prequery=True)
    for aggregation in ['mean', 'median']:
        expected = model.predict(X, return_std=True, aggregation=aggregation, temporal_window_prequery=True)
        actual = model.predict(X, return_std=True, aggregation=aggregation,
                              temporal_window_prequery=True, prediction_backend='memmap')
        for left, right in zip(actual, expected):
            np.testing.assert_allclose(left, right, equal_nan=True, atol=1e-13, rtol=0)
    model = make_model()
    model.base_model = SpecialClassifier()
    model.fit(X, y, temporal_window_prequery=True)
    for offset in [0.2, np.nan]:
        expected = model.predict_proba(X, base_model_method='special', offset=offset)
        actual = model.predict_proba(X, base_model_method='special', offset=offset,
                                     prediction_backend='memmap')
        np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-13, rtol=0)
    model = make_model().fit(X, np.zeros(len(X)), temporal_window_prequery=True)
    expected = model.predict_proba(X, return_by_separate_ensembles=True)
    actual = model.predict_proba(X, prediction_backend='memmap', return_by_separate_ensembles=True)
    np.testing.assert_array_equal(actual, expected)
    assert np.nanmax(actual) == 0


def test_shared_worker_failure_cleans_temporary_files(sample, tmp_path):
    import multiprocessing
    X, y = sample
    model = make_model(n_jobs=2)
    model.base_model = ExplodingClassifier()
    model.fit(X, y, temporal_window_prequery=True)
    existing_children = {child.pid for child in multiprocessing.active_children()}
    destination = tmp_path / 'failed.npy'
    with pytest.raises(RuntimeError, match='Intentional prediction failure'):
        model.predict_proba(X, prediction_backend='memmap', return_by_separate_ensembles=True,
                            prediction_output_path=destination)
    assert not destination.exists()
    assert not list(tmp_path.glob('.failed_*'))
    assert not Path(model.joblib_tmp_dir).exists()
    assert {child.pid for child in multiprocessing.active_children()} <= existing_children


def test_shared_validates_output_options_and_ids(sample, tmp_path):
    X, y = sample
    model = make_model().fit(X, y)
    expected = model.predict_proba(X, prediction_backend='memmap')
    for alias in ['shared', 'bounded']:
        actual = model.predict_proba(X, prediction_backend=alias)
        np.testing.assert_allclose(actual, expected, equal_nan=True)
        assert model.prediction_execution_info_['prediction_backend'] == 'memmap'
    with pytest.raises(ValueError, match='requires return_by_separate_ensembles'):
        model.predict_proba(X, prediction_backend='memmap', prediction_output_path=tmp_path / 'x.npy')
    duplicate = X.copy()
    duplicate.index = np.zeros(len(duplicate), dtype=int)
    with pytest.raises(ValueError, match='unique'):
        model.predict_proba(duplicate, prediction_backend='memmap')
    assert not Path(model.joblib_tmp_dir).exists()


def test_shared_does_not_return_unfilled_columns_for_an_incomplete_grid(sample):
    X, y = sample
    model = make_model().fit(X, y)
    model.ensemble_df = model.ensemble_df[model.ensemble_df.ensemble_index ==
                                           model.ensemble_df.ensemble_index.iloc[0]]
    expected = model.predict_proba(X)
    actual = model.predict_proba(X, prediction_backend='memmap')
    np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-13, rtol=0)
    with pytest.raises(ValueError, match='must contain ensemble_fold ensembles'):
        model.predict_proba(X, prediction_backend='memmap', return_by_separate_ensembles=True)
    assert not Path(model.joblib_tmp_dir).exists()


@pytest.mark.parametrize('logit', [False, True])
def test_shared_moments_minimum_coverage_and_chunk_boundaries(logit):
    from stemflow.lazyloading.prediction import PredictionMoments
    values = np.array([[0., 1., .2], [np.nan, np.nan, .5],
                       [.5, .5, .5], [np.nan, np.nan, np.nan]])
    oracle = PredictionMoments(pd.Index([42, 4, 100, 8]), with_std=True, logit=logit)
    candidate = _DenseMoments(len(values), with_std=True, logit=logit)
    for column in range(values.shape[1]):
        oracle.update(pd.DataFrame({'pred': values[:, column]}, index=oracle.index))
        candidate.update(values[:2, column], 0)
        candidate.update(values[2:, column], 2)
    expected, expected_std = oracle.finish(2)
    actual, actual_std = candidate.finish(2)
    np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-13, rtol=0)
    np.testing.assert_allclose(actual_std, expected_std, equal_nan=True, atol=1e-13, rtol=0)


@pytest.mark.parametrize('regression', [False, True])
def test_shared_forest_float32_gather_preserves_predictions(sample, tmp_path, regression):
    X, y = sample
    params = make_model().get_params(deep=False)
    params['lazy_loading'] = True
    params['lazy_loading_dir'] = str(tmp_path / 'models')
    if regression:
        params.pop('task', None)
        params['base_model'] = RandomForestRegressor(n_estimators=7, max_depth=3, n_jobs=1, random_state=5)
        model = AdaSTEMRegressor(**params).fit(X, X.env.to_numpy(), persist_prediction_state=False)
    else:
        params['base_model'] = RandomForestClassifier(n_estimators=7, max_depth=3, n_jobs=1, random_state=5)
        model = type(make_model())(**params).fit(X, y, persist_prediction_state=False)
    expected = model.predict_proba(X, return_std=True, temporal_window_prequery=True)
    for backend in ['loky', 'threading']:
        model.joblib_backend = backend
        actual = model.predict_proba(X, return_std=True, temporal_window_prequery=True,
                                     prediction_backend='memmap', n_jobs=2)
        for left, right in zip(actual, expected):
            np.testing.assert_allclose(left, right, equal_nan=True, atol=1e-13, rtol=0)


@pytest.mark.parametrize('persist', [False, True])
def test_shared_respects_prediction_state_persistence(sample, persist):
    from .test_prediction_streaming import StatefulClassifier
    X, y = sample
    model = make_model(lazy_loading=True)
    model.base_model = StatefulClassifier()
    model.fit(X, y, persist_prediction_state=persist)
    first = model.predict_proba(X, prediction_backend='memmap')[:, 1]
    second = model.predict_proba(X, prediction_backend='memmap')[:, 1]
    valid = (first > 0) & (first < 1)
    assert valid.any()
    np.testing.assert_allclose(second[valid]-first[valid], .1 if persist else 0, atol=1e-14, rtol=0)


def test_thread_tasks_do_not_release_another_readers_maps(sample, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from stemflow.lazyloading.shared_prediction import _ensemble_to_slot, _shared_input
    X, y = sample
    model = make_model().fit(X, y)
    model.base_model_method, model.base_model_prediction_param = None, {}
    shared = _shared_input(model, X, tmp_path / 'input')
    retained = shared.array('env')
    readers = dict(shared._arrays)
    grids = [grid for _, grid in model.ensemble_df.groupby('ensemble_index')]
    paths = [tmp_path / f'slot_{i}.npy' for i in range(len(grids))]
    for path in paths:
        np.save(path, np.zeros(len(X)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_ensemble_to_slot, model, grid, shared, str(path))
                   for grid, path in zip(grids, paths)]
        assert all(f.result()['estimator_calls'] > 0 for f in futures)
    assert set(shared._arrays) == set(readers)
    assert shared.array('env') is retained
    np.testing.assert_array_equal(retained, X.env)


def test_shared_monitor_report_survives_prediction_and_pickle(sample):
    pytest.importorskip('psutil')
    X, y = sample
    model = make_model().fit(X, y)
    with model.monitor_execution(sample_interval=.05, log_interval=None):
        model.predict_proba(X, prediction_backend='memmap')
    report = model.execution_monitor_
    assert not report['running']
    assert any('shared_prediction_input' in step['name'] for step in report['stages'])
    assert any('prediction_and_aggregation' in step['name'] for step in report['stages'])
    restored = pickle.loads(pickle.dumps(model))
    assert restored.execution_monitor_ == report
    assert restored.prediction_execution_info_['prediction_backend'] == 'memmap'


def test_shared_process_workers_retain_optional_profiling(sample, tmp_path, monkeypatch):
    X, y = sample
    model = make_model().fit(X, y)
    monkeypatch.setenv('STEMFLOW_PROFILE_DIR', str(tmp_path / 'profiles'))
    model.predict_proba(X, n_jobs=2, prediction_backend='memmap')
    records = [json.loads(path.read_text()) for path in (tmp_path / 'profiles').glob('*.json')]
    assert any(record['function'] == '_ensemble_to_slot' for record in records)
    assert any(record['function'] == 'AdaSTEM.predict_proba' for record in records)
    assert all(record['error'] is None for record in records)
