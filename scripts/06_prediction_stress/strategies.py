"""Experimental prediction strategies; production package code is unchanged."""

import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stemflow.lazyloading.open_db_connection import open_db_connection
from stemflow.lazyloading.prediction import SpatialWindowIndex
from stemflow.lazyloading.training_data import (
    INDEX, SharedTrainingData, TrainingWindow, _write_columns, prediction_window,
    quote_identifier,
)
from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier
from stemflow.model.static_func_AdaSTEM import (
    get_model_and_stixel_specific_x_names, transform_pred_set_to_STEM_quad,
)
from stemflow.model.dummy_model import dummy_model1
from stemflow.utils.execution_profile import profile_execution


def build_shared_prediction(data, config, directory, validate_index=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open_db_connection(data, config) as (source, con):
        paths, n = _write_columns(source, con, 'prediction_cache', directory,
                                 'prediction', 65536, {INDEX, 'DOY', 'longitude', 'latitude'})
    # Only frame()/array() are used; no label allocation or access is needed.
    payload = sum(Path(p if isinstance(p, str) else p[0]).stat().st_size
                  for p in {v if isinstance(v, str) else v[0] for v in paths.values()})
    shared = SharedTrainingData(paths, '', n, payload)
    if validate_index:
        index = pd.Index(shared.index)
        if not index.is_unique or index.hasnans:
            raise ValueError('Prediction observation IDs must be unique and non-null.')
    return shared


def rotate_arrays(frame, grid, spatial1, spatial2):
    angle = float(grid.rotation.iloc[0]) / 360 * np.pi * 2
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    coordinates = np.array([frame[spatial1], frame[spatial2]]).T @ rotation
    return (coordinates[:, 0] + float(grid.calibration_point_x_jitter.iloc[0]),
            coordinates[:, 1] + float(grid.calibration_point_y_jitter.iloc[0]))


def arrow_window(source, con, start, end, columns):
    if isinstance(source, pd.DataFrame):
        return prediction_window(source, con, start, end, columns, True)
    con.register('experimental_prediction_source', source)
    condition = f'{quote_identifier(columns[0])} >= {float(start)!r} AND {quote_identifier(columns[0])} < {float(end)!r}'
    table = con.sql(f'SELECT * FROM experimental_prediction_source WHERE {condition}').to_arrow_table()
    frame = table.to_pandas(split_blocks=True).set_index(INDEX)
    coordinates = frame[columns]
    return TrainingWindow(coordinates, lambda positions: frame.iloc[positions])


def shared_window(shared, start, end, columns):
    times = shared.array(columns[0])
    rows = np.flatnonzero((times >= start) & (times < end))
    coordinates = shared.frame(rows, columns)
    return TrainingWindow(coordinates, lambda positions: shared.frame(rows[positions]))


class SharedPositionWindow:
    """Use source positions after the parent has validated observation IDs once."""

    def __init__(self, shared, start, end, columns):
        times = shared.array(columns[0])
        self.source_positions = np.flatnonzero((times >= start) & (times < end))
        self.coordinates = shared.frame(self.source_positions, columns)
        self.shared = shared

    def select_positions(self, positions):
        return self.shared.frame(self.source_positions[positions])


def predict_values(model, frame, stixel_id):
    estimator, names = get_model_and_stixel_specific_x_names(
        model.model_dict, stixel_id, model.stixel_specific_x_names, model.x_names)
    if estimator is None or not len(frame):
        return None
    method = model.base_model_method
    if method is not None and hasattr(estimator, method):
        return np.asarray(getattr(estimator, method)(frame[names],
                          **model.base_model_prediction_param)).reshape(-1)
    if method is not None and not isinstance(estimator, dummy_model1):
        raise TypeError(f'{method} does not exist for {type(estimator)}')
    if model.task == 'classification':
        return np.asarray(estimator.predict_proba(frame[names],
                          **model.base_model_prediction_param))[:, 1]
    return np.asarray(estimator.predict(frame[names],
                      **model.base_model_prediction_param)).reshape(-1)


class ExperimentalAdaSTEMClassifier(AdaSTEMClassifier):
    """Same trained models and stixel memberships; experimental execution only."""

    def _prediction_worker(self, grid):
        worker = copy.copy(self)
        names = set(grid.unique_stixel_id)
        worker.model_dict = {f'{name}_model': self.model_dict[f'{name}_model']
                             for name in names if f'{name}_model' in self.model_dict}
        worker.stixel_specific_x_names = {name: self.stixel_specific_x_names[name]
                                         for name in names if name in self.stixel_specific_x_names}
        worker.ensemble_df = grid
        worker.__dict__.pop('_finalizer', None)
        return worker

    @profile_execution
    def SAC_ensemble_predict(self, single_ensemble_df, data, temporal_window_prequery=False):
        strategy = self._stress_strategy
        config = self.duckdb_config.copy()
        config['temp_directory'] = os.path.join(config['temp_directory'], f'experimental_{os.getpid()}')
        columns = [self.Temporal1, self.Spatio1, self.Spatio2]
        ensemble_id = single_ensemble_df.ensemble_index.iloc[0]
        lean = strategy != 'rotation'
        pieces = []
        timings = dict(query=0., rotation=0., selection=0., gather=0., estimator=0., reduction=0.)
        calls, largest_frame = 0, 0
        with open_db_connection(data, config) as (source, con):
            if lean:
                if strategy == 'shared_positions':
                    all_ids = pd.Index(self._stress_shared.index)
                elif isinstance(source, pd.DataFrame):
                    all_ids = source.index
                else:
                    con.register('experimental_index_source', source)
                    all_ids = pd.Index(con.sql(f'SELECT {INDEX} FROM experimental_index_source').fetchnumpy()[INDEX])
                sums = np.zeros(len(all_ids))
                counts = np.zeros(len(all_ids), dtype=np.int32)
                # Construct the ID lookup once, not once per stixel.
                if strategy != 'shared_positions':
                    all_ids.get_indexer(all_ids[:1])
            for start, time_grid in single_ensemble_df.groupby(f'{self.Temporal1}_start', sort=True):
                started = time.perf_counter()
                if strategy == 'shared_positions':
                    window = SharedPositionWindow(self._stress_shared, start,
                                                  start + self.temporal_bin_interval, columns)
                elif strategy.startswith('shared'):
                    window = shared_window(self._stress_shared, start, start + self.temporal_bin_interval, columns)
                elif strategy.startswith('arrow'):
                    window = arrow_window(source, con, start, start + self.temporal_bin_interval, columns)
                else:
                    window = prediction_window(source, con, start, start + self.temporal_bin_interval,
                                               columns, temporal_window_prequery)
                timings['query'] += time.perf_counter() - started
                started = time.perf_counter()
                x, y = rotate_arrays(window.coordinates, single_ensemble_df, self.Spatio1, self.Spatio2)
                spatial = SpatialWindowIndex(x, y, len(time_grid))
                timings['rotation'] += time.perf_counter() - started
                for stixel_id, grid in time_grid.groupby('unique_stixel_id', sort=True):
                    if lean and f'{stixel_id}_model' not in self.model_dict:
                        continue
                    started = time.perf_counter()
                    positions = spatial.select(grid.iloc[0])
                    timings['selection'] += time.perf_counter() - started
                    if not len(positions):
                        continue
                    started = time.perf_counter()
                    stixel = window.select_positions(positions)
                    if self._stress_detail:
                        largest_frame = max(largest_frame, int(stixel.memory_usage(index=True, deep=True).sum()))
                    timings['gather'] += time.perf_counter() - started
                    started = time.perf_counter()
                    if lean:
                        values = predict_values(self, stixel, stixel_id)
                    else:
                        stixel['ensemble_index'], stixel['unique_stixel_id'] = ensemble_id, stixel_id
                        prediction = self.stixel_predict(stixel)
                    timings['estimator'] += time.perf_counter() - started
                    started = time.perf_counter()
                    if lean and values is not None:
                        rows = (window.source_positions[positions] if strategy == 'shared_positions'
                                else all_ids.get_indexer(stixel.index))
                        valid = ~np.isnan(values)
                        sums[rows[valid]] += values[valid]
                        counts[rows[valid]] += 1
                    elif not lean and prediction is not None:
                        pieces.append(prediction)
                    timings['reduction'] += time.perf_counter() - started
                    calls += 1
                    del stixel
                del window, x, y, spatial
        if self._stress_detail:
            import json
            report = dict(pid=os.getpid(), ensemble=int(ensemble_id), timings=timings,
                          estimator_calls=calls, largest_stixel_bytes=largest_frame)
            Path(self._stress_detail, f'worker_{os.getpid()}_{ensemble_id}.json').write_text(json.dumps(report))
        if lean:
            keep = counts > 0
            return pd.DataFrame({'pred': sums[keep] / counts[keep]}, index=all_ids[keep])
        return (pd.concat(pieces).groupby(level=0, sort=False).mean() if pieces
                else pd.DataFrame({'pred': pd.Series(dtype=float)}))


def configure(model, strategy, shared=None, detail=None):
    if strategy not in {'baseline', 'rotation', 'lean', 'arrow', 'shared', 'shared_positions',
                        'shared_threads', 'lean_threads'}:
        raise ValueError(f'Unknown prediction experiment: {strategy}')
    if strategy == 'baseline':
        return model
    model.__class__ = ExperimentalAdaSTEMClassifier
    model._stress_strategy = strategy
    model._stress_shared = shared
    model._stress_detail = detail
    if strategy.endswith('threads'):
        model.joblib_backend = 'threading'
    return model
