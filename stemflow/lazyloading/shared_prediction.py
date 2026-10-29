"""Share prediction input and reuse a fixed number of ensemble result buffers."""

import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, nullcontext
import os
from pathlib import Path
import tempfile
import time
import warnings

from joblib.externals.loky import ProcessPoolExecutor
from joblib import cpu_count
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tqdm.auto import tqdm

from .lazyloading import LazyLoadingEstimator
from .open_db_connection import open_db_connection
from .prediction import SpatialWindowIndex
from .training_data import INDEX, SharedTrainingData, _write_columns, quote_identifier
from ..model.dummy_model import dummy_model1
from ..model.static_func_AdaSTEM import get_model_and_stixel_specific_x_names
from ..utils.execution_monitor import execution_step
from ..utils.execution_profile import profile_execution


BLOCK_ROWS = 65536


def _shared_input(model, data, directory):
    columns = list(dict.fromkeys([model.Temporal1, model.Spatio1, model.Spatio2,
        *model.x_names, *(name for names in model.stixel_specific_x_names.values() for name in names)]))
    directory.mkdir()
    with open_db_connection(data, model.duckdb_config) as (source, con):
        if isinstance(source, pd.DataFrame):
            source = source.loc[:, columns]
        else:
            con.register('shared_prediction_input', source)
            names = ', '.join(quote_identifier(c) for c in [INDEX, *columns])
            source = con.sql(f'SELECT {names} FROM shared_prediction_input')
        paths, n = _write_columns(source, con, 'shared_prediction_columns', directory,
            'input', BLOCK_ROWS, {INDEX, model.Temporal1, model.Spatio1, model.Spatio2})
    files = {value if isinstance(value, str) else value[0] for value in paths.values()}
    shared = SharedTrainingData(paths, '', n, sum(Path(p).stat().st_size for p in files))
    index = pd.Index(shared.index)
    if not index.is_unique or index.hasnans:
        raise ValueError('Prediction observation IDs must be unique and non-null.')
    # The hash table is needed only for validation, not in every worker.
    del index
    return shared


def _worker_for(model, grid):
    worker = copy.copy(model)
    names = set(grid.unique_stixel_id)
    worker.model_dict = {f'{name}_model': model.model_dict[f'{name}_model']
                         for name in names if f'{name}_model' in model.model_dict}
    worker.stixel_specific_x_names = {name: model.stixel_specific_x_names[name]
        for name in names if name in model.stixel_specific_x_names}
    worker.ensemble_df = grid
    worker.__dict__.pop('_finalizer', None)
    return worker


def _values(estimator, frame, task, method, parameters):
    context = (estimator._loaded_estimator() if isinstance(estimator, LazyLoadingEstimator)
               else nullcontext(estimator))
    with context as fitted:
        values = None
        if method is not None:
            if hasattr(fitted, method):
                values = getattr(fitted, method)(frame, **parameters)
            elif not isinstance(fitted, dummy_model1):
                raise TypeError(f'{method} does not exist for {type(fitted)}')
        if values is None:
            values = (fitted.predict_proba(frame, **parameters)[:, 1] if task == 'classification'
                      else fitted.predict(frame, **parameters))
        values = np.asarray(values).reshape(-1)
        if len(values) != len(frame):
            raise ValueError('The base model must return one prediction per stixel row.')
        return values


def _forest_frame(shared, rows, names):
    """Gather directly into the float32 input required by sklearn forests."""
    values = np.empty((len(rows), len(names)), dtype=np.float32)
    groups = {}
    for destination, name in enumerate(names):
        location = shared.paths[name]
        path, column = (location, None) if isinstance(location, str) else location
        groups.setdefault(path, []).append((destination, column))
    for path, members in groups.items():
        if path not in shared._arrays:
            shared._arrays[path] = np.load(path, mmap_mode='r', allow_pickle=False)
        source = shared._arrays[path]
        destinations, columns = zip(*members)
        for offset in range(0, len(rows), BLOCK_ROWS):
            block = source[rows[offset:offset + BLOCK_ROWS]]
            target = values[offset:offset + len(block)]
            if source.ndim == 1:
                target[:, destinations[0]] = block
            else:
                target[:, list(destinations)] = block[:, list(columns)]
    return pd.DataFrame(values, columns=names,
                        index=pd.Index(shared.index[rows], name=INDEX), copy=False)


def _predict_rows(model, estimator, shared, rows, names):
    context = (estimator._loaded_estimator() if isinstance(estimator, LazyLoadingEstimator)
               else nullcontext(estimator))
    with context as fitted:
        compact = (getattr(model, '_shared_compact_forest', True) and model.base_model_method is None
                   and type(fitted) in {RandomForestClassifier, RandomForestRegressor})
        frame = _forest_frame(shared, rows, names) if compact else shared.frame(rows, names)
        return _values(fitted, frame, model.task, model.base_model_method, model.base_model_prediction_param)


@profile_execution
def _ensemble_to_slot(model, grid, shared, slot_path):
    """Write exclusively owned output; only a small completion record is returned."""
    # Each task owns its mapping handles, including when workers are threads.
    shared = copy.copy(shared)
    shared._arrays = {}
    sums = np.load(slot_path, mmap_mode='r+')
    sums[:] = 0
    counts = np.zeros(shared.n_rows, dtype=np.uint32)
    columns = [model.Temporal1, model.Spatio1, model.Spatio2]
    ensemble = grid.ensemble_index.iloc[0]
    start, stixel_id, calls = None, None, 0
    try:
        times = shared.array(model.Temporal1)
        angle = float(grid.rotation.iloc[0]) / 360 * np.pi * 2
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        for start, time_grid in grid.groupby(f'{model.Temporal1}_start', sort=True):
            rows = np.flatnonzero((times >= start) & (times < start + model.temporal_bin_interval))
            coordinates = shared.frame(rows, columns)
            xy = coordinates[[model.Spatio1, model.Spatio2]].to_numpy() @ rotation
            x = xy[:, 0] + float(grid.calibration_point_x_jitter.iloc[0])
            y = xy[:, 1] + float(grid.calibration_point_y_jitter.iloc[0])
            spatial = SpatialWindowIndex(x, y, len(time_grid))
            for stixel_id, stixel_grid in time_grid.groupby('unique_stixel_id', sort=True):
                estimator, names = get_model_and_stixel_specific_x_names(
                    model.model_dict, stixel_id, model.stixel_specific_x_names, model.x_names)
                if estimator is None:
                    continue
                positions = spatial.select(stixel_grid.iloc[0])
                if not len(positions):
                    continue
                source_rows = rows[positions]
                values = _predict_rows(model, estimator, shared, source_rows, names)
                valid = ~np.isnan(values)
                selected = source_rows[valid]
                sums[selected] += values[valid]
                counts[selected] += 1
                calls += 1
                del values, selected, positions, source_rows
            del coordinates, xy, x, y, spatial, rows
        covered = 0
        for offset in range(0, shared.n_rows, BLOCK_ROWS):
            target = sums[offset:offset + BLOCK_ROWS]
            count = counts[offset:offset + BLOCK_ROWS]
            good = count > 0
            np.divide(target, count, out=target, where=good)
            target[~good] = np.nan
            covered += int(good.sum())
        return dict(covered=covered, estimator_calls=calls)
    except BaseException as error:
        if hasattr(error, 'add_note'):
            error.add_note(f'AdaSTEM shared prediction ensemble={ensemble}, '
                           f'window_start={start}, stixel={stixel_id}')
        raise
    finally:
        # Drop mappings at each task boundary; queued results never own arrays.
        shared._arrays.clear()
        del sums


class _DenseMoments:
    def __init__(self, n, with_std, logit):
        self.count = np.zeros(n, dtype=np.uint32)
        self.mean = np.zeros(n)
        self.m2 = np.zeros(n) if with_std else None
        self.logit = logit

    def update(self, values, offset):
        good = ~np.isnan(values)
        positions = np.flatnonzero(good) + offset
        values = np.asarray(values[good])
        if self.logit:
            values = np.clip(values, 1e-16, 1 - 1e-16)
            values = np.log(values / (1 - values))
        count = self.count[positions] + 1
        delta = values - self.mean[positions]
        self.mean[positions] += delta / count
        if self.m2 is not None:
            self.m2[positions] += delta * (values - self.mean[positions])
        self.count[positions] = count

    def finish(self, minimum):
        if self.logit:
            expit(self.mean, out=self.mean)
            self.mean[self.mean <= 1e-16] = 0
        self.mean[self.count < minimum] = np.nan
        std = None
        if self.m2 is not None:
            std = np.full(len(self.mean), np.nan)
            for offset in range(0, len(std), BLOCK_ROWS):
                count = self.count[offset:offset + BLOCK_ROWS]
                valid = (count >= minimum) & (count > 1)
                block = std[offset:offset + BLOCK_ROWS]
                block[valid] = np.sqrt(np.maximum(0, self.m2[offset:offset + BLOCK_ROWS][valid]) /
                                       (count[valid] - 1))
        return self.mean, std


def _consume(slot, position, matrix, moments):
    for offset in range(0, len(slot), BLOCK_ROWS):
        block = slot[offset:offset + BLOCK_ROWS]
        if matrix is not None:
            matrix[offset:offset + len(block), position] = block
        else:
            moments.update(block, offset)


def _matrix_summary(matrix, aggregation, logit, minimum, with_std):
    mean = np.empty(len(matrix))
    std = np.empty(len(matrix)) if with_std else None
    for offset in range(0, len(matrix), BLOCK_ROWS):
        block = np.array(matrix[offset:offset + BLOCK_ROWS], copy=True)
        count = np.sum(~np.isnan(block), axis=1)
        if logit:
            np.clip(block, 1e-16, 1 - 1e-16, out=block)
            block = np.log(block / (1 - block))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = (np.nanmean(block, axis=1) if aggregation == 'mean'
                      else np.nanmedian(block, axis=1))
            deviations = np.nanstd(block, axis=1, ddof=1) if with_std else None
        if logit:
            expit(values, out=values)
            values[values <= 1e-16] = 0
        values[count < minimum] = np.nan
        mean[offset:offset + len(block)] = values
        if with_std:
            deviations[count < minimum] = np.nan
            std[offset:offset + len(block)] = deviations
    return mean, std


def shared_predict(model, data, verbosity, n_jobs, aggregation, separate, with_std,
                    logit, streaming=True, output_path=None):
    """Bound pending tasks/results independently of the ensemble count.

    Base estimators still receive their entire stixel in one call. Disk buffers
    bound result transport; they do not cap estimator or operating-system RAM.
    """
    if model.joblib_backend not in {'loky', 'threading'}:
        raise ValueError('Shared prediction supports loky and threading backends.')
    if output_path is not None and not separate:
        raise ValueError('prediction_output_path requires return_by_separate_ensembles=True.')
    destination = Path(output_path) if output_path is not None else None
    if destination is not None:
        if destination.suffix != '.npy':
            raise ValueError('prediction_output_path must end in .npy.')
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
    count_ensembles = model.ensemble_df.ensemble_index.nunique()
    if (separate or aggregation == 'median' or not streaming) and count_ensembles != model.ensemble_fold:
        raise ValueError('The prediction grid must contain ensemble_fold ensembles for matrix output.')
    workers = min(n_jobs, count_ensembles)
    if workers < 1:
        raise ValueError('No ensembles are available for prediction.')
    mapped_output, partial_path, executor = None, None, None
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory(
                prefix='shared_', dir=model.joblib_tmp_dir)))
            with execution_step('shared_prediction_input'):
                shared = _shared_input(model, data, directory / 'input')
            preparation = time.perf_counter() - started
            matrix = None
            if separate or aggregation == 'median' or not streaming:
                if destination is None:
                    matrix = np.empty((shared.n_rows, model.ensemble_fold), order='F')
                else:
                    handle, partial_path = tempfile.mkstemp(prefix=f'.{destination.stem}_',
                                                            suffix='.npy', dir=destination.parent)
                    os.close(handle)
                    mapped_output = matrix = np.lib.format.open_memmap(partial_path, mode='w+',
                        dtype=float, shape=(shared.n_rows, model.ensemble_fold), fortran_order=True)
            moments = (_DenseMoments(shared.n_rows, with_std, logit) if matrix is None else None)
            slots = []
            for number in range(workers):
                path = directory / f'result_{number}.npy'
                array = np.lib.format.open_memmap(path, mode='w+', dtype=float, shape=(shared.n_rows,))
                del array
                slots.append(str(path))
            groups = iter(enumerate(model.ensemble_df.groupby('ensemble_index', sort=True)))
            covered = 0
            estimator_calls = 0
            max_pending = 0
            with execution_step('prediction_and_aggregation'), tqdm(
                    total=count_ensembles, desc='Predicting: ', disable=verbosity <= 0) as progress:
                if workers == 1:
                    for position, (_, grid) in groups:
                        result = _ensemble_to_slot(_worker_for(model, grid), grid, shared, slots[0])
                        slot = np.load(slots[0], mmap_mode='r')
                        _consume(slot, position, matrix, moments)
                        del slot
                        covered += result['covered']
                        estimator_calls += result['estimator_calls']
                        progress.update()
                    max_pending = 1
                else:
                    thread_limit = str(max(cpu_count() // workers, 1))
                    environment = {name: os.environ.get(name, thread_limit) for name in (
                        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                        'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS',
                        'NUMEXPR_NUM_THREADS')}
                    executor = (ThreadPoolExecutor(max_workers=workers) if model.joblib_backend == 'threading'
                                else ProcessPoolExecutor(max_workers=workers, env=environment))
                    def stop_executor():
                        nonlocal executor
                        if executor is not None:
                            if isinstance(executor, ThreadPoolExecutor):
                                executor.shutdown(wait=True, cancel_futures=True)
                            else:
                                # Join owned workers before removing their mapped files.
                                executor.shutdown(wait=True)
                            executor = None
                    stack.callback(stop_executor)
                    pending = {}
                    def submit(slot_path):
                        try:
                            position, (_, grid) = next(groups)
                        except StopIteration:
                            return
                        future = executor.submit(_ensemble_to_slot, _worker_for(model, grid),
                                                 grid, shared, slot_path)
                        pending[future] = (position, slot_path)
                    for slot_path in slots:
                        submit(slot_path)
                    max_pending = len(pending)
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            position, slot_path = pending.pop(future)
                            result = future.result()
                            slot = np.load(slot_path, mmap_mode='r')
                            _consume(slot, position, matrix, moments)
                            del slot
                            covered += result['covered']
                            estimator_calls += result['estimator_calls']
                            progress.update()
                            # Reuse only after the parent has consumed every row.
                            submit(slot_path)
                        max_pending = max(max_pending, len(pending))
                    executor.shutdown(wait=True)
                    executor = None
            if not estimator_calls:
                raise ValueError('All samples are not predictable based on current settings!')
            info = dict(aggregation_backend='memmap_matrix' if matrix is not None else 'memmap_streaming',
                prediction_backend='memmap', rows=shared.n_rows, ensembles=model.ensemble_fold,
                workers=workers, max_pending_tasks=max_pending, shared_input_bytes=shared.nbytes,
                shared_preparation_seconds=preparation, result_buffer_bytes=workers * shared.n_rows * 8,
                worker_count_bytes=shared.n_rows * 4, result_transport='completion_records',
                estimator_calls=estimator_calls,
                output_storage='memmap' if destination is not None else 'memory')
            shared._arrays.clear()
            if separate:
                info['output_bytes'] = matrix.nbytes
                model.prediction_execution_info_ = info
                if destination is None:
                    return matrix
                mapped_output.flush()
                # Close before rename so this also works on Windows filesystems.
                mapped_output._mmap.close()
                mapped_output = matrix = None
                # Publish atomically without overwriting a concurrently created file.
                os.link(partial_path, destination)
                Path(partial_path).unlink()
                partial_path = None
                return np.load(destination, mmap_mode='r+')
            mean, std = (moments.finish(model.min_ensemble_required) if moments is not None
                         else _matrix_summary(matrix, aggregation, logit, model.min_ensemble_required, with_std))
            result = np.column_stack([1 - mean, mean]) if model.task == 'classification' else mean
            info['output_bytes'] = result.nbytes + (std.nbytes if std is not None else 0)
            model.prediction_execution_info_ = info
            missing = np.count_nonzero(np.isnan(mean))
            warnings.warn(f'There are {100 * missing / len(mean):.3g}% points ({missing} points) '
                          'falling out of predictable range.')
            return (result, std) if with_std else result
    finally:
        if mapped_output is not None:
            mapped_output._mmap.close()
        if partial_path is not None:
            Path(partial_path).unlink(missing_ok=True)
