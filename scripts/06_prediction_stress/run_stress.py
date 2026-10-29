"""Isolated, memory-guarded prediction stress tests using unchanged fitted models."""

import argparse
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import pickle
import platform
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
for path in ['/private/tmp/stemflow_monitor_deps', '/private/tmp/codex_rf_tuner_deps']:
    if Path(path).is_dir():
        sys.path.append(path)

spec = importlib.util.spec_from_file_location('execution_benchmark', ROOT / 'scripts/00_benchmarks/compare_execution.py')
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)

CASES = {
    'wide': dict(train_rows=12000, rows=300000, features=116, dtype='float64', ensembles=6,
                 trees=20, depth=4, base='beta', upper=25, lower=6, minimum=200, period=80, step=40, interval=60),
    'wide_large': dict(train_rows=12000, rows=600000, features=116, dtype='float64', ensembles=6,
                       trees=20, depth=4, base='beta', upper=25, lower=6, minimum=200, period=80, step=40, interval=60),
    'wide_million': dict(train_rows=12000, rows=1000000, features=116, dtype='float64', ensembles=6,
                         trees=20, depth=4, base='beta', upper=25, lower=6, minimum=200, period=80, step=40, interval=60),
    'annual_million': dict(train_rows=12000, rows=1000000, features=116, dtype='float64', ensembles=6,
                           trees=20, depth=4, base='beta', upper=25, lower=6, minimum=200, period=366, step=40, interval=60),
    'result_pressure': dict(train_rows=4000, rows=1000000, features=8, dtype='float32', ensembles=50,
                            trees=3, depth=2, base='xgb', upper=100, lower=100, minimum=1000, period=80, step=40, interval=60),
    'huge_forests': dict(train_rows=25000, rows=200000, features=16, dtype='float32', ensembles=4,
                         trees=500, depth=24, base='forest', upper=100, lower=100, minimum=1000, period=80, step=40, interval=60),
    'tuned_beta': dict(train_rows=10000, rows=300000, features=116, dtype='float64', ensembles=6,
                       trees=60, depth=6, base='beta_tuned', upper=100, lower=100, minimum=1000, period=80, step=40, interval=60),
    'many_stixels': dict(train_rows=20000, rows=300000, features=24, dtype='float32', ensembles=12,
                        trees=8, depth=3, base='xgb', upper=5, lower=1, minimum=25, period=160, step=40, interval=60),
    'large_models': dict(train_rows=16000, rows=300000, features=32, dtype='float32', ensembles=6,
                        trees=100, depth=20, base='forest', upper=100, lower=100, minimum=1000, period=80, step=40, interval=60),
    'overlap': dict(train_rows=10000, rows=600000, features=48, dtype='float64', ensembles=12,
                   trees=10, depth=4, base='xgb', upper=25, lower=8, minimum=200, period=80, step=10, interval=60),
    'real_acadian': dict(train_rows=20000, rows=500000, features=116, dtype='float32', ensembles=6,
                        trees=30, depth=4, base='beta', upper=25, lower=5, minimum=200, period=80, step=40, interval=60),
    'combined_forest': dict(train_rows=12000, rows=1000000, features=48, dtype='float32', ensembles=20,
                           trees=150, depth=20, base='forest', upper=100, lower=100, minimum=1000,
                           period=80, step=40, interval=60),
    'combined_tuned': dict(train_rows=8000, rows=1000000, features=48, dtype='float32', ensembles=50,
                          trees=60, depth=6, base='beta_tuned', upper=100, lower=100, minimum=1000,
                          period=80, step=40, interval=60),
}


def atomic_json(path, value):
    target = Path(path)
    temporary = target.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(target)


def phase(work, name):
    atomic_json(work / 'phase.json', dict(phase=name))


def frame_for(rng, n, case, offset=0, total=None):
    import numpy as np
    import pandas as pd
    values = rng.normal(size=(n, case['features'])).astype(case['dtype'])
    frame = pd.DataFrame(values, columns=[f'feature_{i}' for i in range(case['features'])])
    frame['longitude'] = rng.uniform(-110, -70, n)
    frame['latitude'] = rng.uniform(25, 55, n)
    frame['DOY'] = rng.uniform(1, case['period'] + 1, n)
    frame.index = ((total or n) - np.arange(offset, offset + n)) * 7 + 13
    return frame


def estimator_inventory(estimator):
    pending, fitted, trees, trials = [estimator], 0, 0, 0
    while pending:
        est = pending.pop()
        if hasattr(est, 'fold_models_'):
            pending.extend(est.fold_models_)
        elif hasattr(est, 'estimators_'):
            fitted += 1
            trees += len(est.estimators_)
        elif hasattr(est, 'estimator_'):
            trials += len(getattr(est, 'trials_', []))
            pending.append(est.estimator_)
        elif hasattr(est, '_Booster'):
            fitted += 1
            trees += est.get_booster().num_boosted_rounds()
    return dict(fitted_estimators=fitted, retained_trees=trees, trial_records=trials)


def prepare(work, case_name, fit_jobs=2):
    import duckdb
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from stemflow.model.AdaSTEM import AdaSTEMClassifier
    from stemflow.lazyloading import make_training_databases
    from stemflow.tuning import local_beta_xgboost
    case = CASES[case_name]
    phase(work, 'fixture_preparation')
    rng = np.random.default_rng(315)
    real = None
    if case_name == 'real_acadian':
        from types import SimpleNamespace
        sampler.build_fixture(SimpleNamespace(source='real', x_database=ROOT.parent / 'X_train.duckdb',
            data_dir=ROOT.parent, species_parquet=ROOT.parent / 'COMMON%20NAME=Acadian%20Flycatcher/data_0.parquet',
            sample_per_mille=600, rows=50000, features=116, species='Acadian%20Flycatcher'), work)
        real = pd.read_parquet(work / 'fixture.parquet')
        real['DOY'] -= 120
        real.index = np.arange(len(real)) * 7 + 13
        X = real.iloc[:case['train_rows']].drop(columns='occurrence').copy()
        y = real.iloc[:case['train_rows']].occurrence.to_numpy()
    else:
        X = frame_for(rng, case['train_rows'], case)
        y = rng.binomial(1, 1 / (1 + np.exp(-(-2 + X.feature_0 - .5 * X.feature_1))))
    estimator = (local_beta_xgboost(tuning=case['base'] == 'beta_tuned', n_trials=3, tuning_n_estimators=20,
                                  n_estimators=case['trees'], n_splits=3,
                                  Spatio_blocks_count=3, Temporal_blocks_count=3,
                                  Spatio1='longitude_tmp', Spatio2='latitude_tmp', random_state=42) if case['base'].startswith('beta') else
                 RandomForestClassifier(n_estimators=case['trees'], max_depth=case['depth'],
                                        n_jobs=1, random_state=42) if case['base'] == 'forest' else
                 XGBClassifier(n_estimators=case['trees'], max_depth=case['depth'],
                               n_jobs=1, tree_method='hist', random_state=42))
    if case['base'].startswith('beta'):
        X['longitude_tmp'], X['latitude_tmp'] = X.longitude, X.latitude
    training_sources = make_training_databases(X, pd.DataFrame({'occ': y}, index=X.index), work / 'train')
    with duckdb.connect(str(work / 'prediction.duckdb'), config={'threads': '1', 'memory_limit': '300MB'}) as con:
        for offset in range(0, case['rows'], 50000):
            n = min(50000, case['rows'] - offset)
            block = (frame_for(rng, n, case, offset, case['rows']) if real is None else
                     real.iloc[np.arange(offset, offset+n) % len(real)].drop(columns='occurrence').copy())
            if real is not None:
                block.index = (case['rows'] - np.arange(offset, offset+n)) * 7 + 13
            if case['base'].startswith('beta'):
                block['longitude_tmp'], block['latitude_tmp'] = block.longitude, block.latitude
            block['__index_level_0__'] = block.index
            con.register('block', block)
            con.execute(('CREATE TABLE tmp AS SELECT * FROM block' if offset == 0 else 'INSERT INTO tmp SELECT * FROM block'))
    model = AdaSTEMClassifier(base_model=estimator, ensemble_fold=case['ensembles'], min_ensemble_required=1,
        grid_len_upper_threshold=case['upper'], grid_len_lower_threshold=case['lower'],
        points_lower_threshold=case['minimum'], temporal_start=1, temporal_end=case['period']+1,
        temporal_step=case['step'], temporal_bin_interval=case['interval'], temporal_bin_start_jitter=0,
        n_jobs=fit_jobs, sample_weights_for_classifier=False, save_gridding_plot=False,
        random_state=42, max_mem='300MB', lazy_loading=True, lazy_loading_dir=str(work / 'models'), verbosity=0)
    phase(work, 'training')
    started = time.perf_counter()
    model.fit(*training_sources,
              temporal_window_prequery=True, persist_prediction_state=False, inner_max_num_threads=1)
    elapsed = time.perf_counter() - started
    model._finalizer.detach()
    with (work / 'model.pkl').open('wb') as handle:
        pickle.dump(model, handle)
    sizes = [p.stat().st_size for p in (work / 'models').rglob('*.pkl')]
    inventory = []
    for path in (work / 'models').rglob('*.pkl'):
        estimator = joblib.load(path)
        inventory.append(estimator_inventory(estimator))
        del estimator
    atomic_json(work / 'model_inventory.json', inventory)
    metadata = dict(case=case_name, specification=case, training_seconds=elapsed,
                    trained_stixels=len(model.model_dict), partition_stixels=len(model.ensemble_df),
                    model_files_bytes=sum(sizes), largest_model_file_bytes=max(sizes, default=0),
                    prediction_database_bytes=(work / 'prediction.duckdb').stat().st_size,
                    train_detections=int(y.sum()),
                    inventory_version=2,
                    training_columns=len(X.columns),
                    model_metadata_bytes=(work / 'model.pkl').stat().st_size,
                    inventory_totals={k:sum(r[k] for r in inventory) for k in inventory[0]} if inventory else {},
                    inventory_maxima={k:max(r[k] for r in inventory) for k in inventory[0]} if inventory else {})
    metadata['training_dataframe_bytes'] = int(X.memory_usage(index=True, deep=True).sum())
    with duckdb.connect(str(work / 'prediction.duckdb'), read_only=True) as con:
        metadata['largest_temporal_window_rows'] = max(
            con.execute('SELECT COUNT(*) FROM tmp WHERE DOY >= ? AND DOY < ?',
                        [float(start), float(start + case['interval'])]).fetchone()[0]
            for start in model.ensemble_df.DOY_start.unique())
    atomic_json(work / 'fixture.json', metadata)
    print(json.dumps(metadata), flush=True)


def sandbox_safe_executor_cleanup():
    if sys.platform != 'darwin':
        return
    from joblib.externals.loky.backend import utils
    def stop_owned_tree(process):
        _, _, descendants = sampler.process_memory(process.pid, details=True)
        for pid in reversed(list(descendants)):
            if pid != process.pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.join()
    utils._kill_process_tree_with_psutil = stop_owned_tree


def prediction_child(args):
    import gc
    import numpy as np
    import duckdb
    import pandas as pd
    import sklearn
    import xgboost
    from strategies import build_shared_prediction, configure
    sandbox_safe_executor_cleanup()
    work = args.work
    phase(work, 'model_loading')
    with (work / 'model.pkl').open('rb') as handle:
        model = pickle.load(handle)
    if args.resident:
        phase(work, 'resident_model_loading')
        for estimator in model.model_dict.values():
            if hasattr(estimator, 'load'):
                estimator.load()
                estimator.keep_loaded = True
    gc.collect()
    shared_backend = args.strategy.startswith(('memmap', 'shared_backend', 'bounded'))
    file_output = args.strategy in {'memmap_disk', 'shared_backend_disk', 'bounded_disk'}
    shared, cache_seconds, cache_bytes = None, 0., 0
    if args.strategy.startswith('shared') and not shared_backend:
        phase(work, 'shared_preparation')
        started = time.perf_counter()
        shared = build_shared_prediction(str(work / 'prediction.duckdb'), model.duckdb_config,
                                         work / 'shared_prediction', validate_index=args.strategy == 'shared_positions')
        cache_seconds = time.perf_counter() - started
        cache_bytes = shared.nbytes
    detail = str(work / 'detail') if args.detail else None
    if detail:
        Path(detail).mkdir(exist_ok=True)
    model = configure(model, args.strategy, shared, detail) if not shared_backend else model
    if args.profile:
        os.environ['STEMFLOW_PROFILE_DIR'] = str(work / 'profiles')
    phase(work, 'prediction')
    started = time.perf_counter()
    options = dict(n_jobs=args.jobs, temporal_window_prequery=True, logit_agg=args.logit)
    if shared_backend:
        options['prediction_backend'] = 'memmap'
    if args.strategy in {'memmap_threads', 'shared_backend_threads', 'bounded_threads'}:
        model.joblib_backend = 'threading'
    if args.strategy in {'memmap_legacy_gather', 'shared_backend_legacy_gather', 'bounded_legacy_gather'}:
        model._shared_compact_forest = False
    if file_output:
        if not args.separate:
            raise ValueError('File output requires --separate.')
        options['prediction_output_path'] = work / 'prediction.npy'
    result = model.predict_proba(str(work / 'prediction.duckdb'),
        return_std=not args.separate, return_by_separate_ensembles=args.separate, **options)
    seconds = time.perf_counter() - started
    phase(work, 'verification_output')
    probabilities, deviations = (result, np.empty(0)) if args.separate else result
    output = probabilities if args.separate else probabilities[:, 1]
    if not file_output:
        np.save(work / 'prediction.npy', output)
    np.save(work / 'deviation.npy', deviations)
    finite = sum(int(np.isfinite(output[offset:offset+65536]).sum()) for offset in range(0, len(output), 65536))
    report = dict(strategy=args.strategy, jobs=args.jobs, resident=args.resident,
                  prediction_seconds=seconds, preparation_seconds=cache_seconds,
                  total_seconds=seconds + cache_seconds, cache_disk_bytes=cache_bytes,
                  coverage=float(finite/output.size), logit=args.logit, separate=args.separate,
                  output_bytes=output.nbytes, execution_info=model.prediction_execution_info_,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                duckdb=duckdb.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__))
    import hashlib
    report['source_sha256'] = {str(path): hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in [
        Path('stemflow/model/AdaSTEM.py'), Path('stemflow/lazyloading/shared_prediction.py'),
        Path('scripts/06_prediction_stress/strategies.py')]}
    atomic_json(work / 'child_result.json', report)
    phase(work, 'done')


def compare_arrays(observed, expected):
    """Validate large ensemble matrices without loading or copying them in full."""
    import numpy as np
    if observed.shape != expected.shape:
        return False, float('inf'), False, False
    masks, maximum, strict, relaxed = True, 0., True, True
    for offset in range(0, len(observed), 16384):
        left, right = observed[offset:offset+16384], expected[offset:offset+16384]
        masks = masks and bool(np.array_equal(np.isnan(left), np.isnan(right)))
        difference = np.abs(left-right)
        valid = ~np.isnan(difference)
        if valid.any():
            maximum = max(maximum, float(difference[valid].max()))
        strict = strict and bool(np.allclose(left, right, equal_nan=True, rtol=0, atol=1e-12))
        relaxed = relaxed and bool(np.allclose(left, right, equal_nan=True, rtol=0, atol=1e-6))
    return masks, maximum, strict, relaxed


def run_child(command, work, memory_gib, timeout):
    import psutil
    peaks = defaultdict(lambda: dict(rss_bytes=0, footprint_bytes=0, parent_rss_bytes=0, descendants_rss_bytes=0))
    start = time.perf_counter()
    reason = None
    stop_started = None
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT)
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        environment[key] = '1'
    with (work / 'child.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=ROOT)
        while process.poll() is None:
            rss, footprint, processes = sampler.process_memory(process.pid, details=True)
            try:
                name = json.loads((work / 'phase.json').read_text())['phase']
            except (FileNotFoundError, json.JSONDecodeError):
                name = 'startup'
            current = peaks[name]
            current['rss_bytes'] = max(current['rss_bytes'], rss)
            current['footprint_bytes'] = max(current['footprint_bytes'], footprint)
            parent_rss = processes.get(process.pid, (0, 0))[0]
            current['parent_rss_bytes'] = max(current['parent_rss_bytes'], parent_rss)
            current['descendants_rss_bytes'] = max(current['descendants_rss_bytes'], rss-parent_rss)
            if reason is None and (footprint > memory_gib * 2**30 or psutil.virtual_memory().available < 1.5 * 2**30):
                reason = 'stopped_at_local_memory_guard'
                stop_started = time.perf_counter()
                atomic_json(work / 'guard.json', dict(reason=reason, phase=name, phase_peaks=dict(peaks)))
                process.send_signal(signal.SIGINT)
            if reason is None and time.perf_counter() - start > timeout:
                reason = 'stopped_at_time_guard'
                stop_started = time.perf_counter()
                process.send_signal(signal.SIGINT)
            if stop_started is not None and time.perf_counter() - stop_started > 15:
                for pid in reversed(list(processes)):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            time.sleep(.05)
        code = process.wait()
    return dict(exit_code=code, guard_reason=reason, wall_seconds=time.perf_counter()-start, phase_peaks=dict(peaks))


def main(args):
    import numpy as np
    import shutil
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for case_name in args.cases.split(','):
        with tempfile.TemporaryDirectory(prefix='stemflow_prediction_stress_') as temporary:
            work = Path(temporary)
            print(f'Preparing {case_name}', flush=True)
            command = [sys.executable, __file__, '--mode', 'prepare', '--work', str(work), '--case', case_name]
            for option, value in [('rows', args.rows), ('train-rows', args.train_rows), ('ensembles', args.ensembles)]:
                if value is not None:
                    command.extend([f'--{option}', str(value)])
            if not args.end_to_end:
                prepared = run_child(command, work, args.memory_gib, args.timeout)
                if prepared['exit_code']:
                    shutil.copy2(work / 'child.log', output / f'{case_name}_prepare_failure.log')
                    raise RuntimeError(f'Fixture preparation failed: {prepared}')
                fixture = json.loads((work / 'fixture.json').read_text())
                atomic_json(output / f'{case_name}_fixture.json', dict(**fixture, measurement=prepared))
            for repeat in range(1, args.repeat+1):
                strategies = args.strategies.split(',')
                if repeat % 2 == 0:
                    strategies = strategies[::-1]
                for strategy in strategies:
                    for jobs in map(int, args.jobs.split(',')):
                        name = f'{case_name}_{strategy}_j{jobs}_r{repeat}'
                        target = output / f'{name}.json'
                        if target.exists() and not args.overwrite:
                            continue
                        print(f'Running {name}', flush=True)
                        for path in [work / 'child_result.json', work / 'phase.json',
                                     work / 'prediction.npy', work / 'deviation.npy']:
                            path.unlink(missing_ok=True)
                        command = [sys.executable, __file__, '--mode', 'predict', '--work', str(work),
                                   '--strategy', strategy, '--jobs', str(jobs)]
                        if args.end_to_end:
                            for directory in ['train', 'models', 'shared_prediction']:
                                shutil.rmtree(work / directory, ignore_errors=True)
                            for filename in ['prediction.duckdb', 'prediction.duckdb.wal', 'model.pkl', 'fixture.parquet']:
                                (work / filename).unlink(missing_ok=True)
                            command[command.index('predict')] = 'workflow'
                            command.extend(['--case', case_name])
                            for option, value in [('rows', args.rows), ('train-rows', args.train_rows), ('ensembles', args.ensembles)]:
                                if value is not None:
                                    command.extend([f'--{option}', str(value)])
                        if args.resident:
                            command.append('--resident')
                        if args.detail:
                            command.append('--detail')
                        if args.logit:
                            command.append('--logit')
                        if args.separate:
                            command.append('--separate')
                        if args.profile:
                            command.append('--profile')
                        measured = run_child(command, work, args.memory_gib, args.timeout)
                        report = dict(case=case_name, repeat=repeat, strategy=strategy, jobs=jobs,
                                      end_to_end=args.end_to_end, measurement=measured)
                        if args.end_to_end and (work / 'fixture.json').exists():
                            fixture = json.loads((work / 'fixture.json').read_text())
                            atomic_json(output / f'{case_name}_fixture.json', dict(**fixture, measurement=measured))
                            report['training_seconds'] = fixture['training_seconds']
                        if measured['exit_code'] == 0:
                            report.update(json.loads((work / 'child_result.json').read_text()))
                            observed = np.load(work / 'prediction.npy', mmap_mode='r')
                            deviation = np.load(work / 'deviation.npy', mmap_mode='r')
                            reference = work / 'reference.npy'
                            if not reference.exists():
                                if strategy != 'baseline':
                                    raise ValueError('The first strategy must be baseline.')
                                shutil.copy2(work / 'prediction.npy', reference)
                                shutil.copy2(work / 'deviation.npy', work / 'reference_std.npy')
                            expected, expected_std = np.load(reference, mmap_mode='r'), np.load(work / 'reference_std.npy', mmap_mode='r')
                            masks_match, difference, strict, relaxed = compare_arrays(observed, expected)
                            std_masks_match, std_difference, std_strict, std_relaxed = compare_arrays(deviation, expected_std)
                            report['equivalence'] = dict(missing_masks_equal=masks_match, std_missing_masks_equal=std_masks_match,
                                maximum_prediction_difference=difference, maximum_std_difference=std_difference)
                            report['equivalence']['strict_tolerance'] = 1e-12
                            report['equivalence']['float32_tolerance'] = 1e-6
                            report['equivalence']['strict_passed'] = bool(masks_match and std_masks_match and strict and std_strict)
                            report['equivalence']['passed'] = bool(masks_match and std_masks_match and relaxed and std_relaxed)
                            del observed, deviation, expected, expected_std
                        else:
                            shutil.copy2(work / 'child.log', output / f'{name}_failure.log')
                        if (work / 'detail').exists():
                            report['worker_details'] = [json.loads(p.read_text()) for p in (work / 'detail').glob('*.json')]
                            shutil.rmtree(work / 'detail')
                        if (work / 'profiles').exists():
                            shutil.copytree(work / 'profiles', output / f'{name}_profiles', dirs_exist_ok=True)
                            shutil.rmtree(work / 'profiles')
                        atomic_json(target, report)
                        all_results.append(report)
                        print(json.dumps({k: report[k] for k in ['case', 'strategy', 'jobs', 'total_seconds', 'equivalence'] if k in report}), flush=True)
                        shutil.rmtree(work / 'shared_prediction', ignore_errors=True)
                        if strategy in {'memmap_disk', 'shared_backend_disk', 'bounded_disk'}:
                            (work / 'prediction.npy').unlink(missing_ok=True)
                        if measured['guard_reason']:
                            print('Local guard triggered; continuing only with smaller strategies.', flush=True)
    atomic_json(output / 'run_summary.json', all_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['run', 'prepare', 'predict', 'workflow'], default='run')
    parser.add_argument('--case', default='wide')
    parser.add_argument('--cases', default='wide,large_models,many_stixels,overlap')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--strategies', default='baseline,rotation,lean,arrow,shared,shared_positions,shared_threads,lean_threads')
    parser.add_argument('--strategy', default='baseline')
    parser.add_argument('--jobs', default='4')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--rows', type=int, help='Override prediction row count for every selected case.')
    parser.add_argument('--train-rows', type=int, help='Override training row count for every selected case.')
    parser.add_argument('--ensembles', type=int, help='Override ensemble count for every selected case.')
    parser.add_argument('--memory-gib', type=float, default=5.5)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/06_prediction_stress/stress_v1')
    parser.add_argument('--resident', action='store_true')
    parser.add_argument('--end-to-end', action='store_true', help='Fit and predict in one child using the normal worker lifecycle.')
    parser.add_argument('--detail', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--logit', action='store_true')
    parser.add_argument('--separate', action='store_true', help='Stress the full observation-by-ensemble return.')
    parser.add_argument('--overwrite', action='store_true')
    arguments = parser.parse_args()
    for key, value in [('rows', arguments.rows), ('train_rows', arguments.train_rows),
                       ('ensembles', arguments.ensembles)]:
        if value is not None:
            if value < 1:
                parser.error(f'{key} must be positive.')
            for case_name in arguments.cases.split(',') if arguments.mode == 'run' else [arguments.case]:
                CASES[case_name][key] = value
    if arguments.mode == 'prepare':
        prepare(arguments.work, arguments.case)
    elif arguments.mode in {'predict', 'workflow'}:
        arguments.jobs = int(arguments.jobs)
        if arguments.mode == 'workflow':
            prepare(arguments.work, arguments.case, fit_jobs=arguments.jobs)
        prediction_child(arguments)
    else:
        main(arguments)
