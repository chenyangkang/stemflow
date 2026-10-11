"""Summarize completed stress runs without retaining models or prediction arrays."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def collect(root):
    rows = []
    for path in sorted(root.glob('*/*_j*_r*.json')):
        d = json.loads(path.read_text())
        measured = d['measurement']
        peaks = measured['phase_peaks']
        prediction = peaks.get('prediction', {})
        equivalence = d.get('equivalence', {})
        fixture = json.loads((path.parent / f"{d['case']}_fixture.json").read_text())
        spec = fixture['specification']
        rows.append(dict(run=path.parent.name, case=d['case'], strategy=d['strategy'],
            workers=d['jobs'], repeat=d['repeat'], success=measured['exit_code'] == 0,
            guard=measured.get('guard_reason'), logit=d.get('logit', False),
            resident=d.get('resident', False), profiled=path.parent.name.startswith('profile'),
            end_to_end=d.get('end_to_end', False), training_seconds=d.get('training_seconds'),
            separate=d.get('separate', False), output_bytes=d.get('output_bytes', 0),
            rows=spec['rows'], covariates=spec['features'], ensembles=spec['ensembles'],
            stixels=fixture['trained_stixels'], model_mib=fixture['model_files_bytes']/2**20,
            prediction_seconds=d.get('prediction_seconds'), preparation_seconds=d.get('preparation_seconds'),
            total_seconds=d.get('total_seconds'), child_wall_seconds=measured['wall_seconds'],
            cache_gib=d.get('cache_disk_bytes', 0)/2**30,
            prediction_footprint_gib=prediction.get('footprint_bytes', 0)/2**30,
            prediction_rss_gib=prediction.get('rss_bytes', 0)/2**30,
            job_footprint_gib=max((p['footprint_bytes'] for p in peaks.values()), default=0)/2**30,
            job_rss_gib=max((p['rss_bytes'] for p in peaks.values()), default=0)/2**30,
            training_footprint_gib=peaks.get('training', {}).get('footprint_bytes', 0)/2**30,
            parent_rss_gib=prediction.get('parent_rss_bytes', 0)/2**30,
            workers_rss_gib=prediction.get('descendants_rss_bytes', 0)/2**30,
            masks_equal=equivalence.get('missing_masks_equal'),
            std_masks_equal=equivalence.get('std_missing_masks_equal'),
            prediction_difference=equivalence.get('maximum_prediction_difference'),
            std_difference=equivalence.get('maximum_std_difference'),
            source=str(path)))
    return pd.DataFrame(rows)


def main(args):
    frame = collect(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'trials.csv', index=False)
    valid = frame[frame.success & ~frame.profiled].copy()
    keys = ['run', 'case', 'strategy', 'workers', 'rows', 'ensembles', 'stixels', 'resident', 'logit', 'separate']
    summary = valid.groupby(keys, dropna=False).agg(
        trials=('total_seconds', 'size'), total_seconds=('total_seconds', 'median'),
        minimum_seconds=('total_seconds', 'min'), maximum_seconds=('total_seconds', 'max'),
        prediction_seconds=('prediction_seconds', 'median'),
        preparation_seconds=('preparation_seconds', 'median'),
        median_footprint_gib=('prediction_footprint_gib', 'median'),
        maximum_job_footprint_gib=('job_footprint_gib', 'max'),
        median_rss_gib=('prediction_rss_gib', 'median'), cache_gib=('cache_gib', 'max'),
        max_prediction_difference=('prediction_difference', 'max'),
        max_std_difference=('std_difference', 'max'),
        masks_equal=('masks_equal', 'all'), std_masks_equal=('std_masks_equal', 'all')).reset_index()
    summary.to_csv(args.output / 'comparison.csv', index=False)
    print(f'Saved {len(frame)} trials and {len(summary)} comparison rows to {args.output}')
    choices = [('repeated_v1', 'wide_million', 2), ('large_positions_v1', 'result_pressure', 1),
               ('positions_v1', 'many_stixels', 4), ('workflow_v1', 'huge_forests', 4)]
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    colors = dict(baseline='#555555', arrow='#e69f00', shared='#0072b2',
                  shared_positions='#56b4e9', shared_threads='#009e73')
    titles = ['1M rows; 116 covariates\n2 workers; 3 repetitions',
              '5M rows; 8 covariates\n1 worker; 6 ensembles',
              '6,796 stixels; 300k rows\n4 workers; 2 repetitions',
              '847 MiB forests; 200k rows\n4 workers; same-job fit + predict']
    for column, (run, case, workers) in enumerate(choices):
        part = summary[(summary.run == run) & (summary.case == case) & (summary.workers == workers)]
        part = part.set_index('strategy').reindex(list(colors)).dropna(subset='total_seconds')
        for row, (value, label) in enumerate([('total_seconds', 'Prediction + preparation (s)'),
                                            ('median_footprint_gib', 'Process-tree footprint (GiB)')]):
            ax = axes[row, column]
            bars = ax.bar(range(len(part)), part[value], color=[colors[s] for s in part.index])
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
            ax.margins(y=.15)
            labels = {'shared_threads': 'shared\nthreads', 'shared_positions': 'shared\npositions'}
            ax.set_xticks(range(len(part)), [labels.get(s, s) for s in part.index], rotation=25)
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(titles[column], fontsize=11)
            ax.spines[['top', 'right']].set_visible(False)
    figure.supxlabel('Times include cache preparation. macOS footprint excludes shared file cache; RSS is also recorded in the CSV.', fontsize=10)
    args.figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figures / 'prediction_stress.png', dpi=180)
    figure.savefig(args.figures / 'prediction_stress.pdf')
    plt.close(figure)

    profiles = []
    worker_phases = []
    for path in sorted(args.root.glob('profile*/*_j*_r*.json')):
        d = json.loads(path.read_text())
        for worker in d.get('worker_details', []):
            worker_phases.append(dict(case=d['case'], strategy=d['strategy'], **worker))
        directory = path.with_name(path.stem + '_profiles')
        for profile in directory.glob('*.json'):
            p = json.loads(profile.read_text())
            for function in p['functions']:
                profiles.append(dict(case=d['case'], strategy=d['strategy'], scope=p['function'], **function))
    if profiles:
        pd.DataFrame(profiles).groupby(['case', 'strategy', 'scope', 'file', 'line', 'function'], as_index=False).agg(
            calls=('calls', 'sum'), own_seconds=('own_seconds', 'sum'),
            cumulative_seconds=('cumulative_seconds', 'sum')).to_csv(args.output / 'profile_functions.csv', index=False)
    if worker_phases:
        records = []
        for worker in worker_phases:
            records.append({**{k:v for k,v in worker.items() if k != 'timings'}, **worker['timings']})
        pd.DataFrame(records).to_csv(args.output / 'worker_phase_times.csv', index=False)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=root / 'data/06_prediction_stress')
    parser.add_argument('--output', type=Path, default=root / 'data/06_prediction_stress/comparison')
    parser.add_argument('--figures', type=Path, default=root / 'figures/06_prediction_stress')
    main(parser.parse_args())
