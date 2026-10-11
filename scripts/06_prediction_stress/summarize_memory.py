"""Compare shared prediction with the native route on the same fitted models."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from summarize import collect


CHOICES = [
    ('bounded_large_separate_v1', 'result_pressure', '3M rows, 50 ensembles\nSeparate ensemble output'),
    ('bounded_repeated_v2', 'combined_tuned', '1M rows, 50 ensembles\nTuned XGBoost + local beta'),
    ('bounded_forest_v2', 'combined_forest', '1M rows, 20 ensembles\n458 MiB fitted forests'),
    ('bounded_repeated_v2', 'many_stixels', '300k rows, 12 ensembles\n6,796 fitted stixels'),
]
COLORS = {'baseline': '#666666', 'bounded': '#0072b2',
          'bounded_threads': '#009e73', 'bounded_disk': '#e69f00'}
LABELS = {'baseline': 'Native', 'bounded': 'Memmap\nprocesses',
          'bounded_threads': 'Memmap\nthreads', 'bounded_disk': 'Memmap\nfile output'}


def main(args):
    trials = collect(args.root)
    trials = trials[trials.run.str.startswith('bounded_')].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output / 'trials.csv', index=False)
    valid = trials[trials.success & trials.guard.isna() & ~trials.profiled]
    keys = ['run', 'case', 'strategy', 'workers', 'rows', 'ensembles', 'stixels', 'separate']
    summary = valid.groupby(keys, dropna=False).agg(
        repetitions=('total_seconds', 'size'), seconds=('total_seconds', 'median'),
        minimum_seconds=('total_seconds', 'min'), maximum_seconds=('total_seconds', 'max'),
        rss_gib=('prediction_rss_gib', 'median'),
        minimum_rss_gib=('prediction_rss_gib', 'min'), maximum_rss_gib=('prediction_rss_gib', 'max'),
        footprint_gib=('prediction_footprint_gib', 'median'),
        parent_rss_gib=('parent_rss_gib', 'median'), descendants_rss_gib=('workers_rss_gib', 'median'),
        model_mib=('model_mib', 'max'), output_bytes=('output_bytes', 'max'),
        masks_equal=('masks_equal', 'all'), std_masks_equal=('std_masks_equal', 'all'),
        max_prediction_difference=('prediction_difference', 'max'),
        max_std_difference=('std_difference', 'max')).reset_index()
    summary.to_csv(args.output / 'comparison.csv', index=False)
    selected = []
    figure, axes = plt.subplots(3, len(CHOICES), figsize=(16, 10), constrained_layout=True)
    measures = [('seconds', 'Prediction including preparation (s)'),
                ('rss_gib', 'Peak summed process RSS (GiB)'),
                ('footprint_gib', 'Peak macOS process footprint (GiB)')]
    for column, (run, case, title) in enumerate(CHOICES):
        part = summary[(summary.run == run) & (summary.case == case) & (summary.workers == 4)]
        part = part.set_index('strategy').reindex(list(COLORS)).dropna(subset='seconds')
        if part.empty:
            raise ValueError(f'No completed four-worker comparison for {run}/{case}')
        selected.append(part.reset_index())
        for row, (measure, ylabel) in enumerate(measures):
            ax = axes[row, column]
            bars = ax.bar(range(len(part)), part[measure], color=[COLORS[s] for s in part.index])
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
            if measure in {'seconds', 'rss_gib'}:
                lower = part[measure] - part[f'minimum_{measure}']
                upper = part[f'maximum_{measure}'] - part[measure]
                ax.errorbar(range(len(part)), part[measure], yerr=[lower, upper],
                            fmt='none', ecolor='black', capsize=3, linewidth=1)
            ax.set_xticks(range(len(part)), [LABELS[s] for s in part.index], fontsize=9)
            if column == 0:
                ax.set_ylabel(ylabel)
            ax.spines[['top', 'right']].set_visible(False)
            ax.margins(y=.25)
            if row == 0:
                count = int(part.repetitions.max())
                ax.set_title(f'{title}\n4 workers; {count} repetition(s)', fontsize=11)
    figure.suptitle('Memory-mapped prediction input and reusable result buffers', fontsize=15)
    figure.supxlabel('Bars are medians; whiskers show observed min–max, not confidence intervals. '
                      'RSS counts shared pages repeatedly. Footprint excludes much file cache.\n'
                      'These Apple Silicon measurements are not Slurm/cgroup memory charges.', fontsize=10)
    args.figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figures / 'prediction_memory.png', dpi=170)
    figure.savefig(args.figures / 'prediction_memory.pdf')
    plt.close(figure)
    selected = pd.concat(selected, ignore_index=True)
    selected.to_csv(args.output / 'selected_comparison.csv', index=False)
    print(selected[['case', 'strategy', 'repetitions', 'seconds', 'rss_gib', 'footprint_gib',
                    'masks_equal', 'max_prediction_difference']].to_string(index=False))
    print(f'Completed shared-backend trials: {len(valid)}/{len(trials)}')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=root / 'data/06_prediction_stress')
    parser.add_argument('--output', type=Path, default=root / 'data/06_prediction_stress/memory_summary')
    parser.add_argument('--figures', type=Path, default=root / 'figures/06_prediction_stress')
    main(parser.parse_args())
