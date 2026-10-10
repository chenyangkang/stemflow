"""Compare legacy and corrected spatial jitter on the same mapped observations."""

import argparse
import json
from itertools import combinations
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import duckdb
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier

from audit_rotation import build_grid, LegacyQTree
from stemflow.gridding.QTree import QTree
from stemflow.model.AdaSTEM import AdaSTEMClassifier


def observations(database, rows):
    if database is None:
        return np.random.default_rng(42).uniform([-105, 25], [-70, 55], (rows, 2)), "synthetic uniform locations"
    with duckdb.connect(str(database), read_only=True,
                        config={"threads": "1", "memory_limit": "100MB"}) as con:
        data = con.sql(f"""
            SELECT longitude, latitude FROM (
                SELECT longitude, latitude FROM tmp
                WHERE longitude BETWEEN -105 AND -70 AND latitude BETWEEN 25 AND 55
            ) USING SAMPLE reservoir({int(rows)} ROWS) REPEATABLE (42)
        """).df()
    return data.to_numpy(), f"sampled checklist coordinates from {database.name}"


def similarity(a, b):
    valid = (a >= 0) & (b >= 0)
    return float(adjusted_rand_score(a[valid], b[valid]))


def pipeline_metadata_audit(xy, output):
    data = pd.DataFrame(dict(longitude=xy[:, 0], latitude=xy[:, 1],
                             DOY=np.linspace(1, 40, len(xy))))
    frames = []
    with tempfile.TemporaryDirectory() as temporary:
        for corrected in [False, True]:
            model = AdaSTEMClassifier(base_model=DecisionTreeClassifier(), ensemble_fold=6,
                min_ensemble_required=1, grid_len_upper_threshold=12, grid_len_lower_threshold=2,
                points_lower_threshold=20, temporal_start=1, temporal_end=41, temporal_step=40,
                temporal_bin_interval=40, temporal_bin_start_jitter=0, save_gridding_plot=False,
                completely_random_rotation=False, n_jobs=1, random_state=42, verbosity=0,
                lazy_loading_dir=str(Path(temporary)/str(corrected)))
            if corrected:
                model.split(data, verbosity=0)
            else:
                with patch.object(QTree, "_grid_phase", LegacyQTree._grid_phase):
                    model.split(data, verbosity=0)
            frame = model.ensemble_df.copy(deep=True)
            frame.to_csv(output/f"ensemble_df_{'after' if corrected else 'before'}.csv", index=False)
            frames.append(frame)
    before, after = frames
    assert list(before.columns) == list(after.columns)
    assert list(before.dtypes) == list(after.dtypes)
    metadata = ["rotation", "calibration_point_x_jitter", "calibration_point_y_jitter", "DOY_start", "DOY_end"]
    pd.testing.assert_frame_equal(before.groupby("ensemble_index")[metadata].first(),
                                  after.groupby("ensemble_index")[metadata].first())
    return dict(columns_unchanged=True, dtypes_unchanged=True, transform_metadata_unchanged=True,
                stixels_before=len(before), stixels_after=len(after))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--output", type=Path, default=Path("data/03_gridding/jitter_correction"))
    parser.add_argument("--figures", type=Path, default=Path("figures/03_gridding/jitter_correction"))
    args = parser.parse_args()
    if args.rows < 100:
        parser.error("Use at least 100 rows.")
    xy, source = observations(args.database, args.rows)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    magnitude = float(np.ptp(xy, axis=0).max())
    jitters = np.random.default_rng(73).uniform(-magnitude, magnitude, (6, 2))
    schedules = {
        "Jitter only: all angles 27.5°": np.full(6, 27.5),
        "Neighboring angles: 0°, 1.8°, …, 9°": np.arange(6)*1.8,
        "Spread angles: 0°, 18°, 36°, 54°, 72°, 88.2°": np.array([0., 18., 36., 54., 72., 88.2]),
    }
    records, ablations = [], []
    colors = plt.get_cmap("tab10")(np.arange(6))
    for kind in ["adaptive", "fixed"]:
        fig, axes = plt.subplots(3, 2, figsize=(13, 13), sharex=True, sharey=True,
                                 constrained_layout=True)
        fig.suptitle(f"{'AdaSTEM' if kind == 'adaptive' else 'STEM'} spatial jitter: before and after\n"
                     f"{len(xy):,} {source}; six grids per panel", fontsize=15)
        for row, (scenario, angles) in enumerate(schedules.items()):
            for col, corrected in enumerate([False, True]):
                ax = axes[row, col]
                labels, cells, elapsed = [], [], []
                ax.scatter(xy[:, 0], xy[:, 1], s=2, color=".5", alpha=.15, zorder=1)
                for i, (angle, jitter) in enumerate(zip(angles, jitters)):
                    start = time.perf_counter()
                    groups, polygons, partition = build_grid(xy, angle, jitter, kind,
                                                             corrected=corrected)
                    elapsed.append(time.perf_counter()-start)
                    labels.append(partition)
                    cells.append(sum(bool(group) for group in groups))
                    ax.add_collection(LineCollection(polygons, colors=[colors[i]],
                                                       linewidths=.7, alpha=.8, zorder=2))
                ari = [similarity(labels[i], labels[j]) for i, j in combinations(range(6), 2)]
                records.append(dict(kind=kind, scenario=scenario, corrected=corrected,
                    mean_pair_ari=float(np.mean(ari)),
                    min_assigned_fraction=float(min(np.mean(label >= 0) for label in labels)),
                    mean_occupied_cells=float(np.mean(cells)),
                    mean_build_seconds=float(np.mean(elapsed))))
                ax.set_title(f"{'Corrected' if corrected else 'Before correction'} — {scenario}\n"
                             f"Mean partition agreement: {np.mean(ari):.3f}", fontsize=10)
                ax.set(xlim=(-105, -70), ylim=(25, 55), aspect="equal",
                       xlabel="Longitude", ylabel="Latitude")
                ax.grid(alpha=.15)
        axes[0, 0].legend([Line2D([], [], color=color) for color in colors],
                          [f"Grid {i+1}" for i in range(6)], ncol=3, fontsize=8, loc="lower left")
        fig.savefig(args.figures/f"{kind}_ensemble_overlay.png", dpi=180)
        plt.close(fig)

        # Reproduce distinct legacy grids when the angle or input rows change.
        baseline = build_grid(xy, 27.5, jitters[0], kind)
        shifted = build_grid(xy, 27.5, jitters[1], kind)
        nearby = build_grid(xy, 29.3, jitters[0], kind)
        rotated = build_grid(xy, 60., jitters[0], kind)
        selected = np.random.default_rng(19).choice(len(xy), int(.8*len(xy)), replace=False)
        subset = build_grid(xy[selected], 27.5, jitters[0], kind)
        bootstrap_ids = np.random.default_rng(20).choice(len(xy), len(xy), replace=True)
        bootstrap = build_grid(xy[bootstrap_ids], 27.5, jitters[0], kind)
        ablations.append(dict(kind=kind,
            jitter_only_identical_memberships=baseline[0] == shifted[0],
            jitter_only_ari=similarity(baseline[2], shifted[2]),
            nearby_angle_ari=similarity(baseline[2], nearby[2]),
            distant_angle_ari=similarity(baseline[2], rotated[2]),
            subset_same_angle_ari=similarity(baseline[2][selected], subset[2]),
            bootstrap_same_angle_ari=similarity(baseline[2][bootstrap_ids], bootstrap[2])))

    pd.DataFrame(records).to_csv(args.output/"comparison.csv", index=False)
    report = dict(source=source, rows=len(xy), jitters=jitters.tolist(),
        comparisons=records, legacy_ablations=ablations,
        pipeline_metadata=pipeline_metadata_audit(xy, args.output),
        note="Angles and observations are held identical between before/after panels. "
        "ARI measures membership agreement, not area overlap or model accuracy. "
        "Build time includes polygon extraction and is an exploratory single run.")
    (args.output/"summary.json").write_text(json.dumps(report, indent=2))
    print(pd.DataFrame(records).to_string(index=False))
    print(json.dumps(ablations, indent=2))


if __name__ == "__main__":
    main()
