"""Audit planar angles and ineffective spatial jitter without fitting models."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier

from stemflow.gridding.QTree import QTree, find_children
from stemflow.gridding.QuadGrid import QuadGrid
from stemflow.model.AdaSTEM import AdaSTEMClassifier
from stemflow.utils.jitterrotation.jitterrotator import JitterRotator


def build_grid(xy, angle, jitter=(0., 0.), kind="adaptive"):
    common = dict(rotation_angle=angle, calibration_point_x_jitter=jitter[0],
                  calibration_point_y_jitter=jitter[1], points_lower_threshold=20,
                  plot_empty=True)
    grid = (QTree(12, 2, 12, 2, **common) if kind == "adaptive"
            else QuadGrid(grid_len=5, **common))
    grid.add_lon_lat_data(np.arange(len(xy)), xy[:, 0], xy[:, 1])
    grid.generate_gridding_params()
    grid.subdivide()
    nodes = find_children(grid.root) if kind == "adaptive" else grid.grids
    memberships, polygons = [], []
    labels = np.full(len(xy), -1, dtype=int)
    for cell, node in enumerate(nodes):
        ids = tuple(sorted(point.index for point in node.points))
        memberships.append(ids)
        labels[list(ids)] = cell
        if kind == "adaptive":
            x, y, width, height = node.x0, node.y0, node.width, node.height
        else:
            x, y, width, height = node.x_range[0], node.y_range[0], 5., 5.
        corners = np.array([[x, y], [x+width, y], [x+width, y+height],
                            [x, y+height], [x, y]])
        original = JitterRotator.inverse_jitter_rotate(
            corners[:, 0], corners[:, 1], angle, *jitter)
        polygons.append(np.column_stack(original))
    return memberships, np.asarray(polygons), labels


def angle_audit(xy, ensembles):
    data = pd.DataFrame(dict(longitude=xy[:, 0], latitude=xy[:, 1],
                             DOY=np.linspace(1, 40, len(xy))))
    model = AdaSTEMClassifier(base_model=DecisionTreeClassifier(),
        ensemble_fold=ensembles, min_ensemble_required=1,
        grid_len_upper_threshold=12, grid_len_lower_threshold=2,
        points_lower_threshold=20, temporal_start=1, temporal_end=41,
        temporal_step=40, temporal_bin_interval=40, temporal_bin_start_jitter=0,
        save_gridding_plot=False, completely_random_rotation=False,
        n_jobs=1, random_state=42, verbosity=0)
    try:
        model.split(data, verbosity=0)
        actual = model.ensemble_df.groupby("ensemble_index").rotation.agg(["first", "nunique"])
        expected = np.arange(ensembles)*90/ensembles
        np.testing.assert_allclose(actual["first"], expected, atol=1e-12, rtol=0)
        assert actual["nunique"].eq(1).all()
        return pd.DataFrame(dict(ensemble=np.arange(ensembles), expected=expected,
                                 actual=actual["first"].to_numpy()))
    finally:
        if getattr(model, "_finalizer", None) is not None:
            model._finalizer()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--ensembles", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("data/03_gridding/rotation_audit"))
    parser.add_argument("--figures", type=Path, default=Path("figures/03_gridding/rotation_audit"))
    args = parser.parse_args()
    if args.rows < 100 or args.ensembles < 2:
        parser.error("Use at least 100 rows and 2 ensembles.")
    rng = np.random.default_rng(42)
    xy = rng.uniform([-105, 25], [-70, 55], (args.rows, 2))
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    angles = angle_audit(xy, args.ensembles)
    angles.to_csv(args.output/"angles.csv", index=False)
    jitter_records = []
    for kind in ["adaptive", "fixed"]:
        for angle in [0., 27.5, 88.2]:
            reference = build_grid(xy, angle, kind=kind)
            shifted = build_grid(xy, angle, (17.25, -31.5), kind)
            jitter_records.append(dict(kind=kind, angle=angle,
                identical_memberships=reference[0] == shifted[0],
                max_vertex_difference=float(np.max(np.abs(reference[1]-shifted[1])))))
    jitter = pd.DataFrame(jitter_records)
    jitter.to_csv(args.output/"jitter.csv", index=False)

    # Isolate spatial partitions: identical observations, no temporal jitter or bootstrap.
    schedules = dict(evenly_spaced=angles.actual.to_numpy(),
                     random=np.random.default_rng(59).uniform(0, 90, args.ensembles))
    pairs, metrics = [], {}
    for name, schedule in schedules.items():
        labels = [build_grid(xy, angle)[2] for angle in schedule]
        scores, adjacent = [], []
        for i in range(args.ensembles):
            for j in range(i+1, args.ensembles):
                delta = abs(schedule[i]-schedule[j])
                valid = (labels[i] >= 0) & (labels[j] >= 0)
                score = adjusted_rand_score(labels[i][valid], labels[j][valid])
                pairs.append(dict(schedule=name, first=i, second=j,
                    angle_difference=min(delta, 90-delta), adjusted_rand=score,
                    common_rows=int(valid.sum())))
                scores.append(score)
                if j == i+1:
                    adjacent.append(score)
        metrics[name] = dict(mean_all_pairs=float(np.mean(scores)),
            mean_adjacent_ensembles=float(np.mean(adjacent)),
            max_unassigned_rows=int(max(np.sum(label < 0) for label in labels)))
    pd.DataFrame(pairs).to_csv(args.output/"partition_similarity.csv", index=False)
    report = dict(rows=args.rows, ensembles=args.ensembles,
        max_angle_error=float(np.max(np.abs(angles.expected-angles.actual))),
        all_jitter_memberships_identical=bool(jitter.identical_memberships.all()),
        max_vertex_difference=float(jitter.max_vertex_difference.max()),
        similarity=metrics,
        interpretation="Synthetic planar spatial partitions, not prediction accuracy. "
        "ARI 1 means identical partitions. Jitter cancels because the origin follows "
        "the shifted data. Random orientation does not guarantee less all-pair overlap.")
    (args.output/"summary.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(angles.ensemble, angles.actual, "o-", ms=3, label="Evenly spaced")
    axes[0].scatter(np.arange(args.ensembles), schedules["random"], s=12,
                    color="tab:orange", label="Random")
    axes[0].set(xlabel="Ensemble index", ylabel="Rotation (degrees)", title="Angle assignment is correct")
    axes[0].legend()
    for offset, color, style, label in [((0., 0.), "tab:blue", "solid", "Jitter (0, 0)"),
            ((17.25, -31.5), "tab:orange", "dashed", "Jitter (17.25, −31.5)")]:
        polygons = build_grid(xy, 27.5, offset)[1]
        axes[1].add_collection(LineCollection(polygons, colors=color, linestyles=style,
                                               linewidths=1, label=label))
    axes[1].set(xlim=(-105, -70), ylim=(25, 55), xlabel="Longitude", ylabel="Latitude",
                title="Changing jitter leaves the same grid", aspect="equal")
    axes[1].legend(fontsize=8)
    frame = pd.DataFrame(pairs)
    for name, group in frame.groupby("schedule", sort=False):
        axes[2].scatter(group.angle_difference, group.adjusted_rand, s=5, alpha=.2,
                        label=name.replace("_", " "))
    axes[2].set(xlabel="Orientation difference modulo 90°", ylabel="Partition similarity (ARI)",
                title="Nearby angles often give similar cells")
    axes[2].legend()
    fig.savefig(args.figures/"rotation_audit.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
