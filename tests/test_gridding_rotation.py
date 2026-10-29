"""Check planar angle scheduling, transforms and plotted cell orientation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.tree import DecisionTreeClassifier

from stemflow.gridding.QTree import QTree, find_children
from stemflow.gridding.QuadGrid import QuadGrid
from stemflow.model.AdaSTEM import AdaSTEMClassifier
from stemflow.model.STEM import STEMClassifier
from stemflow.model.static_func_AdaSTEM import transform_pred_set_to_STEM_quad
from stemflow.utils.jitterrotation.jitterrotator import JitterRotator


@pytest.mark.parametrize("angle", [0., 1.8, 27.5, 88.2, 90.])
def test_planar_rotation_roundtrip_and_distances(angle):
    xy = np.random.default_rng(31).uniform([-105, 25], [-70, 55], (100, 2))
    forward = np.column_stack(JitterRotator.rotate_jitter(xy[:, 0], xy[:, 1], angle, 17.25, -31.5))
    restored = np.column_stack(JitterRotator.inverse_jitter_rotate(
        forward[:, 0], forward[:, 1], angle, 17.25, -31.5))
    assert_allclose(restored, xy, atol=1e-12, rtol=0)
    assert_allclose(np.linalg.norm(forward[1:]-forward[:-1], axis=1),
                    np.linalg.norm(xy[1:]-xy[:-1], axis=1), atol=1e-12, rtol=0)


@pytest.mark.parametrize("jobs,ensembles", [(1, 50), (2, 6)])
def test_even_angles_reach_generated_stixels(tmp_path, jobs, ensembles):
    rng = np.random.default_rng(42)
    data = pd.DataFrame(dict(longitude=rng.uniform(-105, -70, 300),
        latitude=rng.uniform(25, 55, 300), DOY=np.linspace(1, 40, 300)))
    model = AdaSTEMClassifier(base_model=DecisionTreeClassifier(), ensemble_fold=ensembles,
        min_ensemble_required=1, grid_len_upper_threshold=25, grid_len_lower_threshold=3,
        points_lower_threshold=10, temporal_start=1, temporal_end=41, temporal_step=40,
        temporal_bin_interval=40, temporal_bin_start_jitter=0, save_gridding_plot=False,
        completely_random_rotation=False, n_jobs=jobs, random_state=42, verbosity=0,
        lazy_loading_dir=str(tmp_path))
    model.split(data, verbosity=0)
    angles = model.ensemble_df.groupby("ensemble_index").rotation.agg(["first", "nunique"])
    assert angles["nunique"].eq(1).all()
    assert_allclose(angles["first"], np.arange(ensembles)*90/ensembles, atol=1e-12, rtol=0)


def test_plotted_rectangle_matches_inverse_transformed_cell():
    xy = np.random.default_rng(42).uniform([-105, 25], [-70, 55], (300, 2))
    tree = QTree(12, 2, 12, 2, 20, rotation_angle=27.5,
                 calibration_point_x_jitter=17.25, calibration_point_y_jitter=-31.5)
    tree.add_lon_lat_data(np.arange(len(xy)), xy[:, 0], xy[:, 1])
    tree.generate_gridding_params()
    tree.subdivide()
    fig, ax = plt.subplots()
    try:
        tree.graph(scatter=False, ax=ax)
        for node, patch in zip(find_children(tree.root), ax.patches):
            corners = np.array([[node.x0, node.y0], [node.x0+node.width, node.y0],
                [node.x0+node.width, node.y0+node.height],
                [node.x0, node.y0+node.height], [node.x0, node.y0]])
            expected = np.column_stack(JitterRotator.inverse_jitter_rotate(
                corners[:, 0], corners[:, 1], 27.5, 17.25, -31.5))
            actual = patch.get_patch_transform().transform(patch.get_path().vertices)
            assert_allclose(actual, expected, atol=1e-12, rtol=0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["adaptive", "fixed", "rectangular"])
@pytest.mark.parametrize("angle", [0., 27.5, 88.2])
@pytest.mark.parametrize("jitter", [(1.25, 2.75), (-17.25, -31.5), (170.25, -315.75)])
def test_jitter_changes_membership_and_retains_points(kind, angle, jitter):
    rng = np.random.default_rng(42)
    xy = rng.uniform([-105, 25], [-70, 55], (600, 2))
    groups = []
    for offset in [(0., 0.), jitter]:
        common = dict(rotation_angle=angle, calibration_point_x_jitter=offset[0],
                      calibration_point_y_jitter=offset[1], points_lower_threshold=20,
                      plot_empty=True)
        if kind == "fixed":
            tree = QuadGrid(5, **common)
        else:
            tree = QTree(12, 2, 12, 2, lon_lat_equal_grid=kind != "rectangular", **common)
        tree.add_lon_lat_data(np.arange(len(xy)), xy[:, 0], xy[:, 1])
        tree.generate_gridding_params()
        tree.subdivide()
        nodes = tree.grids if kind == "fixed" else find_children(tree.root)
        partition = {frozenset(p.index for p in node.points) for node in nodes if node.points}
        groups.append(partition)
        if offset == jitter:
            assert set().union(*partition) == set(range(len(xy)))
    assert groups[0] != groups[1]


def test_fixed_grid_keeps_exact_outer_maximum():
    tree = QuadGrid(5, points_lower_threshold=0)
    xy = np.array([[0., 0.], [5., 5.], [10., 10.]])
    tree.add_lon_lat_data(np.arange(3), xy[:, 0], xy[:, 1])
    tree.subdivide()
    assert sorted(p.index for grid in tree.grids for p in grid.points) == [0, 1, 2]


def test_fixed_grid_jitter_is_periodic_in_cell_width():
    xy = np.random.default_rng(51).uniform([-105, 25], [-70, 55], (600, 2))
    partitions = []
    for jitter in [(1.25, 2.75), (11.25, -12.25)]:
        tree = QuadGrid(5, 20, rotation_angle=27.5,
            calibration_point_x_jitter=jitter[0], calibration_point_y_jitter=jitter[1])
        tree.add_lon_lat_data(np.arange(len(xy)), xy[:, 0], xy[:, 1])
        tree.subdivide()
        partitions.append({frozenset(p.index for p in grid.points) for grid in tree.grids if grid.points})
    assert partitions[0] == partitions[1]


def test_zero_jitter_retains_unpadded_adaptive_root():
    tree = QTree(12, 2, 12, 2, 20)
    tree.add_lon_lat_data([0, 1], [-105., -70.], [25., 55.])
    tree.generate_gridding_params()
    assert (tree.root.x0, tree.root.y0, tree.root.width, tree.root.height) == (-105., 25., 35., 35.)


@pytest.mark.parametrize("kind", ["adaptive", "fixed"])
def test_ensemble_schema_transforms_and_old_geometry_remain_compatible(tmp_path, monkeypatch, kind):
    rng = np.random.default_rng(42)
    n = 600
    X = pd.DataFrame(dict(longitude=rng.uniform(-105, -70, n),
        latitude=rng.uniform(25, 55, n), DOY=np.linspace(1, 40, n), signal=rng.normal(size=n)))
    y = (X.signal > 0).astype(int)
    def make(name):
        model_cls = AdaSTEMClassifier if kind == "adaptive" else STEMClassifier
        gridding = (dict(grid_len_upper_threshold=25, grid_len_lower_threshold=3)
                    if kind == "adaptive" else dict(grid_len=5))
        return model_cls(base_model=DecisionTreeClassifier(max_depth=2, random_state=31),
            ensemble_fold=2, min_ensemble_required=1, **gridding,
            points_lower_threshold=10, temporal_start=1,
            temporal_end=41, temporal_step=40, temporal_bin_interval=40,
            temporal_bin_start_jitter=0, save_gridding_plot=False,
            completely_random_rotation=False, n_jobs=1, random_state=42, verbosity=0,
            sample_weights_for_classifier=False, lazy_loading_dir=str(tmp_path/name))
    with monkeypatch.context() as patch:
        patch.setattr(QTree, "_grid_phase", lambda self, width, height: (0., 0.))
        patch.setattr(QuadGrid, "_grid_phase", lambda self, width, height: (0., 0.))
        old = make("old").fit(X, y, temporal_window_prequery=True)
        geometry = old.ensemble_df.copy(deep=True)
        expected = old.predict_proba(X, temporal_window_prequery=True)
    reused = make("reused").fit(X, y, ensemble_df=geometry, temporal_window_prequery=True)
    assert_allclose(reused.predict_proba(X, temporal_window_prequery=True), expected, equal_nan=True)
    fixed = make("fixed").fit(X, y, temporal_window_prequery=True)
    assert list(fixed.ensemble_df.columns) == list(geometry.columns)
    assert list(fixed.ensemble_df.dtypes) == list(geometry.dtypes)
    fields = ["rotation", "calibration_point_x_jitter", "calibration_point_y_jitter", "DOY_start", "DOY_end"]
    pd.testing.assert_frame_equal(fixed.ensemble_df.groupby("ensemble_index")[fields].first(),
                                  geometry.groupby("ensemble_index")[fields].first())
    for _, info in fixed.ensemble_df.groupby("ensemble_index"):
        transformed = transform_pred_set_to_STEM_quad("longitude", "latitude", X, info)
        row = info.iloc[0]
        direct = JitterRotator.rotate_jitter(X.longitude, X.latitude, row.rotation,
            row.calibration_point_x_jitter, row.calibration_point_y_jitter)
        assert_allclose(transformed[["longitude_new", "latitude_new"]], np.column_stack(direct))
    assert np.isfinite(fixed.predict_proba(X, temporal_window_prequery=True)).any()
