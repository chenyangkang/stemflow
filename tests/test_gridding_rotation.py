"""Check planar angle scheduling, transforms and plotted cell orientation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.tree import DecisionTreeClassifier

from stemflow.gridding.QTree import QTree, find_children
from stemflow.model.AdaSTEM import AdaSTEMClassifier
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
