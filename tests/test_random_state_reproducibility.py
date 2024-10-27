import numpy as np
import pandas as pd

from stemflow.model.AdaSTEM import AdaSTEM
from stemflow.model_selection import ST_train_test_split

from .make_models import make_AdaSTEMRegressor
from .set_up_data import set_up_data

x_names, (X, y) = set_up_data()

def test_random_state_reproducibility():
    model = make_AdaSTEMRegressor()
    model = model.set_params(random_state=42)
    model.split(X, verbosity=1)
    ensemble_df1 = model.ensemble_df.copy()

    model = model.set_params(random_state=990324)
    model.split(X, verbosity=1)
    ensemble_df2 = model.ensemble_df.copy()

    model = model.set_params(random_state=42)
    model.split(X, verbosity=1)
    ensemble_df3 = model.ensemble_df.copy()

    # 1 and 3 (with the same random state)
    assert ensemble_df1.shape == ensemble_df3.shape
    assert np.allclose(ensemble_df1['calibration_point_x_jitter'].values,
                ensemble_df3['calibration_point_x_jitter'].values)
    assert np.allclose(ensemble_df1['stixel_checklist_count'].values,
                ensemble_df3['stixel_checklist_count'].values)

    # 1 and 2 (with the different random state)
    assert np.sum(ensemble_df1['calibration_point_x_jitter'].values[0] - \
            ensemble_df2['calibration_point_x_jitter'].values[0]) != 0
    assert np.sum(ensemble_df1['stixel_checklist_count'].values[0] - \
            ensemble_df2['stixel_checklist_count'].values[0]) != 0


def test_random_state_reproducibility_completely_random_rotation_angle():
    model = make_AdaSTEMRegressor()
    model = model.set_params(random_state=42, completely_random_rotation=True)
    model.split(X, verbosity=1)
    ensemble_df1 = model.ensemble_df.copy()

    model = model.set_params(random_state=990324, completely_random_rotation=True)
    model.split(X, verbosity=1)
    ensemble_df2 = model.ensemble_df.copy()

    model = model.set_params(random_state=42, completely_random_rotation=True)
    model.split(X, verbosity=1)
    ensemble_df3 = model.ensemble_df.copy()

    # 1 and 3 (with the same random state)
    assert ensemble_df1.shape == ensemble_df3.shape
    assert np.allclose(ensemble_df1['calibration_point_x_jitter'].values,
                ensemble_df3['calibration_point_x_jitter'].values)
    assert np.allclose(ensemble_df1['stixel_checklist_count'].values,
                ensemble_df3['stixel_checklist_count'].values)

    # 1 and 2 (with the different random state)
    assert np.sum(ensemble_df1['calibration_point_x_jitter'].values[0] -
            ensemble_df2['calibration_point_x_jitter'].values[0]) != 0
    assert np.sum(ensemble_df1['stixel_checklist_count'].values[0] -
            ensemble_df2['stixel_checklist_count'].values[0]) != 0

