import numpy as np
import pandas as pd
import os

from stemflow.model.AdaSTEM import AdaSTEM
from stemflow.model_selection import ST_train_test_split

from .make_models import (
    make_AdaSTEMClassifier,
    make_AdaSTEMRegressor,
    make_parallel_SphereAdaClassifier,
    make_parallel_STEMClassifier,
    make_SphereAdaClassifier,
    make_SphereAdaSTEMRegressor,
    make_STEMClassifier,
    make_STEMRegressor,
)
from .set_up_data import set_up_data

x_names, (X, y) = set_up_data()
X_train, X_test, y_train, y_test = ST_train_test_split(
    X, y, Spatio_blocks_count=100, Temporal_blocks_count=100, random_state=42, test_size=0.3
)


def test_AdaSTEMRegressor_custom_temp_folder1():
    model = make_AdaSTEMRegressor(lazy_loading=True, joblib_temp_folder='lazy_loading_dir')
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)

def test_AdaSTEMRegressor_custom_temp_folder2():
    model = make_AdaSTEMRegressor(lazy_loading=True, joblib_temp_folder='./test_tmp_folder')
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))
    assert os.path.exists('./test_tmp_folder')

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
