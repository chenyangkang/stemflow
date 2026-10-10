import numpy as np
import pandas as pd

from stemflow.model.AdaSTEM import AdaSTEM
from stemflow.model_selection import ST_train_test_split
import pytest
from .make_models import (
    make_AdaSTEMClassifier,
    make_AdaSTEMRegressor,
    make_parallel_AdaSTEMClassifier,
    make_parallel_SphereAdaClassifier,
    make_parallel_STEMClassifier,
    make_SphereAdaClassifier,
    make_SphereAdaSTEMRegressor,
    make_STEMClassifier,
    make_STEMRegressor,
    make_AdaSTEMRegressor_Hurdle_for_AdaSTEM
)
from .set_up_data import set_up_data

x_names, (X, y) = set_up_data()
X_train, X_test, y_train, y_test = ST_train_test_split(
    X, y, Spatio_blocks_count=100, Temporal_blocks_count=100, random_state=42, test_size=0.3
)

def test_quadtree_not_supported_for_stem():
    X_train.loc[:, 'aaa'] = np.random.normal(size=len(X_train))
    model = make_STEMClassifier()
    y_train.loc[:] = np.where(y_train > 0, 1, 0)
    with pytest.raises(AttributeError):
        model.split(X_train, quadtree_arg_dict={'additional_features':['aaa'], 'addiitonal_quadtree_criteria':lambda x:len(x[x['aaa']>-0.3])>5})

def test_AdaSTEMClassifier():
    model = make_AdaSTEMClassifier()
    y_train.loc[:] = np.where(y_train > 0, 1, 0)
    X_train.loc[:, 'aaa'] = np.random.normal(size=len(X_train))
    model.split(X_train, quadtree_arg_dict={'additional_features':['aaa'], 'addiitonal_quadtree_criteria':lambda x:len(x[x['aaa']>-0.3])>5})
    del X_train['aaa']
    model = model.fit(X_train, y_train, ensemble_df=model.ensemble_df) # Use the splitted ensemble_df

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3


def test_AdaSTEMRegressor():
    model = make_AdaSTEMRegressor()
    y_train.loc[:] = np.where(y_train > 0, 1, 0)
    X_train.loc[:, 'aaa'] = np.random.normal(size=len(X_train))
    model.split(X_train, quadtree_arg_dict={'additional_features':['aaa'], 'addiitonal_quadtree_criteria':lambda x:len(x[x['aaa']>-0.3])>5})
    del X_train['aaa']
    model = model.fit(X_train, y_train, ensemble_df=model.ensemble_df) # Use the splitted ensemble_df

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0
