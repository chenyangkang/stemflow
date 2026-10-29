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


@pytest.fixture
def filtered_data():
    # Keep filter inputs independent of test order and process-global randomness.
    rng = np.random.default_rng(42)
    train, test = X_train.copy(), X_test.copy()
    train['aaa'] = rng.normal(size=len(train))
    test['aaa'] = rng.normal(size=len(test))
    labels = (y_train > 0).astype(int)
    return train, test, labels, y_test.copy()


def test_STEMRegressor(filtered_data):
    X_train, X_test, y_train, y_test = filtered_data
    model = make_STEMRegressor()
    model = model.fit(X_train, y_train, stixel_filter_func=lambda x: x[x['aaa']>-0.3])

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": np.array(y_test).flatten(), "y_pred": np.where(pred < 0, 0, pred).flatten()}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("hurdle", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    assert eval["Spearman_r"] >= 0.2

    model.calculate_feature_importances()
    assert model.feature_importances_.shape[0] > 0


def test_AdaSTEMClassifier(filtered_data):
    X_train, X_test, y_train, y_test = filtered_data
    model = make_AdaSTEMClassifier()
    model = model.fit(X_train, y_train, stixel_filter_func=lambda x: x[x['aaa']>-0.3])

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": np.array(y_test).flatten(), "y_pred": np.where(pred < 0, 0, pred).flatten()}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    # assert eval["Spearman_r"] >= 0.2

    model.calculate_feature_importances()
    assert model.feature_importances_.shape[0] > 0
