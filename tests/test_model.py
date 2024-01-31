import numpy as np
import pandas as pd

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


def test_STEMClassifier():
    model = make_STEMClassifier()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, njobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    # assert eval["Spearman_r"] >= 0.2

    model.calculate_feature_importances()
    assert model.feature_importances_.shape[0] > 0

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, njobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3


def test_parallel_STEMClassifier():
    model = make_parallel_STEMClassifier()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    # assert eval["Spearman_r"] >= 0.2

    model.calculate_feature_importances()
    assert model.feature_importances_.shape[0] > 0

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3


def test_STEMRegressor():
    model = make_STEMRegressor()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, njobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("hurdle", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    assert eval["Spearman_r"] >= 0.2

    model.calculate_feature_importances()
    assert model.feature_importances_.shape[0] > 0

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, njobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3


def test_AdaSTEMClassifier():
    model = make_AdaSTEMClassifier()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, njobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    # assert eval["Spearman_r"] >= 0.2

    model.calculate_feature_importances()
    assert model.feature_importances_.shape[0] > 0

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, njobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3


def test_AdaSTEMRegressor():
    model = make_AdaSTEMRegressor()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, njobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("hurdle", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    assert eval["Spearman_r"] >= 0.2

    model.calculate_feature_importances()
    assert model.feature_importances_.shape[0] > 0

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, njobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3


def test_SphereAdaClassifier():
    model = make_SphereAdaClassifier()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, njobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    # assert eval["Spearman_r"] >= 0.2

    model.calculate_feature_importances()
    assert model.feature_importances_.shape[0] > 0

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, njobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3


def test_parallel_SphereAdaClassifier():
    model = make_parallel_SphereAdaClassifier()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    # assert eval["Spearman_r"] >= 0.2

    model.calculate_feature_importances()
    assert model.feature_importances_.shape[0] > 0

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3


def test_SphereAdaSTEMRegressor():
    model = make_SphereAdaSTEMRegressor()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, njobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("hurdle", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.2
    assert eval["Spearman_r"] >= 0.2

    model.calculate_feature_importances()
    assert model.feature_importances_.shape[0] > 0

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, njobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3
