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
    make_AdaSTEMRegressor_Hurdle_for_AdaSTEM
)
from .set_up_data import set_up_data

x_names, (X, y) = set_up_data()
X_train, X_test, y_train, y_test = ST_train_test_split(
    X, y, Spatio_blocks_count=100, Temporal_blocks_count=100, random_state=42, test_size=0.3
)


def test_STEMClassifier():
    model = make_STEMClassifier()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
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

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
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

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
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

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3


def test_AdaSTEMClassifier():
    model = make_AdaSTEMClassifier()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
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

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3


def test_AdaSTEMRegressor():
    model = make_AdaSTEMRegressor()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred_ = model.predict(X_test, aggregation='median')
    assert len(pred_) == len(X_test)
    assert np.sum(np.isnan(pred_)) / len(pred_) <= 0.3
    
    pred_return_by_separate_ensembles = model.predict(X_test, return_by_separate_ensembles=True)
    assert pred_return_by_separate_ensembles.shape[1]==model.ensemble_fold
    
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

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3
    
    # score
    score_df = model.score(X_test, y_test)
    
        


def test_SphereAdaClassifier():
    model = make_SphereAdaClassifier()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
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

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=2, aggregation='median')
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3
    
def test_parallel_SphereAdaClassifier():
    model = make_parallel_SphereAdaClassifier()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))
    model = model.set_params(save_gridding_plot=True)

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

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
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

    importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
    assert importances_by_points.shape[0] > 0
    assert importances_by_points.shape[1] == len(x_names) + 3
    

def test_AdaSTEMRegressor_Hurdle_for_AdaSTEM():
    model = make_AdaSTEMRegressor_Hurdle_for_AdaSTEM()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean = model.predict(X_test.reset_index(drop=True))
    assert np.sum(~np.isnan(pred_mean)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("hurdle", pred_df.y_true, pred_df.y_pred)
    assert eval["AUC"] >= 0.5
    assert eval["kappa"] >= 0.1
    assert eval["Spearman_r"] >= 0.1



# def test_AdaSTEMRegressor_median():
#     model = make_AdaSTEMRegressor()
#     model = model.fit(X_train, np.where(y_train > 0, 1, 0))

#     pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1)
#     assert np.sum(~np.isnan(pred_mean)) > 0
#     assert np.sum(~np.isnan(pred_std)) > 0

#     pred = model.predict(X_test)
#     assert len(pred) == len(X_test)
#     assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

#     pred_df = pd.DataFrame(
#         {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
#     ).dropna()
#     assert len(pred_df) > 0

#     eval = AdaSTEM.eval_STEM_res("hurdle", pred_df.y_true, pred_df.y_pred)
#     assert eval["AUC"] >= 0.5
#     assert eval["kappa"] >= 0.2
#     assert eval["Spearman_r"] >= 0.2

#     model.calculate_feature_importances()
#     assert model.feature_importances_.shape[0] > 0

#     importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
#     assert importances_by_points.shape[0] > 0
#     assert importances_by_points.shape[1] == len(x_names) + 3


