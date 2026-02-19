import numpy as np
import pandas as pd
import tempfile
import os
from contextlib import contextmanager
import shutil

from stemflow.model.AdaSTEM import AdaSTEM
from stemflow.model_selection import ST_train_test_split
from stemflow.utils.generate_random import generate_random_saving_code
from .make_models import (
    make_AdaSTEMClassifier,
    make_AdaSTEMRegressor,
    make_parallel_AdaSTEMClassifier,
    make_parallel_STEMClassifier,
    make_STEMClassifier,
)
from .set_up_data import set_up_data

x_names, (X, y) = set_up_data()
X_train, X_test, y_train, y_test = ST_train_test_split(
    X, y, Spatio_blocks_count=100, Temporal_blocks_count=100, random_state=42, test_size=0.3
)

@contextmanager
def make_tmp_parquet_files():
    try:
        tmp_dir = os.path.join(tempfile.gettempdir(), f'{generate_random_saving_code()}')
        os.makedirs(tmp_dir)
        
        target_X_train_path = os.path.join(tmp_dir, 'X_train.parquet')
        target_y_train_path = os.path.join(tmp_dir, 'y_train.parquet')
        target_X_test_path = os.path.join(tmp_dir, 'X_test.parquet')
        target_y_test_path = os.path.join(tmp_dir, 'y_test.parquet')
        X_train.to_parquet(target_X_train_path, engine="pyarrow", index=True)
        y_train.loc[:] = np.where(y_train > 0, 1, 0)
        y_train.to_parquet(target_y_train_path, engine="pyarrow", index=True)
        X_test.to_parquet(target_X_test_path, engine="pyarrow", index=True)
        y_test.loc[:] = np.where(y_test > 0, 1, 0)
        y_test.to_parquet(target_y_test_path, engine="pyarrow", index=True)
        
        yield target_X_train_path, target_y_train_path, target_X_test_path, target_y_test_path
        
    finally:
        shutil.rmtree(tmp_dir)


def test_STEMClassifier_Parquet_input():
    with make_tmp_parquet_files() as (target_X_train_path, target_y_train_path, target_X_test_path, target_y_test_path):
        model = make_STEMClassifier()
        model = model.fit(target_X_train_path, target_y_train_path)

        pred_mean, pred_std = model.predict(target_X_test_path, return_std=True, verbosity=1, n_jobs=1)
        assert np.sum(~np.isnan(pred_mean)) > 0
        assert np.sum(~np.isnan(pred_std)) > 0

        pred = model.predict(target_X_test_path)
        assert len(pred) == len(X_test)
        assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

        pred_df = pd.DataFrame(
            {"y_true": np.array(y_test).flatten(), "y_pred": np.where(pred < 0, 0, pred).flatten()}
        ).dropna()
        assert len(pred_df) > 0

        eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
        assert eval["AUC"] >= 0.51
        assert eval["kappa"] >= 0.2
        # assert eval["Spearman_r"] >= 0.2

        model.calculate_feature_importances()
        assert model.feature_importances_.shape[0] > 0

        importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
        assert importances_by_points.shape[0] > 0
        assert importances_by_points.shape[1] == len(x_names) + 3


def test_parallel_STEMClassifier_Parquet_input():
    with make_tmp_parquet_files() as (target_X_train_path, target_y_train_path, target_X_test_path, target_y_test_path):
        model = make_parallel_STEMClassifier()
        model = model.fit(target_X_train_path, target_y_train_path)

        pred_mean, pred_std = model.predict(target_X_test_path, return_std=True, verbosity=1)
        assert np.sum(~np.isnan(pred_mean)) > 0
        assert np.sum(~np.isnan(pred_std)) > 0

        pred = model.predict(target_X_test_path)
        assert len(pred) == len(X_test)
        assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

        pred_df = pd.DataFrame(
            {"y_true": np.array(y_test).flatten(), "y_pred": np.where(pred < 0, 0, pred).flatten()}
        ).dropna()
        assert len(pred_df) > 0

        eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
        assert eval["AUC"] >= 0.51
        assert eval["kappa"] >= 0.2
        # assert eval["Spearman_r"] >= 0.2

        model.calculate_feature_importances()
        assert model.feature_importances_.shape[0] > 0

        importances_by_points = model.assign_feature_importances_by_points(verbosity=0)
        assert importances_by_points.shape[0] > 0
        assert importances_by_points.shape[1] == len(x_names) + 3


def test_AdaSTEMClassifier_Parquet_input():
    with make_tmp_parquet_files() as (target_X_train_path, target_y_train_path, target_X_test_path, target_y_test_path):
        model = make_AdaSTEMClassifier()
        model = model.fit(target_X_train_path, target_y_train_path)

        pred_mean, pred_std = model.predict(target_X_test_path, return_std=True, verbosity=1, n_jobs=1)
        assert np.sum(~np.isnan(pred_mean)) > 0
        assert np.sum(~np.isnan(pred_std)) > 0

        pred = model.predict(target_X_test_path)
        assert len(pred) == len(X_test)
        assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

        pred_df = pd.DataFrame(
            {"y_true": np.array(y_test).flatten(), "y_pred": np.where(pred < 0, 0, pred).flatten()}
        ).dropna()
        assert len(pred_df) > 0

        eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
        assert eval["AUC"] >= 0.51
        assert eval["kappa"] >= 0.2
        # assert eval["Spearman_r"] >= 0.2

        model.calculate_feature_importances()
        assert model.feature_importances_.shape[0] > 0

        importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
        assert importances_by_points.shape[0] > 0
        assert importances_by_points.shape[1] == len(x_names) + 3


def test_AdaSTEMRegressor_Parquet_input():
    with make_tmp_parquet_files() as (target_X_train_path, target_y_train_path, target_X_test_path, target_y_test_path):
        model = make_AdaSTEMRegressor()
        model = model.fit(target_X_train_path, target_y_train_path)

        pred_mean, pred_std = model.predict(target_X_test_path, return_std=True, verbosity=1, n_jobs=1)
        assert np.sum(~np.isnan(pred_mean)) > 0
        assert np.sum(~np.isnan(pred_std)) > 0

        pred_ = model.predict(target_X_test_path, aggregation='median')
        assert len(pred_) == len(X_test)
        assert np.sum(np.isnan(pred_)) / len(pred_) <= 0.3
        
        pred_return_by_separate_ensembles = model.predict(target_X_test_path, return_by_separate_ensembles=True)
        assert pred_return_by_separate_ensembles.shape[1]==model.ensemble_fold
        
        pred = model.predict(target_X_test_path)
        assert len(pred) == len(X_test)
        assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

        pred_df = pd.DataFrame(
            {"y_true": np.array(y_test).flatten(), "y_pred": np.where(pred < 0, 0, pred).flatten()}
        ).dropna()
        assert len(pred_df) > 0

        eval = AdaSTEM.eval_STEM_res("hurdle", pred_df.y_true, pred_df.y_pred)
        assert eval["AUC"] >= 0.51
        assert eval["kappa"] >= 0.2
        assert eval["Spearman_r"] >= 0.2

        model.calculate_feature_importances()
        assert model.feature_importances_.shape[0] > 0

        importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
        assert importances_by_points.shape[0] > 0
        assert importances_by_points.shape[1] == len(x_names) + 3
        
        # score
        score_df = model.score(X_test, y_test)
    

def test_parallel_AdaSTEMClassifier_Parquet_input():
    with make_tmp_parquet_files() as (target_X_train_path, target_y_train_path, target_X_test_path, target_y_test_path):
        model = make_parallel_AdaSTEMClassifier()
        model = model.fit(target_X_train_path, target_y_train_path)

        pred_mean, pred_std = model.predict(target_X_test_path, return_std=True, verbosity=1, n_jobs=1)
        assert np.sum(~np.isnan(pred_mean)) > 0
        assert np.sum(~np.isnan(pred_std)) > 0

        pred = model.predict(target_X_test_path)
        assert len(pred) == len(X_test)
        assert np.sum(np.isnan(pred)) / len(pred) <= 0.3

        pred_df = pd.DataFrame(
        {"y_true": np.array(y_test).flatten(), "y_pred": np.where(pred < 0, 0, pred).flatten()}
        ).dropna()
        assert len(pred_df) > 0

        eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
        assert eval["AUC"] >= 0.51
        assert eval["kappa"] >= 0.2
        # assert eval["Spearman_r"] >= 0.2

        model.calculate_feature_importances()
        assert model.feature_importances_.shape[0] > 0

        importances_by_points = model.assign_feature_importances_by_points(verbosity=0, n_jobs=1)
        assert importances_by_points.shape[0] > 0
        assert importances_by_points.shape[1] == len(x_names) + 3

