import numpy as np
import pandas as pd
import tempfile
import os
from contextlib import contextmanager
import shutil
import duckdb
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
def make_tmp_duckdb_files(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    try:
        tmp_dir = os.path.join(tempfile.gettempdir(), f'{generate_random_saving_code()}')
        os.makedirs(tmp_dir)
        
        X_train = X_train.copy()
        X_train['__index_level_0__'] = X_train.index
        X_test = X_test.copy()
        X_test['__index_level_0__'] = X_test.index
        
        y_train_regressor = y_train.copy()
        y_train_classifier = y_train.copy()
        y_train_classifier.loc[:] = np.where(y_train_regressor>0, 1, 0)
        y_train_regressor['__index_level_0__'] = y_train.index
        y_train_classifier['__index_level_0__'] = y_train.index
        
        y_test_regressor = y_test.copy()
        y_test_classifier = y_test.copy()
        y_test_classifier.loc[:] = np.where(y_test_classifier>0, 1, 0)
        y_test_regressor['__index_level_0__'] = y_test.index
        y_test_classifier['__index_level_0__'] = y_test.index
        
        target_X_train_path = os.path.join(tmp_dir, 'X_train.duckdb')
        con = duckdb.connect(target_X_train_path)
        con.execute("CREATE TABLE tmp AS SELECT * FROM X_train;")
        con.execute("CREATE INDEX IF NOT EXISTS idx_idx ON tmp(__index_level_0__);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_doy ON tmp(DOY);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_longitude ON tmp(longitude);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_latitude ON tmp(latitude);")
        con.close()

        target_y_train_regressor_path = os.path.join(tmp_dir, 'y_train_regressor.duckdb')
        con = duckdb.connect(target_y_train_regressor_path)
        con.execute("CREATE TABLE tmp AS SELECT * FROM y_train_regressor;")
        con.execute("CREATE INDEX IF NOT EXISTS idx_idx ON tmp(__index_level_0__);")
        con.close()
        
        target_y_train_classifier_path = os.path.join(tmp_dir, 'y_train_classifier.duckdb')
        con = duckdb.connect(target_y_train_classifier_path)
        con.execute("CREATE TABLE tmp AS SELECT * FROM y_train_classifier;")
        con.execute("CREATE INDEX IF NOT EXISTS idx_idx ON tmp(__index_level_0__);")
        con.close()
        
        target_X_test_path = os.path.join(tmp_dir, 'X_test.duckdb')
        con = duckdb.connect(target_X_test_path)
        con.execute("CREATE TABLE tmp AS SELECT * FROM X_test;")
        con.execute("CREATE INDEX IF NOT EXISTS idx_idx ON tmp(__index_level_0__);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_doy ON tmp(DOY);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_longitude ON tmp(longitude);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_latitude ON tmp(latitude);")
        con.close()

        target_y_test_regressor_path = os.path.join(tmp_dir, 'y_test_regressor.duckdb')
        con = duckdb.connect(target_y_test_regressor_path)
        con.execute("CREATE TABLE tmp AS SELECT * FROM y_test_regressor;")
        con.execute("CREATE INDEX IF NOT EXISTS idx_idx ON tmp(__index_level_0__);")
        con.close()
        
        target_y_test_classifier_path = os.path.join(tmp_dir, 'y_test_classifier.duckdb')
        con = duckdb.connect(target_y_test_classifier_path)
        con.execute("CREATE TABLE tmp AS SELECT * FROM y_test_classifier;")
        con.execute("CREATE INDEX IF NOT EXISTS idx_idx ON tmp(__index_level_0__);")
        con.close()

        yield target_X_train_path, target_y_train_regressor_path, target_y_train_classifier_path, target_X_test_path, target_y_test_regressor_path, target_y_test_classifier_path
        
    finally:
        shutil.rmtree(tmp_dir)


def test_STEMClassifier_duckdb_input():
    with make_tmp_duckdb_files() as (target_X_train_path, target_y_train_regressor_path, target_y_train_classifier_path, target_X_test_path, target_y_test_regressor_path, target_y_test_classifier_path):
        model = make_STEMClassifier()
        model = model.fit(target_X_train_path, target_y_train_classifier_path)

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


def test_parallel_STEMClassifier_duckdb_input():
    with make_tmp_duckdb_files() as (target_X_train_path, target_y_train_regressor_path, target_y_train_classifier_path, target_X_test_path, target_y_test_regressor_path, target_y_test_classifier_path):
        model = make_parallel_STEMClassifier()
        model = model.fit(target_X_train_path, target_y_train_classifier_path)

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


def test_AdaSTEMClassifier_duckdb_input():
    with make_tmp_duckdb_files() as (target_X_train_path, target_y_train_regressor_path, target_y_train_classifier_path, target_X_test_path, target_y_test_regressor_path, target_y_test_classifier_path):
        model = make_AdaSTEMClassifier()
        model = model.fit(target_X_train_path, target_y_train_classifier_path)

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


def test_AdaSTEMRegressor_duckdb_input():
    with make_tmp_duckdb_files() as (target_X_train_path, target_y_train_regressor_path, target_y_train_classifier_path, target_X_test_path, target_y_test_regressor_path, target_y_test_classifier_path):
        model = make_AdaSTEMRegressor()
        model = model.fit(target_X_train_path, target_y_train_regressor_path)

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
    

def test_parallel_AdaSTEMClassifier_duckdb_input():
    with make_tmp_duckdb_files() as (target_X_train_path, target_y_train_regressor_path, target_y_train_classifier_path, target_X_test_path, target_y_test_regressor_path, target_y_test_classifier_path):
        model = make_parallel_AdaSTEMClassifier()
        model = model.fit(target_X_train_path, target_y_train_classifier_path)

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


def test_parallel_duckdb_temporal_window_prequery_works():
    with make_tmp_duckdb_files() as (target_X_train_path, target_y_train_regressor_path, target_y_train_classifier_path, target_X_test_path, target_y_test_regressor_path, target_y_test_classifier_path):
        model = make_parallel_AdaSTEMClassifier()
        model = model.fit(target_X_train_path, target_y_train_classifier_path)
        pred1 = model.predict(target_X_test_path)
        
        model = model.fit(target_X_train_path, target_y_train_classifier_path, overwrite=True)
        pred2 = model.predict(target_X_test_path)
        assert np.allclose(pred1.flatten(), pred2.flatten(), equal_nan=True)
        
        model = model.fit(target_X_train_path, target_y_train_classifier_path, overwrite=True, temporal_window_prequery=True)
        pred3 = model.predict(target_X_test_path)
        assert np.allclose(pred1.flatten(), pred3.flatten(), equal_nan=True)
        

