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


def test_AdaSTEMClassifier_duckdb_input_no_lazy():
    with make_tmp_duckdb_files() as (target_X_train_path, target_y_train_regressor_path, target_y_train_classifier_path, target_X_test_path, target_y_test_regressor_path, target_y_test_classifier_path):
        model = make_AdaSTEMClassifier()
        model = model.fit(target_X_train_path, target_y_train_classifier_path)

        model.save(tar_gz_file='./my_model1.tar.gz', remove_temporary_file=True)
        assert not os.path.exists(model.lazy_loading_dir)
        
        model = AdaSTEM.load(tar_gz_file='./my_model1.tar.gz', new_lazy_loading_path='./new_lazyloading_ensemble_folder1', remove_original_file=False)
        assert os.path.exists(model.lazy_loading_dir)
        pred = model.predict(target_X_test_path)
        the_lazy_loading_dir = model.lazy_loading_dir
        del model
        assert not os.path.exists(the_lazy_loading_dir)


def test_AdaSTEMClassifier_duckdb_input_with_lazy():
    with make_tmp_duckdb_files() as (target_X_train_path, target_y_train_regressor_path, target_y_train_classifier_path, target_X_test_path, target_y_test_regressor_path, target_y_test_classifier_path):
        model = make_AdaSTEMClassifier(lazy_loading=True)
        model = model.fit(target_X_train_path, target_y_train_classifier_path)

        model.save(tar_gz_file='./my_model2.tar.gz', remove_temporary_file=True)
        assert not os.path.exists(model.lazy_loading_dir)
        
        model = AdaSTEM.load(tar_gz_file='./my_model2.tar.gz', new_lazy_loading_path='./new_lazyloading_ensemble_folder2', remove_original_file=False)
        assert os.path.exists(model.lazy_loading_dir)
        pred = model.predict(target_X_test_path)
        the_lazy_loading_dir = model.lazy_loading_dir
        del model
        assert not os.path.exists(the_lazy_loading_dir)


def test_AdaSTEMClassifier_no_lazy():
    model = make_AdaSTEMClassifier()
    y_train.loc[:] = np.where(y_train>0, 1, 0)
    model = model.fit(X_train, y_train)

    model.save(tar_gz_file='./my_model3.tar.gz', remove_temporary_file=True)
    assert not os.path.exists(model.lazy_loading_dir)
    
    model = AdaSTEM.load(tar_gz_file='./my_model3.tar.gz', new_lazy_loading_path='./new_lazyloading_ensemble_folder3', remove_original_file=False)
    assert os.path.exists(model.lazy_loading_dir)
    pred = model.predict(X_test)
    the_lazy_loading_dir = model.lazy_loading_dir
    del model
    assert not os.path.exists(the_lazy_loading_dir)


def test_AdaSTEMClassifier_with_lazy():
    model = make_AdaSTEMClassifier(lazy_loading=True)
    y_train.loc[:] = np.where(y_train>0, 1, 0)
    model = model.fit(X_train, y_train)

    model.save(tar_gz_file='./my_model4.tar.gz', remove_temporary_file=True)
    assert not os.path.exists(model.lazy_loading_dir)
    
    model = AdaSTEM.load(tar_gz_file='./my_model4.tar.gz', new_lazy_loading_path='./new_lazyloading_ensemble_folder4', remove_original_file=False)
    assert os.path.exists(model.lazy_loading_dir)
    pred = model.predict(X_test)
    the_lazy_loading_dir = model.lazy_loading_dir
    del model
    assert not os.path.exists(the_lazy_loading_dir)