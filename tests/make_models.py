import os
import pickle
import time
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np

# %%
import pandas as pd
import pytest
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor

import stemflow
from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor
from stemflow.model.Hurdle import Hurdle, Hurdle_for_AdaSTEM
from stemflow.model.SphereAdaSTEM import SphereAdaSTEM, SphereAdaSTEMClassifier, SphereAdaSTEMRegressor
from stemflow.model.STEM import STEM, STEMClassifier, STEMRegressor
from stemflow.model_selection import ST_train_test_split

fold_ = 2
min_req = 1


def make_STEMClassifier(fold_=2, min_req=1, **kwargs):
    model = STEMClassifier(
        base_model=XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len=30,
        temporal_start=1,
        temporal_end=366,
        temporal_step=40,
        temporal_bin_interval=80,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        use_temporal_to_train=True,
        n_jobs=1,
        **kwargs
    )

    return model


def make_parallel_STEMClassifier(
    fold_=2, min_req=1, **kwargs
):
    model = STEMClassifier(
        base_model=XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len=30,
        temporal_start=1,
        temporal_end=366,
        temporal_step=40,
        temporal_bin_interval=80,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        use_temporal_to_train=True,
        n_jobs=2,
        **kwargs
    )

    return model


def make_STEMRegressor(fold_=2, min_req=1, **kwargs):
    model = STEMRegressor(
        base_model=Hurdle(
            classifier=XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
            regressor=XGBRegressor(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        ),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len=30,
        temporal_start=1,
        temporal_end=366,
        temporal_step=40,
        temporal_bin_interval=80,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        use_temporal_to_train=True,
        n_jobs=1, 
        **kwargs
    )

    return model


def make_AdaSTEMClassifier(fold_=2, min_req=1, **kwargs):
    model = AdaSTEMClassifier(
        base_model=XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len_upper_threshold=50,
        grid_len_lower_threshold=20,
        temporal_start=1,
        temporal_end=366,
        temporal_step=40,
        temporal_bin_interval=80,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        use_temporal_to_train=True,
        n_jobs=1, 
        **kwargs
    )
    return model


def make_AdaSTEMRegressor(fold_=2, min_req=1, **kwargs):
    model = AdaSTEMRegressor(
        base_model=Hurdle(
            classifier=XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
            regressor=XGBRegressor(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        ),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len_upper_threshold=50,
        grid_len_lower_threshold=20,
        temporal_start=1,
        temporal_end=366,
        temporal_step=40,
        temporal_bin_interval=80,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        use_temporal_to_train=True,
        n_jobs=1,
        **kwargs
    )
    return model


def make_SphereAdaSTEMRegressor(
    fold_=2, min_req=1, **kwargs
):
    model = SphereAdaSTEMRegressor(
        base_model=Hurdle(
            classifier=XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
            regressor=XGBRegressor(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        ),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len_upper_threshold=8000,
        grid_len_lower_threshold=500,
        temporal_start=1,
        temporal_end=366,
        temporal_step=40,
        temporal_bin_interval=80,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        use_temporal_to_train=True,
        n_jobs=1,
        **kwargs
    )
    return model


def make_SphereAdaClassifier(fold_=2, min_req=1, **kwargs):
    model = SphereAdaSTEMClassifier(
        base_model=XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len_upper_threshold=8000,
        grid_len_lower_threshold=500,
        temporal_start=1,
        temporal_end=366,
        temporal_step=40,
        temporal_bin_interval=80,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        use_temporal_to_train=True,
        n_jobs=1,
        **kwargs
    )
    return model


def make_parallel_SphereAdaClassifier(
    fold_=2, min_req=1, **kwargs
):
    model = SphereAdaSTEMClassifier(
        base_model=XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len_upper_threshold=8000,
        grid_len_lower_threshold=500,
        temporal_start=1,
        temporal_end=366,
        temporal_step=40,
        temporal_bin_interval=80,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        use_temporal_to_train=True,
        n_jobs=2,
        **kwargs
    )
    return model

def make_AdaSTEMRegressor_Hurdle_for_AdaSTEM(fold_=2, min_req=1, **kwargs):
    
    classifier = AdaSTEMClassifier(
        base_model=XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len_upper_threshold=50,
        grid_len_lower_threshold=20,
        temporal_start=1,
        temporal_end=366,
        temporal_step=40,
        temporal_bin_interval=80,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        use_temporal_to_train=True,
        n_jobs=1,
        **kwargs
    )
    
    regressor = AdaSTEMRegressor(
        base_model=XGBRegressor(tree_method="hist", random_state=42, verbosity=0, n_jobs=1),
        save_gridding_plot=True,
        ensemble_fold=fold_,
        min_ensemble_required=min_req,
        grid_len_upper_threshold=50,
        grid_len_lower_threshold=20,
        temporal_start=1,
        temporal_end=366,
        temporal_step=40,
        temporal_bin_interval=80,
        points_lower_threshold=30,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        use_temporal_to_train=True,
        n_jobs=1,
        **kwargs
    )
    
    model = Hurdle_for_AdaSTEM(
        classifier=classifier, regressor=regressor
    )
    
    return model
