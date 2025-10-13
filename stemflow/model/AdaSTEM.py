import multiprocessing as mp
import os
import pickle
import time
import warnings
from functools import partial
from itertools import repeat
import tarfile 
from pathlib import Path
import shutil
import weakref
#
from multiprocessing import Lock, Pool, Process, cpu_count, shared_memory
from typing import Callable, Tuple, Union, Optional, List
from abc import ABC, abstractmethod
import string
from joblib.externals.loky import get_reusable_executor

import joblib
import matplotlib.pyplot as plt

#
import numpy as np
import pandas as pd
from numpy import ndarray
import duckdb

# validation check
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    d2_tweedie_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

#
from ..utils.quadtree import get_one_ensemble_quadtree
from ..utils.validation import (
    check_base_model,
    check_prediciton_aggregation,
    check_prediction_return,
    check_random_state,
    check_spatial_scale,
    check_spatio_bin_jitter_magnitude,
    check_task,
    check_temporal_bin_start_jitter,
    check_temporal_scale,
    check_transform_n_jobs,
    check_transform_spatio_bin_jitter_magnitude,
    check_verbosity,
    check_X_test,
    check_X_train,
    check_y_train,
    check_mem_string,
    check_X_y_format_match,
    check_X_y_indexes_match,
    initiate_lazy_loading_dir,
    initiate_joblib_tmp_dir
)
from ..utils.wrapper import model_wrapper
from ..lazyloading.open_db_connection import open_db_connection, duckdb_config, open_both_Xy_db_connection
from ..lazyloading.lazyloading import LazyLoadingEstimator
    
from .dummy_model import dummy_model1
from .Hurdle import Hurdle
from .static_func_AdaSTEM import (  # predict_one_ensemble
    assign_points_to_one_ensemble,
    get_model_and_stixel_specific_x_names,
    predict_one_stixel,
    train_one_stixel,
    transform_pred_set_to_STEM_quad,
)

from ..utils.generate_random import generate_random_saving_code

class AdaSTEM(BaseEstimator):
    """An AdaSTEM model class inherited by AdaSTEMClassifier and AdaSTEMRegressor"""

    def __init__(
        self,
        base_model: BaseEstimator,
        task: str = "hurdle",
        ensemble_fold: int = 10,
        min_ensemble_required: int = 7,
        grid_len_upper_threshold: Union[float, int] = 25,
        grid_len_lower_threshold: Union[float, int] = 5,
        points_lower_threshold: int = 50,
        stixel_training_size_threshold: int = None,
        temporal_start: Union[float, int] = 1,
        temporal_end: Union[float, int] = 366,
        temporal_step: Union[float, int] = 20,
        temporal_bin_interval: Union[float, int] = 50,
        temporal_bin_start_jitter: Union[float, int, str] = "adaptive",
        spatio_bin_jitter_magnitude: Union[float, int] = "adaptive",
        random_state=None,
        save_gridding_plot: bool = True,
        sample_weights_for_classifier: bool = True,
        Spatio1: str = "longitude",
        Spatio2: str = "latitude",
        Temporal1: str = "DOY",
        use_temporal_to_train: bool = True,
        n_jobs: int = 1,
        subset_x_names: bool = False,
        plot_xlims: Tuple[Union[float, int], Union[float, int]] = None,
        plot_ylims: Tuple[Union[float, int], Union[float, int]] = None,
        verbosity: int = 1,
        plot_empty: bool = False,
        completely_random_rotation: bool = False,
        lazy_loading: bool = False,
        lazy_loading_dir: Union[str, None] = None,
        min_class_sample: int = 1,
        ensemble_bootstrap: bool = False,
        joblib_backend: str = 'loky',
        max_mem: str = '2GB'
    ):
        """Make an AdaSTEM object

        Args:
            base_model:
                base model estimator
            task:
                task of the model. One of 'classifier', 'regressor' and 'hurdle'. Defaults to 'hurdle'.
            ensemble_fold:
                Ensembles count. Higher, better for the model performance. Time complexity O(N). Defaults to 10.
            min_ensemble_required:
                Only points with more than this number of model ensembles available are predicted.
                In the training phase, if stixels contain less than `points_lower_threshold` of data records,
                the results are set to np.nan, making them `unpredictable`. Defaults to 7.
            grid_len_upper_threshold:
                force divide if grid length larger than the threshold. Defaults to 25.
            grid_len_lower_threshold:
                stop divide if grid length **will** be below than the threshold. Defaults to 5.
            points_lower_threshold:
                Do not further split the gird if split results in less samples than this threshold.
                Overriden by grid_len_*_upper_threshold parameters. Defaults to 50.
            stixel_training_size_threshold:
                Do not train the model if the available data records for this stixel is less than this threshold,
                and directly set the value to np.nan. Defaults to 50.
            temporal_start:
                start of the temporal sequence. Defaults to 1.
            temporal_end:
                end of the temporal sequence. Defaults to 366.
            temporal_step:
                step of the sliding window. Defaults to 20.
            temporal_bin_interval:
                size of the sliding window. Defaults to 50.
            temporal_bin_start_jitter:
                jitter of the start of the sliding window.
                If 'adaptive', a random jitter of range (-bin_interval, 0) will be generated
                for the start. Defaults to 'adaptive'.
            spatio_bin_jitter_magnitude:
                jitter of the spatial gridding. Defaults to 'adaptive'.
            random_state:
                None or int. After setting the same seed, the model will generate the same results each time. For reproducibility.
            save_gridding_plot:
                Whether ot save gridding plots. Defaults to True.
            sample_weights_for_classifier:
                Whether to adjust for unbanlanced data for the classifier. Default to True.
            Spatio1:
                Spatial column name 1 in data. Defaults to 'longitude'.
            Spatio2:
                Spatial column name 2 in data. Defaults to 'latitude'.
            Temporal1:
                Temporal column name 1 in data.  Defaults to 'DOY'.
            use_temporal_to_train:
                Whether to use temporal variable to train. For example in modeling the daily abundance of bird population,
                whether use 'day of year (DOY)' as a training variable. Defaults to True.
            n_jobs:
                Number of multiprocessing in fitting the model. Defaults to 1.
            subset_x_names:
                Whether to only store variables with std > 0 for each stixel. Set to False will significantly increase the training speed.
            plot_xlims:
                If save_gridding_plot=true, what is the xlims of the plot. Defaults to the extent of input X varibale.
            plot_ylims:
                If save_gridding_plot=true, what is the ylims of the plot. Defaults to the extent of input Y varibale.
            verbosity:
                Verbosity of the logging information to print. 0 to output nothing and everything otherwise.
            plot_empty:
                Whether to plot the empty grid
            completely_random_rotation:
                If True, the rotation angle will be generated completely randomly, as in paper https://doi.org/10.1002/eap.2056. If False, the ensembles will split the 90 degree with equal angle intervals. e.g., if ensemble_fold=9, then each ensemble will rotate 10 degree futher than the previous ensemble. Defalt to False, because if ensemble fold is small, it will be more robust to equally devide the data; and if ensemble fold is large, they are effectively similar than complete random.
            lazy_loading:
                If True, ensembles of models will be saved in disk, and only loaded when being used (e.g., prediction phase), and the ensembles of models are dump to disk once it is used.
            lazy_loading_dir:
                If lazy_loading, the directory of the model to temporary save to. Default to None, where a folder in /tmp will be created and used. This folder can exist even with lazy_loading==False.
            min_class_sample:
                Minimum umber of samples needed to train the classifier in each stixel. If the sample does not satisfy, fit a dummy one. This parameter does not influence regression tasks.
            ensemble_bootstrap:
                Whether to bootstrap the data at each ensemble level to account for uncertainty. Defaults to False.
            joblib_backend:
                The backend of joblib. Defaults to 'loky'. Other options include 'threading'. ('multiprocessing' not supported because it does not allow generator format).
            max_mem:
                The maximum memory use during the training or prediction process. Should be format like '60GB', '512MB', '1.5GB'.
        Raises:
            AttributeError: Base model do not have method 'fit' or 'predict'
            AttributeError: task not in one of ['regression', 'classification', 'hurdle']
            AttributeError: temporal_bin_start_jitter not in one of [str, float, int]
            AttributeError: temporal_bin_start_jitter is type str, but not 'random'

        Attributes:
            x_names (list):
                All training variables used.
            stixel_specific_x_names (dict):
                stixel specific x_names (predictor variable names) for each stixel.
                We remove the variables that have no variation for each stixel.
                Therefore, the x_names are different for each stixel.
            ensemble_df (pd.DataFrame):
                A dataframe storing the stixel gridding information.
            gridding_plot (matplotlib.figure.Figure):
                Ensemble plot.
            model_dict (dict):
                Dictionary of {stixel_index: trained_model}.
            grid_dict (dict):
                An array of stixels assigned to each ensemble.
            feature_importances_ (pd.DataFrame):
                feature importance dataframe for each stixel.

        """
        # 1. Check random state
        self.random_state = random_state

        # 2. Base model
        check_base_model(base_model)
        base_model = model_wrapper(base_model)
        self.base_model = base_model

        # 3. Model params
        check_task(task)
        self.task = task
        self.Temporal1 = Temporal1
        self.Spatio1 = Spatio1
        self.Spatio2 = Spatio2

        # 4. Gridding params
        if min_ensemble_required > ensemble_fold:
            raise ValueError("Not satisfied: min_ensemble_required <= ensemble_fold")

        self.ensemble_fold = ensemble_fold
        self.min_ensemble_required = min_ensemble_required
        self.grid_len_upper_threshold = grid_len_upper_threshold
        self.grid_len_lower_threshold = grid_len_lower_threshold
        self.grid_len = None # Just a place holder. This will not be used for AdaSTEM and will be override by grid_len in STEM for fixed grid size.
        self.points_lower_threshold = points_lower_threshold
        self.temporal_start = temporal_start
        self.temporal_end = temporal_end
        self.temporal_step = temporal_step
        self.temporal_bin_interval = temporal_bin_interval
        self.completely_random_rotation = completely_random_rotation

        check_spatio_bin_jitter_magnitude(spatio_bin_jitter_magnitude)
        self.spatio_bin_jitter_magnitude = spatio_bin_jitter_magnitude
        check_temporal_bin_start_jitter(temporal_bin_start_jitter)
        self.temporal_bin_start_jitter = temporal_bin_start_jitter

        # 5. Training params
        if stixel_training_size_threshold is None:
            self.stixel_training_size_threshold = points_lower_threshold
        else:
            self.stixel_training_size_threshold = stixel_training_size_threshold
        self.use_temporal_to_train = use_temporal_to_train
        self.subset_x_names = subset_x_names
        self.sample_weights_for_classifier = sample_weights_for_classifier
        self.min_class_sample = min_class_sample
        self.ensemble_bootstrap = ensemble_bootstrap

        # 6. Multi-processing params
        n_jobs = check_transform_n_jobs(self, n_jobs)
        self.n_jobs = n_jobs
        self.joblib_backend = joblib_backend

        # 7. Plotting params
        self.plot_xlims = plot_xlims
        self.plot_ylims = plot_ylims
        self.save_gridding_plot = save_gridding_plot
        self.plot_empty = plot_empty

        # X. miscellaneous
        self.lazy_loading = lazy_loading
        self.lazy_loading_dir = lazy_loading_dir
        self.joblib_tmp_dir = None
        self.duckdb_config = None
        check_mem_string(max_mem)
        self.max_mem = max_mem

        if not verbosity == 0:
            self.verbosity = 1
        else:
            self.verbosity = 0

    def split(self, X_train: Union[pd.DataFrame, str], verbosity: Union[None, int] = None, ax=None, n_jobs: Union[None, int] = None):
        """QuadTree indexing the input data

        Args:
            X_train: Training variables. Can be either a pd.DataFrame object or a string that indicate the path to the database (.duckdb or .parquet).
            verbosity: 0 to output nothing, everything other wise. Default None set it to the verbosity of AdaSTEM model class.
            ax: matplotlit Axes to add to.
            n_jobs: number of processors for parallel computing

        Returns:
            self.grid_dict, a dictionary of one DataFrame for each grid, containing the gridding information
        """
        self.rng = check_random_state(self.random_state)
        n_jobs = check_transform_n_jobs(self, n_jobs)
        
        # If the .split is being called by itself, need to initiate the joblib_tmp_dir and duckdb_config
        remove_joblib_tmp_dir = False
        if self.lazy_loading_dir is None:
            self.lazy_loading_dir = initiate_lazy_loading_dir(self.lazy_loading_dir)
            self._finalizer = weakref.finalize(self, self._cleanup, self.lazy_loading_dir) # run self._cleanup when the object is being garbage collected
        if self.duckdb_config is None:
            self.joblib_tmp_dir = initiate_joblib_tmp_dir(self.lazy_loading_dir)
            self.duckdb_config = duckdb_config(self.max_mem, self.joblib_tmp_dir)
            remove_joblib_tmp_dir = True
            
        if verbosity is None:
            verbosity = self.verbosity
            
        # Determine grid_len based on conditions
        if self.grid_len is None:
            # We are using AdaSTEM
            grid_len_upper = self.grid_len_upper_threshold
            grid_len_lower = self.grid_len_lower_threshold
        else:
            # We are using STEM
            grid_len_upper = self.grid_len
            grid_len_lower = self.grid_len
            
        ## Open connection
        with open_db_connection(X_train, self.duckdb_config) as (X_train_df, con):
            # Here X_train_df can be either pd.DataFrame or duckdb.DuckDBPyRelation
            con.register("X_train_df", X_train_df)
            
            # spatial & temporal min max
            spatial1_min = con.sql(f"select MIN({self.Spatio1}) from X_train_df;").fetchone()[0]
            spatial1_max = con.sql(f"select MAX({self.Spatio1}) from X_train_df;").fetchone()[0]
            spatial2_min = con.sql(f"select MIN({self.Spatio2}) from X_train_df;").fetchone()[0]
            spatial2_max = con.sql(f"select MAX({self.Spatio2}) from X_train_df;").fetchone()[0]
            temporal1_min = con.sql(f"select MIN({self.Temporal1}) from X_train_df;").fetchone()[0]
            temporal1_max = con.sql(f"select MAX({self.Temporal1}) from X_train_df;").fetchone()[0]
        
        # Call spatial and temporal scale checks
        check_spatial_scale(
            spatial1_min,
            spatial1_max,
            spatial2_min,
            spatial2_max,
            grid_len_upper,
            grid_len_lower,
        )
        
        check_temporal_scale(temporal1_min, temporal1_max, self.temporal_bin_interval)

        spatio_bin_jitter_magnitude = check_transform_spatio_bin_jitter_magnitude(
           spatial1_max, spatial1_min, spatial2_max, spatial2_min, self.spatio_bin_jitter_magnitude
        )

        if self.save_gridding_plot:
            if self.plot_xlims is None:
                self.plot_xlims = (spatial1_min, spatial1_max)
            if self.plot_ylims is None:
                self.plot_ylims = (spatial2_min, spatial2_max)

            if ax is None:
                plt.figure(figsize=(20, 20))
                plt.xlim([self.plot_xlims[0], self.plot_xlims[1]])
                plt.ylim([self.plot_ylims[0], self.plot_ylims[1]])
                plt.title("Quadtree", fontsize=20)
            else:
                pass
        
        if isinstance(X_train, pd.DataFrame):
            X_train_indexes = X_train[[self.Temporal1, self.Spatio1, self.Spatio2]]
        else:
            X_train_indexes = X_train
        
        partial_get_one_ensemble_quadtree = partial(
            get_one_ensemble_quadtree,
            size=self.ensemble_fold,
            spatio_bin_jitter_magnitude=spatio_bin_jitter_magnitude,
            temporal_start=self.temporal_start,
            temporal_end=self.temporal_end,
            temporal_step=self.temporal_step,
            temporal_bin_interval=self.temporal_bin_interval,
            temporal_bin_start_jitter=self.temporal_bin_start_jitter,
            data=X_train_indexes,
            duckdb_config=self.duckdb_config,
            Temporal1=self.Temporal1,
            grid_len=self.grid_len,
            grid_len_lon_upper_threshold=self.grid_len_upper_threshold,
            grid_len_lon_lower_threshold=self.grid_len_lower_threshold,
            grid_len_lat_upper_threshold=self.grid_len_upper_threshold,
            grid_len_lat_lower_threshold=self.grid_len_lower_threshold,
            points_lower_threshold=self.points_lower_threshold,
            plot_empty=self.plot_empty,
            Spatio1=self.Spatio1,
            Spatio2=self.Spatio2,
            save_gridding_plot=self.save_gridding_plot,
            ax=ax,
            completely_random_rotation=self.completely_random_rotation,
            ensemble_bootstrap=self.ensemble_bootstrap
        )

        if n_jobs > 1 and isinstance(n_jobs, int):
            parallel = joblib.Parallel(n_jobs=n_jobs, return_as="generator", backend=self.joblib_backend, temp_folder=self.joblib_tmp_dir)
            output_generator = parallel(
                joblib.delayed(partial_get_one_ensemble_quadtree)(
                    ensemble_count=ensemble_count, rng=np.random.default_rng(self.rng.integers(1e9) + ensemble_count)
                )
                for ensemble_count in list(range(self.ensemble_fold))
            )
            if verbosity > 0:
                output_generator = tqdm(output_generator, total=self.ensemble_fold, desc="Generating Ensembles: ")

            ensemble_all_df_list = [i for i in output_generator]
            get_reusable_executor().shutdown(wait=True)

        else:
            iter_func_ = (
                tqdm(range(self.ensemble_fold), total=self.ensemble_fold, desc="Generating Ensembles: ")
                if verbosity > 0
                else range(self.ensemble_fold)
            )
            ensemble_all_df_list = [
                partial_get_one_ensemble_quadtree(
                    ensemble_count=ensemble_count, rng=np.random.default_rng(self.rng.integers(1e9) + ensemble_count)
                )
                for ensemble_count in iter_func_
            ]

        # concat
        ensemble_df = pd.concat(ensemble_all_df_list).reset_index(drop=True)
        del ensemble_all_df_list

        # processing
        ensemble_df = ensemble_df.reset_index(drop=True)

        if self.save_gridding_plot:
            if ax is None:
                plt.tight_layout()
                plt.gca().set_aspect("equal")
                ax = plt.gcf()
                plt.close()

            else:
                pass

            self.ensemble_df, self.gridding_plot = ensemble_df, ax

        else:
            self.ensemble_df, self.gridding_plot = ensemble_df, np.nan

        # Finally, if joblib_tmp_dir is created only for this .split, clean it up
        if remove_joblib_tmp_dir:
            if os.path.exists(self.joblib_tmp_dir):
                shutil.rmtree(self.joblib_tmp_dir)
            
            
    def store_x_names(self, X_train: Union[pd.DataFrame, str]):
        """Store the training variables

        Args:
            X_train (pd.DataFrame, str): input training data.
        """
        # store x_names
        if isinstance(X_train, pd.DataFrame):
            self.x_names = list(X_train.columns)
        else:
            with open_db_connection(X_train, self.duckdb_config) as (X_train_df, con):
                con.register("X_train_df", X_train_df)
                self.x_names = [i for i in con.sql("DESCRIBE X_train_df").df()["column_name"].tolist() if not i=='__index_level_0__']
            
        if not self.use_temporal_to_train:
            if self.Temporal1 in list(self.x_names):
                del self.x_names[self.x_names.index(self.Temporal1)]

        for i in [self.Spatio1, self.Spatio2]:
            if i in self.x_names:
                del self.x_names[self.x_names.index(i)]
        

    def stixel_fitting(self, stixel):
        """A sub module of SAC training. Fit one stixel

        Args:
            stixel (pd.DataFrame): data sjoined with ensemble_df.
            For a single stixel.
        """
        
        if '__index_level_0__' in stixel.columns:
            raise AttributeError('__index_level_0__ should not apprear in the final training data!')

        unique_stixel_id = stixel["unique_stixel_id"].iloc[0]
        name = unique_stixel_id

        if self.lazy_loading:
            base_model = LazyLoadingEstimator(estimator=self.base_model, 
                                               dump_dir=os.path.join(self.lazy_loading_dir, 'models', 'ensemble_' + name.split('_')[1]), 
                                               filename=f"model_{name}.pkl", 
                                               auto_dump=True, auto_load=True, keep_loaded=False)
        else:
            base_model = self.base_model
            
        model, stixel_specific_x_names, status = train_one_stixel(
            stixel_training_size_threshold=self.stixel_training_size_threshold,
            x_names=self.x_names,
            task=self.task,
            base_model=base_model,
            sample_weights_for_classifier=self.sample_weights_for_classifier,
            subset_x_names=self.subset_x_names,
            stixel_X_train=stixel,
            min_class_sample=self.min_class_sample
        )

        if not status == "Success":
            # print(f'Fitting: {ensemble_index}. Not pass: {status}')
            pass
        else:
            return (name, model, stixel_specific_x_names)

    def SAC_ensemble_training(self, single_ensemble_df: pd.DataFrame, X_train: Union[pd.DataFrame, str], y_train: Union[pd.DataFrame, str],
                              temporal_window_prequery: bool = False):
        """A sub-module of SAC training function.
        Train only one ensemble.

        Args:
            single_ensemble_df (pd.DataFrame): ensemble data (from model.ensemble_df)
            X_train (pd.DataFrame, str): input covariates to train
            y_train (pd.DataFrame, str): input ground truth labels
            temporal_window_prequery (bool): Whether to prequery the temporal windows as pd.DataFrame object to speed-up the stixel query. If set to True, query speed will be faster but with a moderate memory usage increase.
        """

        # Calculate the start indices for the sliding window
        unique_start_indices = np.sort(single_ensemble_df[f"{self.Temporal1}_start"].unique())
        
        # Initiate duckdb connection
        duckdb_config = self.duckdb_config.copy()
        duckdb_config['temp_directory'] = os.path.join(duckdb_config['temp_directory'], generate_random_saving_code())
        with open_both_Xy_db_connection(X_train, y_train, duckdb_config) as (X_train_df, y_train_df, con):
            con.register("X_train_df", X_train_df)
            con.register("y_train_df", y_train_df)

            # Get indexes and total_length
            if isinstance(X_train_df, pd.DataFrame):
                indexes = np.array(X_train_df.index)
            else:
                indexes = con.sql(f"SELECT __index_level_0__ FROM X_train_df;").df().values.flatten()
            total_length = con.sql("SELECT COUNT(*) FROM X_train_df;").fetchone()[0]
        
            # training, window by window
            if self.ensemble_bootstrap:
                bootstrap_random_state = single_ensemble_df['bootstrap_random_state'].iloc[0]
                rng = np.random.default_rng(bootstrap_random_state)  # NumPy's random generator
                bootstrap_indices = rng.choice(indexes, size=total_length, replace=True)  # Full bootstrap sample
            else:
                bootstrap_indices = None # Place holder
                
            res_list = []
            for start in unique_start_indices:
                # Select the temporal window
                if isinstance(X_train_df, pd.DataFrame):
                    temporal_window_indexes = np.array(X_train_df.index[
                        (X_train_df[self.Temporal1] >= start) & 
                        (X_train_df[self.Temporal1] < start + self.temporal_bin_interval)
                        ])
                    # Apply bootstrap
                    temporal_window_indexes = bootstrap_indices[np.isin(bootstrap_indices, temporal_window_indexes)] if self.ensemble_bootstrap else temporal_window_indexes
                    window_X_df = X_train_df.loc[temporal_window_indexes] if temporal_window_prequery else X_train_df
                    window_y_df = y_train_df.loc[temporal_window_indexes] if temporal_window_prequery else y_train_df
                    window_X_df_indexes_only = X_train_df.loc[temporal_window_indexes][[self.Temporal1, self.Spatio1, self.Spatio2]]
                else:
                    temporal_window_indexes = con.sql(f"SELECT __index_level_0__ FROM X_train_df WHERE {self.Temporal1} >= {start} AND {self.Temporal1} < {start + self.temporal_bin_interval};").df().values.flatten()
                    # Apply bootstrap
                    temporal_window_indexes = bootstrap_indices[np.isin(bootstrap_indices, temporal_window_indexes)] if self.ensemble_bootstrap else temporal_window_indexes                
                    temporal_window_indexes_df = pd.DataFrame(temporal_window_indexes, columns=['__index_level_0__'])
                    con.register("temporal_window_indexes_df", temporal_window_indexes_df)
                    
                    window_X_df = con.sql(f"""
                        SELECT temporal_window_indexes_df.__index_level_0__,
                            X_train_df.* EXCLUDE(__index_level_0__)
                        FROM temporal_window_indexes_df
                        LEFT JOIN X_train_df
                        ON X_train_df.__index_level_0__ = temporal_window_indexes_df.__index_level_0__
                    """).df().set_index('__index_level_0__') if temporal_window_prequery else X_train_df
                    window_y_df = con.sql(f"""
                        SELECT temporal_window_indexes_df.__index_level_0__,
                            y_train_df.* EXCLUDE(__index_level_0__)
                        FROM temporal_window_indexes_df
                        LEFT JOIN y_train_df
                        ON y_train_df.__index_level_0__ = temporal_window_indexes_df.__index_level_0__
                    """).df().set_index('__index_level_0__') if temporal_window_prequery else y_train_df
                    window_X_df_indexes_only = con.sql(f"""
                                                        SELECT X_train_df.{self.Temporal1}, X_train_df.{self.Spatio1}, X_train_df.{self.Spatio2}, X_train_df.__index_level_0__ FROM X_train_df 
                                                        JOIN temporal_window_indexes_df
                                                        ON X_train_df.__index_level_0__ = temporal_window_indexes_df.__index_level_0__;
                                                       """).df().set_index('__index_level_0__')

                # Transform to STEM gridding coordinates
                window_X_df_indexes_only = transform_pred_set_to_STEM_quad(self.Spatio1, self.Spatio2, window_X_df_indexes_only, single_ensemble_df)
                window_single_ensemble_df = single_ensemble_df[single_ensemble_df[f"{self.Temporal1}_start"] == start]
                
                # Merge
                def find_belonged_points(df, st_indexes_df, X_df, y_df):
                    this_stixel_point_indexes = st_indexes_df.index[
                        (st_indexes_df[f"{self.Spatio1}_new"] >= df["stixel_calibration_point_transformed_left_bound"].iloc[0])
                        & (st_indexes_df[f"{self.Spatio1}_new"] < df["stixel_calibration_point_transformed_right_bound"].iloc[0])
                        & (st_indexes_df[f"{self.Spatio2}_new"] >= df["stixel_calibration_point_transformed_lower_bound"].iloc[0])
                        & (st_indexes_df[f"{self.Spatio2}_new"] < df["stixel_calibration_point_transformed_upper_bound"].iloc[0])
                    ]
                    
                    if isinstance(X_df, pd.DataFrame):
                        X_y = pd.concat([X_df.loc[this_stixel_point_indexes], y_df.loc[this_stixel_point_indexes].set_axis(['true_y'], axis=1)], axis=1)
                    elif isinstance(X_df, duckdb.DuckDBPyRelation):  
                        this_stixel_point_indexes_df = pd.DataFrame(this_stixel_point_indexes, columns=['__index_level_0__'])
                        con.register("this_stixel_point_indexes_df", this_stixel_point_indexes_df)
                        con.register('X_df', X_df)
                        con.register('y_df', y_df)

                        y_column = [i for i in y_df.columns if not i=='__index_level_0__'][0]
                        
                        X_y = con.sql(f"""
                        SELECT this_stixel_point_indexes_df.__index_level_0__, 
                            X_df.* EXCLUDE(__index_level_0__), 
                            y_df.{y_column} as true_y
                        FROM this_stixel_point_indexes_df
                        LEFT JOIN X_df
                            ON X_df.__index_level_0__ = this_stixel_point_indexes_df.__index_level_0__
                        LEFT JOIN y_df
                            ON y_df.__index_level_0__ = this_stixel_point_indexes_df.__index_level_0__;
                        """).df().set_index('__index_level_0__')

                        con.unregister("this_stixel_point_indexes_df")
                        con.unregister("X_df")
                        con.unregister("y_df")
                        
                    else:
                        raise
                    
                    return X_y
                
                def find_belonged_points_and_fit(df, st_indexes_df, X_df, y_df):
                    X_y = find_belonged_points(df, st_indexes_df, X_df, y_df)
                    X_y['ensemble_index'] = df['ensemble_index'].iloc[0]
                    X_y['unique_stixel_id'] = df['unique_stixel_id'].iloc[0]
                    X_y = X_y.sort_index() # To ensure the input dataframes for the two method (temporal_window_prequery or not) are the same so the tained base models are identical, at least with the same input data
                    return self.stixel_fitting(X_y)

                # train
                res = (
                    window_single_ensemble_df[
                        [
                            "ensemble_index",
                            "unique_stixel_id",
                            "stixel_calibration_point_transformed_left_bound",
                            "stixel_calibration_point_transformed_right_bound",
                            "stixel_calibration_point_transformed_lower_bound",
                            "stixel_calibration_point_transformed_upper_bound",
                        ]
                    ]
                    .groupby(["ensemble_index", "unique_stixel_id"], as_index=True)
                    .pipe(lambda x: x[x.obj.columns]) # Explicitly select all the columns in the original df to include. To overcome the include_groups=True deprecation warning
                    .apply(find_belonged_points_and_fit, st_indexes_df=window_X_df_indexes_only, X_df=window_X_df, y_df=window_y_df, include_groups=False)  # although ["ensemble_index", "unique_stixel_id"] will be passed into `find_belonged_points` due to `.pipe(lambda x: x[x.obj.columns])`, the output will not have them so we still set `as_index=True` in `groupby`
                ).values

                res_list.append(list(res))
            
        return res_list

    def SAC_training(
        self, ensemble_df: pd.DataFrame, X_train: Union[pd.DataFrame, str], y_train: Union[pd.DataFrame, str], verbosity: int = 0, n_jobs: int = 1,
        temporal_window_prequery: bool = False
    ):
        """This function is a training function with SAC strategy:
        Split (S), Apply(A), Combine (C). At ensemble level.
        It is built on pandas `apply` method.

        Args:
            ensemble_df (pd.DataFrame): gridding information for all ensemble
            X_train (pd.DataFrame, str): X_train
            y_train (pd.DataFrame, str): y_train
            verbosity (int, optional): Defaults to 0.
            temporal_window_prequery (bool): Whether to prequery the temporal windows as pd.DataFrame object to speed-up the stixel query. If set to True, query speed will be faster but with a moderate memory usage increase.
        """
        assert isinstance(n_jobs, int)

        groups = ensemble_df.groupby("ensemble_index")

        # Parallel wrapper
        if n_jobs == 1:
            output_generator = (self.SAC_ensemble_training(single_ensemble_df=ensemble[1], X_train=X_train, y_train=y_train, temporal_window_prequery=temporal_window_prequery) for ensemble in groups)
        else:
            def mp_train(ensemble, self=self):
                res = self.SAC_ensemble_training(single_ensemble_df=ensemble[1], X_train=X_train, y_train=y_train, temporal_window_prequery=temporal_window_prequery)
                return res
            
            parallel = joblib.Parallel(n_jobs=n_jobs, return_as="generator", backend=self.joblib_backend, temp_folder=self.joblib_tmp_dir)
            output_generator = parallel(joblib.delayed(mp_train)(i) for i in groups)

        # tqdm wrapper
        if verbosity > 0:
            output_generator = tqdm(
                output_generator, total=len(ensemble_df["ensemble_index"].unique()), desc="Training: "
            )

        # iterate through
        model_dict = {}
        stixel_specific_x_names = {}

        for ensemble_id, ensemble in enumerate(output_generator):
            for time_block in ensemble:
                for feature_tuple in time_block:
                    if feature_tuple is None:
                        continue
                    name = feature_tuple[0]
                    model = feature_tuple[1]
                    x_names = feature_tuple[2]
                    model_dict[f"{name}_model"] = model
                    stixel_specific_x_names[name] = x_names

        get_reusable_executor().shutdown(wait=True)
        self.model_dict = model_dict
        self.stixel_specific_x_names = stixel_specific_x_names # Do it at the end to avoid dictionary changes during the pickling
        return self

    def fit(
        self,
        X_train: Union[pd.DataFrame, str],
        y_train: Union[pd.DataFrame, pd.Series, np.ndarray, str],
        verbosity: Union[None, int] = None,
        ax=None,
        n_jobs: Union[None, int] = None,
        overwrite = False,
        temporal_window_prequery: bool = False
    ):
        """Fitting method

        Args:
            X_train: Training variables. Can be either a pd.DataFrame object or a string that indicate the path to the database (.duckdb or .parquet).
            y_train: Training target. Can be either a pd.DataFrame object, a pd.Series object, a np.ndarray, or a string that indicate the path to the database (.duckdb or .parquet). It has to have indexes that match with the X_train.
            ax: matplotlib Axes to add to
            verbosty: whether to show progress bar. 0 for no and 1 for yes.
            ax: matplotlib ax for adding grid plot on that.
            n_jobs: multiprocessing thread count. Default the n_jobs of model object.
            overwrite: overwrite files in lazy_loading_dir. If set to False and any file exists in lazy_loading_dir, an error will be raise.
            temporal_window_prequery: Whether to prequery the temporal windows as pd.DataFrame object to speed-up the stixel query. If set to True, query speed will be faster but with a moderate memory usage increase.

        Raises:
            TypeError: X_train is not a type of pd.DataFrame
            TypeError: y_train is not a type of np.ndarray or pd.DataFrame
        """
        # Setup lazy_loading_dir and joblib_tmp_dir
        if overwrite and self.lazy_loading_dir and os.path.isdir(self.lazy_loading_dir):
            for file in os.listdir(self.lazy_loading_dir):
                shutil.rmtree(os.path.join(self.lazy_loading_dir, file))
                    
        self.lazy_loading_dir = initiate_lazy_loading_dir(self.lazy_loading_dir)
        self._finalizer = weakref.finalize(self, self._cleanup, self.lazy_loading_dir) # run self._cleanup when the object is being garbage collected
        self.joblib_tmp_dir = initiate_joblib_tmp_dir(self.lazy_loading_dir)

        try:
            self.duckdb_config = duckdb_config(self.max_mem, self.joblib_tmp_dir)
            
            # Input check
            self.rng = check_random_state(self.random_state)
            verbosity = check_verbosity(self, verbosity)
            self.data_format = check_X_y_format_match(X_train, y_train)
            check_X_train(X_train, self)
            check_y_train(y_train, self)
            X_train, y_train = check_X_y_indexes_match(X_train, y_train, self)
            n_jobs = check_transform_n_jobs(self, n_jobs)
            self.store_x_names(X_train)
            
            # Quadtree            
            self.split(X_train, verbosity=verbosity, ax=ax, n_jobs=n_jobs)
            
            # stixel specific x_names list
            for rm_target in ['model_dict', 'stixel_specific_x_names']:
                if hasattr(self, rm_target):
                    delattr(self, rm_target)

            # Training
            self.SAC_training(self.ensemble_df, X_train, y_train, verbosity, n_jobs, temporal_window_prequery)
            self.classes_ = np.unique(y_train)
        except: # Remove the entire lazy_loading_dir since it includes failed models in this case
            if os.path.exists(self.lazy_loading_dir):
                shutil.rmtree(self.lazy_loading_dir)
            raise
        finally: # Remove the joblib_tmp_dir anyway
            if os.path.exists(self.joblib_tmp_dir):
                shutil.rmtree(self.joblib_tmp_dir)

        return self

    def stixel_predict(self, stixel: pd.DataFrame) -> Union[None, pd.DataFrame]:
        """A sub module of SAC prediction. Predict one stixel

        Args:
            stixel (pd.DataFrame): data joined with ensemble_df.
            For a single stixel.

        Returns:
            pd.DataFrame: the prediction result of this stixel
        """
        if '__index_level_0__' in stixel.columns:
            raise AttributeError('__index_level_0__ should not apprear in the final training data!')

        stixel['unique_stixel_id'] = stixel.name
        unique_stixel_id = stixel["unique_stixel_id"].iloc[0]

        model_x_names_tuple = get_model_and_stixel_specific_x_names(
            self.model_dict,
            unique_stixel_id,
            self.stixel_specific_x_names,
            self.x_names,
        )

        if model_x_names_tuple[0] is None:
            return None

        pred = predict_one_stixel(X_test_stixel=stixel,
                                  task=self.task,
                                  model_x_names_tuple=model_x_names_tuple,
                                  base_model_method=self.base_model_method,
                                  **self.base_model_prediction_param)

        if pred is None:
            return None
        else:
            return pred

    def SAC_ensemble_predict(
        self, single_ensemble_df: pd.DataFrame, data: Union[pd.DataFrame, str]
        ) -> pd.DataFrame:
        """A sub-module of SAC prediction function.
        Predict only one ensemble.

        Args:
            single_ensemble_df (pd.DataFrame): ensemble data (model.ensemble_df)
            data (pd.DataFrame, str): input covariates to predict
        Returns:
            pd.DataFrame: Prediction result of one ensemble.
        """
        # Calculate the start indices for the sliding window
        start_indices = sorted(single_ensemble_df[f"{self.Temporal1}_start"].unique())

        # Initiate duckdb connection
        duckdb_config = self.duckdb_config.copy()
        duckdb_config['temp_directory'] = os.path.join(duckdb_config['temp_directory'], generate_random_saving_code())

        with open_db_connection(data, duckdb_config) as (data_df, con):
            con.register("data_df", data_df)

            # prediction, window by window
            window_prediction_list = []
            for start in start_indices:
                if isinstance(data_df, pd.DataFrame):
                    temporal_window_indexes = np.array(data_df.index[
                        (data_df[self.Temporal1] >= start) & 
                        (data_df[self.Temporal1] < start + self.temporal_bin_interval)
                        ])
                    window_X_df_indexes_only = data_df.loc[temporal_window_indexes][[self.Temporal1, self.Spatio1, self.Spatio2]]
                else:
                    temporal_window_indexes = con.sql(f"SELECT __index_level_0__ FROM data_df WHERE {self.Temporal1} >= {start} AND {self.Temporal1} < {start + self.temporal_bin_interval};").df().values.flatten()
                    temporal_window_indexes_df = pd.DataFrame(temporal_window_indexes, columns=['__index_level_0__'])
                    con.register("temporal_window_indexes_df", temporal_window_indexes_df)
                    window_X_df_indexes_only = con.sql(f"""
                                                       SELECT data_df.{self.Temporal1}, data_df.{self.Spatio1}, data_df.{self.Spatio2}, data_df.__index_level_0__ 
                                                       FROM data_df
                                                       JOIN temporal_window_indexes_df
                                                       ON data_df.__index_level_0__ = temporal_window_indexes_df.__index_level_0__;
                                                       """).df().set_index('__index_level_0__')

                    
                window_X_df_indexes_only = transform_pred_set_to_STEM_quad(self.Spatio1, self.Spatio2, window_X_df_indexes_only, single_ensemble_df)
                window_single_ensemble_df = single_ensemble_df[single_ensemble_df[f"{self.Temporal1}_start"] == start]

                def find_belonged_points(df, st_indexes_df, X_df):
                    this_stixel_point_indexes = st_indexes_df.index[
                        (st_indexes_df[f"{self.Spatio1}_new"] >= df["stixel_calibration_point_transformed_left_bound"].iloc[0])
                        & (st_indexes_df[f"{self.Spatio1}_new"] < df["stixel_calibration_point_transformed_right_bound"].iloc[0])
                        & (st_indexes_df[f"{self.Spatio2}_new"] >= df["stixel_calibration_point_transformed_lower_bound"].iloc[0])
                        & (st_indexes_df[f"{self.Spatio2}_new"] < df["stixel_calibration_point_transformed_upper_bound"].iloc[0])
                    ]
                    
                    if isinstance(X_df, pd.DataFrame):
                        X = X_df.loc[this_stixel_point_indexes]
                    elif isinstance(X_df, duckdb.DuckDBPyRelation):      
                        this_stixel_point_indexes_df = pd.DataFrame(this_stixel_point_indexes, columns=['__index_level_0__'])
                        con.register("this_stixel_point_indexes_df", this_stixel_point_indexes_df)
                        con.register('X_df', X_df)
                        X = con.sql(f"""
                                    SELECT X_df.* FROM X_df 
                                    JOIN this_stixel_point_indexes_df
                                    ON X_df.__index_level_0__ = this_stixel_point_indexes_df.__index_level_0__
                                    """).df().set_index('__index_level_0__')
                        con.unregister("this_stixel_point_indexes_df")
                        con.unregister("X_df")
                    else:
                        raise
                    
                    return X

                query_results = (
                    window_single_ensemble_df[
                        [
                            "ensemble_index",
                            "unique_stixel_id",
                            "stixel_calibration_point_transformed_left_bound",
                            "stixel_calibration_point_transformed_right_bound",
                            "stixel_calibration_point_transformed_lower_bound",
                            "stixel_calibration_point_transformed_upper_bound",
                        ]
                    ]
                    .groupby(["ensemble_index", "unique_stixel_id"], as_index=True)
                    .pipe(lambda x: x[x.obj.columns]) # Explicitly select all the columns in the original df to include. To overcome the include_groups=True deprecation warning
                    .apply(find_belonged_points, st_indexes_df=window_X_df_indexes_only, X_df=data_df, include_groups=False) # although ["ensemble_index", "unique_stixel_id"] will be passed into `find_belonged_points` due to `.pipe(lambda x: x[x.obj.columns])`, the output will not have them so we still set `as_index=True` in `groupby`
                    .reset_index(level=["ensemble_index", "unique_stixel_id"]) # Turn these indexes into columns and keep the original df indexing
                )
            
                if len(query_results) == 0:
                    """All points fall out of the grids"""
                    continue
                
                # predict            
                window_prediction = (
                    query_results
                    .dropna(subset="unique_stixel_id")
                    .groupby("unique_stixel_id", as_index=False)
                    .pipe(lambda x: x[x.obj.columns]) # Explicitly select all the columns in the original df to include. To overcome the include_groups=True deprecation warning
                    .apply(lambda stixel: self.stixel_predict(stixel), include_groups=False) #
                    .droplevel(0) # If using as_index=False duing groupby, pandas will automatically generate a group indexing column, so drop the indexing of the new groups
                )

                window_prediction_list.append(window_prediction)

            if any([i is not None for i in window_prediction_list]):
                ensemble_prediction = pd.concat(window_prediction_list, axis=0)
                ensemble_prediction = ensemble_prediction.groupby("index").mean().reset_index(drop=False)
            else:
                ensmeble_index = list(window_single_ensemble_df["ensemble_index"])[0]
                warnings.warn(f"No prediction for this ensemble: {ensmeble_index}")
                ensemble_prediction = None

        return ensemble_prediction

    def SAC_predict(
        self, ensemble_df: pd.DataFrame, data: Union[pd.DataFrame, str], verbosity: int = 0, n_jobs: int = 1
    ) -> pd.DataFrame:
        """This function is a prediction function with SAC strategy:
        Split (S), Apply(A), Combine (C). At ensemble level.
        It is built on pandas `apply` method.

        Args:
            ensemble_df (pd.DataFrame): gridding information for all ensemble
            data (pd.DataFrame, str): data
            verbosity (int, optional): Defaults to 0.
            n_jobs (int): number of processors for parallel computing

        Returns:
            pd.DataFrame: prediction results.
        """
        assert isinstance(n_jobs, int)

        groups = ensemble_df.groupby("ensemble_index")

        # Parallel maker
        if n_jobs == 1:
            output_generator = (self.SAC_ensemble_predict(single_ensemble_df=ensemble[1], data=data) for ensemble in groups)
        else:
            def mp_predict(ensemble, self=self):
                res = self.SAC_ensemble_predict(single_ensemble_df=ensemble[1], data=data)
                return res

            parallel = joblib.Parallel(n_jobs=n_jobs, return_as="generator", backend=self.joblib_backend, temp_folder=self.joblib_tmp_dir)
            output_generator = parallel(joblib.delayed(mp_predict)(i) for i in groups)

        # tqdm wrapper
        if verbosity > 0:
            output_generator = tqdm(
                output_generator, total=len(ensemble_df["ensemble_index"].unique()), desc="Predicting: "
            )

        # Prediction
        pred = [i.set_index("index") for i in output_generator]
        get_reusable_executor().shutdown(wait=True)
        
        pred = pd.concat(pred, axis=1)
        
        if len(pred) == 0:
            raise ValueError(
                "All samples are not predictable based on current settings!\nTry adjusting the 'points_lower_threshold', increase the grid size, or increase sample size!"
            )

        pred.columns = list(range(self.ensemble_fold))
        return pred

    def predict_proba(
        self,
        X_test: Union[pd.DataFrame, str],
        verbosity: Union[int, None] = None,
        return_std: bool = False,
        n_jobs: Union[None, int] = None,
        aggregation: str = "mean",
        return_by_separate_ensembles: bool = False,
        logit_agg: bool = False,
        base_model_method: Union[None, str] = None,
        **base_model_prediction_param
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Predict probability

        Args:
            X_test (pd.DataFrame):
                Testing variables. Can be either a pd.DataFrame object or a string that indicate the path to the database (.duckdb or .parquet).
            verbosity (int, optional):
                show progress bar or not. Yes for 0, and No for other. Defaults to None, which set it as the verbosity of the main model class.
            return_std (bool, optional):
                Whether return the standard deviation among ensembles. Defaults to False.
            n_jobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.n_jobs. Default to 1.
                I do not recommend setting value larger than 1.
                In practice, multi-processing seems to slow down the process instead of speeding up.
                Could be more practical with large amount of data.
                Still in experiment.
            aggregation (str, optional):
                'mean' or 'median' for aggregation method across ensembles.
            return_by_separate_ensembles (bool, optional):
                Experimental function. return not by aggregation, but by separate ensembles.
            logit_agg:
                Whether to use logit aggregation for the classification task. Most likely only used when you are predicting "real" calibrated probability. If True, the model is averaging the probability prediction estimated by all ensembles in logit scale, and then back-tranforms it to probability scale. It's recommended to be jointly used with the CalibratedClassifierCV class in sklearn as a wrapper of the classifier to estimate the calibrated probability. Default is False, but can be set to true for "real" probability averaging.
            base_model_method:
                The name of the prediction method for base models. If None, `predict` or `predict_proba` will be used depending on the tasks. This argument is handy if you have a custom base model class that has a special prediction function. Notice that dummy model will still predict 0, so the ensemble-aggregated result is still an average of zeros and your special prediction function output. Therefore, it may only make sense if your special prediction function predicts 0 as the absense/control value. Defaults to None.
            base_model_prediction_param:
                Any other paramters to pass into the prediction method of the base models. e.g., base_model_prediction_param={'n_jobs':1} (set n_jobs=1 for the *base model*). 
        Raises:
            TypeError:
                X_test is not of type pd.DataFrame or str.
            ValueError:
                aggregation is not in ['mean','median'].

        Returns:
            predicted results. (pred_mean, pred_std) if return_std==true, and pred_mean if return_std==False.

            If return_by_separate_ensembles == True:
                Return numpy.ndarray of shape (n_samples, n_ensembles)

        """
        check_X_test(X_test, self)
        check_prediciton_aggregation(aggregation)
        return_by_separate_ensembles, return_std = check_prediction_return(return_by_separate_ensembles, return_std)
        verbosity = check_verbosity(self, verbosity)
        n_jobs = check_transform_n_jobs(self, n_jobs)
        self.base_model_method = base_model_method
        self.base_model_prediction_param = base_model_prediction_param

        # Setup joblib_tmp_dir
        self.joblib_tmp_dir = initiate_joblib_tmp_dir(self.lazy_loading_dir)
        self.duckdb_config = duckdb_config(self.max_mem, self.joblib_tmp_dir)
        
        try:
            # predict
            res = self.SAC_predict(self.ensemble_df, X_test, verbosity=verbosity, n_jobs=n_jobs)
        except: # Remove the entire lazy_loading_dir since it includes failed models
            raise
        finally: # Remove the joblib_tmp_dir anyway
            if os.path.exists(self.joblib_tmp_dir):
                shutil.rmtree(self.joblib_tmp_dir)
                
        # Get X_test indexes
        if isinstance(X_test, pd.DataFrame):
            X_test_indexes = np.array(X_test.index)
        elif isinstance(X_test, str):
            with open_db_connection(X_test, self.duckdb_config) as (X_test_df, con):
                con.register("X_test_df", X_test_df)
                X_test_indexes = con.sql("SELECT __index_level_0__ FROM X_test_df;").df().values.flatten()
        else: 
            raise
        
        # Experimental Function
        if return_by_separate_ensembles:
            new_res = pd.DataFrame({"index": list(X_test_indexes)}).set_index("index")
            new_res = new_res.merge(res, left_on="index", right_on="index", how="left")
            return new_res.values

        # Transform to logit space if classification:
        if self.task=='classification' and logit_agg:
            for col_index in range(res.shape[1]):
                prob = np.clip(res.iloc[:,col_index], 1e-8, 1 - 1e-8)
                res.iloc[:,col_index] = np.log(prob / (1-prob)) # logit space
                
            # Aggregate
            if aggregation == "mean":
                res_mean = res.mean(axis=1, skipna=True)  # mean of all grid model that predicts this stixel
            elif aggregation == "median":
                res_mean = res.median(axis=1, skipna=True)
                
            # Transform back to 0-1:
            res_mean = 1/(1+np.exp(-res_mean)) # notice that the res_std is not transformed!
            res_mean = res_mean.where(res_mean<=1e-8, 0)
            
        else:
            # don't need to aggregate at logit scale
            # Aggregate
            if aggregation == "mean":
                res_mean = res.mean(axis=1, skipna=True)  # mean of all grid model that predicts this stixel
            elif aggregation == "median":
                res_mean = res.median(axis=1, skipna=True)
        
        res_std = res.std(axis=1, skipna=True)
        
        # Nan count
        res_nan_count = res.isnull().sum(axis=1)
        pred_mean = np.where(
            self.ensemble_fold - res_nan_count.values >= self.min_ensemble_required, res_mean.values, np.nan
        )
        pred_std = np.where(
            self.ensemble_fold - res_nan_count.values >= self.min_ensemble_required, res_std.values, np.nan
        )
        
        res = pd.DataFrame({"index": list(res_mean.index), "pred_mean": pred_mean, "pred_std": pred_std}).set_index(
            "index"
        )

        # Preparing output (formatting)
        new_res = pd.DataFrame({"index": list(X_test_indexes)}).set_index("index")
        new_res = new_res.merge(res, left_on="index", right_on="index", how="left")
        nan_count = np.sum(np.isnan(new_res["pred_mean"].values))
        nan_frac = nan_count / len(new_res["pred_mean"].values)
        warnings.warn(f"There are {nan_frac}% points ({nan_count} points) falling out of predictable range.")

        if return_std:
            if self.task=='classification':
                return np.array([1-new_res["pred_mean"].values.flatten(), new_res["pred_mean"].values.flatten()]).T, new_res["pred_std"].values
            else:
                return new_res["pred_mean"].values.flatten(), new_res["pred_std"].values.flatten()
        else:
            if self.task=='classification':
                return np.array([1-new_res["pred_mean"].values.flatten(), new_res["pred_mean"].values.flatten()]).T
            else:
                return new_res["pred_mean"].values.flatten()
        

    @abstractmethod
    def predict(
        self,
        X_test: pd.DataFrame,
        verbosity: Union[None, int] = None,
        return_std: bool = False,
        n_jobs: Union[None, int] = None,
        aggregation: str = "mean",
        return_by_separate_ensembles: bool = False,
        logit_agg: bool = False,
        base_model_method: Union[None, str] = None,
        **base_model_prediction_param
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        pass


    @classmethod
    def eval_STEM_res(
        self,
        task: str,
        y_test: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        cls_threshold: Union[float, None] = None,
    ) -> dict:
        """Evaluation using multiple metrics

        Classification metrics used:
        1. AUC
        2. Cohen's Kappa
        3. F1
        4. precision
        5. recall
        6. average precision

        Regression metrics used:
        1. spearman's r
        2. peason's r
        3. R2
        4. mean absolute error (MAE)
        5. mean squared error (MSE)
        6. poisson deviance explained (PDE)

        Args:
            task (str):
                one of 'regression', 'classification' or 'hurdle'.
            y_test (Union[pd.Series, np.ndarray]):
                y true
            y_pred (Union[pd.Series, np.ndarray]):
                y predicted
            cls_threshold (Union[float, None], optional):
                Cutting threshold for the classification.
                Values above cls_threshold will be labeled as 1 and 0 otherwise.
                Defaults to None (0.5 for classification and 0 for hurdle).

        Raises:
            AttributeError: task not one of 'regression', 'classification' or 'hurdle'.

        Returns:
            dict: dictionary containing the metric names and their values.
        """

        if task not in ["regression", "classification", "hurdle"]:
            raise AttributeError(
                f"task type must be one of 'regression', 'classification', or 'hurdle'! Now it is {task}"
            )

        if cls_threshold is None:
            if task == "classification":
                cls_threshold = 0.5
            elif task == "hurdle":
                cls_threshold = 0

        if task == "regression":
            auc, kappa, f1, precision, recall, average_precision = [np.nan] * 6
        else:
            a = pd.DataFrame({"y_true": np.array(y_test).flatten(), "pred": np.array(y_pred).flatten()}).dropna()

            y_test_b = np.where(a.y_true > cls_threshold, 1, 0)
            y_pred_b = np.where(a.pred > cls_threshold, 1, 0)

            if len(np.unique(y_test_b)) == 1 and len(np.unique(y_pred_b)) == 1:
                auc, kappa, f1, precision, recall, average_precision = [np.nan] * 6

            else:
                auc = roc_auc_score(y_test_b, np.array(a.pred)) # AUC can be calculated with probability
                kappa = cohen_kappa_score(y_test_b, y_pred_b, weights='linear')
                f1 = f1_score(y_test_b, y_pred_b)
                precision = precision_score(y_test_b, y_pred_b)
                recall = recall_score(y_test_b, y_pred_b)
                average_precision = average_precision_score(y_test_b, y_pred_b)

        if not task == "classification":
            a = pd.DataFrame({"y_true": y_test, "pred": y_pred}).dropna()
            s_r, _ = spearmanr(np.array(a.y_true), np.array(a.pred))
            p_r, _ = pearsonr(np.array(a.y_true), np.array(a.pred))
            r2 = r2_score(a.y_true, a.pred)
            MAE = mean_absolute_error(a.y_true, a.pred)
            MSE = mean_squared_error(a.y_true, a.pred)
            try:
                poisson_deviance_explained = d2_tweedie_score(a[a.pred > 0].y_true, a[a.pred > 0].pred, power=1)
            except Exception as e:
                warnings.warn(f"PED estimation fail: {e}")
                poisson_deviance_explained = np.nan
        else:
            s_r, p_r, r2, MAE, MSE, poisson_deviance_explained = [np.nan] * 6

        return {
            "AUC": auc,
            "kappa": kappa,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "average_precision": average_precision,
            "Spearman_r": s_r,
            "Pearson_r": p_r,
            "R2": r2,
            "MAE": MAE,
            "MSE": MSE,
            "poisson_deviance_explained": poisson_deviance_explained,
        }

    def score(self, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray]) -> dict:
        """Combine predicting and evaluating in one method

        Args:
            X_test (pd.DataFrame): Testing variables
            y_test (Union[pd.Series, np.ndarray]): y true

        Returns:
            dict: dictionary containing the metric names and their values.
        """

        y_pred = self.predict(X_test)
        score_dict = AdaSTEM.eval_STEM_res(self.task, np.array(y_test).flatten(), np.array(y_pred).flatten())
        self.score_dict = score_dict
        return self.score_dict

    def calculate_feature_importances(self):
        """A method to generate feature importance values for each stixel.

        feature importances are saved in self.feature_importances_.

        Attribute dependence:
            1. self.ensemble_df
            2. self.model_dict
            3. self.stixel_specific_x_names
            4. The input base model should have attribute `feature_importances_`

        """
        # generate feature importance dict
        feature_importance_list = []
        
        for ensemble_id in self.ensemble_df['ensemble_index'].unique():
            for index, ensemble_row in self.ensemble_df[
                (self.ensemble_df['ensemble_index']==ensemble_id) &
                (self.ensemble_df["stixel_checklist_count"] >= self.stixel_training_size_threshold)
                ].iterrows():
                if ensemble_row["stixel_checklist_count"] < self.stixel_training_size_threshold:
                    continue
                
                try:
                    stixel_index = ensemble_row["unique_stixel_id"]
                    the_model = self.model_dict[f"{stixel_index}_model"]
                    x_names = self.stixel_specific_x_names[stixel_index]
                    
                    if isinstance(the_model, dummy_model1):
                        importance_dict = dict(zip(self.x_names, [1 / len(self.x_names)] * len(self.x_names)))
                    elif isinstance(the_model, Hurdle):
                        if "feature_importances_" in the_model.__dir__():
                            importance_dict = dict(zip(x_names, the_model.feature_importances_))
                        else:
                            if isinstance(the_model.classifier, dummy_model1):
                                importance_dict = dict(zip(self.x_names, [1 / len(self.x_names)] * len(self.x_names)))
                            else:
                                importance_dict = dict(zip(x_names, the_model.classifier.feature_importances_))
                    else:
                        importance_dict = dict(zip(x_names, the_model.feature_importances_))

                    importance_dict["stixel_index"] = stixel_index
                    feature_importance_list.append(importance_dict)

                except Exception as e:
                    warnings.warn(f"{e}")
                    continue
        
        self.feature_importances_ = (
            pd.DataFrame(feature_importance_list).set_index("stixel_index").reset_index(drop=False).fillna(0)
        )

    def assign_feature_importances_by_points(
        self,
        Sample_ST_df: Union[pd.DataFrame, None] = None,
        verbosity: Union[None, int] = None,
        aggregation: str = "mean",
        n_jobs: Union[int, None] = 1,
        assign_function: Callable = assign_points_to_one_ensemble,
    ) -> pd.DataFrame:
        """Assign feature importance to the input spatio-temporal points

        Args:
            Sample_ST_df (Union[pd.DataFrame, None], optional):
                Dataframe that indicate the spatio-temporal points of interest.
                Must contain `self.Spatio1`, `self.Spatio2`, and `self.Temporal1` in columns.
                If None, the resolution will be:

                | variable|values|
                |---------|--------|
                |Spatio_var1|np.arange(-180,180,1)|
                |Spatio_var2|np.arange(-90,90,1)|
                |Temporal_var1|np.arange(1,366,7)|

                Defaults to None.
            verbosity (Union[None, int], optional):
                0 to output nothing, everything other wise. Default None set it to the verbosity of AdaSTEM model class.
            aggregation (str, optional):
                One of 'mean' and 'median' to aggregate feature importance across ensembles.
            n_jobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.n_jobs. Default to 1.

        Raises:
            NameError:
                feature_importances_ attribute is not calculated. Try model.calculate_feature_importances() first.
            ValueError:
                f'aggregation not one of [\'mean\',\'median\'].'
            KeyError:
                One of [`self.Spatio1`, `self.Spatio2`, `self.Temporal1`] not found in `Sample_ST_df.columns`

        Returns:
            DataFrame with feature importance assigned.
        """
        #
        verbosity = check_verbosity(self, verbosity=verbosity)
        n_jobs = check_transform_n_jobs(self, n_jobs)
        check_prediciton_aggregation(aggregation)

        #
        if "feature_importances_" not in dir(self):
            raise NameError(
                "feature_importances_ attribute is not calculated. Try model.calculate_feature_importances() first."
            )

        #
        if Sample_ST_df is None:
            Spatio_var1 = np.arange(-180, 180, 1)
            Spatio_var2 = np.arange(-90, 90, 1)
            Temporal_var1 = np.arange(1, 366, 7)
            new_Spatio_var1, new_Spatio_var2, new_Temporal_var1 = np.meshgrid(Spatio_var1, Spatio_var2, Temporal_var1)

            Sample_ST_df = pd.DataFrame(
                {
                    self.Temporal1: new_Temporal_var1.flatten(),
                    self.Spatio1: new_Spatio_var1.flatten(),
                    self.Spatio2: new_Spatio_var2.flatten(),
                }
            )
        else:
            for var_name in [self.Spatio1, self.Spatio2, self.Temporal1]:
                if var_name not in Sample_ST_df.columns:
                    raise KeyError(f"{var_name} not found in Sample_ST_df.columns")
                
        partial_assign_func = partial(
            assign_function,
            ensemble_df=self.ensemble_df,
            Sample_ST_df=Sample_ST_df,
            Temporal1=self.Temporal1,
            Spatio1=self.Spatio1,
            Spatio2=self.Spatio2,
            feature_importances_=self.feature_importances_,
        )
    
        # assign input spatio-temporal points to stixels
        if n_jobs > 1:
            parallel = joblib.Parallel(n_jobs=n_jobs, return_as="generator", backend=self.joblib_backend, temp_folder=self.joblib_tmp_dir)
            output_generator = parallel(joblib.delayed(partial_assign_func)(i) for i in list(range(self.ensemble_fold)))
            if verbosity > 0:
                output_generator = tqdm(output_generator, total=self.ensemble_fold, desc="Querying ensembles: ")
            round_res_list = [i for i in output_generator]

        else:
            iter_func_ = (
                tqdm(range(self.ensemble_fold), total=self.ensemble_fold, desc="Querying ensembles: ")
                if verbosity > 0
                else range(self.ensemble_fold)
            )
            round_res_list = [partial_assign_func(ensemble_count) for ensemble_count in iter_func_]

        round_res_df = pd.concat(round_res_list, axis=0)
        del round_res_list

        ensemble_available_count = round_res_df.groupby("sample_index").count().iloc[:, 0]

        # Only points with more than self.min_ensemble_required ensembles available are used
        usable_sample = ensemble_available_count[ensemble_available_count >= self.min_ensemble_required]  #
        round_res_df = round_res_df[round_res_df["sample_index"].isin(list(usable_sample.index))]

        # aggregate across ensembles
        if aggregation == "mean":
            mean_feature_importances_across_ensembles = round_res_df.groupby("sample_index").mean()
        elif aggregation == "median":
            mean_feature_importances_across_ensembles = round_res_df.groupby("sample_index").median()

        if self.use_temporal_to_train:
            mean_feature_importances_across_ensembles = mean_feature_importances_across_ensembles.rename(
                columns={self.Temporal1: f"{self.Temporal1}_predictor"}
            )
        out_ = pd.concat([Sample_ST_df, mean_feature_importances_across_ensembles], axis=1).dropna()
        return out_

    @staticmethod
    def load(tar_gz_file, new_lazy_loading_path=None, remove_original_file=False):
        
        if new_lazy_loading_path is None:
            new_lazy_loading_path = initiate_lazy_loading_dir(new_lazy_loading_path)
        new_lazy_loading_path = str(Path(new_lazy_loading_path.rstrip('/\\')))
            
        file = tarfile.open(tar_gz_file) 
        file.extractall(new_lazy_loading_path, filter=tarfile.data_filter) 
        file.close()
        
        with open(os.path.join(new_lazy_loading_path, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
            
        model.set_params(lazy_loading_dir=new_lazy_loading_path)
        model._finalizer = weakref.finalize(model, model._cleanup, model.lazy_loading_dir)
        
        if model.lazy_loading:
            for model_name in model.model_dict:
                if isinstance(model.model_dict[model_name], LazyLoadingEstimator):
                    model.model_dict[model_name].dump_dir = Path(os.path.join(new_lazy_loading_path, 'models', 'ensemble_' + model_name.split('_')[1]))
    
        if remove_original_file:
            os.remove(tar_gz_file)
            
        return model
    
    def save(self, tar_gz_file, remove_temporary_file = True):
        if not os.path.exists(self.lazy_loading_dir):
            os.makedirs(self.lazy_loading_dir, exist_ok=False)

        # temporary save the model using pickle
        model_path = os.path.join(self.lazy_loading_dir, f'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
                
        # save the main model class and potentially lazyloading pieces to the tar.gz file
        with tarfile.open(tar_gz_file, "w:gz") as tar:
            for pieces in os.listdir(self.lazy_loading_dir):
                tar.add(os.path.join(self.lazy_loading_dir, pieces), arcname=pieces)

        if remove_temporary_file:
            if self.lazy_loading_dir is not None:
                if os.path.exists(self.lazy_loading_dir):
                    shutil.rmtree(self.lazy_loading_dir)

    @staticmethod
    def _cleanup(lazy_loading_dir):
        if lazy_loading_dir is not None:
            if os.path.exists(lazy_loading_dir):
                shutil.rmtree(lazy_loading_dir)

class AdaSTEMClassifier(AdaSTEM):
    """AdaSTEM model Classifier interface

    Example:
        ```
        >>> from stemflow.model.AdaSTEM import AdaSTEMClassifier
        >>> from xgboost import XGBClassifier
        >>> model = AdaSTEMClassifier(base_model=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
                                save_gridding_plot = True,
                                ensemble_fold=10,
                                min_ensemble_required=7,
                                grid_len_upper_threshold=25,
                                grid_len_lower_threshold=5,
                                points_lower_threshold=50,
                                Spatio1='longitude',
                                Spatio2 = 'latitude',
                                Temporal1 = 'DOY',
                                use_temporal_to_train=True)
        >>> model.fit(X_train, y_train)
        >>> pred = model.predict(X_test)
        ```

    """

    def __init__(
        self,
        base_model,
        task="classification",
        ensemble_fold=10,
        min_ensemble_required=7,
        grid_len_upper_threshold=25,
        grid_len_lower_threshold=5,
        points_lower_threshold=50,
        stixel_training_size_threshold=None,
        temporal_start=1,
        temporal_end=366,
        temporal_step=20,
        temporal_bin_interval=50,
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        random_state=None,
        save_gridding_plot=False,
        sample_weights_for_classifier=True,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        use_temporal_to_train=True,
        n_jobs=1,
        subset_x_names=False,
        plot_xlims=None,
        plot_ylims=None,
        verbosity=0,
        plot_empty=False,
        completely_random_rotation=False,
        lazy_loading = False,
        lazy_loading_dir = None,
        min_class_sample = 1,
        ensemble_bootstrap = False,
        joblib_backend = 'loky',
        max_mem = '2GB'
    ):
        super().__init__(
            base_model=base_model,
            task=task,
            ensemble_fold=ensemble_fold,
            min_ensemble_required=min_ensemble_required,
            grid_len_upper_threshold=grid_len_upper_threshold,
            grid_len_lower_threshold=grid_len_lower_threshold,
            points_lower_threshold=points_lower_threshold,
            stixel_training_size_threshold=stixel_training_size_threshold,
            temporal_start=temporal_start,
            temporal_end=temporal_end,
            temporal_step=temporal_step,
            temporal_bin_interval=temporal_bin_interval,
            temporal_bin_start_jitter=temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude=spatio_bin_jitter_magnitude,
            random_state=random_state,
            save_gridding_plot=save_gridding_plot,
            sample_weights_for_classifier=sample_weights_for_classifier,
            Spatio1=Spatio1,
            Spatio2=Spatio2,
            Temporal1=Temporal1,
            use_temporal_to_train=use_temporal_to_train,
            n_jobs=n_jobs,
            subset_x_names=subset_x_names,
            plot_xlims=plot_xlims,
            plot_ylims=plot_ylims,
            verbosity=verbosity,
            plot_empty=plot_empty,
            completely_random_rotation=completely_random_rotation,
            lazy_loading=lazy_loading,
            lazy_loading_dir=lazy_loading_dir,
            min_class_sample=min_class_sample,
            ensemble_bootstrap=ensemble_bootstrap,
            joblib_backend=joblib_backend,
            max_mem=max_mem
        )
        
        self._estimator_type = 'classifier'

    def predict(
        self,
        X_test: pd.DataFrame,
        verbosity: Union[None, int] = None,
        return_std: bool = False,
        cls_threshold: float = 0.5,
        n_jobs: Union[int, None] = None,
        aggregation: str = "mean",
        return_by_separate_ensembles: bool = False,
        logit_agg: bool = False,
        base_model_method: Union[None, str] = None,
        **base_model_prediction_param
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """A rewrite of predict_proba adapted for Classifier

        Args:
            X_test (pd.DataFrame):
                Testing variables.
            verbosity (int, optional):
                0 to output nothing, everything other wise. Default None set it to the verbosity of AdaSTEM model class.
            return_std (bool, optional):
                Whether return the standard deviation among ensembles. Defaults to False.
            cls_threshold (float, optional):
                Cutting threshold for the classification.
                Values above cls_threshold will be labeled as 1 and 0 otherwise.
                Defaults to 0.5.
            n_jobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.n_jobs. Default to None.
                In practice, multi-processing might to slow down the process instead of speeding up. Suggest some adapted experiments.
                Could be more practical with large amount of data.
            aggregation (str, optional):
                'mean' or 'median' for aggregation method across ensembles.
            return_by_separate_ensembles (bool, optional):
                Experimental function. return not by aggregation, but by separate ensembles.
            logit_agg:
                Whether to use logit aggregation for the classification task. If True, the model is averaging the probability prediction estimated by all ensembles in logit scale, and then back-tranform it to probability scale. It's recommened to be combinedly used with the CalibratedClassifierCV class in sklearn as a wrapper of the classifier to estimate the calibrated probability.
            base_model_method:
                The name of the prediction method for base models. If None, `predict` or `predict_proba` will be used depending on the tasks. This argument is handy if you have a custom base model class that has a special prediction function. Defaults to None.
            base_model_prediction_param:
                Any other paramters to pass into the prediction method of the base models. e.g., base_model_prediction_param={'n_jobs':1}.
        Raises:
            TypeError:
                X_test is not of type pd.DataFrame.
            ValueError:
                aggregation is not in ['mean','median'].

        Returns:
            predicted results. (pred_mean, pred_std) if return_std==true, and pred_mean if return_std==False.

        """
        if return_by_separate_ensembles!=False:
            raise AttributeError('If you want to return by separate ensembles in this classifier, use it in .predict_proba, instead of .predict.')
        
        if return_std:
            mean, std = self.predict_proba(
                X_test,
                verbosity=verbosity,
                return_std=True,
                n_jobs=n_jobs,
                aggregation=aggregation,
                return_by_separate_ensembles=return_by_separate_ensembles,
                logit_agg=logit_agg,
                base_model_method=base_model_method,
                **base_model_prediction_param
            )
            mean = mean[:,1].flatten()
            mean = np.where(mean < cls_threshold, 0, mean)
            mean = np.where(mean >= cls_threshold, 1, mean)
            warnings.warn('This is a classification task. The standard deviation of the prediction is output at logit scale! The mean prediction is output at probability scale.')
            return mean, std # notice! the std
        else:
            mean = self.predict_proba(
                X_test,
                verbosity=verbosity,
                return_std=False,
                n_jobs=n_jobs,
                aggregation=aggregation,
                return_by_separate_ensembles=return_by_separate_ensembles,
                logit_agg=logit_agg,
                base_model_method=base_model_method,
                **base_model_prediction_param
            )
            mean = mean[:,1].flatten()
            mean = np.where(mean < cls_threshold, 0, mean)
            mean = np.where(mean >= cls_threshold, 1, mean)
            return mean


class AdaSTEMRegressor(AdaSTEM):
    """AdaSTEM model Regressor interface

    Example:
    ```
    >>> from stemflow.model.AdaSTEM import AdaSTEMRegressor
    >>> from xgboost import XGBRegressor
    >>> model = AdaSTEMRegressor(base_model=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
                            save_gridding_plot = True,
                            ensemble_fold=10,
                            min_ensemble_required=7,
                            grid_len_upper_threshold=25,
                            grid_len_lower_threshold=5,
                            points_lower_threshold=50,
                            Spatio1='longitude',
                            Spatio2 = 'latitude',
                            Temporal1 = 'DOY',
                            use_temporal_to_train=True)
    >>> model.fit(X_train, y_train)
    >>> pred = model.predict(X_test)
    ```

    """

    def __init__(
        self,
        base_model,
        task="regression",
        ensemble_fold=10,
        min_ensemble_required=7,
        grid_len_upper_threshold=25,
        grid_len_lower_threshold=5,
        points_lower_threshold=50,
        stixel_training_size_threshold=None,
        temporal_start=1,
        temporal_end=366,
        temporal_step=20,
        temporal_bin_interval=50,
        temporal_bin_start_jitter="adaptive",
        spatio_bin_jitter_magnitude="adaptive",
        random_state=None,
        save_gridding_plot=False,
        sample_weights_for_classifier=True,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        use_temporal_to_train=True,
        n_jobs=1,
        subset_x_names=False,
        plot_xlims=None,
        plot_ylims=None,
        verbosity=0,
        plot_empty=False,
        completely_random_rotation=False,
        lazy_loading=False,
        lazy_loading_dir=None,
        min_class_sample=1,
        ensemble_bootstrap=False,
        joblib_backend='loky',
        max_mem='2GB'
    ):
        super().__init__(
            base_model=base_model,
            task=task,
            ensemble_fold=ensemble_fold,
            min_ensemble_required=min_ensemble_required,
            grid_len_upper_threshold=grid_len_upper_threshold,
            grid_len_lower_threshold=grid_len_lower_threshold,
            points_lower_threshold=points_lower_threshold,
            stixel_training_size_threshold=stixel_training_size_threshold,
            temporal_start=temporal_start,
            temporal_end=temporal_end,
            temporal_step=temporal_step,
            temporal_bin_interval=temporal_bin_interval,
            temporal_bin_start_jitter=temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude=spatio_bin_jitter_magnitude,
            random_state=random_state,
            save_gridding_plot=save_gridding_plot,
            sample_weights_for_classifier=sample_weights_for_classifier,
            Spatio1=Spatio1,
            Spatio2=Spatio2,
            Temporal1=Temporal1,
            use_temporal_to_train=use_temporal_to_train,
            n_jobs=n_jobs,
            subset_x_names=subset_x_names,
            plot_xlims=plot_xlims,
            plot_ylims=plot_ylims,
            verbosity=verbosity,
            plot_empty=plot_empty,
            completely_random_rotation=completely_random_rotation,
            lazy_loading=lazy_loading,
            lazy_loading_dir=lazy_loading_dir,
            min_class_sample=min_class_sample,
            ensemble_bootstrap=ensemble_bootstrap,
            joblib_backend=joblib_backend,
            max_mem=max_mem
        )
        
        self._estimator_type = 'regressor'
            
            
    def predict(
        self,
        X_test: pd.DataFrame,
        verbosity: Union[None, int] = None,
        return_std: bool = False,
        n_jobs: Union[None, int] = None,
        aggregation: str = "mean",
        return_by_separate_ensembles: bool = False,
        base_model_method: Union[None, str] = None,
        **base_model_prediction_param
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """A rewrite of predict_proba

        Args:
            X_test (pd.DataFrame):
                Testing variables.
            verbosity (Union[None, int], optional):
                0 to output nothing, everything other wise. Default None set it to the verbosity of AdaSTEM model class.
            return_std (bool, optional):
                Whether return the standard deviation among ensembles. Defaults to False.
            n_jobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.n_jobs. Default to None.
                In practice, multi-processing might to slow down the process instead of speeding up. Suggest some adapted experiments.
                Could be more practical with large amount of data.
            aggregation (str, optional):
                'mean' or 'median' for aggregation method across ensembles.
            return_by_separate_ensembles (bool, optional):
                Experimental function. return not by aggregation, but by separate ensembles.
            base_model_method:
                The name of the prediction method for base models. If None, `predict` or `predict_proba` will be used depending on the tasks. This argument is handy if you have a custom base model class that has a special prediction function. Defaults to None.
            base_model_prediction_param:
                Any other paramters to pass into the prediction method of the base models. e.g., base_model_prediction_param={'n_jobs':1}.
                
        Raises:
            TypeError:
                X_test is not of type pd.DataFrame.
            ValueError:
                aggregation is not in ['mean','median'].

        Returns:
            predicted results. (pred_mean, pred_std) if return_std==true, and pred_mean if return_std==False.

            If return_by_separate_ensembles == True:
                Return numpy.ndarray of shape (n_samples, n_ensembles)

        """
        
        prediciton = self.predict_proba(
            X_test,
            verbosity=verbosity,
            return_std=return_std,
            n_jobs=n_jobs,
            aggregation=aggregation,
            return_by_separate_ensembles=return_by_separate_ensembles,
            base_model_method = base_model_method,
            **base_model_prediction_param
        )
        
        # if return_by_separate_ensembles, this will be the dataframe for ensemble
        # if return_std, this wil be a tuple of mean and std of prediction
        # if none of these, then it ill output the mean prediction
        return prediciton
