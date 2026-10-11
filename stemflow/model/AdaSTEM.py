import multiprocessing as mp
import copy
import os
import pickle
import time
import warnings
from contextlib import nullcontext
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
    check_prediction_aggregation,
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
from ..utils.validation import normalize_quadtree_args
from ..utils.wrapper import model_wrapper
from ..lazyloading.open_db_connection import open_db_connection, duckdb_config, open_both_Xy_db_connection
from ..lazyloading.lazyloading import LazyLoadingEstimator
from ..lazyloading.training_data import SharedTrainingData, TrainingWindow, native_window, iter_prediction_windows, quote_identifier
from ..lazyloading.prediction import PredictionMoments, SpatialWindowIndex
    
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
from ..utils.execution_profile import profile_execution
from ..utils.execution_monitor import ExecutionMonitor, execution_step, monitor_step

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
        completely_random_rotation: bool = True,
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
                and directly set the value to np.nan. Defaults to None, which will be set as the same value as `points_lower_threshold`.
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
                If True, the rotation angle will be generated completely randomly, as in paper https://doi.org/10.1002/eap.2056. If False, the ensembles will split the 90 degree with equal angle intervals. e.g., if ensemble_fold=9, then each ensemble will rotate 10 degree futher than the previous ensemble. Defalt to True.
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
                The maximum memory use during the training or prediction process. Should be format like '60GB', '512MB', '1.5GB'. This argument is only valid if your input training/prediction data is the path to duckdb files. But even if your input are duckdb files, this argument does not guarantee that the memory use for each joblib worker will be lower than that due to various intermediate objects. This argument should only be considered a relative constrain.
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

    def monitor_execution(self, sample_interval=1.0, log_interval=30.0,
                          output_path=None, max_samples=0, detailed_memory=False):
        """Return an opt-in context for wall timings and local process-tree memory.

        Wrap fit and prediction in the same context to measure their combined
        peak. Stages are recorded for all AdaSTEM calls in the calling thread;
        use monitor.phase('evaluation') for additional workflow stages.
        Requires the optional monitoring extra: pip install 'stemflow[monitoring]'.
        A plain report is stored as execution_monitor_ after each outer model
        call and finalized on context exit; a later context replaces it. Save
        the model after exiting to include the complete report. Set
        log_interval=None for silent monitoring. The live sampler is never
        attached to or serialized with the model.
        """
        monitor = ExecutionMonitor(sample_interval=sample_interval, log_interval=log_interval,
                                   output_path=output_path, max_samples=max_samples,
                                   detailed_memory=detailed_memory)
        monitor._track_model(self)
        return monitor

    @monitor_step("partitioning")
    def split(self, X_train: Union[pd.DataFrame, str], 
              verbosity: Union[None, int] = None, 
              ax=None, 
              n_jobs: Union[None, int] = None,
              quadtree_arg_dict = None):
        """QuadTree indexing the input data

        Args:
            X_train: Training variables. Can be either a pd.DataFrame object or a string that indicate the path to the database (.duckdb or .parquet).
            verbosity: 0 to output nothing, everything other wise. Default None set it to the verbosity of AdaSTEM model class.
            ax: matplotlit Axes to add to.
            n_jobs: number of processors for parallel computing
            quadtree_arg_dict: a dictionary to pass into quadtree splitting algorithm for additional conditional gridding. Must be None or a dictionary that have keys "additional_features" and "addiitonal_quadtree_criteria". For example, this can be {'additional_features': ['co_occurrence'], 'addiitonal_quadtree_criteria': lamba x:np.sum(x['co_occurrence']==1)>10} to take the 'co_occurrence' column into consideration during quadtree gridding, with the criteria that the number of 'co_occurrence' record must be more than 10, and if further splitting will fail that criterion, the splitting is stopped. The function for 'addiitonal_quadtree_criteria' must return a True or False boolean for that stixel. The additional column must exist in the X_train dataframe or database.

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

        quadtree_arg_dict = normalize_quadtree_args(quadtree_arg_dict)
            
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
            X_train_indexes = X_train[[self.Temporal1, self.Spatio1, self.Spatio2] + quadtree_arg_dict['additional_features']]
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
            ensemble_bootstrap=self.ensemble_bootstrap,
            quadtree_arg_dict=quadtree_arg_dict
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
        # Estimator-declared metadata supports local CV without manual coordinate copies.
        for name in getattr(self.base_model, "required_metadata_columns", ()):
            if name not in self.x_names:
                self.x_names.append(name)
        

    def stixel_fitting(self, stixel, stixel_filter_func = None):
        """A sub module of SAC training. Fit one stixel

        Args:
            stixel (pd.DataFrame): data sjoined with ensemble_df. For a single stixel.
            stixel_filter_func: filtering function applied on data.
        """
        
        if '__index_level_0__' in stixel.columns:
            raise AttributeError('__index_level_0__ should not apprear in the final training data!')

        unique_stixel_id = stixel["unique_stixel_id"].iloc[0]
        ensemble_index = unique_stixel_id.split('_')[0]

        if self.lazy_loading:
            base_model = LazyLoadingEstimator(estimator=self.base_model, 
                                               dump_dir=os.path.join(self.lazy_loading_dir, 'models', 'ensemble_' + ensemble_index), 
                                               filename=f"model_{unique_stixel_id}.pkl", 
                                               auto_dump=True, auto_load=True, keep_loaded=False,
                                               dump_on_read=getattr(self, "persist_prediction_state_", True))
            base_model.required_metadata_columns = getattr(self.base_model, "required_metadata_columns", ())
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
            min_class_sample=self.min_class_sample,
            stixel_filter_func=stixel_filter_func
        )

        if not status == "Success":
            # print(f'Fitting: {ensemble_index}. Not pass: {status}')
            pass
        else:
            return [unique_stixel_id, model, stixel_specific_x_names]


    @profile_execution
    def SAC_ensemble_training(self, single_ensemble_df: pd.DataFrame, X_train, y_train,
                              temporal_window_prequery=False, stixel_filter_func=None):
        """Fit one ensemble, holding one window and one stixel workspace at a time."""
        starts = np.sort(single_ensemble_df[f"{self.Temporal1}_start"].unique())
        shared = getattr(self, "_shared_training_data", None)
        config = self.duckdb_config.copy()
        config['temp_directory'] = os.path.join(config['temp_directory'], generate_random_saving_code())
        context = (nullcontext((None, None, None)) if shared is not None else
                   open_both_Xy_db_connection(X_train, y_train, config))
        columns = [self.Temporal1, self.Spatio1, self.Spatio2]
        ensemble_id = single_ensemble_df['ensemble_index'].iloc[0]
        start, stixel_id = None, None
        try:
            with context as (x_source, y_source, con):
                bootstrap_ids = None
                if self.ensemble_bootstrap:
                    if shared is not None:
                        ids = shared.index
                    elif isinstance(x_source, pd.DataFrame):
                        ids = x_source.index.to_numpy()
                    else:
                        con.register("bootstrap_source", x_source)
                        ids = con.sql("SELECT __index_level_0__ FROM bootstrap_source").fetchnumpy()['__index_level_0__']
                    rng = np.random.default_rng(single_ensemble_df['bootstrap_random_state'].iloc[0])
                    bootstrap_ids = rng.choice(ids, size=len(ids), replace=True)

                results = []
                for start in starts:
                    if shared is not None:
                        times = shared.array(self.Temporal1)
                        rows = np.flatnonzero((times >= start) & (times < start + self.temporal_bin_interval))
                        coordinates = shared.frame(rows, columns)
                        window = TrainingWindow(coordinates,
                            lambda positions: shared.training_rows(rows[positions]), bootstrap_ids)
                    else:
                        window = native_window(x_source, y_source, con, start,
                            start + self.temporal_bin_interval, columns,
                            temporal_window_prequery, bootstrap_ids)
                    coordinates = transform_pred_set_to_STEM_quad(
                        self.Spatio1, self.Spatio2, window.coordinates, single_ensemble_df)
                    x = coordinates[f"{self.Spatio1}_new"].to_numpy()
                    y = coordinates[f"{self.Spatio2}_new"].to_numpy()
                    time_grid = single_ensemble_df[single_ensemble_df[f"{self.Temporal1}_start"] == start]
                    spatial_index = SpatialWindowIndex(x, y, len(time_grid))
                    time_results = []
                    for stixel_id, grid in time_grid.groupby("unique_stixel_id", sort=True):
                        grid = grid.iloc[0]
                        positions = spatial_index.select(grid)
                        stixel = window.select_positions(positions)
                        if len(stixel):
                            stixel['ensemble_index'] = ensemble_id
                            stixel['unique_stixel_id'] = stixel_id
                            fitted = self.stixel_fitting(stixel.sort_index(), stixel_filter_func)
                            time_results.append(fitted)
                        del stixel
                    results.append(time_results)
                    # Drop the old cache before allocating the following window.
                    del window, coordinates, x, y, spatial_index
                return results
        except Exception as error:
            if hasattr(error, "add_note"):
                error.add_note(f"AdaSTEM ensemble={ensemble_id}, window_start={start}, "
                               f"stixel={stixel_id}, backend={'memmap' if shared else 'native'}, "
                               f"temporal_window_prequery={temporal_window_prequery}")
            raise

    def SAC_training(
        self, ensemble_df: pd.DataFrame, X_train: Union[pd.DataFrame, str], y_train: Union[pd.DataFrame, str], verbosity: int = 0, n_jobs: int = 1,
        temporal_window_prequery: bool = False, stixel_filter_func = None
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
            stixel_filter_func: filtering function applied on data.
        """
        assert isinstance(n_jobs, int)

        groups = ensemble_df.groupby("ensemble_index")

        # Parallel wrapper
        if n_jobs == 1:
            output_generator = (self.SAC_ensemble_training(single_ensemble_df=ensemble[1], X_train=X_train, y_train=y_train, temporal_window_prequery=temporal_window_prequery, stixel_filter_func=stixel_filter_func) for ensemble in groups)
        else:
            def mp_train(ensemble, self=self):
                res = self.SAC_ensemble_training(single_ensemble_df=ensemble[1], X_train=X_train, y_train=y_train, temporal_window_prequery=temporal_window_prequery, stixel_filter_func=stixel_filter_func)
                return res
            
            options = dict(n_jobs=n_jobs, return_as="generator_unordered",
                           backend=self.joblib_backend, temp_folder=self.joblib_tmp_dir,
                           pre_dispatch=n_jobs, batch_size=1)
            threads = getattr(self, "_inner_max_num_threads", None)
            context = (nullcontext() if threads is None else
                       joblib.parallel_config(backend=options.pop("backend"),
                                              inner_max_num_threads=threads))
            with context:
                parallel = joblib.Parallel(**options)
                output_generator = parallel(joblib.delayed(mp_train)(i) for i in groups)

        # tqdm wrapper
        if verbosity > 0:
            output_generator = tqdm(
                output_generator, total=len(ensemble_df["ensemble_index"].unique()), desc="Training: "
            )

        # iterate through
        model_dict = {}
        stixel_specific_x_names = {}

        try:
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
        finally:
            get_reusable_executor().shutdown(wait=True)
        names = [name for _, group in ensemble_df.groupby("ensemble_index")
                 for _, window in group.groupby(f"{self.Temporal1}_start")
                 for name in sorted(window.unique_stixel_id.unique())]
        self.model_dict = {f"{name}_model": model_dict[f"{name}_model"]
                           for name in names if f"{name}_model" in model_dict}
        self.stixel_specific_x_names = {name: stixel_specific_x_names[name]
                                       for name in names if name in stixel_specific_x_names}
        return self

    @monitor_step("fit")
    @profile_execution
    def fit(
        self,
        X_train: Union[pd.DataFrame, str],
        y_train: Union[pd.DataFrame, pd.Series, np.ndarray, str],
        verbosity: Union[None, int] = None,
        ax=None,
        n_jobs: Union[None, int] = None,
        overwrite = False,
        temporal_window_prequery: bool = False,
        ensemble_df: Union[None, pd.DataFrame] = None,
        stixel_filter_func = None,
        training_data_backend: str = "native",
        inner_max_num_threads: Optional[int] = None,
        max_concurrent_ensembles: Optional[int] = None,
        persist_prediction_state: bool = True
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
            ensemble_df: Instead of runing model.split within the `fit` function, you can pass in a customized ensemble_df to skip the whole splitting process.
            stixel_filter_func: filtering function applied on input data when fitting stixels. For example, you can pass `stixel_filter_func=lambda x:x[x['temp']*x['prec']]>25` to fit only using data where temperature * precipication is over 25, within each stixel. The function must return a dataframe.
            training_data_backend: 'native' retains Pandas/DuckDB window queries. Opt-in
                'memmap' streams numeric inputs once to temporary read-only shared arrays.
                Estimators still receive ordinary Pandas DataFrames. Currently planar models only.
            inner_max_num_threads: Optional native-library thread limit in loky workers.
                Also set the base estimator's own n_jobs when it exposes that parameter.
            max_concurrent_ensembles: Optional cap on concurrent partitioning/training tasks,
                independently of the total ensemble_fold. This is not a hard RAM limit.
            persist_prediction_state: Preserve legacy lazy-model writes after prediction.
                Set False for read-only prediction estimators to release models without rewriting.
            
        Raises:
            TypeError: X_train is not a type of pd.DataFrame
            TypeError: y_train is not a type of np.ndarray or pd.DataFrame
        """
        if training_data_backend not in {"native", "memmap"}:
            raise ValueError("training_data_backend must be 'native' or 'memmap'.")
        if training_data_backend == "memmap" and type(self).SAC_ensemble_training is not AdaSTEM.SAC_ensemble_training:
            raise NotImplementedError("Shared training is currently supported by planar AdaSTEM and STEM.")
        for name, value in [("inner_max_num_threads", inner_max_num_threads),
                            ("max_concurrent_ensembles", max_concurrent_ensembles)]:
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None.")
        if inner_max_num_threads is not None and self.joblib_backend != "loky":
            raise ValueError("inner_max_num_threads requires the loky backend.")
        if not isinstance(persist_prediction_state, bool):
            raise ValueError("persist_prediction_state must be a bool.")
        self.persist_prediction_state_ = persist_prediction_state
        fit_start = time.perf_counter()
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
            with execution_step("input_validation"):
                self.rng = check_random_state(self.random_state)
                verbosity = check_verbosity(self, verbosity)
                self.data_format = check_X_y_format_match(X_train, y_train)
                check_X_train(X_train, self)
                check_y_train(y_train, self)
                X_train, y_train = check_X_y_indexes_match(X_train, y_train, self)
                n_jobs = check_transform_n_jobs(self, n_jobs)
                if max_concurrent_ensembles is not None:
                    n_jobs = min(n_jobs, max_concurrent_ensembles)
                self.store_x_names(X_train)
            
            if stixel_filter_func is not None:
                assert callable(stixel_filter_func)
            
            partition_start = time.perf_counter()
            if ensemble_df is None:
                # Quadtree            
                self.split(X_train, verbosity=verbosity, ax=ax, n_jobs=n_jobs)
            else:
                assert isinstance(ensemble_df, pd.DataFrame)
                self.ensemble_df = ensemble_df
            partition_seconds = time.perf_counter() - partition_start
            
            # Deside whether to terminate the training process if avaliable ensmebles does not match with the targeted ensemble_fold
            if len(self.ensemble_df["ensemble_index"].unique()) < self.ensemble_fold:
                raise ValueError(f"Not enough ensembles generated. The number of ensembles generated is {len(self.ensemble_df['ensemble_index'].unique())}, which is less than the required minimum of {self.ensemble_fold}. This indicates that the input data might be too sparse or the gridding parameters are too strict. Please consider adjusting the gridding parameters or providing more data, otherwise, AdaSTEM may not be the optimum method for your purpose, since it require a lot of data.")
            
            # stixel specific x_names list
            for rm_target in ['model_dict', 'stixel_specific_x_names']:
                if hasattr(self, rm_target):
                    delattr(self, rm_target)

            workers = min(joblib.effective_n_jobs(n_jobs), self.ensemble_fold)
            if max_concurrent_ensembles is not None:
                workers = min(workers, max_concurrent_ensembles)
            self._inner_max_num_threads = inner_max_num_threads
            self.execution_info_ = dict(training_data_backend=training_data_backend,
                                        workers=workers, shared_input_bytes=0,
                                        persist_prediction_state=persist_prediction_state,
                                        partition_seconds=partition_seconds)
            preparation_start = time.perf_counter()
            with execution_step("training_data_preparation"):
                if training_data_backend == "memmap":
                    self._shared_training_data = SharedTrainingData.build(X_train, y_train,
                        self.duckdb_config, os.path.join(self.joblib_tmp_dir, 'shared_training'),
                        coordinate_columns=(self.Temporal1, self.Spatio1, self.Spatio2))
                    self.execution_info_["shared_input_bytes"] = self._shared_training_data.nbytes
                    # Only metadata and filenames need to be serialized to workers.
                    training_X, training_y = None, None
                else:
                    training_X, training_y = X_train, y_train
            self.execution_info_["shared_preparation_seconds"] = time.perf_counter() - preparation_start
            # Training
            training_start = time.perf_counter()
            with execution_step("training"):
                self.SAC_training(self.ensemble_df, training_X, training_y, verbosity, workers, temporal_window_prequery, stixel_filter_func)
            self.execution_info_["training_seconds"] = time.perf_counter() - training_start
            if isinstance(y_train, str):
                with open_db_connection(y_train, self.duckdb_config) as (source, con):
                    con.register("class_source", source)
                    target = [c for c in source.columns if c != '__index_level_0__'][0]
                    self.classes_ = np.sort(con.sql(f"SELECT DISTINCT {quote_identifier(target)} FROM class_source").df().iloc[:, 0].to_numpy())
            else:
                self.classes_ = np.unique(y_train)
            
        except: # Remove the entire lazy_loading_dir since it includes failed models in this case
            if os.path.exists(self.lazy_loading_dir):
                shutil.rmtree(self.lazy_loading_dir)
            raise
        finally: # Remove temporary shared inputs after all workers finish.
            with execution_step("fit_cleanup"):
                self.__dict__.pop("_shared_training_data", None)
                self.__dict__.pop("_inner_max_num_threads", None)
                if os.path.exists(self.joblib_tmp_dir):
                    shutil.rmtree(self.joblib_tmp_dir)

        self.execution_info_["fit_seconds"] = time.perf_counter() - fit_start
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
        
        if len(stixel) == 0:
            return None # No data to predict

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

    @profile_execution
    def SAC_ensemble_predict(
        self, single_ensemble_df: pd.DataFrame, data: Union[pd.DataFrame, str], temporal_window_prequery: bool = False
    ) -> pd.DataFrame:
        """Predict one ensemble with one coordinate/covariate query per window.

        Temporal-window prequerying is retained. Each base estimator receives
        the same Pandas rows, with the original half-open stixel boundaries.
        """
        config = self.duckdb_config.copy()
        config['temp_directory'] = os.path.join(config['temp_directory'], generate_random_saving_code())
        pieces = []
        columns = [self.Temporal1, self.Spatio1, self.Spatio2]
        ensemble_id = single_ensemble_df['ensemble_index'].iloc[0]
        start, stixel_id = None, None
        try:
            with open_db_connection(data, config) as (source, con):
                for start in sorted(single_ensemble_df[f"{self.Temporal1}_start"].unique()):
                    time_grid = single_ensemble_df[single_ensemble_df[f"{self.Temporal1}_start"] == start]
                    stixels = [(name, rows.iloc[0]) for name, rows in
                               time_grid.groupby("unique_stixel_id", sort=True)]
                    if type(self).stixel_predict is AdaSTEM.stixel_predict:
                        stixels = [(name, grid) for name, grid in stixels
                                   if f"{name}_model" in self.model_dict]
                    if not stixels:
                        continue
                    windows = iter_prediction_windows(source, con, start,
                        start + self.temporal_bin_interval, columns, temporal_window_prequery,
                        getattr(self, "_prediction_batch_size", None))
                    for window in windows:
                        coordinates = transform_pred_set_to_STEM_quad(
                            self.Spatio1, self.Spatio2, window.coordinates, single_ensemble_df)
                        x = coordinates[f"{self.Spatio1}_new"].to_numpy()
                        y = coordinates[f"{self.Spatio2}_new"].to_numpy()
                        spatial_index = SpatialWindowIndex(x, y, len(time_grid))
                        for stixel_id, grid in stixels:
                            positions = spatial_index.select(grid)
                            if not len(positions):
                                continue
                            stixel = window.select_positions(positions)
                            stixel['ensemble_index'] = ensemble_id
                            stixel['unique_stixel_id'] = stixel_id
                            prediction = self.stixel_predict(stixel)
                            if prediction is not None:
                                pieces.append(prediction)
                            del stixel
                        del window, coordinates, x, y, spatial_index
        except Exception as error:
            if hasattr(error, "add_note"):
                error.add_note(f"AdaSTEM prediction ensemble={ensemble_id}, window_start={start}, "
                               f"stixel={stixel_id}, temporal_window_prequery={temporal_window_prequery}")
            raise
        if pieces:
            return pd.concat(pieces, axis=0).groupby(level=0, sort=False).mean()
        warnings.warn(f"No prediction for this ensemble: {ensemble_id}")
        return pd.DataFrame({'pred': pd.Series(dtype="float64")})

    def _prediction_worker(self, single_ensemble_df):
        """Serialize only this ensemble's fitted models for built-in planar prediction."""
        if (type(self).SAC_ensemble_predict is not AdaSTEM.SAC_ensemble_predict or
                type(self).stixel_predict is not AdaSTEM.stixel_predict):
            return self
        worker = copy.copy(self)
        names = set(single_ensemble_df.unique_stixel_id)
        worker.model_dict = {f"{name}_model": self.model_dict[f"{name}_model"]
                             for name in names if f"{name}_model" in self.model_dict}
        worker.stixel_specific_x_names = {name: self.stixel_specific_x_names[name]
                                         for name in names if name in self.stixel_specific_x_names}
        worker.ensemble_df = single_ensemble_df
        worker.__dict__.pop("_finalizer", None)
        return worker

    def _iter_ensemble_predictions(
        self, ensemble_df: pd.DataFrame, data: Union[pd.DataFrame, str], verbosity: int = 0, n_jobs: int = 1, temporal_window_prequery: bool = False
        
    ) -> pd.DataFrame:
        """This function is a prediction function with SAC strategy:
        Split (S), Apply(A), Combine (C). At ensemble level.
        It is built on pandas `apply` method.

        Args:
            ensemble_df (pd.DataFrame): gridding information for all ensemble
            data (pd.DataFrame, str): data
            verbosity (int, optional): Defaults to 0.
            n_jobs (int): number of processors for parallel computing
            temporal_window_prequery (bool): Whether to prequery the temporal windows as pd.DataFrame object to speed-up the stixel query. If set to True, query speed will be faster but with a moderate memory usage increase.

        Returns:
            pd.DataFrame: prediction results.
        """
        assert isinstance(n_jobs, int)

        groups = ensemble_df.groupby("ensemble_index")

        # Parallel maker
        if n_jobs == 1:
            output_generator = ((position, self.SAC_ensemble_predict(single_ensemble_df=ensemble[1], data=data, temporal_window_prequery=temporal_window_prequery)) for position, ensemble in enumerate(groups))
        else:
            def mp_predict(position, ensemble, worker):
                res = worker.SAC_ensemble_predict(single_ensemble_df=ensemble, data=data, temporal_window_prequery=temporal_window_prequery)
                return position, res

            parallel = joblib.Parallel(n_jobs=n_jobs, return_as="generator_unordered",
                backend=self.joblib_backend, temp_folder=self.joblib_tmp_dir, pre_dispatch=n_jobs, batch_size=1)
            output_generator = parallel(joblib.delayed(mp_predict)(position, ensemble[1],
                self._prediction_worker(ensemble[1])) for position, ensemble in enumerate(groups))

        # tqdm wrapper
        if verbosity > 0:
            output_generator = tqdm(
                output_generator, total=len(ensemble_df["ensemble_index"].unique()), desc="Predicting: "
            )

        try:
            yield from output_generator
        finally:
            get_reusable_executor().shutdown(wait=True)

    def SAC_predict(self, ensemble_df, data, verbosity=0, n_jobs=1,
                    temporal_window_prequery=False):
        """Return all ensemble columns for callers that require the full matrix."""
        pred = dict(self._iter_ensemble_predictions(ensemble_df, data, verbosity, n_jobs,
                                                    temporal_window_prequery))
        pred = pd.concat([pred[position] for position in sorted(pred)], axis=1)
        
        if len(pred) == 0:
            raise ValueError(
                "All samples are not predictable based on current settings!\nTry adjusting the 'points_lower_threshold', increase the grid size, or increase sample size!"
            )

        pred.columns = list(range(self.ensemble_fold))
        return pred

    def _stream_prediction(self, X_test, verbosity, n_jobs, prequery, return_std, logit):
        with execution_step("prediction_input_index"):
            if isinstance(X_test, pd.DataFrame):
                index = X_test.index
            else:
                with open_db_connection(X_test, self.duckdb_config) as (source, con):
                    con.register("prediction_index_source", source)
                    index = con.sql("SELECT __index_level_0__ FROM prediction_index_source").fetchnumpy()["__index_level_0__"]
            reducer = PredictionMoments(index, with_std=return_std, logit=logit)
        try:
            with execution_step("prediction_and_aggregation"):
                for _, prediction in self._iter_ensemble_predictions(
                        self.ensemble_df, X_test, verbosity, n_jobs, prequery):
                    reducer.update(prediction)
                    del prediction
        finally:
            with execution_step("prediction_cleanup"):
                if os.path.exists(self.joblib_tmp_dir):
                    shutil.rmtree(self.joblib_tmp_dir)
        if not reducer.seen:
            raise ValueError("All samples are not predictable based on current settings!")
        with execution_step("prediction_finalize"):
            mean, std = reducer.finish(self.min_ensemble_required)
            self.prediction_execution_info_ = dict(aggregation_backend="streaming",
                accumulator_bytes=reducer.count.nbytes + reducer.mean.nbytes +
                    (reducer.m2.nbytes if reducer.m2 is not None else 0),
                rows=len(index), ensembles=self.ensemble_fold,
                prediction_batch_size=getattr(self, "_prediction_batch_size", None))
            missing = np.count_nonzero(np.isnan(mean))
            warnings.warn(f"There are {100 * missing / len(mean):.3g}% points ({missing} points) falling out of predictable range.")
            result = np.column_stack([1 - mean, mean]) if self.task == "classification" else mean
        return (result, std) if return_std else result

    @monitor_step("prediction")
    @profile_execution
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
        temporal_window_prequery: bool = False,
        streaming: bool = True,
        prediction_batch_size: Optional[int] = None,
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
            temporal_window_prequery:
                Whether to prequery the temporal windows as pd.DataFrame object to speed-up the stixel query. If set to True, query speed will be faster but with a moderate memory usage increase.
            streaming:
                Reduce mean predictions as ensembles finish, without a full sample-by-ensemble
                matrix. Standard deviations use ddof=1 (logit scale when requested).
                Median, separate-ensemble returns and custom SAC_predict overrides retain
                their original matrix route. Set False for comparison with that route.
            prediction_batch_size:
                Optional positive row limit for each temporal-query batch. Requires
                temporal_window_prequery=True and built-in planar ensemble prediction.
                Each window is still queried once; workers keep only one batch of
                covariates. Default None retains whole-window prediction calls.
                Use for estimators whose predictions are independent of other rows
                in the prediction batch, such as XGBoost and Random Forests. Custom
                batch-dependent or prediction-stateful estimators should retain None.
                This bounds input batches, not total worker or job RAM.
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
        with execution_step("prediction_setup"):
            # Setup joblib_tmp_dir
            self.joblib_tmp_dir = initiate_joblib_tmp_dir(self.lazy_loading_dir)
            self.duckdb_config = duckdb_config(self.max_mem, self.joblib_tmp_dir)
            check_X_test(X_test, self)
            check_prediction_aggregation(aggregation)
            return_by_separate_ensembles, return_std = check_prediction_return(return_by_separate_ensembles, return_std)
            verbosity = check_verbosity(self, verbosity)
            n_jobs = check_transform_n_jobs(self, n_jobs)
            if prediction_batch_size is not None:
                if (isinstance(prediction_batch_size, bool) or
                        not isinstance(prediction_batch_size, (int, np.integer)) or
                        prediction_batch_size <= 0):
                    raise ValueError("prediction_batch_size must be a positive integer or None.")
                if not temporal_window_prequery:
                    raise ValueError("prediction_batch_size requires temporal_window_prequery=True.")
                if type(self).SAC_ensemble_predict is not AdaSTEM.SAC_ensemble_predict:
                    raise NotImplementedError("prediction_batch_size requires built-in planar ensemble prediction.")
            self._prediction_batch_size = prediction_batch_size
            self.base_model_method = base_model_method
            self.base_model_prediction_param = base_model_prediction_param

        if (streaming and aggregation == "mean" and not return_by_separate_ensembles
                and type(self).SAC_predict is AdaSTEM.SAC_predict):
            return self._stream_prediction(X_test, verbosity, n_jobs, temporal_window_prequery,
                return_std, self.task == "classification" and logit_agg)
        self.prediction_execution_info_ = dict(aggregation_backend="matrix")
        
        try:
            # predict
            with execution_step("ensemble_prediction"):
                res = self.SAC_predict(self.ensemble_df, X_test, verbosity=verbosity, n_jobs=n_jobs, temporal_window_prequery=temporal_window_prequery)
        except: # Remove the entire lazy_loading_dir since it includes failed models
            raise
        finally: # Remove the joblib_tmp_dir anyway
            with execution_step("prediction_cleanup"):
                if os.path.exists(self.joblib_tmp_dir):
                    shutil.rmtree(self.joblib_tmp_dir)
                
        with execution_step("prediction_aggregation"):
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
                indexer = pd.DataFrame({"index": list(X_test_indexes)}).set_index("index")
                res = pd.concat([indexer, res], axis=1)
                return res.values

            # Transform to logit space if classification:
            if self.task=='classification' and logit_agg:
                for col_index in range(res.shape[1]):
                    prob = np.clip(res.iloc[:,col_index], 1e-16, 1 - 1e-16)
                    res.iloc[:,col_index] = np.log(prob / (1-prob)) # logit space

                # Aggregate
                if aggregation == "mean":
                    res_mean = res.mean(axis=1, skipna=True)  # mean of all grid model that predicts this stixel
                elif aggregation == "median":
                    res_mean = res.median(axis=1, skipna=True)

                # Transform back to 0-1:
                res_mean = 1/(1+np.exp(-res_mean)) # notice that the res_std is not transformed!
                res_mean = res_mean.mask(res_mean <= 1e-16, 0)

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
            indexer = pd.DataFrame({"index": list(X_test_indexes)}).set_index("index")
            res = pd.concat([indexer, res], axis=1)

            nan_count = np.sum(np.isnan(res["pred_mean"].values))
            nan_frac = nan_count / len(res["pred_mean"].values)
            warnings.warn(f"There are {nan_frac}% points ({nan_count} points) falling out of predictable range.")

            if return_std:
                if self.task=='classification':
                    return np.array([1-res["pred_mean"].values.flatten(), res["pred_mean"].values.flatten()]).T, res["pred_std"].values
                else:
                    return res["pred_mean"].values.flatten(), res["pred_std"].values.flatten()
            else:
                if self.task=='classification':
                    return np.array([1-res["pred_mean"].values.flatten(), res["pred_mean"].values.flatten()]).T
                else:
                    return res["pred_mean"].values.flatten()


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
        temporal_window_prequery: bool = False,
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

    @monitor_step("evaluation")
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

    @staticmethod
    def eval_calibration(y_test, y_pred, n_bins=8):
        """Binary probability scores and bin support; evaluate on held-out rows."""
        from ..calibration import calibration_scores
        return calibration_scores(y_test, y_pred, n_bins)

    @monitor_step("calibration_diagnostics")
    def calibration_diagnostics(self, X_test, y_test, pred=None, **kwargs):
        """Held-out overall, spatial, temporal and actual-stixel calibration."""
        from ..calibration import stixel_calibration_diagnostics
        return stixel_calibration_diagnostics(self, X_test, y_test, pred, **kwargs)

    @monitor_step("feature_importances")
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

    @monitor_step("feature_importances_by_points")
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
        check_prediction_aggregation(aggregation)

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
                    model.model_dict[model_name].dump_dir = Path(os.path.join(new_lazy_loading_path, 'models', 'ensemble_' + model_name.split('_')[0]))
    
        if remove_original_file:
            os.remove(tar_gz_file)
            
        return model
    
    @monitor_step("model_save")
    def save(self, tar_gz_file, remove_temporary_file = True, verbosity=1, compresslevel=2):
        os.makedirs(self.lazy_loading_dir, exist_ok=True)

        # dump the main object
        model_path = os.path.join(self.lazy_loading_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

        # collect files recursively (deterministic order)
        root = os.path.abspath(self.lazy_loading_dir)
        files = []
        for dp, dns, fns in os.walk(root):
            dns.sort(); fns.sort()
            for fn in fns:
                files.append(os.path.join(dp, fn))

        it = files
        if verbosity > 0 and tqdm is not None:
            it = tqdm(files, desc="Archiving", unit="file")

        with tarfile.open(tar_gz_file, "w:gz", compresslevel=compresslevel) as tar:
            for fp in it:
                arcname = os.path.relpath(fp, root)
                tar.add(fp, arcname=arcname, recursive=False)

        if remove_temporary_file and os.path.exists(self.lazy_loading_dir):
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
        completely_random_rotation=True,
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
        temporal_window_prequery: bool = False,
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
            temporal_window_prequery: 
                Whether to prequery the temporal windows as pd.DataFrame object to speed-up the stixel query. If set to True, query speed will be faster but with a moderate memory usage increase.
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
                temporal_window_prequery=temporal_window_prequery,
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
                temporal_window_prequery=temporal_window_prequery,
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
        completely_random_rotation=True,
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
        temporal_window_prequery: bool = False,
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
            temporal_window_prequery: 
                Whether to prequery the temporal windows as pd.DataFrame object to speed-up the stixel query. If set to True, query speed will be faster but with a moderate memory usage increase.
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
        
        prediction = self.predict_proba(
            X_test,
            verbosity=verbosity,
            return_std=return_std,
            n_jobs=n_jobs,
            aggregation=aggregation,
            return_by_separate_ensembles=return_by_separate_ensembles,
            base_model_method = base_model_method,
            temporal_window_prequery=temporal_window_prequery,
            **base_model_prediction_param
        )
        
        # if return_by_separate_ensembles, this will be the dataframe for ensemble
        # if return_std, this wil be a tuple of mean and std of prediction
        # if none of these, then it ill output the mean prediction
        return prediction
