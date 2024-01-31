import multiprocessing as mp
import os
import pickle
import time
import warnings
from functools import partial
from itertools import repeat

#
from multiprocessing import Lock, Pool, Process, cpu_count, shared_memory
from typing import Callable, Tuple, Union

import joblib
import matplotlib.pyplot as plt

#
# import dask.dataframe as dd
import numpy as np
import pandas as pd
from numpy import ndarray

# validation check
from pandas.core.frame import DataFrame
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
    check_spatio_bin_jitter_magnitude,
    check_task,
    check_temporal_bin_start_jitter,
    check_transform_njobs,
    check_transform_spatio_bin_jitter_magnitude,
    check_verbosity,
    check_X_test,
    check_X_train,
    check_y_train,
)
from ..utils.wrapper import model_wrapper
from .dummy_model import dummy_model1
from .Hurdle import Hurdle
from .static_func_AdaSTEM import (  # predict_one_ensemble
    assign_points_to_one_ensemble,
    get_model_and_stixel_specific_x_names,
    predict_one_stixel,
    train_one_stixel,
    transform_pred_set_to_STEM_quad,
)


class AdaSTEM(BaseEstimator):
    """A AdaSTEM model class inherited by AdaSTEMClassifier and AdaSTEMRegressor"""

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
        save_gridding_plot: bool = True,
        save_tmp: bool = False,
        save_dir: str = "./",
        sample_weights_for_classifier: bool = True,
        Spatio1: str = "longitude",
        Spatio2: str = "latitude",
        Temporal1: str = "DOY",
        use_temporal_to_train: bool = True,
        njobs: int = 1,
        subset_x_names: bool = False,
        ensemble_models_disk_saver: bool = False,
        ensemble_models_disk_saving_dir: str = "./",
        plot_xlims: Tuple[Union[float, int], Union[float, int]] = (-180, 180),
        plot_ylims: Tuple[Union[float, int], Union[float, int]] = (-90, 90),
        verbosity: int = 0,
        plot_empty: bool = False,
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
            save_gridding_plot:
                Whether ot save gridding plots. Defaults to True.
            save_tmp:
                Whether to save the ensemble dataframe. Defaults to False.
            save_dir:
                If save_tmp==True, save the ensemble dataframe to this path. Defaults to './'.
            ensemble_models_disk_saver:
                Whether to balance the sample weights of classifier for imbalanced datasets. Defaults to True.
            Spatio1:
                Spatial column name 1 in data. Defaults to 'longitude'.
            Spatio2:
                Spatial column name 2 in data. Defaults to 'latitude'.
            Temporal1:
                Temporal column name 1 in data.  Defaults to 'DOY'.
            use_temporal_to_train:
                Whether to use temporal variable to train. For example in modeling the daily abundance of bird population,
                whether use 'day of year (DOY)' as a training variable. Defaults to True.
            njobs:
                Number of multiprocessing in fitting the model. Defaults to 1.
            subset_x_names:
                Whether to only store variables with std > 0 for each stixel. Set to False will significantly increase the training speed.
            ensemble_disk_saver:
                Whether to save each ensemble of models to dicts instead of saving them in memory.
            ensemble_models_disk_saving_dir:
                Where to save the ensemble models. Only valid if ensemble_disk_saver is True.
            plot_xlims:
                If save_gridding_plot=true, what is the xlims of the plot. Defaults to (-180,180).
            plot_ylims:
                If save_gridding_plot=true, what is the ylims of the plot. Defaults to (-90,90).
            verbosity:
                0 to output nothing and everything otherwise.
            plot_empty:
                Whether to plot the empty grid

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
            ensemble_df (pd.core.frame.DataFrame):
                A dataframe storing the stixel gridding information.
            gridding_plot (matplotlib.figure.Figure):
                Ensemble plot.
            model_dict (dict):
                Dictionary of {stixel_index: trained_model}.
            grid_dict (dict):
                An array of stixels assigned to each ensemble.
            feature_importances_ (pd.core.frame.DataFrame):
                feature importance dataframe for each stixel.

        """
        # 1. Base model
        check_base_model(base_model)
        base_model = model_wrapper(base_model)
        self.base_model = base_model

        # 2. Model params
        check_task(task)
        self.task = task
        self.Temporal1 = Temporal1
        self.Spatio1 = Spatio1
        self.Spatio2 = Spatio2

        # 3. Gridding params
        if min_ensemble_required > ensemble_fold:
            raise ValueError("Not satisfied: min_ensemble_required <= ensemble_fold")

        self.ensemble_fold = ensemble_fold
        self.min_ensemble_required = min_ensemble_required
        self.grid_len_upper_threshold = (
            self.grid_len_lon_upper_threshold
        ) = self.grid_len_lat_upper_threshold = grid_len_upper_threshold
        self.grid_len_lower_threshold = (
            self.grid_len_lon_lower_threshold
        ) = self.grid_len_lat_lower_threshold = grid_len_lower_threshold
        self.points_lower_threshold = points_lower_threshold
        self.temporal_start = temporal_start
        self.temporal_end = temporal_end
        self.temporal_step = temporal_step
        self.temporal_bin_interval = temporal_bin_interval

        check_spatio_bin_jitter_magnitude(spatio_bin_jitter_magnitude)
        self.spatio_bin_jitter_magnitude = spatio_bin_jitter_magnitude
        check_temporal_bin_start_jitter(temporal_bin_start_jitter)
        self.temporal_bin_start_jitter = temporal_bin_start_jitter

        # 4. Training params
        if stixel_training_size_threshold is None:
            self.stixel_training_size_threshold = points_lower_threshold
        else:
            self.stixel_training_size_threshold = stixel_training_size_threshold
        self.use_temporal_to_train = use_temporal_to_train
        self.subset_x_names = subset_x_names
        self.sample_weights_for_classifier = sample_weights_for_classifier

        # 5. Multi-threading params (not implemented yet)
        njobs = check_transform_njobs(self, njobs)
        self.njobs = njobs

        # 6. Plotting params
        self.plot_xlims = plot_xlims
        self.plot_ylims = plot_ylims
        self.save_tmp = save_tmp
        self.save_dir = save_dir
        self.save_gridding_plot = save_gridding_plot
        self.plot_empty = plot_empty

        # X. miscellaneous
        self.ensemble_models_disk_saver = ensemble_models_disk_saver
        self.ensemble_models_disk_saving_dir = ensemble_models_disk_saving_dir
        if self.ensemble_models_disk_saver:
            self.saving_code = np.random.randint(1, 1e8, 1)

        if not verbosity == 0:
            self.verbosity = 1
        else:
            self.verbosity = 0

    def split(
        self, X_train: pd.core.frame.DataFrame, verbosity: Union[None, int] = None, ax=None, njobs: int = 1
    ) -> dict:
        """QuadTree indexing the input data

        Args:
            X_train: Input training data
            verbosity: 0 to output nothing, everything other wise. Default None set it to the verbosity of AdaSTEM model class.
            ax: matplotlit Axes to add to.

        Returns:
            self.grid_dict, a dictionary of one DataFrame for each grid, containing the gridding information
        """
        njobs = check_transform_njobs(self, njobs)

        if verbosity is None:
            verbosity = self.verbosity

        # fold = self.ensemble_fold
        save_path = os.path.join(self.save_dir, "ensemble_quadtree_df.csv") if self.save_tmp else ""

        if "grid_len" not in self.__dir__():
            # We har using AdaSTEM
            self.grid_len = None
        else:
            # We har using STEM
            pass

        spatio_bin_jitter_magnitude = check_transform_spatio_bin_jitter_magnitude(
            X_train, self.Spatio1, self.Spatio2, self.spatio_bin_jitter_magnitude
        )

        if self.save_gridding_plot:
            if ax is None:
                plt.figure(figsize=(20, 20))
                plt.xlim([self.plot_xlims[0], self.plot_xlims[1]])
                plt.ylim([self.plot_ylims[0], self.plot_ylims[1]])
                plt.title("Quadtree", fontsize=20)
            else:
                pass

        partial_get_one_ensemble_quadtree = partial(
            get_one_ensemble_quadtree,
            size=self.ensemble_fold,
            spatio_bin_jitter_magnitude=spatio_bin_jitter_magnitude,
            temporal_start=self.temporal_start,
            temporal_end=self.temporal_end,
            temporal_step=self.temporal_step,
            temporal_bin_interval=self.temporal_bin_interval,
            temporal_bin_start_jitter=self.temporal_bin_start_jitter,
            data=X_train,
            Temporal1=self.Temporal1,
            grid_len=self.grid_len,
            grid_len_lon_upper_threshold=self.grid_len_lon_upper_threshold,
            grid_len_lon_lower_threshold=self.grid_len_lon_lower_threshold,
            grid_len_lat_upper_threshold=self.grid_len_lat_upper_threshold,
            grid_len_lat_lower_threshold=self.grid_len_lat_lower_threshold,
            points_lower_threshold=self.points_lower_threshold,
            plot_empty=self.plot_empty,
            Spatio1=self.Spatio1,
            Spatio2=self.Spatio2,
            save_gridding_plot=self.save_gridding_plot,
            ax=ax,
        )

        if njobs > 1 and isinstance(njobs, int):
            parallel = joblib.Parallel(n_jobs=njobs, return_as="generator")
            output_generator = parallel(
                joblib.delayed(partial_get_one_ensemble_quadtree)(i) for i in list(range(self.ensemble_fold))
            )
            if verbosity > 0:
                output_generator = tqdm(output_generator, total=self.ensemble_fold, desc="Generating Ensemble: ")

            ensemble_all_df_list = [i for i in output_generator]

        else:
            iter_func_ = (
                tqdm(range(self.ensemble_fold), total=self.ensemble_fold, desc="Generating Ensemble: ")
                if verbosity > 0
                else range(self.ensemble_fold)
            )
            ensemble_all_df_list = [partial_get_one_ensemble_quadtree(ensemble_count) for ensemble_count in iter_func_]

        # concat
        ensemble_df = pd.concat(ensemble_all_df_list).reset_index(drop=True)
        del ensemble_all_df_list

        # processing
        ensemble_df = ensemble_df.reset_index(drop=True)

        if not save_path == "":
            ensemble_df.to_csv(save_path, index=False)
            print(f"Saved! {save_path}")

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

    def store_x_names(self, X_train: pd.core.frame.DataFrame):
        """Store the training variables

        Args:
            X_train (pd.core.frame.DataFrame): input training data.
        """
        # store x_names
        self.x_names = list(X_train.columns)
        if not self.use_temporal_to_train:
            if self.Temporal1 in list(self.x_names):
                del self.x_names[self.x_names.index(self.Temporal1)]

        for i in [self.Spatio1, self.Spatio2]:
            if i in self.x_names:
                del self.x_names[self.x_names.index(i)]

    def stixel_fitting(self, stixel):
        """A sub module of SAC training. Fit one stixel

        Args:
            stixel (pd.core.frame.DataFrame): data sjoined with ensemble_df.
            For a single stixel.
        """

        ensemble_index = int(stixel["ensemble_index"].iloc[0])
        unique_stixel_id = stixel["unique_stixel_id"].iloc[0]
        name = f"{ensemble_index}_{unique_stixel_id}"

        model, stixel_specific_x_names, status = train_one_stixel(
            stixel_training_size_threshold=self.stixel_training_size_threshold,
            x_names=self.x_names,
            task=self.task,
            base_model=self.base_model,
            sample_weights_for_classifier=self.sample_weights_for_classifier,
            subset_x_names=self.subset_x_names,
            stixel_X_train=stixel,
        )

        if not status == "Success":
            # print(f'Fitting: {ensemble_index}. Not pass: {status}')
            pass

        else:
            # self.model_dict[f"{name}_model"] = model
            # self.stixel_specific_x_names[name] = stixel_specific_x_names
            return (name, model, stixel_specific_x_names)

    def SAC_ensemble_training(self, index_df: pd.core.frame.DataFrame, data: pd.core.frame.DataFrame):
        """A sub-module of SAC training function.
        Train only one ensemble.

        Args:
            index_df (pd.core.frame.DataFrame): ensemble data (model.ensemble_df)
            data (pd.core.frame.DataFrame): input covariates to train
        """

        # Calculate the start indices for the sliding window

        unique_start_indices = np.sort(index_df[f"{self.Temporal1}_start"].unique())
        # training, window by window

        res_list = []
        for start in unique_start_indices:
            window_data_df = data[
                (data[self.Temporal1] >= start) & (data[self.Temporal1] < start + self.temporal_bin_interval)
            ]
            window_data_df = transform_pred_set_to_STEM_quad(self.Spatio1, self.Spatio2, window_data_df, index_df)
            window_index_df = index_df[index_df[f"{self.Temporal1}_start"] == start]

            # Merge
            def find_belonged_points(df, df_a):
                return df_a[
                    (df_a[f"{self.Spatio1}_new"] >= df["stixel_calibration_point_transformed_left_bound"].iloc[0])
                    & (df_a[f"{self.Spatio1}_new"] < df["stixel_calibration_point_transformed_right_bound"].iloc[0])
                    & (df_a[f"{self.Spatio2}_new"] >= df["stixel_calibration_point_transformed_lower_bound"].iloc[0])
                    & (df_a[f"{self.Spatio2}_new"] < df["stixel_calibration_point_transformed_upper_bound"].iloc[0])
                ]

            query_results = (
                window_index_df[
                    [
                        "ensemble_index",
                        "unique_stixel_id",
                        "stixel_calibration_point_transformed_left_bound",
                        "stixel_calibration_point_transformed_right_bound",
                        "stixel_calibration_point_transformed_lower_bound",
                        "stixel_calibration_point_transformed_upper_bound",
                    ]
                ]
                .groupby(["ensemble_index", "unique_stixel_id"])
                .apply(find_belonged_points, df_a=window_data_df)
            )

            if len(query_results) == 0:
                """All points fall out of the grids"""
                continue

            # train
            res = (
                query_results.reset_index(drop=False, level=[0, 1])
                .dropna(subset="unique_stixel_id")
                .groupby("unique_stixel_id")
                .apply(lambda stixel: self.stixel_fitting(stixel))
            ).values

            res_list.append(list(res))

        return res_list

    def SAC_training(
        self, ensemble_df: pd.core.frame.DataFrame, data: pd.core.frame.DataFrame, verbosity: int = 0, njobs: int = 1
    ):
        """This function is a training function with SAC strategy:
        Split (S), Apply(A), Combine (C). At ensemble level.
        It is built on pandas `apply` method.

        Args:
            ensemble_df (pd.core.frame.DataFrame): gridding information for all ensemble
            data (pd.core.frame.DataFrame): data
            verbosity (int, optional): Defaults to 0.

        """
        assert isinstance(njobs, int)

        groups = ensemble_df.groupby("ensemble_index")

        # Parallel wrapper
        if njobs == 1:
            output_generator = (self.SAC_ensemble_training(index_df=ensemble[1], data=data) for ensemble in groups)
        else:

            def mp_train(ensemble, self=self, data=data):
                res = self.SAC_ensemble_training(index_df=ensemble[1], data=data)
                return res

            parallel = joblib.Parallel(n_jobs=njobs, return_as="generator")
            output_generator = parallel(joblib.delayed(mp_train)(i) for i in groups)

        # tqdm wrapper
        if verbosity > 0:
            output_generator = tqdm(
                output_generator, total=len(ensemble_df["ensemble_index"].unique()), desc="Training: "
            )

        # iterate through
        model_dict = {}
        stixel_specific_x_names = {}

        for ensemble in output_generator:
            for time_block in ensemble:
                for feature_tuple in time_block:
                    if feature_tuple is None:
                        continue
                    name = feature_tuple[0]
                    model = feature_tuple[1]
                    x_names = feature_tuple[2]
                    model_dict[f"{name}_model"] = model
                    stixel_specific_x_names[name] = x_names

        self.model_dict = model_dict
        self.stixel_specific_x_names = stixel_specific_x_names
        return self

    def fit(
        self,
        X_train: pd.core.frame.DataFrame,
        y_train: Union[pd.core.frame.DataFrame, np.ndarray],
        verbosity: Union[None, int] = None,
        ax=None,
        njobs: int = 1,
    ):
        """Fitting method

        Args:
            X_train: Training variables
            y_train: Training target
            ax: matplotlib Axes to add to

        Raises:
            TypeError: X_train is not a type of pd.core.frame.DataFrame
            TypeError: y_train is not a type of np.ndarray or pd.core.frame.DataFrame
        """
        #
        verbosity = check_verbosity(self, verbosity)
        check_X_train(X_train)
        check_y_train(y_train)
        njobs = check_transform_njobs(self, njobs)
        self.store_x_names(X_train)

        # quadtree
        X_train = X_train.reset_index(drop=True)  # I reset index here!! caution!
        X_train["true_y"] = np.array(y_train).flatten()
        self.split(X_train, verbosity=verbosity, ax=ax, njobs=njobs)

        # define model dict
        self.model_dict = {}
        # stixel specific x_names list
        self.stixel_specific_x_names = {}

        self.SAC_training(self.ensemble_df, X_train, verbosity, njobs)

        return self

    def stixel_predict(self, stixel: pd.core.frame.DataFrame) -> Union[None, pd.core.frame.DataFrame]:
        """A sub module of SAC prediction. Predict one stixel

        Args:
            stixel (pd.core.frame.DataFrame): data joined with ensemble_df.
            For a single stixel.

        Returns:
            pd.core.frame.DataFrame: the prediction result of this stixel
        """
        ensemble_index = stixel["ensemble_index"].iloc[0]
        unique_stixel_id = stixel["unique_stixel_id"].iloc[0]

        model_x_names_tuple = get_model_and_stixel_specific_x_names(
            self.model_dict,
            ensemble_index,
            unique_stixel_id,
            self.stixel_specific_x_names,
            self.x_names,
        )

        if model_x_names_tuple[0] is None:
            return None

        pred = predict_one_stixel(stixel, self.task, model_x_names_tuple)

        if pred is None:
            return None
        else:
            return pred

    def SAC_ensemble_predict(
        self, index_df: pd.core.frame.DataFrame, data: pd.core.frame.DataFrame
    ) -> pd.core.frame.DataFrame:
        """A sub-module of SAC prediction function.
        Predict only one ensemble.

        Args:
            index_df (pd.core.frame.DataFrame): ensemble data (model.ensemble_df)
            data (pd.core.frame.DataFrame): input covariates to predict
        Returns:
            pd.core.frame.DataFrame: Prediction result of one ensemble.
        """

        # Calculate the start indices for the sliding window
        start_indices = sorted(index_df[f"{self.Temporal1}_start"].unique())

        # prediction, window by window
        window_prediction_list = []
        for start in start_indices:
            window_data_df = data[
                (data[self.Temporal1] >= start) & (data[self.Temporal1] < start + self.temporal_bin_interval)
            ]
            window_data_df = transform_pred_set_to_STEM_quad(self.Spatio1, self.Spatio2, window_data_df, index_df)
            window_index_df = index_df[index_df[f"{self.Temporal1}_start"] == start]

            def find_belonged_points(df, df_a):
                return df_a[
                    (df_a[f"{self.Spatio1}_new"] >= df["stixel_calibration_point_transformed_left_bound"].iloc[0])
                    & (df_a[f"{self.Spatio1}_new"] < df["stixel_calibration_point_transformed_right_bound"].iloc[0])
                    & (df_a[f"{self.Spatio2}_new"] >= df["stixel_calibration_point_transformed_lower_bound"].iloc[0])
                    & (df_a[f"{self.Spatio2}_new"] < df["stixel_calibration_point_transformed_upper_bound"].iloc[0])
                ]

            query_results = (
                window_index_df[
                    [
                        "ensemble_index",
                        "unique_stixel_id",
                        "stixel_calibration_point_transformed_left_bound",
                        "stixel_calibration_point_transformed_right_bound",
                        "stixel_calibration_point_transformed_lower_bound",
                        "stixel_calibration_point_transformed_upper_bound",
                    ]
                ]
                .groupby(["ensemble_index", "unique_stixel_id"])
                .apply(find_belonged_points, df_a=window_data_df)
            )

            if len(query_results) == 0:
                """All points fall out of the grids"""
                continue

            # predict
            window_prediction = (
                query_results.reset_index(drop=False, level=[0, 1])
                .dropna(subset="unique_stixel_id")
                .groupby("unique_stixel_id")
                .apply(lambda stixel: self.stixel_predict(stixel))
            )

            window_prediction_list.append(window_prediction)

        if any([i is not None for i in window_prediction_list]):
            ensemble_prediction = pd.concat(window_prediction_list, axis=0)
            ensemble_prediction = ensemble_prediction.droplevel(0, axis=0)
            ensemble_prediction = ensemble_prediction.groupby("index").mean().reset_index(drop=False)
        else:
            ensmeble_index = list(window_index_df["ensemble_index"])[0]
            warnings.warn(f"No prediction for this ensemble: {ensmeble_index}")
            ensemble_prediction = None

        return ensemble_prediction

    def SAC_predict(
        self, ensemble_df: pd.core.frame.DataFrame, data: pd.core.frame.DataFrame, verbosity: int = 0, njobs: int = 1
    ) -> pd.core.frame.DataFrame:
        """This function is a prediction function with SAC strategy:
        Split (S), Apply(A), Combine (C). At ensemble level.
        It is built on pandas `apply` method.

        Args:
            ensemble_df (pd.core.frame.DataFrame): gridding information for all ensemble
            data (pd.core.frame.DataFrame): data
            verbosity (int, optional): Defaults to 0.

        Returns:
            pd.core.frame.DataFrame: prediction results.
        """
        assert isinstance(njobs, int)

        groups = ensemble_df.groupby("ensemble_index")

        # Parallel maker
        if njobs == 1:
            output_generator = (self.SAC_ensemble_predict(index_df=ensemble[1], data=data) for ensemble in groups)
        else:

            def mp_predict(ensemble, self=self, data=data):
                res = self.SAC_ensemble_predict(index_df=ensemble[1], data=data)
                return res

            parallel = joblib.Parallel(n_jobs=njobs, return_as="generator")
            output_generator = parallel(joblib.delayed(mp_predict)(i) for i in groups)

        # tqdm wrapper
        if verbosity > 0:
            output_generator = tqdm(
                output_generator, total=len(ensemble_df["ensemble_index"].unique()), desc="Predicting: "
            )

        # Prediction
        pred = [i.set_index("index") for i in output_generator]
        pred = pd.concat(pred, axis=1)
        if len(pred) == 0:
            raise ValueError(
                "All samples are not predictable based on current settings!\nTry adjusting the 'points_lower_threshold', increase the grid size, or increase sample size!"
            )

        pred.columns = list(range(self.ensemble_fold))
        return pred

    def predict_proba(
        self,
        X_test: pd.core.frame.DataFrame,
        verbosity: Union[int, None] = None,
        return_std: bool = False,
        njobs: Union[None, int] = 1,
        aggregation: str = "mean",
        return_by_separate_ensembles: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Predict probability

        Args:
            X_test (pd.core.frame.DataFrame):
                Testing variables.
            verbosity (int, optional):
                show progress bar or not. Yes for 0, and No for other. Defaults to None, which set it as the verbosity of the main model class.
            return_std (bool, optional):
                Whether return the standard deviation among ensembles. Defaults to False.
            njobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.njobs. Default to 1.
                I do not recommend setting value larger than 1.
                In practice, multi-processing seems to slow down the process instead of speeding up.
                Could be more practical with large amount of data.
                Still in experiment.
            aggregation (str, optional):
                'mean' or 'median' for aggregation method across ensembles.
            return_by_separate_ensembles (bool, optional):
                Experimental function. return not by aggregation, but by separate ensembles.

        Raises:
            TypeError:
                X_test is not of type pd.core.frame.DataFrame.
            ValueError:
                aggregation is not in ['mean','median'].

        Returns:
            predicted results. (pred_mean, pred_std) if return_std==true, and pred_mean if return_std==False.

            If return_by_separate_ensembles == True:
                Return numpy.ndarray of shape (n_samples, n_ensembles)

        """
        check_X_test(X_test)
        check_prediciton_aggregation(aggregation)
        return_by_separate_ensembles, return_std = check_prediction_return(return_by_separate_ensembles, return_std)
        verbosity = check_verbosity(self, verbosity)
        njobs = check_transform_njobs(self, njobs)

        # predict
        res = self.SAC_predict(self.ensemble_df, X_test, verbosity=verbosity, njobs=njobs)

        # Experimental Function
        if return_by_separate_ensembles:
            new_res = pd.DataFrame({"index": list(X_test.index)}).set_index("index")
            new_res = new_res.merge(res, left_on="index", right_on="index", how="left")
            return new_res.values

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
        new_res = pd.DataFrame({"index": list(X_test.index)}).set_index("index")
        new_res = new_res.merge(res, left_on="index", right_on="index", how="left")

        nan_count = np.sum(np.isnan(new_res["pred_mean"].values))
        nan_frac = nan_count / len(new_res["pred_mean"].values)
        warnings.warn(f"There are {nan_frac}% points ({nan_count} points) falling out of predictable range.")

        if return_std:
            return new_res["pred_mean"].values, new_res["pred_std"].values
        else:
            return new_res["pred_mean"].values

    def predict(
        self,
        X_test: pd.core.frame.DataFrame,
        verbosity: Union[None, int] = None,
        return_std: bool = False,
        njobs: Union[None, int] = 1,
        aggregation: str = "mean",
        return_by_separate_ensembles: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """A rewrite of predict_proba

        Args:
            X_test (pd.core.frame.DataFrame):
                Testing variables.
            verbosity (Union[None, int], optional):
                0 to output nothing, everything other wise. Default None set it to the verbosity of AdaSTEM model class.
            return_std (bool, optional):
                Whether return the standard deviation among ensembles. Defaults to False.
            njobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.njobs. Default to 1.
                I do not recommend setting value larger than 1.
                In practice, multi-processing seems to slow down the process instead of speeding up.
                Could be more practical with large amount of data.
                Still in experiment.
            aggregation (str, optional):
                'mean' or 'median' for aggregation method across ensembles.
            return_by_separate_ensembles (bool, optional):
                Experimental function. return not by aggregation, but by separate ensembles.

        Raises:
            TypeError:
                X_test is not of type pd.core.frame.DataFrame.
            ValueError:
                aggregation is not in ['mean','median'].

        Returns:
            predicted results. (pred_mean, pred_std) if return_std==true, and pred_mean if return_std==False.

            If return_by_separate_ensembles == True:
                Return numpy.ndarray of shape (n_samples, n_ensembles)

        """

        return self.predict_proba(
            X_test,
            verbosity=verbosity,
            return_std=return_std,
            njobs=njobs,
            aggregation=aggregation,
            return_by_separate_ensembles=return_by_separate_ensembles,
        )

    @classmethod
    def eval_STEM_res(
        self,
        task: str,
        y_test: Union[pd.core.series.Series, np.ndarray],
        y_pred: Union[pd.core.series.Series, np.ndarray],
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
            y_test (Union[pd.core.series.Series, np.ndarray]):
                y true
            y_pred (Union[pd.core.series.Series, np.ndarray]):
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

        if not task == "regression":
            a = pd.DataFrame({"y_true": np.array(y_test).flatten(), "pred": np.array(y_pred).flatten()}).dropna()

            y_test_b = np.where(a.y_true > cls_threshold, 1, 0)
            y_pred_b = np.where(a.pred > cls_threshold, 1, 0)

            if len(np.unique(y_test_b)) == 1 and len(np.unique(y_pred_b)) == 1:
                auc, kappa, f1, precision, recall, average_precision = [np.nan] * 6

            else:
                auc = roc_auc_score(y_test_b, y_pred_b)
                kappa = cohen_kappa_score(y_test_b, y_pred_b)
                f1 = f1_score(y_test_b, y_pred_b)
                precision = precision_score(y_test_b, y_pred_b)
                recall = recall_score(y_test_b, y_pred_b)
                average_precision = average_precision_score(y_test_b, y_pred_b)

        else:
            auc, kappa, f1, precision, recall, average_precision = [np.nan] * 6

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

    def score(self, X_test: pd.core.frame.DataFrame, y_test: Union[pd.core.series.Series, np.ndarray]) -> dict:
        """Combine predicting and evaluating in one method

        Args:
            X_test (pd.core.frame.DataFrame): Testing variables
            y_test (Union[pd.core.series.Series, np.ndarray]): y true

        Returns:
            dict: dictionary containing the metric names and their values.
        """

        y_pred, _ = self.predict(X_test)
        score_dict = AdaSTEM.eval_STEM_res(self.task, y_test, y_pred)
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

        for index, ensemble_row in self.ensemble_df[
            self.ensemble_df["stixel_checklist_count"] >= self.stixel_training_size_threshold
        ].iterrows():
            if ensemble_row["stixel_checklist_count"] < self.stixel_training_size_threshold:
                continue

            try:
                ensemble_index = ensemble_row["ensemble_index"]
                stixel_index = ensemble_row["unique_stixel_id"]
                the_model = self.model_dict[f"{ensemble_index}_{stixel_index}_model"]
                x_names = self.stixel_specific_x_names[f"{ensemble_index}_{stixel_index}"]

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
                # print(e)
                continue

        self.feature_importances_ = (
            pd.DataFrame(feature_importance_list).set_index("stixel_index").reset_index(drop=False).fillna(0)
        )

    def assign_feature_importances_by_points(
        self,
        Sample_ST_df: Union[pd.core.frame.DataFrame, None] = None,
        verbosity: Union[None, int] = None,
        aggregation: str = "mean",
        njobs: Union[int, None] = 1,
        assign_function: Callable = assign_points_to_one_ensemble,
    ) -> pd.core.frame.DataFrame:
        """Assign feature importance to the input spatio-temporal points

        Args:
            Sample_ST_df (Union[pd.core.frame.DataFrame, None], optional):
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
            njobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.njobs. Default to 1.

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
        njobs = check_transform_njobs(self, njobs)
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
        if njobs > 1:
            parallel = joblib.Parallel(n_jobs=njobs, return_as="generator")
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
        save_gridding_plot=False,
        save_tmp=False,
        save_dir="./",
        sample_weights_for_classifier=True,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        use_temporal_to_train=True,
        njobs=1,
        subset_x_names=False,
        ensemble_models_disk_saver=False,
        ensemble_models_disk_saving_dir="./",
        plot_xlims=(-180, 180),
        plot_ylims=(-90, 90),
        verbosity=0,
        plot_empty=False,
    ):
        super().__init__(
            base_model,
            task,
            ensemble_fold,
            min_ensemble_required,
            grid_len_upper_threshold,
            grid_len_lower_threshold,
            points_lower_threshold,
            stixel_training_size_threshold,
            temporal_start,
            temporal_end,
            temporal_step,
            temporal_bin_interval,
            temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude,
            save_gridding_plot,
            save_tmp,
            save_dir,
            sample_weights_for_classifier,
            Spatio1,
            Spatio2,
            Temporal1,
            use_temporal_to_train,
            njobs,
            subset_x_names,
            ensemble_models_disk_saver,
            ensemble_models_disk_saving_dir,
            plot_xlims,
            plot_ylims,
            verbosity,
            plot_empty,
        )

    def predict(
        self,
        X_test: pd.core.frame.DataFrame,
        verbosity: Union[None, int] = None,
        return_std: bool = False,
        cls_threshold: float = 0.5,
        njobs: Union[int, None] = 1,
        aggregation: str = "mean",
        return_by_separate_ensembles: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """A rewrite of predict_proba adapted for Classifier

        Args:
            X_test (pd.core.frame.DataFrame):
                Testing variables.
            verbosity (int, optional):
                0 to output nothing, everything other wise. Default None set it to the verbosity of AdaSTEM model class.
            return_std (bool, optional):
                Whether return the standard deviation among ensembles. Defaults to False.
            cls_threshold (float, optional):
                Cutting threshold for the classification.
                Values above cls_threshold will be labeled as 1 and 0 otherwise.
                Defaults to 0.5.
            njobs (Union[int, None], optional):
                Number of processes used in this task. If None, use the self.njobs. Default to 1.
                I do not recommend setting value larger than 1.
                In practice, multi-processing seems to slow down the process instead of speeding up.
                Could be more practical with large amount of data.
                Still in experiment.
            aggregation (str, optional):
                'mean' or 'median' for aggregation method across ensembles.
            return_by_separate_ensembles (bool, optional):
                Experimental function. return not by aggregation, but by separate ensembles.

        Raises:
            TypeError:
                X_test is not of type pd.core.frame.DataFrame.
            ValueError:
                aggregation is not in ['mean','median'].

        Returns:
            predicted results. (pred_mean, pred_std) if return_std==true, and pred_mean if return_std==False.

        """

        if return_std:
            mean, std = self.predict_proba(
                X_test,
                verbosity=verbosity,
                return_std=True,
                njobs=njobs,
                aggregation=aggregation,
                return_by_separate_ensembles=return_by_separate_ensembles,
            )
            mean = np.where(mean < cls_threshold, 0, mean)
            mean = np.where(mean >= cls_threshold, 1, mean)
            return mean, std
        else:
            mean = self.predict_proba(
                X_test,
                verbosity=verbosity,
                return_std=False,
                njobs=njobs,
                aggregation=aggregation,
                return_by_separate_ensembles=return_by_separate_ensembles,
            )
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
        save_gridding_plot=False,
        save_tmp=False,
        save_dir="./",
        sample_weights_for_classifier=True,
        Spatio1="longitude",
        Spatio2="latitude",
        Temporal1="DOY",
        use_temporal_to_train=True,
        njobs=1,
        subset_x_names=False,
        ensemble_models_disk_saver=False,
        ensemble_models_disk_saving_dir="./",
        plot_xlims=(-180, 180),
        plot_ylims=(-90, 90),
        verbosity=0,
        plot_empty=False,
    ):
        super().__init__(
            base_model,
            task,
            ensemble_fold,
            min_ensemble_required,
            grid_len_upper_threshold,
            grid_len_lower_threshold,
            points_lower_threshold,
            stixel_training_size_threshold,
            temporal_start,
            temporal_end,
            temporal_step,
            temporal_bin_interval,
            temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude,
            save_gridding_plot,
            save_tmp,
            save_dir,
            sample_weights_for_classifier,
            Spatio1,
            Spatio2,
            Temporal1,
            use_temporal_to_train,
            njobs,
            subset_x_names,
            ensemble_models_disk_saver,
            ensemble_models_disk_saving_dir,
            plot_xlims,
            plot_ylims,
            verbosity,
            plot_empty,
        )
