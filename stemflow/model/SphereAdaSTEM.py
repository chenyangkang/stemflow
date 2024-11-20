import os
import warnings
from functools import partial
from types import MethodType
from typing import Callable, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray

# validation check
from sklearn.base import BaseEstimator
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

from ..utils.sphere.discriminant_formula import intersect_triangle_plane

#
from ..utils.sphere_quadtree import get_one_ensemble_sphere_quadtree
from ..utils.validation import (
    check_base_model,
    check_prediciton_aggregation,
    check_random_state,
    check_spatial_scale,
    check_spatio_bin_jitter_magnitude,
    check_task,
    check_temporal_bin_start_jitter,
    check_temporal_scale,
    check_transform_n_jobs,
    check_verbosity,
)
from ..utils.wrapper import model_wrapper
from .AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor
from .static_func_AdaSTEM import (  # predict_one_ensemble
    assign_points_to_one_ensemble_sphere,
    transform_pred_set_to_Sphere_STEM_quad,
)

#


class SphereAdaSTEM(AdaSTEM):
    """A SphereAdaSTEm model class (allow fixed grid size)

    Parents:
        stemflow.model.AdaSTEM

    Children:
        stemflow.model.SphereAdaSTEM.SphereAdaSTEMClassifier

        stemflow.model.SphereAdaSTEM.SphereAdaSTEMRegressor

    """

    def __init__(
        self,
        base_model: BaseEstimator,
        task: str = "hurdle",
        ensemble_fold: int = 10,
        min_ensemble_required: int = 7,
        grid_len_upper_threshold: Union[float, int] = 8000,
        grid_len_lower_threshold: Union[float, int] = 500,
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
        plot_xlims: Tuple[Union[float, int], Union[float, int]] = (-180, 180),
        plot_ylims: Tuple[Union[float, int], Union[float, int]] = (-90, 90),
        verbosity: int = 0,
        plot_empty: bool = False,
        radius: float = 6371.0,
        lazy_loading: bool = False,
        lazy_loading_dir: Union[str, None] = None,
        min_class_sample: int = 1
    ):
        """Make a Spherical AdaSTEM object

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
                force divide if grid length (km) larger than the threshold. Defaults to 8000 km.
            grid_len_lower_threshold:
                stop divide if grid length (km) **will** be below than the threshold. Defaults to 500 km.
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
                Spatial column name 1 in data. For SphereAdaSTEM, this HAS to be 'longitude'.
            Spatio2:
                Spatial column name 2 in data. For SphereAdaSTEM, this HAS to be 'latitude'.
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
                If save_gridding_plot=true, what is the xlims of the plot. Defaults to (-180,180).
            plot_ylims:
                If save_gridding_plot=true, what is the ylims of the plot. Defaults to (-90,90).
            verbosity:
                0 to output nothing and everything otherwise.
            plot_empty:
                Whether to plot the empty grid
            radius:
                radius of earth in km.
            lazy_loading:
                If True, ensembles of models will be saved in disk, and only loaded when being used (e.g., prediction phase), and the ensembles of models are dump to disk once it is used.
            lazy_loading_dir:
                If lazy_loading, the directory of the model to temporary save to. Default to None, where a random number will be generated as folder name.
            min_class_sample:
                Minimum umber of samples needed to train the classifier in each stixel. If the sample does not satisfy, fit a dummy one. This parameter does not influence regression tasks.
                
        Raises:
            AttributeError: Base model do not have method 'fit' or 'predict'
            AttributeError: task not in one of ['regression', 'classification', 'hurdle']
            AttributeError: temporal_bin_start_jitter not in one of [str, float, int]
            AttributeError: temporal_bin_start_jitter is type str, but not 'adaptive'

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
        # Init parent class
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
            lazy_loading=lazy_loading,
            lazy_loading_dir=lazy_loading_dir,
            min_class_sample=min_class_sample
        )

        if not self.Spatio1 == "longitude":
            warnings.warn('the input Spatio1 is not "longitude"! Set to "longitude"')
            self.Spatio1 = "longitude"
        if not self.Spatio2 == "latitude":
            warnings.warn('the input Spatio1 is not "latitude"! Set to "latitude"')
            self.Spatio2 = "latitude"

        self.radius = radius

    def split(
        self, X_train: pd.core.frame.DataFrame, verbosity: Union[None, int] = None, ax=None, n_jobs: int = 1
    ) -> dict:
        """QuadTree indexing the input data

        Args:
            X_train: Input training data
            verbosity: 0 to output nothing, everything other wise. Default None set it to the verbosity of AdaSTEM model class.
            ax: matplotlit Axes to add to.

        Returns:
            self.grid_dict, a dictionary of one DataFrame for each grid, containing the gridding information
        """
        self.rng = check_random_state(self.random_state)
        verbosity = check_verbosity(self, verbosity)
        n_jobs = check_transform_n_jobs(self, n_jobs)

        if "grid_len" not in self.__dir__():
            # We are using AdaSTEM
            self.grid_len = None
            check_spatial_scale(
                X_train[self.Spatio1].min(),
                X_train[self.Spatio1].max(),
                X_train[self.Spatio2].min(),
                X_train[self.Spatio2].max(),
                self.grid_len_upper_threshold,
                self.grid_len_lower_threshold,
            )
            check_temporal_scale(
                X_train[self.Temporal1].min(), X_train[self.Temporal1].min(), self.temporal_bin_interval
            )
        else:
            # We are using STEM
            check_spatial_scale(
                X_train[self.Spatio1].min(),
                X_train[self.Spatio1].max(),
                X_train[self.Spatio2].min(),
                X_train[self.Spatio2].max(),
                self.grid_len,
                self.grid_len,
            )
            check_temporal_scale(
                X_train[self.Temporal1].min(), X_train[self.Temporal1].min(), self.temporal_bin_interval
            )
            pass

        partial_get_one_ensemble_sphere_quadtree = partial(
            get_one_ensemble_sphere_quadtree,
            data=X_train,
            spatio_bin_jitter_magnitude=self.spatio_bin_jitter_magnitude,
            temporal_start=self.temporal_start,
            temporal_end=self.temporal_end,
            temporal_step=self.temporal_step,
            temporal_bin_interval=self.temporal_bin_interval,
            temporal_bin_start_jitter=self.temporal_bin_start_jitter,
            Temporal1=self.Temporal1,
            radius=self.radius,
            grid_len_upper_threshold=self.grid_len_upper_threshold,
            grid_len_lower_threshold=self.grid_len_lower_threshold,
            points_lower_threshold=self.points_lower_threshold,
            plot_empty=self.plot_empty,
            save_gridding_plot=False,
            save_gridding_plotly=self.save_gridding_plot,
            ax=ax,
        )

        if n_jobs > 1 and isinstance(n_jobs, int):
            parallel = joblib.Parallel(n_jobs=n_jobs, return_as="generator")
            output_generator = parallel(
                joblib.delayed(partial_get_one_ensemble_sphere_quadtree)(
                    ensemble_count=ensemble_count, rng=np.random.default_rng(self.rng.integers(1e9) + ensemble_count)
                )
                for ensemble_count in list(range(self.ensemble_fold))
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
            ensemble_all_df_list = [
                partial_get_one_ensemble_sphere_quadtree(
                    ensemble_count=ensemble_count, rng=np.random.default_rng(self.rng.integers(1e9) + ensemble_count)
                )
                for ensemble_count in iter_func_
            ]

        ensemble_df = pd.concat(ensemble_all_df_list).reset_index(drop=True)
        ensemble_df = ensemble_df.reset_index(drop=True)

        del ensemble_all_df_list

        if self.save_gridding_plot:
            self.ensemble_df, self.gridding_plot = ensemble_df, ax

        else:
            self.ensemble_df, self.gridding_plot = ensemble_df, None

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
            window_data_df = transform_pred_set_to_Sphere_STEM_quad(
                self.Spatio1, self.Spatio2, window_data_df, index_df
            )
            window_index_df = index_df[index_df[f"{self.Temporal1}_start"] == start]

            # Merge
            def find_belonged_points(df, df_a):
                P0 = np.array([0, 0, 0]).reshape(1, -1)
                A = np.array(df[["p1x", "p1y", "p1z"]].iloc[0])
                B = np.array(df[["p2x", "p2y", "p2z"]].iloc[0])
                C = np.array(df[["p3x", "p3y", "p3z"]].iloc[0])

                intersect = intersect_triangle_plane(
                    P0=P0, V=df_a[["x_3D_transformed", "y_3D_transformed", "z_3D_transformed"]].values, A=A, B=B, C=C
                )

                return df_a.iloc[np.where(intersect)[0], :]

            query_results = (
                window_index_df[
                    [
                        "ensemble_index",
                        "unique_stixel_id",
                        "p1x",
                        "p1y",
                        "p1z",
                        "p2x",
                        "p2y",
                        "p2z",
                        "p3x",
                        "p3y",
                        "p3z",
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
            )
            res_list.append(list(res))

        return res_list

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
            window_data_df = transform_pred_set_to_Sphere_STEM_quad(
                self.Spatio1, self.Spatio2, window_data_df, index_df
            )
            window_index_df = index_df[index_df[f"{self.Temporal1}_start"] == start]

            def find_belonged_points(df, df_a):
                P0 = np.array([0, 0, 0]).reshape(1, -1)
                A = np.array(df[["p1x", "p1y", "p1z"]].iloc[0])
                B = np.array(df[["p2x", "p2y", "p2z"]].iloc[0])
                C = np.array(df[["p3x", "p3y", "p3z"]].iloc[0])

                intersect = intersect_triangle_plane(
                    P0=P0, V=df_a[["x_3D_transformed", "y_3D_transformed", "z_3D_transformed"]].values, A=A, B=B, C=C
                )

                return df_a.iloc[np.where(intersect)[0], :]

            query_results = (
                window_index_df[
                    [
                        "ensemble_index",
                        "unique_stixel_id",
                        "p1x",
                        "p1y",
                        "p1z",
                        "p2x",
                        "p2y",
                        "p2z",
                        "p3x",
                        "p3y",
                        "p3z",
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
            # print('window_prediction:',window_prediction)
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

    def assign_feature_importances_by_points(
        self,
        Sample_ST_df: Union[pd.core.frame.DataFrame, None] = None,
        verbosity: Union[None, int] = None,
        aggregation: str = "mean",
        n_jobs: Union[int, None] = 1,
        assign_function: Callable = assign_points_to_one_ensemble_sphere,
    ) -> pd.core.frame.DataFrame:
        return super().assign_feature_importances_by_points(
            Sample_ST_df, verbosity, aggregation, n_jobs, assign_function
        )


class SphereAdaSTEMClassifier(SphereAdaSTEM):
    """SphereAdaSTEM model Classifier interface

    Example:
        ```
        >>> from stemflow.model.SphereAdaSTEM import SphereAdaSTEMClassifier
        >>> from xgboost import XGBClassifier
        >>> model = SphereAdaSTEMClassifier(base_model=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
                                save_gridding_plot = True,
                                ensemble_fold=10,
                                min_ensemble_required=7,
                                grid_len_upper_threshold=8000,
                                grid_len_lower_threshold=500,
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
        grid_len_upper_threshold=8000,
        grid_len_lower_threshold=500,
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
        plot_xlims=(-180, 180),
        plot_ylims=(-90, 90),
        verbosity=0,
        plot_empty=False,
        lazy_loading=False,
        lazy_loading_dir=None,
        min_class_sample: int = 1
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
            lazy_loading=lazy_loading,
            lazy_loading_dir=lazy_loading_dir,
            min_class_sample=min_class_sample
        )

        self.predict = MethodType(AdaSTEMClassifier.predict, self)


class SphereAdaSTEMRegressor(SphereAdaSTEM):
    """SphereAdaSTEM model Regressor interface

    Example:
    ```
    >>> from stemflow.model.SphereAdaSTEM import SphereAdaSTEMRegressor
    >>> from xgboost import XGBRegressor
    >>> model = SphereAdaSTEMRegressor(base_model=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
                            save_gridding_plot = True,
                            ensemble_fold=10,
                            min_ensemble_required=7,
                            grid_len_upper_threshold=8000,
                            grid_len_lower_threshold=500,
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
        grid_len_upper_threshold=8000,
        grid_len_lower_threshold=500,
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
        plot_xlims=(-180, 180),
        plot_ylims=(-90, 90),
        verbosity=0,
        plot_empty=False,
        lazy_loading=False,
        lazy_loading_dir=None,
        min_class_sample: int = 1
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
            lazy_loading=lazy_loading,
            lazy_loading_dir=lazy_loading_dir,
            min_class_sample=min_class_sample
        )

        self.predict = MethodType(AdaSTEMRegressor.predict, self)