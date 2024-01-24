import os
import pickle
import warnings
from itertools import repeat

#
from multiprocessing import Pool, cpu_count
from typing import Tuple, Union

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

from ..utils.sphere.coordinate_transform import lonlat_cartesian_3D_transformer
from ..utils.sphere.discriminant_formula import intersect_triangle_plane

#
from ..utils.sphere_quadtree import get_ensemble_sphere_quadtree
from ..utils.validation import (
    check_base_model,
    check_njobs,
    check_prediciton_aggregation,
    check_prediction_return,
    check_random_state,
    check_spatio_bin_jitter_magnitude,
    check_task,
    check_temporal_bin_start_jitter,
    check_verbosity,
    check_X_test,
    check_X_train,
    check_y_train,
)
from ..utils.wrapper import model_wrapper
from .AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor
from .dummy_model import dummy_model1
from .Hurdle import Hurdle
from .static_func_AdaSTEM import (  # predict_one_ensemble
    assign_points_to_one_ensemble_sphere,
    get_model_and_stixel_specific_x_names,
    predict_one_stixel,
    train_one_stixel,
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
                Spatial column name 1 in data. For SphereAdaSTEM, this HAS to be 'longitude'.
            Spatio2:
                Spatial column name 2 in data. For SphereAdaSTEM, this HAS to be 'latitude'.
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
        if not ((self.Spatio1 == "longitude") and (self.Spatio2 == "latitude")):
            raise ValueError("Spatio1 and Spatio2 must be longitude and latitude!")

        # 3. Gridding params
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
        check_njobs(njobs)
        self.njobs = njobs

        # 6. Plotting params
        self.plot_xlims = plot_xlims
        self.plot_ylims = plot_ylims
        self.save_tmp = save_tmp
        self.save_dir = save_dir
        self.save_gridding_plot = save_gridding_plot  # Actually means plotly

        # X. miscellaneous
        self.ensemble_models_disk_saver = ensemble_models_disk_saver
        self.ensemble_models_disk_saving_dir = ensemble_models_disk_saving_dir
        if self.ensemble_models_disk_saver:
            self.saving_code = np.random.randint(1, 1e8, 1)

        if not verbosity == 0:
            self.verbosity = 1
        else:
            self.verbosity = 0

    def split(self, X_train: pd.core.frame.DataFrame, verbosity: Union[None, int] = None, ax=None) -> dict:
        """QuadTree indexing the input data

        Args:
            X_train: Input training data
            verbosity: 0 to output nothing, everything other wise. Default None set it to the verbosity of AdaSTEM model class.
            ax: matplotlit Axes to add to.

        Returns:
            self.grid_dict, a dictionary of one DataFrame for each grid, containing the gridding information
        """
        if verbosity is None:
            verbosity = self.verbosity

        fold = self.ensemble_fold
        save_path = os.path.join(self.save_dir, "ensemble_quadtree_df.csv") if self.save_tmp else ""

        if "grid_len" not in self.__dir__():
            # We har using AdaSTEM
            self.grid_len = None
        else:
            # We har using STEM
            pass

        self.ensemble_df, self.gridding_plot = get_ensemble_sphere_quadtree(
            X_train[[self.Spatio1, self.Spatio2, self.Temporal1]],
            Temporal1=self.Temporal1,
            size=fold,
            grid_len_upper_threshold=self.grid_len_upper_threshold,
            grid_len_lower_threshold=self.grid_len_lower_threshold,
            points_lower_threshold=self.points_lower_threshold,
            temporal_start=self.temporal_start,
            temporal_end=self.temporal_end,
            temporal_step=self.temporal_step,
            temporal_bin_interval=self.temporal_bin_interval,
            temporal_bin_start_jitter=self.temporal_bin_start_jitter,
            spatio_bin_jitter_magnitude=self.spatio_bin_jitter_magnitude,
            save_gridding_plotly=self.save_gridding_plot,  # currently only allow output plotly
            save_gridding_plot=False,
            njobs=self.njobs,
            verbosity=verbosity,
            plot_xlims=self.plot_xlims,
            plot_ylims=self.plot_ylims,
            save_path=save_path,
            ax=ax,
        )

    def SAC_ensemble_training(self, index_df, data):
        # Calculate the start indices for the sliding window
        unique_start_indices = np.sort(index_df[f"{self.Temporal1}_start"].unique())
        # training, window by window
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
            (
                query_results.reset_index(drop=False, level=[0, 1])
                .dropna(subset="unique_stixel_id")
                .groupby("unique_stixel_id")
                .apply(lambda stixel: self.stixel_fitting(stixel))
            )

    def SAC_ensemble_predict(self, index_df, data):
        """Predict one ensemble

        Args:
            index_df (pd.core.frame.DataFrame): ensemble data (model.ensemble_df)
            data (pd.core.frame.DataFrame): input covariates to predict
        """

        temp_start = index_df[f"{self.Temporal1}_start"].min()
        temp_end = index_df[f"{self.Temporal1}_end"].max()

        # Calculate the start indices for the sliding window
        start_indices = np.arange(temp_start, temp_end, self.temporal_step)

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

            window_prediction_list.append(window_prediction)

        ensemble_prediction = pd.concat(window_prediction_list, axis=0)
        ensemble_prediction = ensemble_prediction.droplevel(0, axis=0)
        ensemble_prediction = ensemble_prediction.groupby("index").mean().reset_index(drop=False)
        return ensemble_prediction

    def assign_feature_importances_by_points(
        self,
        Sample_ST_df: Union[pd.core.frame.DataFrame, None] = None,
        verbosity: Union[None, int] = None,
        aggregation: str = "mean",
        njobs: Union[int, None] = 1,
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
        verbosity = check_verbosity(self, verbosity)

        #
        if "feature_importances_" not in dir(self):
            raise NameError(
                "feature_importances_ attribute is not calculated. Try model.calculate_feature_importances() first."
            )
        #
        if aggregation not in ["mean", "median"]:
            raise ValueError("aggregation not one of ['mean','median'].")
        #
        if njobs is None:
            njobs = self.njobs

        #
        if Sample_ST_df is not None:
            for var_name in [self.Spatio1, self.Spatio2, self.Temporal1]:
                if var_name not in Sample_ST_df.columns:
                    raise KeyError(f"{var_name} not found in Sample_ST_df.columns")
        else:
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

        # assign input spatio-temporal points to stixels

        if not njobs > 1:
            # Single processing
            round_res_list = []
            iter_func_ = (
                tqdm(list(self.ensemble_df.ensemble_index.unique()))
                if verbosity > 0
                else list(self.ensemble_df.ensemble_index.unique())
            )
            for ensemble in iter_func_:
                res_list = assign_points_to_one_ensemble_sphere(
                    self.ensemble_df,
                    ensemble,
                    Sample_ST_df,
                    self.Temporal1,
                    self.Spatio1,
                    self.Spatio2,
                    self.feature_importances_,
                )
                round_res_list.append(res_list)

        else:
            raise NotImplementedError("Multi-threading not implemented")

        round_res_df = pd.concat(round_res_list, axis=0)
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


class SphereAdaSTEMClassifier(SphereAdaSTEM):
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
        """A rewrite of predict_proba

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


class SphereAdaSTEMRegressor(SphereAdaSTEM):
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
        )
