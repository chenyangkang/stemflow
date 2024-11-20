"""This file is to store static functions for multi-processing

"""
import copy

#
import warnings
from typing import Tuple, Union
from warnings import simplefilter

import numpy as np
import pandas as pd
from numpy import ndarray

# validation check
from pandas.core.frame import DataFrame
from sklearn.base import BaseEstimator
from sklearn.utils import class_weight

from ..utils.jitterrotation.jitterrotator import JitterRotator, Sphere_Jitterrotator
from ..utils.sphere.coordinate_transform import lonlat_cartesian_3D_transformer
from ..utils.sphere.discriminant_formula import intersect_triangle_plane
from .dummy_model import dummy_model1

# warnings.filterwarnings("ignore")


def train_one_stixel(
    stixel_training_size_threshold: int,
    x_names: Union[list, np.ndarray],
    task: str,
    base_model: BaseEstimator,
    sample_weights_for_classifier: bool,
    subset_x_names: bool,
    stixel_X_train: pd.core.frame.DataFrame,
    min_class_sample: int,
) -> Tuple[Union[None, BaseEstimator], list]:
    """Train one stixel

    Args:
        stixel_training_size_threshold (int): Only stixels with data points above this threshold are trained.
        x_names (Union[list, np.ndarray]): Total x_names. Predictor variable.s
        task (str): One of 'regression', 'classification' and 'hurdle'
        base_model (BaseEstimator): Base model estimator.
        sample_weights_for_classifier (bool): Whether to balance the sample weights in classifier for imbalanced samples.
        subset_x_names (bool): Whether to only store variables with std > 0 for each stixel.
        sub_X_train (pd.core.frame.DataFrame): Input training dataframe for THE stixel.
        min_class_sample (int): Minimum umber of samples needed to train the classifier in each stixel. If the sample does not satisfy, fit a dummy one.

    Returns:
        tuple[Union[None, BaseEstimator], list]: trained_model, stixel_specific_x_names
    """

    if len(stixel_X_train) < stixel_training_size_threshold:  # threshold
        return (None, [], "Not_Enough_Data")

    sub_y_train = stixel_X_train["true_y"]
    sub_X_train = stixel_X_train[x_names]
    unique_sub_y_train_binary = np.unique(np.where(sub_y_train > 0, 1, 0))

    # nan check
    nan_count = np.sum(np.isnan(np.array(sub_X_train))) + np.sum(np.isnan(sub_y_train))
    if nan_count > 0:
        return (None, [], "Contain_Nan")

    sample_count_each_class = {i:np.sum(np.where(sub_y_train > 0, 1, 0)==i) for i in unique_sub_y_train_binary}
    min_sample_count_each_class = min([sample_count_each_class[i] for i in sample_count_each_class])
    
    # fit
    if (not task == "regression") and ((len(unique_sub_y_train_binary) == 1) or min_sample_count_each_class < min_class_sample):
        trained_model = dummy_model1(float(unique_sub_y_train_binary[0]))
        return (trained_model, [], "Success")
    else:
        # Remove the variables that have no variation
        stixel_specific_x_names = x_names.copy()

        if subset_x_names:
            stixel_specific_x_names = [
                i for i in stixel_specific_x_names if i not in list(sub_X_train.columns[sub_X_train.std(axis=0) == 0])
            ]

        # continue, if no variable left
        if len(stixel_specific_x_names) == 0:
            return (None, [], "x_names_length_zero")

        # now we are sure to fit a model
        trained_model = copy.deepcopy(base_model)

        if (not task == "regression") and sample_weights_for_classifier:
            sample_weights = class_weight.compute_sample_weight(
                class_weight="balanced", y=np.where(sub_y_train > 0, 1, 0)
            ).astype('float32')
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced", classes=np.array([0,1]), y=np.where(sub_y_train > 0, 1, 0)
            ).astype('float32')
            trained_model.fit(sub_X_train[stixel_specific_x_names], sub_y_train, sample_weight=sample_weights)
            trained_model.my_class_weights = class_weights
            
        else:
            trained_model.fit(sub_X_train[stixel_specific_x_names], sub_y_train)

    return (trained_model, stixel_specific_x_names, "Success")


def assign_points_to_one_ensemble(
    ensemble: str,
    ensemble_df: pd.core.frame.DataFrame,
    Sample_ST_df: pd.core.frame.DataFrame,
    Temporal1: str,
    Spatio1: str,
    Spatio2: str,
    feature_importances_: pd.core.frame.DataFrame,
) -> pd.core.frame.DataFrame:
    """assign points to one ensemble

    Args:
        ensemble_df (pd.core.frame.DataFrame): ensemble_df
        ensemble (str): name of the ensemble
        Sample_ST_df (pd.core.frame.DataFrame): input sample spatio-temporal points of interest
        Temporal1 (str): Temporal variable name 1
        Spatio1 (str): Spatio variable name 1
        Spatio2 (str): Spatio variable name 2
        feature_importances_ (pd.core.frame.DataFrame): feature_importances_ dataframe

    Returns:
        A DataFrame containing the aggregated feature importance
    """
    this_ensemble = ensemble_df[ensemble_df.ensemble_index == ensemble]
    this_ensemble.loc[:, "stixel_calibration_point_transformed_left_bound"] = [
        i[0] for i in this_ensemble["stixel_calibration_point(transformed)"]
    ]

    this_ensemble.loc[:, "stixel_calibration_point_transformed_lower_bound"] = [
        i[1] for i in this_ensemble["stixel_calibration_point(transformed)"]
    ]

    this_ensemble.loc[:, "stixel_calibration_point_transformed_right_bound"] = (
        this_ensemble["stixel_calibration_point_transformed_left_bound"] + this_ensemble["stixel_width"]
    )

    this_ensemble.loc[:, "stixel_calibration_point_transformed_upper_bound"] = (
        this_ensemble["stixel_calibration_point_transformed_lower_bound"] + this_ensemble["stixel_height"]
    )

    Sample_ST_df_ = transform_pred_set_to_STEM_quad(
        Spatio1, Spatio2, Sample_ST_df.reset_index(drop=True), this_ensemble
    )

    # pred each stixel
    res_list = []
    for index, line in this_ensemble.iterrows():
        stixel_index = line["unique_stixel_id"]
        sub_Sample_ST_df = Sample_ST_df_[
            (Sample_ST_df_[Temporal1] >= line[f"{Temporal1}_start"])
            & (Sample_ST_df_[Temporal1] < line[f"{Temporal1}_end"])
            & (Sample_ST_df_[f"{Spatio1}_new"] >= line["stixel_calibration_point_transformed_left_bound"])
            & (Sample_ST_df_[f"{Spatio1}_new"] <= line["stixel_calibration_point_transformed_right_bound"])
            & (Sample_ST_df_[f"{Spatio2}_new"] >= line["stixel_calibration_point_transformed_lower_bound"])
            & (Sample_ST_df_[f"{Spatio2}_new"] <= line["stixel_calibration_point_transformed_upper_bound"])
        ]

        if len(sub_Sample_ST_df) == 0:
            continue

        # load feature_importances
        try:
            this_feature_importance = feature_importances_[feature_importances_["stixel_index"] == stixel_index]
            if len(this_feature_importance) == 0:
                continue
            this_feature_importance = dict(this_feature_importance.iloc[0, :])
            res_list.append(
                {
                    "sample_index": list(sub_Sample_ST_df.index),
                    **{
                        a: [b] * len(sub_Sample_ST_df)
                        for a, b in zip(this_feature_importance.keys(), this_feature_importance.values())
                    },
                }
            )

        except Exception as e:
            print(e)
            continue

    res_list = pd.concat([pd.DataFrame(i) for i in res_list], axis=0).drop("stixel_index", axis=1)
    res_list = res_list.groupby("sample_index").mean().reset_index(drop=False)
    return res_list


def assign_points_to_one_ensemble_sphere(
    ensemble: str,
    ensemble_df: pd.core.frame.DataFrame,
    Sample_ST_df: pd.core.frame.DataFrame,
    Temporal1: str,
    Spatio1: str,
    Spatio2: str,
    feature_importances_: pd.core.frame.DataFrame,
    radius: Union[int, float] = 6371,
) -> pd.core.frame.DataFrame:
    """assign points to one ensemble, for spherical indexing

    Args:
        ensemble_df (pd.core.frame.DataFrame): ensemble_df
        ensemble (str): name of the ensemble
        Sample_ST_df (pd.core.frame.DataFrame): input sample spatio-temporal points of interest
        Temporal1 (str): Temporal variable name 1
        Spatio1 (str): Spatio variable name 1
        Spatio2 (str): Spatio variable name 2
        feature_importances_ (pd.core.frame.DataFrame): feature_importances_ dataframe
        radius (Union[float, int]): radius of earth in km

    Returns:
        A DataFrame containing the aggregated feature importance
    """
    this_ensemble = ensemble_df[ensemble_df.ensemble_index == ensemble]
    Sample_ST_df_ = transform_pred_set_to_Sphere_STEM_quad(
        Spatio1, Spatio2, Sample_ST_df.reset_index(drop=True), this_ensemble, radius
    )

    def find_belonged_points(df, df_a):
        P0 = np.array([0, 0, 0]).reshape(1, -1)
        A = np.array(df[["p1x", "p1y", "p1z"]].values.astype("float"))
        B = np.array(df[["p2x", "p2y", "p2z"]].values.astype("float"))
        C = np.array(df[["p3x", "p3y", "p3z"]].values.astype("float"))

        intersect = intersect_triangle_plane(
            P0=P0, V=df_a[["x_3D_transformed", "y_3D_transformed", "z_3D_transformed"]].values, A=A, B=B, C=C
        )

        return df_a.iloc[np.where(intersect)[0], :]

    # pred each stixel
    res_list = []

    unique_starts = sorted(this_ensemble[f"{Temporal1}_start"].unique())
    for start in unique_starts:
        this_slice = this_ensemble[this_ensemble[f"{Temporal1}_start"] == start]
        end_ = this_slice[f"{Temporal1}_end"].iloc[0]
        this_slice_sub_Sample_ST_df = Sample_ST_df_[
            (Sample_ST_df_[Temporal1] >= start) & (Sample_ST_df_[Temporal1] < end_)
        ]

        if len(this_slice_sub_Sample_ST_df) == 0:
            continue

        for index, line in this_slice.iterrows():
            stixel_index = line["unique_stixel_id"]
            sub_Sample_ST_df = find_belonged_points(line, this_slice_sub_Sample_ST_df)

            if len(sub_Sample_ST_df) == 0:
                continue

            # load feature_importances
            try:
                this_feature_importance = feature_importances_[feature_importances_["stixel_index"] == stixel_index]
                if len(this_feature_importance) == 0:
                    continue
                this_feature_importance = dict(this_feature_importance.iloc[0, :])
                res_list.append(
                    {
                        "sample_index": list(sub_Sample_ST_df.index),
                        **{
                            a: [b] * len(sub_Sample_ST_df)
                            for a, b in zip(this_feature_importance.keys(), this_feature_importance.values())
                        },
                    }
                )

            except Exception as e:
                print(e)
                continue

    res_list = pd.concat([pd.DataFrame(i) for i in res_list], axis=0).drop("stixel_index", axis=1)
    res_list = res_list.groupby("sample_index").mean().reset_index(drop=False)
    return res_list


def transform_pred_set_to_STEM_quad(
    Spatio1: str, Spatio2: str, X_train: pd.core.frame.DataFrame, ensemble_info: pd.core.frame.DataFrame
) -> pd.core.frame.DataFrame:
    """Project the input data points to the space of quadtree stixels.

    Args:
        Spatio1 (str):
            Name of the spatio column 1
        Spatio2 (str):
            Name of the spatio column 2
        X_train (pd.core.frame.DataFrame):
            Training/Testing variables
        ensemble_info (pd.core.frame.DataFrame):
            the DataFrame with information of the stixel.

    Returns:
        Projected X_train

    """

    angle = float(ensemble_info["rotation"].iloc[0])
    calibration_point_x_jitter = float(ensemble_info["calibration_point_x_jitter"].iloc[0])
    calibration_point_y_jitter = float(ensemble_info["calibration_point_y_jitter"].iloc[0])

    X_train_ = X_train.copy()
    a, b = JitterRotator.rotate_jitter(
        X_train[Spatio1], X_train[Spatio2], angle, calibration_point_x_jitter, calibration_point_y_jitter
    )
    X_train_[f"{Spatio1}_new"] = a
    X_train_[f"{Spatio2}_new"] = b

    return X_train_


def transform_pred_set_to_Sphere_STEM_quad(
    Spatio1: str,
    Spatio2: str,
    X_train: pd.core.frame.DataFrame,
    ensemble_info: pd.core.frame.DataFrame,
    radius: Union[float, int] = 6371.0,
) -> pd.core.frame.DataFrame:
    """Project the input data points to the space of quadtree stixels. For spherical indexing.

    Args:
        Spatio1 (str):
            Name of the spatio column 1
        Spatio2 (str):
            Name of the spatio column 2
        X_train (pd.core.frame.DataFrame):
            Training/Testing variables
        ensemble_info (pd.core.frame.DataFrame):
            the DataFrame with information of the stixel.

    Returns:
        Projected X_train

    """

    angle = float(ensemble_info["rotation_angle"].iloc[0])
    axis = np.array(
        [
            float(ensemble_info["rotaton_axis_x"].iloc[0]),
            float(ensemble_info["rotaton_axis_y"].iloc[0]),
            float(ensemble_info["rotaton_axis_z"].iloc[0]),
        ]
    )

    X_train_ = X_train.copy()
    x, y, z = lonlat_cartesian_3D_transformer.transform(
        X_train_[Spatio1].values, X_train_[Spatio2].values, radius=radius
    )
    X_train_["x_3D"] = x
    X_train_["y_3D"] = y
    X_train_["z_3D"] = z

    rotated_point = Sphere_Jitterrotator.rotate_jitter(
        np.column_stack([x, y, z]),
        axis,
        angle,
    )
    X_train_["x_3D_transformed"] = rotated_point[:, 0]
    X_train_["y_3D_transformed"] = rotated_point[:, 1]
    X_train_["z_3D_transformed"] = rotated_point[:, 2]

    return X_train_


def get_model_by_name(model_dict: dict, grid_index: str) -> Union[None, BaseEstimator]:
    """get_model_by_name

    Args:
        model_dict (dict): self.model_dict. Dictionary of trained models.
        grid_index (str): grid index

    Returns:
        The trained model.
    """
    try:
        model = model_dict[f"{grid_index}_model"]
        return model
    except Exception as e:
        if not isinstance(e, KeyError):
            warnings.warn(f"Cannot find model: {e}")
        return None


def get_stixel_specific_name_by_model(
    model: Union[None, BaseEstimator], stixel_specific_x_names_dict: dict, x_names: list, grid_index: str
) -> Union[None, list]:
    """get_stixel_specific_name_by_model

    Args:
        model (Union[None, BaseEstimator]): model of this stixel
        stixel_specific_x_names_dict (dict): the stixel_specific_x_names dictionary. Generated after training.
        x_names (list): total x_names. All variables.
        grid_index (str): grid index.

    Returns:
        stixel specific x_names.
    """
    if model is None:
        return None

    if isinstance(model, dummy_model1):
        stixel_specific_x_names = x_names
    else:
        stixel_specific_x_names = stixel_specific_x_names_dict[grid_index]

    return stixel_specific_x_names


def get_model_and_stixel_specific_x_names(
    model_dict: dict, grid_index: str, stixel_specific_x_names_dict: dict, x_names: list
) -> Tuple[Union[None, BaseEstimator], list]:
    """get_model_and_stixel_specific_x_names

    Args:
        model_dict (dict): self.model_dict. Dictionary of trained models.
        grid_index (str): grid index.
        stixel_specific_x_names_dict (dict): the stixel_specific_x_names dictionary. Generated after training.
        x_names (list): Total x_names. All variables.

    Returns:
       A tuple of (model, stixel_specific_x_names) for this stixel
    """
    model = get_model_by_name(model_dict, grid_index)
    stixel_specific_x_names = get_stixel_specific_name_by_model(
        model, stixel_specific_x_names_dict, x_names, grid_index
    )
    return model, stixel_specific_x_names


def predict_one_stixel(
    X_test_stixel: pd.core.frame.DataFrame,
    task: str,
    model_x_names_tuple: Tuple[Union[None, BaseEstimator], list],
    **base_model_prediction_param
) -> pd.core.frame.DataFrame:
    """predict_one_stixel

    Args:
        X_test_stixel (pd.core.frame.DataFrame): Input testing variables
        task (str): One of 'regression', 'classification' and 'hurdle'
        model_x_names_tuple (tuple[Union[None, BaseEstimator], list]): A tuple of (model, stixel_specific_x_names)
        base_model_prediction_param: Additional parameter passed to base_model.predict_proba or base_model.predict

    Returns:
        A Dataframe of predicted results. With 'index' the same as the input indexes.
    """
    
    if model_x_names_tuple[0] is None:
        return None

    if len(X_test_stixel) == 0:
        return None

    # get test data
    if task == "regression":
        pred = model_x_names_tuple[0].predict(X_test_stixel[model_x_names_tuple[1]])
    else:
        pred = model_x_names_tuple[0].predict_proba(X_test_stixel[model_x_names_tuple[1]], **base_model_prediction_param)
        pred = pred[:,1]


    res = pd.DataFrame({"index": list(X_test_stixel.index), "pred": np.array(pred).flatten()}).set_index("index")

    return res
