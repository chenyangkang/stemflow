"Validation module. Most of these functions are plain checking and easy to understand."

import warnings
from typing import Union

import numpy as np
import pandas as pd


def check_random_state(seed: Union[None, int, np.random._generator.Generator]) -> np.random._generator.Generator:
    """Turn seed into a np.random.RandomState instance.

    Args:
        seed:
            If seed is None, return a random generator.
            If seed is an int, return a random generator with that seed.
            If seed is already a random generator instance, return it.
            Otherwise raise ValueError.

    Returns:
        The random generator object based on `seed` parameter.
    """
    if seed is None:
        return np.random.default_rng(np.random.randint(0, 2**32 - 1))
    if isinstance(seed, int):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random._generator.Generator):
        return seed
    raise ValueError("%r cannot be used to seed a np.random.default_rng instance" % seed)


def check_task(task):
    if task not in ["regression", "classification", "hurdle"]:
        raise AttributeError(f"task type must be one of 'regression', 'classification', or 'hurdle'! Now it is {task}")
    if task == "hurdle":
        warnings.warn(
            "You have chosen HURDLE task. The goal is to first conduct classification, and then apply regression on points with *positive values*"
        )


def check_base_model(base_model):
    for func in ["fit", "predict"]:
        if func not in dir(base_model):
            raise AttributeError(f"input base model must have method '{func}'!")


def check_transform_n_jobs(self, n_jobs):
    if n_jobs is None:
        if self.n_jobs is None:
            warnings.warn("No n_jobs input. Default to 1.")
            return 1
        else:
            return self.n_jobs
    else:
        if not isinstance(n_jobs, int):
            raise TypeError(f"n_jobs is not a integer. Got {n_jobs}.")
        else:
            if n_jobs == 0:
                raise ValueError("n_jobs cannot be 0!")
            else:
                return n_jobs


def check_verbosity(self, verbosity):
    if verbosity is None:
        verbosity = self.verbosity
    elif verbosity == 0:
        verbosity = 0
    else:
        verbosity = 1
    return verbosity


def check_spatio_bin_jitter_magnitude(spatio_bin_jitter_magnitude):
    if isinstance(spatio_bin_jitter_magnitude, (int, float)):
        pass
    elif isinstance(spatio_bin_jitter_magnitude, str):
        if spatio_bin_jitter_magnitude == "adaptive":
            pass
        else:
            raise ValueError("spatio_bin_jitter_magnitude string must be adaptive!")
    else:
        raise ValueError("spatio_bin_jitter_magnitude string must be one of [int, float, 'adaptive']!")


def check_transform_spatio_bin_jitter_magnitude(data, Spatio1, Spatio2, spatio_bin_jitter_magnitude):
    check_spatio_bin_jitter_magnitude(spatio_bin_jitter_magnitude)
    if isinstance(spatio_bin_jitter_magnitude, str):
        if spatio_bin_jitter_magnitude == "adaptive":
            jit = max(data[Spatio1].max() - data[Spatio1].min(), data[Spatio2].max() - data[Spatio2].min())
            return jit
    return spatio_bin_jitter_magnitude


def check_temporal_bin_start_jitter(temporal_bin_start_jitter):
    # validate temporal_bin_start_jitter
    if not isinstance(temporal_bin_start_jitter, (str, float, int)):
        raise AttributeError(
            f"Input temporal_bin_start_jitter should be 'adaptive', float or int, got {type(temporal_bin_start_jitter)}"
        )
    if isinstance(temporal_bin_start_jitter, str):
        if not temporal_bin_start_jitter == "adaptive":
            raise AttributeError(
                f"The input temporal_bin_start_jitter as string should only be 'adaptive'. Other options include float or int. Got {temporal_bin_start_jitter}"
            )


def check_transform_temporal_bin_start_jitter(temporal_bin_start_jitter, bin_interval, rng):
    check_temporal_bin_start_jitter(temporal_bin_start_jitter)
    if isinstance(temporal_bin_start_jitter, str):
        if temporal_bin_start_jitter == "adaptive":
            jit = rng.uniform(low=0, high=bin_interval)
    elif type(temporal_bin_start_jitter) in [int, float]:
        jit = temporal_bin_start_jitter

    return jit


def check_X_train(X_train):
    # check type

    type_X_train = type(X_train)
    if not isinstance(X_train, pd.core.frame.DataFrame):
        raise TypeError(f"Input X should be type 'pd.core.frame.DataFrame'. Got {str(type_X_train)}")

    if np.sum(np.isnan(np.array(X_train))) > 0:
        raise ValueError(
            "NAs (missing values) detected in input data. stemflow do not support NAs input. Consider filling them with values (e.g., -1 or mean values) or removing the rows."
        )


def check_y_train(y_train):
    type_y_train = str(type(y_train))
    if not isinstance(y_train, (pd.core.frame.DataFrame, pd.core.frame.Series, np.ndarray)):
        raise TypeError(
            f"Input y_train should be type 'pd.core.frame.DataFrame' or 'pd.core.frame.Series', or 'np.ndarray'. Got {str(type_y_train)}"
        )

    if np.sum(np.isnan(np.array(y_train))) > 0:
        raise ValueError("NAs (missing values) detected in input y data. Consider deleting these rows.")


def check_X_test(X_test):
    check_X_train(X_test)


def check_prediciton_aggregation(aggregation):
    if aggregation not in ["mean", "median"]:
        raise ValueError(f"aggregation must be one of 'mean' and 'median'. Got {aggregation}")


def check_prediction_return(return_by_separate_ensembles, return_std):
    if not isinstance(return_by_separate_ensembles, bool):
        type_return_by_separate_ensembles = str(type(return_by_separate_ensembles))
        raise TypeError(f"return_by_separate_ensembles must be bool. Got {type_return_by_separate_ensembles}")
    else:
        if return_by_separate_ensembles and return_std:
            warnings.warn("return_by_separate_ensembles == True. Automatically setting return_std=False")
            return_std = False
    return return_by_separate_ensembles, return_std


def check_X_y_shape_match(X, y):
    # check shape match
    y_size = np.array(y).flatten().shape[0]
    X_size = X.shape[0]
    if not y_size == X_size:
        raise ValueError(f"The shape of X and y should match. Got X: {X_size}, y: {y_size}")


def check_spatial_scale(x_min, x_max, y_min, y_max, grid_length_upper, grid_length_lower):
    if (grid_length_upper <= (x_max - x_min) / 100) or (grid_length_upper <= (y_max - y_min) / 100):
        warnings.warn(
            "The grid_len_upper_threshold is significantly smaller than the scale of longitude and latitude (x and y). Be sure if this is desired."
        )
    if (grid_length_upper >= (x_max - x_min)) or (grid_length_upper >= (y_max - y_min)):
        warnings.warn(
            "The grid_len_upper_threshold is larger than the scale of longitude and latitude (x and y). Be sure if this is desired."
        )
    if (grid_length_lower <= (x_max - x_min) / 100) or (grid_length_lower <= (y_max - y_min) / 100):
        warnings.warn(
            "The grid_len_lower_threshold is significantly smaller than the scale of longitude and latitude (x and y). Be sure if this is desired."
        )
    if (grid_length_lower >= (x_max - x_min)) or (grid_length_lower >= (y_max - y_min)):
        warnings.warn(
            "The grid_len_lower_threshold is larger than the scale of longitude and latitude (x and y). Be sure if this is desired."
        )


def check_temporal_scale(t_min, t_max, temporal_bin_interval):
    if temporal_bin_interval <= (t_max - t_min) / 100:
        warnings.warn(
            "The temporal_bin_interval is significantly smaller than the scale of temporal parameters in provided data. Be sure if this is desired."
        )
    if temporal_bin_interval >= t_max - t_min:
        warnings.warn(
            "The temporal_bin_interval is larger than the scale of temporal parameters in provided data. Be sure if this is desired."
        )
