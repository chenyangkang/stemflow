"Validation module. Most of these functions are plain checking and easy to understand."

import warnings
from typing import Union

import numpy as np
import pandas as pd


def check_random_state(seed: Union[None, int, np.random.RandomState]) -> np.random.RandomState:
    """Turn seed into a np.random.RandomState instance.

    Args:
        seed:
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.

    Returns:
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState instance" % seed)


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


def check_transform_njobs(self, njobs):
    if njobs is None:
        if self.njobs is None:
            warnings.warn("No njobs input. Default to 1.")
            return 1
        else:
            return self.njobs
    else:
        if not isinstance(njobs, int):
            raise TypeError(f"njobs is not a integer. Got {njobs}.")
        else:
            if njobs == 0:
                raise ValueError("njobs cannot be 0!")
            else:
                return njobs


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


def check_transform_temporal_bin_start_jitter(temporal_bin_start_jitter, bin_interval):
    check_temporal_bin_start_jitter(temporal_bin_start_jitter)
    if isinstance(temporal_bin_start_jitter, str):
        if temporal_bin_start_jitter == "adaptive":
            jit = np.random.uniform(low=0, high=bin_interval)
    elif type(temporal_bin_start_jitter) in [int, float]:
        jit = temporal_bin_start_jitter

    return jit


def check_X_train(X_train):
    # check type

    type_X_train = type(X_train)
    if not isinstance(X_train, pd.core.frame.DataFrame):
        raise TypeError(f"Input X should be type 'pd.core.frame.DataFrame'. Got {str(type_X_train)}")


def check_y_train(y_train):
    type_y_train = str(type(y_train))
    if not isinstance(y_train, (pd.core.frame.DataFrame, pd.core.frame.Series, np.ndarray)):
        raise TypeError(
            f"Input y_train should be type 'pd.core.frame.DataFrame' or 'pd.core.frame.Series', or 'np.ndarray'. Got {str(type_y_train)}"
        )


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
            warnings("return_by_separate_ensembles == True. Automatically setting return_std=False")
            return_std = False
    return return_by_separate_ensembles, return_std


def check_X_y_shape_match(X, y):
    # check shape match
    y_size = np.array(y).flatten().shape[0]
    X_size = X.shape[0]
    if not y_size == X_size:
        raise ValueError(f"The shape of X and y should match. Got X: {X_size}, y: {y_size}")
