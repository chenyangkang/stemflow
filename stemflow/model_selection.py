from collections.abc import Sequence
from typing import Generator, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame

from .utils import check_random_state


def ST_train_test_split(
    X: DataFrame,
    y: Sequence,
    Spatio1: str = "longitude",
    Spatio2: str = "latitude",
    Temporal1: str = "DOY",
    Spatio_blocks_count: int = 10,
    Temporal_blocks_count: int = 10,
    test_size: float = 0.3,
    random_state: Union[None, int] = None,
) -> Tuple[DataFrame, DataFrame, ndarray, ndarray]:
    """Spatial Temporal train-test split

    Args:
        X:
            Training variables in DataFrame format
        y:
            Training target in DataFrame or numpy array format
        Spatio1:
            column name of spatial indicator 1
        Spatio2:
            column name of spatial indicator 2
        Temporal1:
            column name of temporal indicator 1
        Spatio_blocks_count:
            How many block to split for spatio indicators
        Temporal_blocks_count:
            How many block to split for temporal indicators
        test_size:
            Fraction of test set in terms of blocks count
        random_state:
            random state for choosing testing blocks

    Returns:
        X_train, X_test, y_train, y_test

    """
    # random seed
    rng = check_random_state(random_state)

    # validate
    if not isinstance(X, DataFrame):
        type_x = str(type(X))
        raise TypeError(f"X input should be pandas.core.frame.DataFrame, Got {type_x}")
    if not (isinstance(y, DataFrame) or isinstance(y, ndarray)):
        type_y = str(type(y))
        raise TypeError(f"y input should be pandas.core.frame.DataFrame or numpy.ndarray, Got {type_y}")

    # check shape match
    y_size = np.array(y).flatten().shape[0]
    X_size = X.shape[0]
    if not y_size == X_size:
        raise ValueError(f"The shape of X and y should match. Got X: {X_size}, y: {y_size}")

    # indexing
    Sindex1 = np.linspace(X[Spatio1].min(), X[Spatio1].max(), Spatio_blocks_count)
    Sindex2 = np.linspace(X[Spatio2].min(), X[Spatio2].max(), Spatio_blocks_count)
    Tindex1 = np.linspace(X[Temporal1].min(), X[Temporal1].max(), Temporal_blocks_count)

    indexes = [
        str(a) + "_" + str(b) + "_" + str(c)
        for a, b, c in zip(
            np.digitize(X[Spatio1].values, Sindex1),
            np.digitize(X[Spatio2].values, Sindex2),
            np.digitize(X[Temporal1].values, Tindex1),
        )
    ]

    unique_indexes = list(np.unique(indexes))

    # get test set record indexes
    test_indexes = []
    test_cell = list(rng.choice(unique_indexes, replace=False, size=int(len(unique_indexes) * test_size)))

    tmp_table = pd.DataFrame({"index": range(len(indexes)), "cell": indexes})

    tmp_table = tmp_table[tmp_table["cell"].isin(test_cell)]
    test_indexes = tmp_table["index"].values

    # get train set record indexes
    train_indexes = list(set(range(len(indexes))) - set(test_indexes))

    # get train test data
    X_train = X.iloc[train_indexes, :]
    y_train = np.array(y).flatten()[train_indexes].reshape(-1, 1)
    X_test = X.iloc[test_indexes, :]
    y_test = np.array(y).flatten()[test_indexes].reshape(-1, 1)

    return X_train, X_test, y_train, y_test


def ST_CV(
    X: DataFrame,
    y: Sequence,
    Spatio1: str = "longitude",
    Spatio2: str = "latitude",
    Temporal1: str = "DOY",
    Spatio_blocks_count: int = 10,
    Temporal_blocks_count: int = 10,
    random_state: Union[np.random.RandomState, None, int] = None,
    CV: int = 3,
) -> Generator[Tuple[DataFrame, DataFrame, ndarray, ndarray], None, None]:
    """Spatial Temporal train-test split

    Args:
        X:
            Training variables
        y:
            Training target
        Spatio1:
            column name of spatial indicator 1
        Spatio2:
            column name of spatial indicator 2
        Temporal1:
            column name of temporal indicator 1
        Spatio_blocks_count:
            How many block to split for spatio indicators
        Temporal_blocks_count:
            How many block to split for temporal indicators
        random_state:
            random state for choosing testing blocks
        CV:
            fold cross validation

    Returns:
        X_train, X_test, y_train, y_test

    """
    # random seed
    rng = check_random_state(random_state)

    # validate
    if not isinstance(X, DataFrame):
        type_x = str(type(X))
        raise TypeError(f"X input should be pandas.core.frame.DataFrame, Got {type_x}")
    if not (isinstance(y, DataFrame) or isinstance(y, ndarray)):
        type_y = str(type(y))
        raise TypeError(f"y input should be pandas.core.frame.DataFrame or numpy.ndarray, Got {type_y}")
    if not (isinstance(CV, int) and CV > 0):
        raise ValueError("CV should be a positive integer")

    # check shape match
    y_size = np.array(y).flatten().shape[0]
    X_size = X.shape[0]
    if not y_size == X_size:
        raise ValueError(f"The shape of X and y should match. Got X: {X_size}, y: {y_size}")

    # indexing
    Sindex1 = np.linspace(X[Spatio1].min(), X[Spatio1].max(), Spatio_blocks_count)
    Sindex2 = np.linspace(X[Spatio2].min(), X[Spatio2].max(), Spatio_blocks_count)
    Tindex1 = np.linspace(X[Temporal1].min(), X[Temporal1].max(), Temporal_blocks_count)

    indexes = [
        str(a) + "_" + str(b) + "_" + str(c)
        for a, b, c in zip(
            np.digitize(X[Spatio1], Sindex1), np.digitize(X[Spatio2], Sindex2), np.digitize(X[Temporal1], Tindex1)
        )
    ]

    unique_indexes = list(np.unique(indexes))
    rng.shuffle(unique_indexes)
    test_size = int(len(unique_indexes) * (1 / CV))

    tmp_table = pd.DataFrame({"index": range(len(indexes)), "cell": indexes})

    for cv_count in range(CV):
        # get test set record indexes
        test_indexes = []
        start = cv_count * test_size
        end = np.min([(cv_count + 1) * test_size, len(unique_indexes) + 1])
        test_cell = unique_indexes[start:end]

        tmp_this_CV_table = tmp_table[tmp_table["cell"].isin(test_cell)]
        test_indexes = tmp_this_CV_table["index"].values

        # get train set record indexes
        train_indexes = list(set(range(len(indexes))) - set(test_indexes))

        # get train test data
        X_train = X.iloc[train_indexes, :]
        y_train = np.array(y).flatten()[train_indexes].reshape(-1, 1)
        X_test = X.iloc[test_indexes, :]
        y_test = np.array(y).flatten()[test_indexes].reshape(-1, 1)

        yield X_train, X_test, y_train, y_test
