from collections.abc import Sequence
from typing import Generator, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame

from .utils.validation import check_random_state


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
    """A function to generate spatiotemporal train-test-split data. To only generate indexes, see class `ST_Kfold`.

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


class ST_KFold:
    def __init__(
        self,
        Spatio1: str = "longitude",
        Spatio2: str = "latitude",
        Temporal1: str = "DOY",
        Spatio_blocks_count: int = 10,
        Temporal_blocks_count: int = 10,
        random_state: Union[np.random.RandomState, None, int] = None,
        n_splits: int = 3,
    ) -> None:
        """Spatial Temporal KFold generator class. While the ST_CV functions yield the data directly (X_train, X_test, y_train, y_test),
            this ST_KFold class generate only indices, which match the KFold class in sklearn.model_selection.

        Args:
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
            n_splits:
                fold cross validation

        Returns:
            train_indexes, test_indexes

        Example:
            ```
            from sklearn.model_selection import KFold
            ST_KFold_generator = ST_KFold(n_splits=5,
                    Spatio1 = "longitude",
                    Spatio2 = "latitude",
                    Temporal1 = "DOY",
                    Spatio_blocks_count = 10,
                    Temporal_blocks_count = 10,
                    random_state = 42).split(X)

            for train_indexes, test_indexes in ST_KFold_generator:
                X_train = X.iloc[train_indexes,:]
                X_test = X.iloc[test_indexes,:]
                ...

            ```

        """
        self.rng = check_random_state(random_state)
        self.Spatio1 = Spatio1
        self.Spatio2 = Spatio2
        self.Temporal1 = Temporal1
        self.Spatio_blocks_count = Spatio_blocks_count
        self.Temporal_blocks_count = Temporal_blocks_count
        self.n_splits = n_splits

        if not (isinstance(n_splits, int) and n_splits > 0):
            raise ValueError("CV should be a positive integer")

    def split(self, X: DataFrame) -> Generator[Tuple[ndarray, ndarray], None, None]:
        """split

        Args:
            X:
                Training variables
        Yields:
            Generator[Tuple[ndarray, ndarray], None, None]: train_index, test_index
        """

        # validate
        if not isinstance(X, DataFrame):
            type_x = str(type(X))
            raise TypeError(f"X input should be pandas.core.frame.DataFrame, Got {type_x}")

        # indexing
        Sindex1 = np.linspace(X[self.Spatio1].min(), X[self.Spatio1].max(), self.Spatio_blocks_count)
        Sindex2 = np.linspace(X[self.Spatio2].min(), X[self.Spatio2].max(), self.Spatio_blocks_count)
        Tindex1 = np.linspace(X[self.Temporal1].min(), X[self.Temporal1].max(), self.Temporal_blocks_count)

        indexes = [
            str(a) + "_" + str(b) + "_" + str(c)
            for a, b, c in zip(
                np.digitize(X[self.Spatio1], Sindex1),
                np.digitize(X[self.Spatio2], Sindex2),
                np.digitize(X[self.Temporal1], Tindex1),
            )
        ]

        unique_indexes = list(np.unique(indexes))
        self.rng.shuffle(unique_indexes)
        test_size = int(len(unique_indexes) * (1 / self.n_splits))

        tmp_table = pd.DataFrame({"index": range(len(indexes)), "cell": indexes})

        for cv_count in range(self.n_splits):
            # get test set record indexes
            test_indexes = []
            start = cv_count * test_size
            end = np.min([(cv_count + 1) * test_size, len(unique_indexes) + 1])
            test_cell = unique_indexes[start:end]

            tmp_this_CV_table = tmp_table[tmp_table["cell"].isin(test_cell)]
            test_indexes = tmp_this_CV_table["index"].values

            # get train set record indexes
            train_indexes = list(set(range(len(indexes))) - set(test_indexes))

            yield train_indexes, test_indexes
