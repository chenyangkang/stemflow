from typing import Union

import numpy as np
from sklearn.base import BaseEstimator


class dummy_model1(BaseEstimator):
    """A dummy model that predict the constant value all the time"""

    def __init__(self, the_value: Union[float, int]):
        """Make dummy model1 class

        Args:
            the_value: The dummy value

        """
        self.the_value = float(the_value)
        pass

    def fit(self, X_train, y_train):
        """Fake fit"""
        pass

    def predict(self, X_test):
        """Fake predict"""
        return np.array([self.the_value] * X_test.shape[0])

    def predict_proba(self, X_test):
        """Fake predict_proba"""
        if self.the_value == 0:
            return np.array([[1, 0]] * X_test.shape[0])
        elif self.the_value == 1:
            return np.array([[0, 1]] * X_test.shape[0])
