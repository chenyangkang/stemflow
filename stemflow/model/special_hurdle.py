import warnings
from collections.abc import Sequence
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator

from .dummy_model import dummy_model1


class LightGBM_Hurdle(BaseEstimator):
    """A simple Hurdle model class"""

    def __init__(
        self,
        classifier_params: dict = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "num_threads": 1,
            "verbosity": -1,
        },
        regressor_params: dict = {"objective": "regression", "metric": ["rmse"], "num_threads": 1, "verbosity": -1},
    ):
        """Make a LightGBM Hurdle class object

        Args:
            classifier_params:
                params for LightGBM for classification tasks
            regressor_params:
                params for LightGBM for regression tasks


        Example:
            ```py
            >>> model = AdaSTEMRegressor(
                    base_model=LightGBM_Hurdle(),
                    save_gridding_plot = True,
                    ensemble_fold=10,
                    min_ensemble_required=7,
                    grid_len_lon_upper_threshold=25,
                    grid_len_lon_lower_threshold=5,
                    grid_len_lat_upper_threshold=25,
                    grid_len_lat_lower_threshold=5,
                    points_lower_threshold=50,
                    Spatio1='longitude',
                    Spatio2 = 'latitude',
                    Temporal1 = 'DOY',
                    use_temporal_to_train=True,
                    njobs=1
                )

            >>> # fit adastem
            >>> model.fit(X_train.reset_index(drop=True), y_train, verbosity=0)

            >>> ...

            ```

        """

        self.classifier_params = classifier_params
        self.regressor_params = regressor_params
        self.classifier = None
        self.regressor = None

    def fit(self, X_train: Union[pd.core.frame.DataFrame, np.ndarray], y_train: Sequence, sample_weight=None):
        """Fitting model

        Args:
            X_train:
                Training variables
            y_train:
                Training target

        """
        binary_ = np.unique(np.where(y_train > 0, 1, 0))

        if len(binary_) == 1:
            warnings.warn("Warning: only one class presented. Replace with dummy classifier & regressor.")
            self.classifier = dummy_model1(binary_[0])
            self.regressor = dummy_model1(binary_[0])
            return

        # cls
        cls_dat = lgb.Dataset(np.array(X_train), label=np.where(np.array(y_train).flatten() > 0, 1, 0))

        if sample_weight is not None:
            cls_dat.set_weight(sample_weight)
        else:
            pass

        cls_ = lgb.train(self.classifier_params, cls_dat)
        self.classifier = cls_

        # reg
        reg_dat = lgb.Dataset(
            np.array(X_train)[np.array(y_train).flatten() > 0, :],
            label=np.array(y_train).flatten()[np.array(y_train).flatten() > 0],
        )
        reg_ = lgb.train(self.regressor_params, reg_dat)
        self.regressor = reg_

        # try:
        #     self.feature_importances_ = (np.array(self.classifier.feature_importances_) + np.array(self.regressor.feature_importances_))/2
        # except Exception as e:
        #     pass

    def predict(self, X_test: Union[pd.core.frame.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicting

        Args:
            X_test: Test variables

        Returns:
            A prediction array with shape (-1,1)
        """
        cls_res = self.classifier.predict(X_test)
        cls_res = np.where(cls_res > 0.5, 1, cls_res)
        cls_res = np.where(cls_res < 0.5, 0, cls_res)
        reg_res = self.regressor.predict(X_test)
        # reg_res = np.where(reg_res>=0, reg_res, 0) # we constrain the reg value to be positive
        res = np.where(cls_res > 0, reg_res, cls_res)
        return res.reshape(-1, 1)

    def predict_proba(self, X_test: Union[pd.core.frame.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicting probability

        This method output a numpy array with shape (n_sample, 2)
        However, user should notice that this is only for structuring the sklearn predict_proba-like method
        Only the res[:,1] is meaningful, aka the last dimension in the two dimensions. The first dimension is always zero.

        Args:
            X_test:
                Testing variables

        Returns:
            Prediction results with shape (n_samples, 2)
        """
        a = np.zeros(len(X_test)).reshape(-1, 1)
        b = self.predict(X_test).reshape(-1, 1)
        res = np.concatenate([a, b], axis=1)
        return res


class XGBoost_Hurdle(BaseEstimator):
    """A simple Hurdle model class"""

    def __init__(
        self,
        classifier_params: dict = {
            "objective": "binary:logistic",
            "metric": ["auc", "logloss"],
            "tree_method": "hist",
            "nthread": 1,
            "verbosity": 0,
        },
        regressor_params: dict = {
            "objective": "reg:squarederror",
            "metric": ["rmse"],
            "tree_method": "hist",
            "nthread": 1,
            "verbosity": 0,
        },
    ):
        """Make a XGBoost Hurdle class object

        Args:
            classifier_params:
                params for XGBoost for classification tasks
            regressor_params:
                params for XGBoost for regression tasks

        Example:
            ```py
            >>> model = AdaSTEMRegressor(
                    base_model=XGBoost_Hurdle(),
                    save_gridding_plot = True,
                    ensemble_fold=10,
                    min_ensemble_required=7,
                    grid_len_lon_upper_threshold=25,
                    grid_len_lon_lower_threshold=5,
                    grid_len_lat_upper_threshold=25,
                    grid_len_lat_lower_threshold=5,
                    points_lower_threshold=50,
                    Spatio1='longitude',
                    Spatio2 = 'latitude',
                    Temporal1 = 'DOY',
                    use_temporal_to_train=True,
                    njobs=1
                )

            >>> # fit adastem
            >>> model.fit(X_train.reset_index(drop=True), y_train, verbosity=0)

            >>> ...

            ```

        """
        self.classifier_params = classifier_params
        self.regressor_params = regressor_params
        self.classifier = None
        self.regressor = None

    def fit(self, X_train: Union[pd.core.frame.DataFrame, np.ndarray], y_train: Sequence, sample_weight=None):
        """Fitting model

        Args:
            X_train:
                Training variables
            y_train:
                Training target

        """
        binary_ = np.unique(np.where(y_train > 0, 1, 0))

        if len(binary_) == 1:
            warnings.warn("Warning: only one class presented. Replace with dummy classifier & regressor.")
            self.classifier = dummy_model1(binary_[0])
            self.regressor = dummy_model1(binary_[0])
            return

        # cls
        if sample_weight is not None:
            cls_dat = xgb.DMatrix(
                np.array(X_train), label=np.where(np.array(y_train).flatten() > 0, 1, 0), weight=sample_weight
            )
        else:
            cls_dat = xgb.DMatrix(np.array(X_train), label=np.where(np.array(y_train).flatten() > 0, 1, 0))

        cls_ = xgb.train(self.classifier_params, cls_dat)
        self.classifier = cls_

        # reg
        reg_dat = xgb.DMatrix(
            np.array(X_train)[np.array(y_train).flatten() > 0, :],
            label=np.array(y_train).flatten()[np.array(y_train).flatten() > 0],
        )
        reg_ = xgb.train(self.regressor_params, reg_dat)
        self.regressor = reg_

        # try:
        #     self.feature_importances_ = (np.array(self.classifier.feature_importances_) + np.array(self.regressor.feature_importances_))/2
        # except Exception as e:
        #     pass

    def predict(self, X_test: Union[pd.core.frame.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicting

        Args:
            X_test: Test variables

        Returns:
            A prediction array with shape (-1,1)
        """
        if isinstance(self.classifier, dummy_model1):
            res = np.array([self.classifier.the_value] * X_test.shape[0])
        else:
            X_test_dmat = xgb.DMatrix(np.array(X_test))
            cls_res = self.classifier.predict(X_test_dmat)
            cls_res = np.where(cls_res > 0.5, 1, cls_res)
            cls_res = np.where(cls_res < 0.5, 0, cls_res)
            reg_res = self.regressor.predict(X_test_dmat)
            res = np.where(cls_res > 0, reg_res, cls_res)

        return res.reshape(-1, 1)

    def predict_proba(self, X_test: Union[pd.core.frame.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicting probability

        This method output a numpy array with shape (n_sample, 2)
        However, user should notice that this is only for structuring the sklearn predict_proba-like method
        Only the res[:,1] is meaningful, aka the last dimension in the two dimensions. The first dimension is always zero.

        Args:
            X_test:
                Testing variables

        Returns:
            Prediction results with shape (n_samples, 2)
        """
        a = np.zeros(len(X_test)).reshape(-1, 1)
        b = self.predict(X_test).reshape(-1, 1)
        res = np.concatenate([a, b], axis=1)
        return res
