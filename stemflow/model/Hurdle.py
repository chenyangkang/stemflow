import warnings
from collections.abc import Sequence
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .dummy_model import dummy_model1


class Hurdle(BaseEstimator):
    """A simple Hurdle model class"""

    def __init__(self, classifier: BaseEstimator, regressor: BaseEstimator):
        """Make a Hurdle class object

        Args:
            classifier:
                A sklearn style classifier estimator. Must have `fit` and `predict` methods.
                Will be better if it has `predict_proba` method, which returns a numpy array of shape (n_sample, 2)
            regressor:
                A sklearn style regressor estimator. Must have `fit` and `predict` methods.

        Example:
            ```
            >> from xgboost import XGBClassifier, XGBRegressor
            >> from stemflow.model.Hurdle import Hurdle
            >> model = Hurdle(classifier = XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
                              regressor = XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1))
            >> model.fit(X_train, y_train)
            >> pred = model.predict(X_test)
            >> ...

            ```

        """
        self.classifier = classifier
        self.regressor = regressor

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

        new_dat = np.concatenate([np.array(X_train), np.array(y_train).reshape(-1, 1)], axis=1)
        if not isinstance(sample_weight, type(None)):
            self.classifier.fit(new_dat[:, :-1], np.where(new_dat[:, -1] > 0, 1, 0), sample_weight=sample_weight)
        else:
            self.classifier.fit(new_dat[:, :-1], np.where(new_dat[:, -1] > 0, 1, 0))

        regressor_y = new_dat[new_dat[:, -1] > 0, :][:, -1].reshape(-1, 1)
        if regressor_y.shape[0] <= 1:
            self.regressor = dummy_model1(regressor_y[0][0])
        else:
            self.regressor.fit(new_dat[new_dat[:, -1] > 0, :][:, :-1], np.array(regressor_y))

        try:
            self.feature_importances_ = (
                np.array(self.classifier.feature_importances_) + np.array(self.regressor.feature_importances_)
            ) / 2
        except Exception as e:
            warnings.warn(f"Cannot calculate feature importance: {e}")
            pass

    def predict(self, X_test: Union[pd.core.frame.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicting

        Args:
            X_test: Test variables

        Returns:
            A prediction array with shape (-1,1)
        """
        cls_res = self.classifier.predict(X_test)
        reg_res = self.regressor.predict(X_test)
        # reg_res = np.where(reg_res>=0, reg_res, 0) ### we constrain the reg value to be positive
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


class Hurdle_for_AdaSTEM(BaseEstimator):
    def __init__(self, classifier: BaseEstimator, regressor: BaseEstimator):
        """Make a Hurdle_for_AdaSTEM class object

        Normally speaking, AdaSTEMClassifier and AdaSTEMRegressor should be passed here if using this class.

        Args:
            classifier:
                A sklearn style classifier estimator (should be AdaSTEMClassifier here). Must have `fit` and `predict` methods.
                Will be better if it has `predict_proba` method, which returns a numpy array of shape (n_sample, 2)
            regressor:
                A sklearn style regressor estimator (should be AdaSTEMRegressor here). Must have `fit` and `predict` methods.

        Example:
            ```
            >> from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor
            >> from stemflow.model.Hurdle import Hurdle_for_AdaSTEM
            >> from xgboost import XGBClassifier, XGBRegressor

            >> SAVE_DIR = './'

            >> model = Hurdle_for_AdaSTEM(
            ...     classifier=AdaSTEMClassifier(base_model=XGBClassifier(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
            ...                                 save_gridding_plot = True,
            ...                                 ensemble_fold=10,
            ...                                 min_ensemble_required=7,
            ...                                 grid_len_lon_upper_threshold=25,
            ...                                 grid_len_lon_lower_threshold=5,
            ...                                 grid_len_lat_upper_threshold=25,
            ...                                 grid_len_lat_lower_threshold=5,
            ...                                 points_lower_threshold=50,
            ...                                 Spatio1='longitude',
            ...                                 Spatio2 = 'latitude',
            ...                                 Temporal1 = 'DOY',
            ...                                 use_temporal_to_train=True),
            ...     regressor=AdaSTEMRegressor(base_model=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
            ...                                 save_gridding_plot = True,
            ...                                 ensemble_fold=10,
            ...                                 min_ensemble_required=7,
            ...                                 grid_len_lon_upper_threshold=25,
            ...                                 grid_len_lon_lower_threshold=5,
            ...                                 grid_len_lat_upper_threshold=25,
            ...                                 grid_len_lat_lower_threshold=5,
            ...                                 points_lower_threshold=50,
            ...                                 Spatio1='longitude',
            ...                                 Spatio2 = 'latitude',
            ...                                 Temporal1 = 'DOY',
            ...                                 use_temporal_to_train=True)
            ... )

            >> ## fit
            >> model.fit(X_train.reset_index(drop=True), y_train)

            >> ## predict
            >> pred = model.predict(X_test)
            >> pred = np.where(pred<0, 0, pred)
            >> eval_metrics = AdaSTEM.eval_STEM_res('hurdle',y_test, pred_mean)
            >> print(eval_metrics)


            ```

        """

        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X_train: Union[pd.core.frame.DataFrame, np.ndarray], y_train: Sequence, verbosity: int = 1):
        """Fitting model
        Args:
            X_train:
                Training variables
            y_train:
                Training target
            verbosity:
                Whether to show progress bar. 0 for No, and Yes other wise.

        """
        binary_ = np.unique(np.where(y_train > 0, 1, 0))
        if len(binary_) == 1:
            warnings.warn("Warning: only one class presented. Replace with dummy classifier & regressor.")
            self.classifier = dummy_model1(binary_[0])
            self.regressor = dummy_model1(binary_[0])
            return

        X_train["y_train"] = y_train

        if verbosity == 0:
            self.classifier.fit(X_train.iloc[:, :-1], np.where(X_train.iloc[:, -1].values > 0, 1, 0), verbosity=0)
            self.regressor.fit(
                X_train[X_train["y_train"] > 0].iloc[:, :-1],
                np.array(X_train[X_train["y_train"] > 0].iloc[:, -1]),
                verbosity=0,
            )
        else:
            self.classifier.fit(X_train.iloc[:, :-1], np.where(X_train.iloc[:, -1].values > 0, 1, 0), verbosity=1)
            self.regressor.fit(
                X_train[X_train["y_train"] > 0].iloc[:, :-1],
                np.array(X_train[X_train["y_train"] > 0].iloc[:, -1]),
                verbosity=1,
            )

    def predict(
        self,
        X_test: Union[pd.core.frame.DataFrame, np.ndarray],
        njobs: int = 1,
        verbosity: int = 1,
        return_by_separate_ensembles: bool = False,
    ) -> np.ndarray:
        """Predict

        Args:
            X_test:
                Test variables
            njobs:
                Multi-processing in prediction.
            verbosity:
                Whether to show progress bar. 0 for No, and Yes other wise.
            return_by_separate_ensembles (bool, optional):
                Test function. return not by aggregation, but by separate ensembles.

        Returns:
            A prediction array with shape (-1,1)
        """
        if verbosity == 0:
            cls_res = self.classifier.predict(
                X_test, njobs=njobs, verbosity=0, return_by_separate_ensembles=return_by_separate_ensembles
            )
            reg_res = self.regressor.predict(
                X_test, njobs=njobs, verbosity=0, return_by_separate_ensembles=return_by_separate_ensembles
            )
        else:
            cls_res = self.classifier.predict(
                X_test, njobs=njobs, verbosity=1, return_by_separate_ensembles=return_by_separate_ensembles
            )
            reg_res = self.regressor.predict(
                X_test, njobs=njobs, verbosity=1, return_by_separate_ensembles=return_by_separate_ensembles
            )
        # reg_res = np.where(reg_res>=0, reg_res, 0) ### we constrain the reg value to be positive
        res = np.where(cls_res < 0.5, 0, cls_res)
        res = np.where(cls_res > 0.5, reg_res, cls_res)
        return res.reshape(-1, 1)

    def predict_proba(
        self,
        X_test: Union[pd.core.frame.DataFrame, np.ndarray],
        njobs: int = 1,
        verbosity: int = 0,
        return_by_separate_ensembles: bool = False,
    ) -> np.ndarray:
        """Just a rewrite of `predict` method

        Args:
            X_test:
                Testing variables
            njobs:
                Multi-processing in prediction.
            verbosity:
                Whether to show progress bar. 0 for No, and Yes other wise.
            return_by_separate_ensembles (bool, optional):
                Test function. return not by aggregation, but by separate ensembles.

        Returns:
            A prediction array with shape (-1,1)
        """

        return self.predict(
            self, X_test, njobs=njobs, verbosity=verbosity, return_by_separate_ensembles=return_by_separate_ensembles
        )
