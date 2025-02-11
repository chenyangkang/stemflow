import numpy as np
import pandas as pd

from stemflow.model.AdaSTEM import AdaSTEM
from stemflow.model_selection import ST_train_test_split
from xgboost import XGBClassifier, XGBRegressor

from .make_models import (
    make_AdaSTEMClassifier,
    make_AdaSTEMClassifier_custom_pred_method
)
from .set_up_data import set_up_data

x_names, (X, y) = set_up_data()
X_train, X_test, y_train, y_test = ST_train_test_split(
    X, y, Spatio_blocks_count=100, Temporal_blocks_count=100, random_state=42, test_size=0.3
)
def test_AdaSTEMClassifier():
    
    class my_base_model:
        def __init__(self):
            self.model = XGBClassifier(tree_method="hist", random_state=42, verbosity=0, n_jobs=1)
            pass
        def fit(self, X_train, y_train):
            self.model.fit(X_train, y_train)
            return self
        def predict(self, X_test):
            return self.model.predict(X_test)
        def predict_proba(self, X_test):
            return self.model.predict_proba(X_test)
        def special_predict(self, X_test):
            # Fold change
            pred1 = self.model.predict_proba(X_test)[:,1]
            pred2 = self.model.predict_proba(X_test + np.random.normal(loc=0, scale=1, size=X_test.shape))[:,1]
            # Interaction
            i_ = np.log(np.clip(1e-6, 1-1e-6, pred1) / np.clip(1e-6, 1-1e-6, pred2))
            pred = i_ # Should be -inf to inf, 0 as no interaction
            return pred
        
    model = make_AdaSTEMClassifier_custom_pred_method(base_model_class=my_base_model)
    # model = make_AdaSTEMClassifier()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean = model.predict_proba(X_test.reset_index(drop=True), return_std=False, verbosity=1, n_jobs=1, base_model_method='special_predict')[:,1]
    pred_mean = pred_mean[~np.isnan(pred_mean)]
    
    # print(pred_mean)
    # assert(np.sum((pred_mean < 1000) & (pred_mean > 1001)))

