import numpy as np
import pandas as pd

from stemflow.model.AdaSTEM import AdaSTEM
from stemflow.model_selection import ST_train_test_split

from .make_models import (
    make_AdaSTEMClassifier_caliP,
    make_SphereAdaClassifier_caliP,
    make_STEMClassifier_caliP,
)
from .set_up_data import set_up_data

x_names, (X, y) = set_up_data()
X_train, X_test, y_train, y_test = ST_train_test_split(
    X, y, Spatio_blocks_count=100, Temporal_blocks_count=100, random_state=42, test_size=0.3
)


def test_STEMClassifier_caliP():
    model = make_STEMClassifier_caliP()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1, logit_agg=True)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.5

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)

def test_AdaSTEMClassifier_caliP():
    model = make_AdaSTEMClassifier_caliP()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1, logit_agg=True)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.5

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)

def test_SphereAdaClassifier_caliP():
    model = make_SphereAdaClassifier_caliP()
    model = model.fit(X_train, np.where(y_train > 0, 1, 0))

    pred_mean, pred_std = model.predict(X_test.reset_index(drop=True), return_std=True, verbosity=1, n_jobs=1, logit_agg=True)
    assert np.sum(~np.isnan(pred_mean)) > 0
    assert np.sum(~np.isnan(pred_std)) > 0

    pred = model.predict(X_test)
    assert len(pred) == len(X_test)
    assert np.sum(np.isnan(pred)) / len(pred) <= 0.5

    pred_df = pd.DataFrame(
        {"y_true": y_test.flatten(), "y_pred": np.where(pred.flatten() < 0, 0, pred.flatten())}
    ).dropna()
    assert len(pred_df) > 0

    eval = AdaSTEM.eval_STEM_res("classification", pred_df.y_true, pred_df.y_pred)
