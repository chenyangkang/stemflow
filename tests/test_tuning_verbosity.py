"""Optuna logging controls must preserve search results and survive pickling."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from stemflow.tuning import SelfTuningClassifier, SelfTuningRegressor, local_beta_xgboost

optuna = pytest.importorskip("optuna")


@pytest.mark.parametrize("regression", [False, True])
def test_optuna_verbosity_preserves_search_predictions_and_level(regression, capsys, tmp_path):
    rng = np.random.default_rng(32)
    X = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    y = X.a.to_numpy() + rng.normal(0, .2, len(X))
    if regression:
        cls, estimator, cv = SelfTuningRegressor, DecisionTreeRegressor(random_state=42), KFold(2)
    else:
        y = (y > 0).astype(int)
        cls, estimator, cv = SelfTuningClassifier, DecisionTreeClassifier(random_state=42), StratifiedKFold(2)
    original = optuna.logging.get_verbosity()
    logger = logging.getLogger("optuna")
    # Capture through this process's stderr, independent of Optuna import time.
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    try:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        models = []
        for verbose in (False, True):
            model = cls(estimator, {"max_depth": [1, 2, 3]}, cv=cv, n_trials=3,
                        search="optuna", verbose=verbose)
            model = clone(model).fit(X, y)
            assert optuna.logging.get_verbosity() == optuna.logging.ERROR
            captured = capsys.readouterr().err
            assert ("A new study created" in captured) is verbose
            assert ("Trial 0 finished" in captured) is verbose
            filename = tmp_path / f"model_{verbose}.joblib"
            joblib.dump(model, filename)
            models.append(joblib.load(filename))
        assert models[0].best_params_ == models[1].best_params_
        assert models[0].trials_ == models[1].trials_
        method = "predict" if regression else "predict_proba"
        np.testing.assert_array_equal(getattr(models[0], method)(X), getattr(models[1], method)(X))
    finally:
        logger.removeHandler(handler)
        optuna.logging.set_verbosity(original)


def test_quiet_search_keeps_failure_warning_and_restores_logging(capsys):
    X = pd.DataFrame({"a": np.arange(20)})
    y = np.arange(20) % 2
    logger = logging.getLogger("optuna")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    previous = optuna.logging.get_verbosity()
    try:
        tuner = SelfTuningClassifier(DecisionTreeClassifier(), {"max_depth": [-1]},
                                    cv=2, n_trials=1, search="optuna", verbose=False)
        with pytest.raises(ValueError):
            tuner.fit(X, y)
        messages = capsys.readouterr().err
        assert "A new study created" not in messages
        assert "Trial 0 failed" in messages
        assert optuna.logging.get_verbosity() == previous
    finally:
        logger.removeHandler(handler)
        optuna.logging.set_verbosity(previous)


def test_factory_forwards_tuning_verbosity():
    pytest.importorskip("xgboost")
    for verbose in (False, True):
        model = clone(local_beta_xgboost(tuning_verbose=verbose))
        assert model.estimator.verbose is verbose
    assert local_beta_xgboost().estimator.verbose is False


def _fit_dump_reload_in_worker(verbose, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    X = pd.DataFrame({"a": np.arange(40), "b": np.arange(40) % 3})
    y = np.arange(40) % 2
    model = SelfTuningClassifier(DecisionTreeClassifier(random_state=42),
        {"max_depth": [1, 2]}, cv=2, n_trials=1, search="optuna", verbose=verbose).fit(X, y)
    path = Path(directory) / f"worker_{verbose}.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    return loaded.predict_proba(X), loaded.best_params_


def test_quiet_tuner_parallel_dump_and_reload(tmp_path, capfd):
    results = joblib.Parallel(n_jobs=2, backend="loky")(
        joblib.delayed(_fit_dump_reload_in_worker)(False, tmp_path / str(i)) for i in range(2))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
    messages = capfd.readouterr()
    assert "A new study created" not in messages.err + messages.out
    assert "Trial 0 finished" not in messages.err + messages.out
