"""Exercise an installed distribution outside the repository checkout."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--forbid-source", type=Path)
    parser.add_argument("--with-extras", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    import stemflow
    from stemflow.calibration import BetaCalibrator
    from stemflow.lazyloading.data_preparation import training_databases
    from stemflow.model.AdaSTEM import AdaSTEMClassifier

    assert stemflow.__version__ == version("stemflow") == args.expected_version
    if args.forbid_source:
        assert args.forbid_source.resolve() not in Path(stemflow.__file__).resolve().parents
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"longitude": rng.uniform(-100, -90, 800),
                      "latitude": rng.uniform(30, 40, 800),
                      "DOY": rng.uniform(1, 40, 800),
                      "habitat": rng.normal(size=800)})
    y = rng.binomial(1, 1 / (1 + np.exp(1 - X.habitat.to_numpy())))
    with tempfile.TemporaryDirectory(prefix="stemflow_install_") as scratch:
        model = AdaSTEMClassifier(
            base_model=DecisionTreeClassifier(max_depth=3, random_state=42),
            ensemble_fold=2, min_ensemble_required=1, points_lower_threshold=30,
            grid_len_upper_threshold=25, grid_len_lower_threshold=5,
            temporal_start=1, temporal_end=41, temporal_step=40,
            temporal_bin_interval=40, temporal_bin_start_jitter=0,
            sample_weights_for_classifier=False, ensemble_bootstrap=True,
            save_gridding_plot=False, n_jobs=2, random_state=42,
            lazy_loading=True, lazy_loading_dir=str(Path(scratch) / "models"), verbosity=0)
        with training_databases(X, y, directory=scratch) as (xp, yp):
            model.fit(xp, yp, temporal_window_prequery=True,
                      training_data_backend="memmap", inner_max_num_threads=1,
                      persist_prediction_state=False)
        native = model.predict_proba(X, temporal_window_prequery=True)
        mapped = model.predict_proba(X, temporal_window_prequery=True,
                                     prediction_backend="memmap")
        np.testing.assert_allclose(native, mapped, atol=1e-12, equal_nan=True)
        assert np.isfinite(mapped[:, 1]).any()
        valid = np.isfinite(native[:, 1])
        calibrated = BetaCalibrator(ridge=100).fit(np.clip(native[valid, 1], 1e-8, 1 - 1e-8), y[valid])
        assert np.isfinite(calibrated.predict([0.1, 0.5, 0.9])).all()
        if args.with_extras:
            from stemflow.tuning import local_beta_xgboost
            recipe = local_beta_xgboost(
                tuning=True, n_trials=1, n_estimators=6, tuning_n_estimators=4,
                n_splits=2, tuning_n_splits=2, tuning_verbose=False)
            recipe.fit(X, y)
            assert np.isfinite(recipe.predict_proba(X)).all()
            with model.monitor_execution(sample_interval=0.1, log_interval=None):
                model.predict_proba(X, temporal_window_prequery=True, prediction_backend="memmap")
            assert model.execution_monitor_["sampling_error"] is None
            assert model.execution_monitor_["peak_tree_rss_bytes"] > 0
        del model
    print(json.dumps(dict(python=platform.python_version(), stemflow=stemflow.__version__,
                          location=stemflow.__file__, extras=args.with_extras)))
    print("Installed-distribution execution checks passed.")


if __name__ == "__main__":
    main()
