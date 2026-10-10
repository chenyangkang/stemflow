"""Compare modular local beta against the supplied standalone paper helper."""

import argparse
import importlib.util
from importlib.machinery import SourceFileLoader
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit

from stemflow.tuning import local_beta_xgboost


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-helper", type=Path, required=True)
    parser.add_argument("--extra-site-packages", type=Path)
    parser.add_argument("--output", type=Path,
        default=ROOT/"data/02_calibration/reference_equivalence.json")
    args = parser.parse_args()
    if args.extra_site_packages is not None:
        sys.path.append(str(args.extra_site_packages.resolve()))
    loader = SourceFileLoader("reference_local_beta", str(args.reference_helper))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(42)
    X = pd.DataFrame(dict(signal=rng.normal(size=1400), DOY=rng.integers(1, 367, 1400),
        longitude_tmp=rng.uniform(-100, -70, 1400), latitude_tmp=rng.uniform(25, 50, 1400)))
    y = rng.binomial(1, expit(-2 + X.signal.to_numpy()))
    records = []
    for tuning in [False, True]:
        options = dict(tuning=tuning, ridge=100, n_trials=4, n_estimators=20,
                       tuning_n_estimators=10, random_state=42)
        started = time.perf_counter()
        reference = module.TunedLocalBetaXGBClassifier(**options).fit(X.iloc[:1000], y[:1000])
        original_seconds = time.perf_counter()-started
        started = time.perf_counter()
        rewritten = local_beta_xgboost(**options, Spatio1="longitude_tmp",
            Spatio2="latitude_tmp").fit(X.iloc[:1000], y[:1000])
        rewrite_seconds = time.perf_counter()-started
        a, b = reference.predict_proba(X.iloc[1000:]), rewritten.predict_proba(X.iloc[1000:])
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(reference.beta_.theta_, rewritten.calibrator_.theta_, atol=1e-12)
        old_params = [v["best_params"] for v in reference.tuning_audit_]
        new_params = [getattr(v, "best_params_", None) for v in rewritten.fold_models_]
        if tuning:
            assert old_params == new_params
        records.append(dict(tuning=tuning, training_rows=1000, independent_prediction_rows=400,
            original_seconds=original_seconds, rewrite_seconds=rewrite_seconds,
            maximum_probability_difference=float(np.max(np.abs(a-b))),
            maximum_beta_parameter_difference=float(np.max(np.abs(
                reference.beta_.theta_-rewritten.calibrator_.theta_))),
            selected_parameters_match=old_params == new_params, selected_parameters=new_params))
    import optuna
    report = dict(results=records, optuna_version=optuna.__version__,
        note="Numerical equivalence probe, not a calibration-quality or copula-recovery experiment.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
