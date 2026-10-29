"""Pandas-to-DuckDB helpers preserve rows and safely manage their files."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from stemflow.lazyloading import make_duckdb_database, make_training_databases, training_databases
from .test_execution_rewrite import make_model, sample


def read(path):
    with duckdb.connect(path, read_only=True) as con:
        return con.sql("SELECT * FROM tmp").df().set_index("__index_level_0__")


def test_pair_aligns_shuffled_named_labels_and_preserves_inputs(sample, tmp_path):
    X, y = sample
    X = X.iloc[::-1]
    original = X.copy(deep=True)
    xs, ys = make_training_databases(X, y.occ.rename("named_target"), tmp_path)
    pd.testing.assert_frame_equal(read(xs).rename_axis(X.index.name), X)
    assert_allclose(read(ys).y, y.reindex(X.index).occ)
    pd.testing.assert_frame_equal(X, original)
    with pytest.raises(FileExistsError):
        make_training_databases(X, y, tmp_path)


@pytest.mark.parametrize("backend", ["native", "memmap"])
def test_prepared_databases_match_pandas_predictions(sample, tmp_path, backend):
    X, y = sample
    X["binary_feature"] = X.env > 0
    reference = make_model().fit(X, y)
    with training_databases(X, y.occ.to_numpy(), directory=tmp_path) as (xs, ys):
        model = make_model(n_jobs=2).fit(xs, ys, training_data_backend=backend,
                                       temporal_window_prequery=True)
        expected = reference.predict_proba(X, return_by_separate_ensembles=True)
        assert_allclose(model.predict_proba(X, return_by_separate_ensembles=True), expected, equal_nan=True)
    assert not Path(xs).exists()
    assert not Path(ys).exists()


def test_database_directory_with_apostrophe(sample, tmp_path):
    X, y = sample
    with training_databases(X, y, directory=tmp_path) as (xs, ys):
        destination = tmp_path / "bird's inputs"
        destination.mkdir()
        new_xs, new_ys = [str(Path(path).rename(destination / Path(path).name)) for path in (xs, ys)]
        model = make_model().fit(new_xs, new_ys, temporal_window_prequery=True)
        assert np.isfinite(model.predict_proba(new_xs)).any()


def test_context_cleans_up_after_error(sample, tmp_path):
    X, y = sample
    with pytest.raises(RuntimeError):
        with training_databases(X, y, directory=tmp_path) as (xs, ys):
            raise RuntimeError("deliberate")
    assert not Path(xs).exists()
    assert not list(tmp_path.iterdir())


def test_context_does_not_retain_callers_dataframes(sample, tmp_path):
    import gc
    import weakref
    X, y = [frame.copy() for frame in sample]
    references = [weakref.ref(X), weakref.ref(y)]
    with training_databases(X, y, directory=tmp_path):
        del X, y
        gc.collect()
        assert all(reference() is None for reference in references)


def test_prediction_database_and_validation(sample, tmp_path):
    X, y = sample
    path = make_duckdb_database(X, tmp_path / "predict.duckdb")
    pd.testing.assert_frame_equal(read(path).rename_axis(X.index.name), X)
    with pytest.raises(FileExistsError):
        make_duckdb_database(X, path)
    make_duckdb_database(X, path, overwrite=True)
    with pytest.raises(ValueError, match="observation IDs"):
        make_training_databases(X, y.reset_index(drop=True), tmp_path / "mismatch")
    with pytest.raises(ValueError, match="missing"):
        make_duckdb_database(X.assign(env=np.nan), tmp_path / "missing.duckdb")
    with pytest.raises(ValueError, match="numeric"):
        make_duckdb_database(X.rename(index=lambda i: str(i)), tmp_path / "string.duckdb")
