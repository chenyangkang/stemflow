"""Create model-ready DuckDB inputs from familiar Pandas objects."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

import duckdb
import numpy as np
import pandas as pd

from .training_data import INDEX


def _validate_frame(frame):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("The DataFrame must be non-empty with unique column names.")
    if INDEX in frame:
        raise ValueError(f"{INDEX!r} is reserved. Use frame.set_index({INDEX!r}) first.")
    if (isinstance(frame.index, pd.MultiIndex) or not frame.index.is_unique or
            frame.index.hasnans or not pd.api.types.is_numeric_dtype(frame.index)):
        raise ValueError("Observation IDs must be a unique, non-null numeric DataFrame index.")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise TypeError("Model inputs must be numeric; encode categorical features first.")
    if any(frame[column].isna().any() for column in frame):
        raise ValueError("Model inputs contain missing values; fill or remove them first.")


def _aligned_target(X, y):
    if isinstance(y, pd.Series):
        y = y.to_frame()
    if isinstance(y, pd.DataFrame):
        if len(y.columns) != 1 or not y.index.is_unique:
            raise ValueError("y must have one target column and unique observation IDs.")
        if len(y) != len(X) or not X.index.isin(y.index).all():
            raise ValueError("X and y observation IDs must match. Pass a NumPy array for positional labels.")
        target = y.reindex(X.index).copy(deep=False)
        target.columns = ["y"]
    else:
        values = np.asarray(y)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or len(values) != len(X):
            raise ValueError("Positional y must contain one value per row of X.")
        target = pd.DataFrame({"y": values}, index=X.index)
    _validate_frame(target)
    return target


def _write_database(frame, path, memory_limit):
    # A shallow copy adds IDs without duplicating all predictor columns.
    stored = frame.copy(deep=False)
    for column in stored:
        values = stored[column].to_numpy(copy=False)
        if any(stride < 0 for stride in values.strides):
            stored[column] = values.copy()
    ids = frame.index.to_numpy(copy=False)
    stored[INDEX] = ids.copy() if any(stride < 0 for stride in ids.strides) else ids
    with duckdb.connect(str(path), config={"threads": "1", "memory_limit": memory_limit}) as con:
        con.register("input_frame", stored)
        con.execute("CREATE TABLE tmp AS SELECT * FROM input_frame")


def make_duckdb_database(frame, path, *, memory_limit="500MB", overwrite=False):
    """Save a numeric DataFrame as one model-ready DuckDB table.

    Observation IDs come from ``frame.index`` and row order is retained.
    No SQL knowledge, manual ID column, or SQL index creation is needed.
    Returns an absolute filename suitable for ``split``/``predict``.
    Existing files are protected unless ``overwrite=True``.
    The caller's DataFrame is not modified or freed.
    """
    _validate_frame(frame)
    path = Path(path).expanduser().resolve()
    if path.suffix != ".duckdb":
        raise ValueError("The destination must end in .duckdb.")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stemflow_write_", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        _write_database(frame, staged, memory_limit)
        staged.replace(path)
    return str(path)


def make_training_databases(X, y, directory, *, memory_limit="500MB", overwrite=False):
    """Create paired X/y DuckDB files and return ``(X_path, y_path)``.

    Pandas labels align by observation ID, including shuffled labels.
    NumPy/list labels align by position. Both database files contain one
    ``tmp`` table and the ID column required by AdaSTEM. They are closed
    before returning, so parallel workers can open them read-only.

    Use ``training_databases`` for automatic temporary-file cleanup.
    """
    _validate_frame(X)
    target = _aligned_target(X, y)
    directory = Path(directory).expanduser().resolve()
    paths = [directory / "X.duckdb", directory / "y.duckdb"]
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(path)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stemflow_write_", dir=directory) as temporary:
        staged = [Path(temporary) / path.name for path in paths]
        for frame, path in zip([X, target], staged):
            _write_database(frame, path, memory_limit)
        for source, destination in zip(staged, paths):
            source.replace(destination)
    return tuple(str(path) for path in paths)


@contextmanager
def training_databases(X, y, *, directory=None, memory_limit="500MB"):
    """Yield temporary ``(X_path, y_path)`` and clean up on success/failure.

    ``directory`` optionally selects node-local scratch, such as
    ``os.environ.get('SLURM_TMPDIR')``. Each context gets its own folder.
    Keep the context open until ``fit`` and other input reads complete.
    """
    with tempfile.TemporaryDirectory(prefix="stemflow_training_", dir=directory) as temporary:
        paths = make_training_databases(X, y, temporary, memory_limit=memory_limit)
        del X, y
        yield paths
