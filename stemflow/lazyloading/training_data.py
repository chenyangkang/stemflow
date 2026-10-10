"""Read-only shared predictors and aligned, position-based training windows."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .open_db_connection import open_both_Xy_db_connection


INDEX = "__index_level_0__"


def quote_identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


def _batches(source, con, view, batch_size):
    if isinstance(source, pd.DataFrame):
        for start in range(0, len(source), batch_size):
            frame = source.iloc[start:start + batch_size]
            yield pd.DataFrame({INDEX: frame.index.to_numpy(),
                                **{c: frame[c].to_numpy() for c in frame}})
    else:
        con.register(view, source)
        reader = con.sql(f"SELECT * FROM {view}").fetch_record_batch(batch_size)
        for batch in reader:
            yield batch.to_pandas()


def _column(paths, name):
    location = paths[name]
    path, column = (location, None) if isinstance(location, str) else location
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return array if column is None else array[:, column]


def _files(paths):
    return {value if isinstance(value, str) else value[0] for value in paths.values()}


def _write_columns(source, con, view, directory, prefix, batch_size, separate):
    n = len(source) if isinstance(source, pd.DataFrame) else source.count('*').fetchone()[0]
    arrays, paths, groups, offset = {}, {}, {}, 0
    for frame in _batches(source, con, view, batch_size):
        for column in frame:
            values = frame[column].to_numpy()
            if not np.issubdtype(values.dtype, np.number) and values.dtype != np.bool_:
                raise TypeError(f"Shared training requires numeric columns; {column!r} has {values.dtype}.")
            if not groups:
                continue
            key, position = groups[column]
            target = arrays[key]
            if values.dtype != target.dtype:
                raise TypeError(f"Column {column!r} changes dtype across input batches.")
            if target.ndim == 1:
                target[offset:offset + len(frame)] = values
            else:
                target[offset:offset + len(frame), position] = values
        if not groups:
            dtype_groups = {}
            for column in frame:
                dtype = frame[column].to_numpy().dtype
                key = ("column", column) if column in separate else ("dtype", dtype.str)
                dtype_groups.setdefault(key, []).append(column)
            for number, (key, names) in enumerate(dtype_groups.items()):
                path = str(Path(directory) / f"{prefix}_{number}.npy")
                shape = (n,) if len(names) == 1 else (n, len(names))
                arrays[key] = np.lib.format.open_memmap(path, mode="w+",
                    dtype=frame[names[0]].to_numpy().dtype, shape=shape)
                for position, column in enumerate(names):
                    groups[column] = (key, position)
                    paths[column] = path if len(names) == 1 else (path, position)
                    if len(names) == 1:
                        arrays[key][:len(frame)] = frame[column].to_numpy()
                    else:
                        arrays[key][:len(frame), position] = frame[column].to_numpy()
            # Preserve original feature order, independently of dtype grouping.
            paths = {column: paths[column] for column in frame}
        offset += len(frame)
    if offset != n or INDEX not in paths:
        raise ValueError("Cannot build shared training data from empty or inconsistent input.")
    for array in arrays.values():
        array.flush()
    arrays.clear()
    return paths, n


@dataclass
class SharedTrainingData:
    """Only filenames cross process boundaries; workers map arrays read-only.

    Original numeric dtypes, observation IDs, and source row order are retained.
    Estimators receive ordinary DataFrames containing just their stixel rows.
    """

    paths: dict
    label_path: str
    n_rows: int
    nbytes: int
    _arrays: dict = field(default_factory=dict, repr=False)
    _labels: object = field(default=None, repr=False)

    def __getstate__(self):
        return {**self.__dict__, "_arrays": {}, "_labels": None}

    def array(self, name):
        if name not in self._arrays:
            self._arrays[name] = _column(self.paths, name)
        return self._arrays[name]

    @property
    def labels(self):
        if self._labels is None:
            self._labels = np.load(self.label_path, mmap_mode="r", allow_pickle=False)
        return self._labels

    @property
    def index(self):
        return self.array(INDEX)

    def frame(self, positions, columns=None):
        columns = [c for c in self.paths if c != INDEX] if columns is None else columns
        selected, values = {}, {}
        for column in columns:
            location = self.paths[column]
            if isinstance(location, str):
                values[column] = self.array(column)[positions]
            else:
                path, position = location
                if path not in selected:
                    if path not in self._arrays:
                        self._arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
                    selected[path] = self._arrays[path][positions]
                values[column] = selected[path][:, position]
        return pd.DataFrame(values,
                            index=pd.Index(self.index[positions], name=INDEX), copy=False)

    def training_rows(self, positions):
        frame = self.frame(positions)
        frame["true_y"] = self.labels[positions]
        return frame

    @classmethod
    def build(cls, X, y, config, directory, batch_size=65536,
              coordinate_columns=("DOY", "longitude", "latitude")):
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open_both_Xy_db_connection(X, y, config) as (x_source, y_source, con):
            separate = {INDEX, *coordinate_columns}
            x_paths, n = _write_columns(x_source, con, "shared_x", directory, "x", batch_size, separate)
            y_paths, yn = _write_columns(y_source, con, "shared_y", directory, "y", batch_size, {INDEX})
        label_columns = [c for c in y_paths if c != INDEX]
        if n != yn or len(label_columns) != 1:
            raise ValueError("Shared training requires matching X/y sizes and one target column.")
        x_ids = _column(x_paths, INDEX)
        y_ids = _column(y_paths, INDEX)
        labels = _column(y_paths, label_columns[0])
        indexer = pd.Index(y_ids).get_indexer(x_ids)
        if np.any(indexer < 0):
            raise ValueError("X and y observation IDs do not match.")
        label_path = str(Path(directory) / "labels.npy")
        aligned = np.lib.format.open_memmap(label_path, mode="w+", dtype=labels.dtype, shape=(n,))
        for start in range(0, n, batch_size):
            aligned[start:start + batch_size] = labels[indexer[start:start + batch_size]]
        aligned.flush()
        del aligned, labels, x_ids, y_ids, indexer
        for path in _files(y_paths):
            Path(path).unlink()
        nbytes = sum(Path(p).stat().st_size for p in [*_files(x_paths), label_path])
        return cls(x_paths, label_path, n, nbytes)


class TrainingWindow:
    """A unique coordinate table plus optional bootstrap occurrence positions."""

    def __init__(self, coordinates, fetch_rows, bootstrap_ids=None):
        if not coordinates.index.is_unique or coordinates.index.hasnans:
            raise ValueError("Temporal-window observation IDs must be unique and non-null.")
        self.coordinates = coordinates
        self.fetch_rows = fetch_rows
        self.sample_positions = None
        if bootstrap_ids is not None:
            positions = coordinates.index.get_indexer(bootstrap_ids)
            self.sample_positions = positions[positions >= 0]

    def select(self, mask):
        positions = np.flatnonzero(mask) if self.sample_positions is None else (
            self.sample_positions[np.asarray(mask)[self.sample_positions]])
        return self.fetch_rows(positions)


def native_window(x_source, y_source, con, start, end, columns, prequery, bootstrap_ids):
    temporal, spatial1, spatial2 = columns
    if isinstance(x_source, pd.DataFrame):
        coordinates = x_source.loc[(x_source[temporal] >= start) &
                                   (x_source[temporal] < end), columns].copy()
        def fetch(positions):
            ids = coordinates.index[positions]
            frame = x_source.loc[ids].copy()
            frame["true_y"] = y_source.loc[ids].iloc[:, 0].to_numpy()
            return frame
        return TrainingWindow(coordinates, fetch, bootstrap_ids)

    con.register("window_x_source", x_source)
    con.register("window_y_source", y_source)
    temporal = quote_identifier(temporal)
    condition = f"{temporal} >= {float(start)!r} AND {temporal} < {float(end)!r}"
    cached = None
    if prequery:
        # One wide query per window; X/y coordinates use this same extraction.
        cached = con.sql(f"SELECT * FROM window_x_source WHERE {condition}").df().set_index(INDEX)
        coordinates = cached[columns].copy()
    else:
        names = ', '.join(quote_identifier(c) for c in [INDEX, *columns])
        coordinates = con.sql(f"SELECT {names} FROM window_x_source WHERE {condition}").df().set_index(INDEX)
    target = [c for c in y_source.columns if c != INDEX]
    if len(target) != 1:
        raise ValueError("Expected one target column.")
    if prequery:
        request = pd.DataFrame({INDEX: coordinates.index})
        con.register("window_request", request)
        labels = con.sql(f"SELECT y.* FROM window_y_source y SEMI JOIN window_request r USING ({INDEX})").df().set_index(INDEX)
        labels = labels.reindex(coordinates.index)
        if labels.isna().any().any():
            raise ValueError("Missing labels in temporal window.")

    def fetch(positions):
        ids = coordinates.index[positions]
        if cached is not None:
            frame = cached.loc[ids].copy()
            frame["true_y"] = labels.iloc[positions, 0].to_numpy()
        else:
            request = pd.DataFrame({INDEX: ids, "__sample_position__": np.arange(len(ids))})
            con.register("stixel_request", request)
            frame = con.sql(f"""
                SELECT x.*, y.{quote_identifier(target[0])} AS true_y
                FROM stixel_request r JOIN window_x_source x USING ({INDEX})
                JOIN window_y_source y USING ({INDEX}) ORDER BY r.__sample_position__
            """).df().set_index(INDEX)
        return frame
    return TrainingWindow(coordinates, fetch, bootstrap_ids)


def prediction_window(source, con, start, end, columns, prequery):
    """Extract coordinates and optional covariates with one temporal query."""
    temporal = columns[0]
    if isinstance(source, pd.DataFrame):
        coordinates = source.loc[(source[temporal] >= start) & (source[temporal] < end), columns].copy()
        return TrainingWindow(coordinates, lambda positions: source.loc[coordinates.index[positions]].copy())
    con.register("prediction_source", source)
    condition = f"{quote_identifier(temporal)} >= {float(start)!r} AND {quote_identifier(temporal)} < {float(end)!r}"
    names = "*" if prequery else ', '.join(quote_identifier(c) for c in [INDEX, *columns])
    frame = con.sql(f"SELECT {names} FROM prediction_source WHERE {condition}").df().set_index(INDEX)
    coordinates = frame[columns].copy() if prequery else frame
    def fetch(positions):
        if prequery:
            return frame.iloc[positions].copy()
        request = pd.DataFrame({INDEX: coordinates.index[positions], "__sample_position__": np.arange(len(positions))})
        con.register("prediction_request", request)
        return con.sql(f"""SELECT x.* FROM prediction_request r
            JOIN prediction_source x USING ({INDEX}) ORDER BY r.__sample_position__""").df().set_index(INDEX)
    return TrainingWindow(coordinates, fetch)
