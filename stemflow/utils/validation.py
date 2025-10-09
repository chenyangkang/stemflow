"Validation module. Most of these functions are plain checking and easy to understand."

import warnings
from typing import Union

import numpy as np
import pandas as pd
import string
import tempfile
import shutil
import os
from pathlib import Path
import duckdb
import re
from .generate_random import generate_random_saving_code

def check_random_state(seed: Union[None, int, np.random._generator.Generator]) -> np.random._generator.Generator:
    """Turn seed into a np.random.RandomState instance.

    Args:
        seed:
            If seed is None, return a random generator.
            If seed is an int, return a random generator with that seed.
            If seed is already a random generator instance, return it.
            Otherwise raise ValueError.

    Returns:
        The random generator object based on `seed` parameter.
    """
    if seed is None:
        return np.random.default_rng(np.random.randint(0, 2**32 - 1))
    if isinstance(seed, int):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random._generator.Generator):
        return seed
    raise ValueError("%r cannot be used to seed a np.random.default_rng instance" % seed)


def check_task(task):
    if task not in ["regression", "classification", "hurdle"]:
        raise AttributeError(f"task type must be one of 'regression', 'classification', or 'hurdle'! Now it is {task}")
    if task == "hurdle":
        warnings.warn(
            "You have chosen HURDLE task. The goal is to first conduct classification, and then apply regression on points with *positive values*"
        )


def check_base_model(base_model):
    for func in ["fit", "predict"]:
        if func not in dir(base_model):
            raise AttributeError(f"input base model must have method '{func}'!")


def check_transform_n_jobs(self, n_jobs):
    if n_jobs is None:
        if self.n_jobs is None:
            warnings.warn("No n_jobs input. Default to 1.")
            return 1
        else:
            return self.n_jobs
    else:
        if not isinstance(n_jobs, int):
            raise TypeError(f"n_jobs is not a integer. Got {n_jobs}.")
        else:
            if n_jobs == 0:
                raise ValueError("n_jobs cannot be 0!")
            elif n_jobs > self.ensemble_fold:
                raise ValueError(f"n_jobs ({n_jobs}) is larger than ensemble_fold ({self.ensemble_fold}). The computational resources are redundant since the algorithm is paralleled by ensemble_fold. Consider using a n_jobs value that is smaller than self.ensemble_fold.")
            else:
                return n_jobs


def check_verbosity(self, verbosity):
    if verbosity is None:
        verbosity = self.verbosity
    elif verbosity == 0:
        verbosity = 0
    else:
        verbosity = 1
    return verbosity


def check_spatio_bin_jitter_magnitude(spatio_bin_jitter_magnitude):
    if isinstance(spatio_bin_jitter_magnitude, (int, float)):
        pass
    elif isinstance(spatio_bin_jitter_magnitude, str):
        if spatio_bin_jitter_magnitude == "adaptive":
            pass
        else:
            raise ValueError("spatio_bin_jitter_magnitude string must be adaptive!")
    else:
        raise ValueError("spatio_bin_jitter_magnitude string must be one of [int, float, 'adaptive']!")


def check_transform_spatio_bin_jitter_magnitude(spatial1_max, spatial1_min, spatial2_max, spatial2_min, 
                                                spatio_bin_jitter_magnitude):
    check_spatio_bin_jitter_magnitude(spatio_bin_jitter_magnitude)
    if isinstance(spatio_bin_jitter_magnitude, str):
        if spatio_bin_jitter_magnitude == "adaptive":
            jit = max(spatial1_max - spatial1_min, spatial2_max - spatial2_min)
            return jit
        
    return spatio_bin_jitter_magnitude


def check_temporal_bin_start_jitter(temporal_bin_start_jitter):
    # validate temporal_bin_start_jitter
    if not isinstance(temporal_bin_start_jitter, (str, float, int)):
        raise AttributeError(
            f"Input temporal_bin_start_jitter should be 'adaptive', float or int, got {type(temporal_bin_start_jitter)}"
        )
    if isinstance(temporal_bin_start_jitter, str):
        if not temporal_bin_start_jitter == "adaptive":
            raise AttributeError(
                f"The input temporal_bin_start_jitter as string should only be 'adaptive'. Other options include float or int. Got {temporal_bin_start_jitter}"
            )


def check_transform_temporal_bin_start_jitter(temporal_bin_start_jitter, bin_interval, rng):
    check_temporal_bin_start_jitter(temporal_bin_start_jitter)
    if isinstance(temporal_bin_start_jitter, str):
        if temporal_bin_start_jitter == "adaptive":
            jit = rng.uniform(low=0, high=bin_interval)
    elif type(temporal_bin_start_jitter) in [int, float]:
        jit = temporal_bin_start_jitter

    return jit


def check_X_train(X_train, self, index_col='__index_level_0__'):
    

    _NUMERIC = {"TINYINT","SMALLINT","INTEGER","BIGINT","HUGEINT",
                "UTINYINT","USMALLINT","UINTEGER","UBIGINT",
                "FLOAT","DOUBLE","DECIMAL"}
                    
    # check type
    if isinstance(X_train, pd.DataFrame):
        if np.sum(np.isnan(np.array(X_train))) > 0:
            raise ValueError(
                "NAs (missing values) detected in input data. stemflow do not support NAs input. Consider filling them with values (e.g., -1 or mean values) or removing the rows."
            )
        if isinstance(X_train.index, pd.MultiIndex):
            raise ValueError("Index must be single-level, not MultiIndex.")
        if not X_train.index.is_unique:
            raise ValueError("Index values must be unique.")
        if not pd.api.types.is_numeric_dtype(X_train.index):
            raise ValueError("Index must be numeric (int or float).")
        
        warnings.warn('Input X is pandas dataframe; self.max_mem is not used.')
        
    elif isinstance(X_train, str):
        if not (X_train.endswith('.duckdb') or X_train.endswith('.parquet')):
            raise ValueError(f"If the input for X is a string, it has to be the path to a database/file that ends with either .duckdb or .parquet; Got {X_train}")

        if X_train.endswith('.duckdb'):
            con = duckdb.connect(X_train, read_only=True, config=self.duckdb_config)
            try:
                all_tables = con.sql("""SHOW TABLES""").fetchall()
                if len(all_tables) > 1:
                    raise AttributeError(f'If the input X is a path to a .duckdb database, the database should only contain one table. X and y should be put in different database files. Got {all_tables}')
                elif len(all_tables) == 0:
                    raise AttributeError(f'No file found in the input database {X_train}!')
                else:
                    the_only_table_name = all_tables[0][0]
                    # type check
                    schema = con.sql(f'DESCRIBE SELECT * FROM "{the_only_table_name}"').fetchall()
                    name2type = {row[0]: row[1] for row in schema}
                    if index_col not in name2type:
                        raise ValueError(f"index_col '{index_col}' not found in table '{the_only_table_name}'.")
                    if str(name2type[index_col]).upper() not in _NUMERIC:
                        raise ValueError(f"Index '{index_col}' must be numeric; found {name2type[index_col]}.")
                    # NAs anywhere?
                    cols = [row[0] for row in schema]
                    na = con.sql(f'SELECT COUNT(*) FROM "{the_only_table_name}" WHERE NOT (' + " AND ".join(f'"{c}" IS NOT NULL' for c in cols) + ")").fetchone()[0]
                    if na > 0:
                        raise ValueError(f"NAs detected in table '{the_only_table_name}' ({na} row(s)).")
                    # unique + non-null index
                    indexes_are_unique = con.sql(f'''
                        SELECT 
                            COUNT(*) = COUNT(DISTINCT "{index_col}") 
                        FROM "{the_only_table_name}"
                    ''').fetchone()[0]
                    if not indexes_are_unique:
                        raise ValueError(f"Index '{index_col}' is not unique.")
                    
                    # con.sql(f"""
                    #         CREATE INDEX IF NOT EXISTS idx_idx ON {the_only_table_name} ({index_col});
                    #         CREATE INDEX IF NOT EXISTS idx_spatial1 ON {the_only_table_name} ({self.Spatio1});
                    #         CREATE INDEX IF NOT EXISTS idx_spatial2 ON {the_only_table_name} ({self.Spatio2});
                    #         CREATE INDEX IF NOT EXISTS idx_temporal1 ON {the_only_table_name} ({self.Temporal1});
                    #         """)
                    #  Since it is read only so i can't do it, but sugges you to do it!
                
            finally:
                con.close()
            
        elif X_train.endswith('.parquet'):
            con = duckdb.connect(config=self.duckdb_config)
            try:
                rel_db = con.read_parquet(X_train, hive_partitioning=False)
                cols = rel_db.columns
                if index_col not in cols:
                    raise ValueError(f"index_col '{index_col}' not found in parquet. You need to turn your index into the column '{index_col}' and save it into the parquet file. This is necessary for database query. This can be done by exporting pandas dataframe using: df.to_parquet(\"./test.parquet\", engine=\"pyarrow\", index=True)")
                # numeric index?
                col2type = dict(zip(cols, rel_db.types))   # DuckDB type names
                if str(col2type[index_col]).upper() not in _NUMERIC:
                    raise ValueError(f"Index '{index_col}' must be numeric; found {col2type[index_col]}.")
                # any NAs anywhere?
                na = con.sql(f"SELECT COUNT(*) FROM rel_db WHERE NOT (" + " AND ".join(f'"{c}" IS NOT NULL' for c in cols) + ")").fetchone()[0]
                if na > 0:
                    raise ValueError(f"NAs detected in parquet ({na} row(s)).")
                # unique + non-null index
                ok = con.sql(f'SELECT COUNT(*)=COUNT(DISTINCT "{index_col}") FROM rel_db').fetchone()[0]
                if not ok:
                    raise ValueError(f"Index '{index_col}' is not unique.")
            finally:
                con.close()
        else:
            raise AttributeError('Not possible! Must be .duckdb or .parquet')
    else:
        raise TypeError(f"Input X should be either type 'pd.DataFrame' or string (path to file with postfix .duckdb or .parquet). Got {str(type(X_train))}")


def check_y_train(y_train, self, index_col='__index_level_0__'):
    
    _NUMERIC = {"TINYINT","SMALLINT","INTEGER","BIGINT","HUGEINT",
                "UTINYINT","USMALLINT","UINTEGER","UBIGINT",
                "FLOAT","DOUBLE","DECIMAL"}
                    
    type_y_train = type(y_train)
    if isinstance(y_train, (pd.DataFrame, pd.Series)):
        if np.sum(np.isnan(np.array(y_train))) > 0:
            raise ValueError("NAs (missing values) detected in input y data. Consider deleting these rows.")
        if isinstance(y_train.index, pd.MultiIndex):
            raise ValueError("Index must be single-level, not MultiIndex.")
        if not y_train.index.is_unique:
            raise ValueError("Index values must be unique.")
        if not pd.api.types.is_numeric_dtype(y_train.index):
            raise ValueError("Index must be numeric (int or float).")
    
    elif isinstance(y_train, str):
        if not (y_train.endswith('.duckdb') or y_train.endswith('.parquet')):
            raise ValueError(f"If the input for y is a string, it has to be the path to a database/file that ends with either .duckdb or .parquet; Got {y_train}")
        
        if y_train.endswith('.duckdb'):
            con = duckdb.connect(y_train, read_only=True, config=self.duckdb_config)
            try:
                all_tables = con.sql("""SHOW TABLES""").fetchall()
                if len(all_tables) > 1:
                    raise AttributeError(f'If the input y is a path to a .duckdb database, the database should only contain one table. Also, X and y should be put in different database files. Got {all_tables}')
                elif len(all_tables) == 0:
                    raise AttributeError(f'No file found in the input database {y_train}!')
                else:
                    the_only_table_name = all_tables[0][0]
                    # type check
                    schema = con.sql(f'DESCRIBE SELECT * FROM "{the_only_table_name}"').fetchall()
                    name2type = {row[0]: row[1] for row in schema}
                    if index_col not in name2type:
                        raise ValueError(f"index_col '{index_col}' not found in table '{the_only_table_name}'.")
                    if str(name2type[index_col]).upper() not in _NUMERIC:
                        raise ValueError(f"Index '{index_col}' must be numeric; found {name2type[index_col]}.")
                    # NAs anywhere?
                    cols = [row[0] for row in schema]
                    na = con.sql(f'SELECT COUNT(*) FROM "{the_only_table_name}" WHERE NOT (' + " AND ".join(f'"{c}" IS NOT NULL' for c in cols) + ")").fetchone()[0]
                    if na > 0:
                        raise ValueError(f"NAs detected in table '{the_only_table_name}' ({na} row(s)).")
                    # unique + non-null index
                    indexes_are_unique = con.sql(f'''
                        SELECT 
                            COUNT(*) = COUNT(DISTINCT "{index_col}") 
                        FROM "{the_only_table_name}"
                    ''').fetchone()[0]
                    if not indexes_are_unique:
                        raise ValueError(f"Index '{index_col}' is not unique.")
            finally:
                con.close()
            
        elif y_train.endswith('.parquet'):
            con = duckdb.connect(config=self.duckdb_config)
            try:
                rel_db = con.read_parquet(y_train, hive_partitioning=False)
                cols = rel_db.columns
                if index_col not in cols:
                    raise ValueError(f"index_col '{index_col}' not found in parquet. You need to turn your index into the column '{index_col}' and save it into the parquet file. This is necessary for database query. This can be done by exporting pandas dataframe using: df.to_parquet(\"./test.parquet\", engine=\"pyarrow\", index=True)")
                # numeric index?
                col2type = dict(zip(cols, rel_db.types))   # DuckDB type names
                if str(col2type[index_col]).upper() not in _NUMERIC:
                    raise ValueError(f"Index '{index_col}' must be numeric; found {col2type[index_col]}.")
                # any NAs anywhere?
                na = con.sql(f"SELECT COUNT(*) FROM rel_db WHERE NOT (" + " AND ".join(f'"{c}" IS NOT NULL' for c in cols) + ")").fetchone()[0]
                if na > 0:
                    raise ValueError(f"NAs detected in parquet ({na} row(s)).")
                # unique + non-null index
                ok = con.sql(f'SELECT COUNT(*)=COUNT(DISTINCT "{index_col}") FROM rel_db').fetchone()[0]
                if not ok:
                    raise ValueError(f"Index '{index_col}' is not unique.")
            finally:
                con.close()
        else:
            raise AttributeError('Not possible! Must be .duckdb or .parquet')
        
    else:
        raise TypeError(
            f"Input y_train should be either type 'pd.DataFrame' or 'pd.Series', or string (path to file with postfix .duckdb or .parquet). Got {str(type_y_train)}"
        )

def check_X_y_format_match(X_train, y_train):
    # 01. Format match, both pandas, both duckdb, or both parquet
    if isinstance(X_train, (pd.DataFrame)) and isinstance(y_train, (pd.DataFrame, pd.Series)):
        return 'pandas'
    elif isinstance(X_train, str) and isinstance(y_train, str):
        if X_train.endswith('.duckdb') and X_train.endswith('.duckdb'):
            return 'duckdb'
        elif X_train.endswith('.parquet') and X_train.endswith('.parquet'):
            return 'parquet'
        
    raise AttributeError('Input X and y must have the same format. Either both pd.DataFrame, .duckdb, or .parquet.')


def check_X_y_indexes_match(X_train, y_train, self, index_col='__index_level_0__'):
    
    # 01. Format match, both pandas, both duckdb, or both parquet
    if self.data_format == 'pandas':
        if not X_train.index.equals(y_train.index):
            raise ValueError("Indexes of X and y must be identical.")
    elif self.data_format == 'duckdb':
        con = duckdb.connect(X_train, read_only=True, config=self.duckdb_config)
        try:
            the_only_table_name = con.sql("""SHOW TABLES""").fetchall()[0][0]
            index1 = con.sql(f'SELECT {index_col} FROM "{the_only_table_name}"').df()[index_col]
        finally:
            con.close()
            
        con = duckdb.connect(y_train, read_only=True, config=self.duckdb_config)
        try:
            the_only_table_name = con.sql("""SHOW TABLES""").fetchall()[0][0]
            index2 = con.sql(f'SELECT {index_col} FROM "{the_only_table_name}"').df()[index_col]
        finally:
            con.close()
            
        if not index1.equals(index2):
            raise ValueError("Indexes of X and y must be identical.")
    elif self.data_format == 'parquet':
        con = duckdb.connect(config=self.duckdb_config)
        try:
            rel_db = con.read_parquet(X_train, hive_partitioning=False)
            index1 = con.sql(f"SELECT {index_col} FROM 'rel_db'").df()[index_col]
        finally:
            con.close()
            
        con = duckdb.connect(config=self.duckdb_config)
        try:
            rel_db = con.read_parquet(y_train, hive_partitioning=False)
            index2 = con.sql(f"SELECT {index_col} FROM 'rel_db'").df()[index_col]
        finally:
            con.close()
            
        if not index1.equals(index2):
            raise ValueError("Indexes of X and y must be identical.")


def check_X_test(X_test, self):
    check_X_train(X_test, self)


def check_prediciton_aggregation(aggregation):
    if aggregation not in ["mean", "median"]:
        raise ValueError(f"aggregation must be one of 'mean' and 'median'. Got {aggregation}")


def check_prediction_return(return_by_separate_ensembles, return_std):
    if not isinstance(return_by_separate_ensembles, bool):
        type_return_by_separate_ensembles = str(type(return_by_separate_ensembles))
        raise TypeError(f"return_by_separate_ensembles must be bool. Got {type_return_by_separate_ensembles}")
    else:
        if return_by_separate_ensembles and return_std:
            warnings.warn("return_by_separate_ensembles == True. Automatically setting return_std=False")
            return_std = False
    return return_by_separate_ensembles, return_std


def check_spatial_scale(x_min, x_max, y_min, y_max, grid_length_upper, grid_length_lower):
    if (grid_length_upper <= (x_max - x_min) / 100) or (grid_length_upper <= (y_max - y_min) / 100):
        warnings.warn(
            "The grid_len_upper_threshold is significantly smaller than the scale of longitude and latitude (x and y). Be sure if this is desired."
        )
    if (grid_length_upper >= (x_max - x_min)) or (grid_length_upper >= (y_max - y_min)):
        warnings.warn(
            "The grid_len_upper_threshold is larger than the scale of longitude and latitude (x and y). Be sure if this is desired."
        )
    if (grid_length_lower <= (x_max - x_min) / 100) or (grid_length_lower <= (y_max - y_min) / 100):
        warnings.warn(
            "The grid_len_lower_threshold is significantly smaller than the scale of longitude and latitude (x and y). Be sure if this is desired."
        )
    if (grid_length_lower >= (x_max - x_min)) or (grid_length_lower >= (y_max - y_min)):
        warnings.warn(
            "The grid_len_lower_threshold is larger than the scale of longitude and latitude (x and y). Be sure if this is desired."
        )


def check_temporal_scale(t_min, t_max, temporal_bin_interval):
    if temporal_bin_interval <= (t_max - t_min) / 100:
        warnings.warn(
            "The temporal_bin_interval is significantly smaller than the scale of temporal parameters in provided data. Be sure if this is desired."
        )
    if temporal_bin_interval >= t_max - t_min:
        warnings.warn(
            "The temporal_bin_interval is larger than the scale of temporal parameters in provided data. Be sure if this is desired."
        )


def check_sql_backend(sql_backend):
    if sql_backend in ['duckdb', 'pandas']:
        return sql_backend
    else:
        raise ValueError("The sql_backend can only be 'duckdb' or 'pandas'!")

def check_mem_string(mem_str: str) -> bool:
    """
    Check if a string is a valid memory specification like '8GB', '512MB', '1.5GB', etc.
    """
    if not isinstance(mem_str, str):
        return False

    pattern = r'^\s*(\d+(\.\d+)?)\s*(KB|MB|GB|TB)\s*$'
    return re.match(pattern, mem_str.upper()) is not None


def initiate_lazy_loading_dir(lazy_loading_dir):
    # Setup lazyloading dir
    if lazy_loading_dir is None:
        saving_code = generate_random_saving_code()
        lazy_loading_dir = os.path.join(tempfile.gettempdir(), f'stemflow_model_{saving_code}')
        warnings.warn(f'lazy_loading_dir not specified during instance initiation. Using the temporary folder: {lazy_loading_dir}')
    else:
        if os.path.exists(lazy_loading_dir):
            shutil.rmtree(lazy_loading_dir)
    lazy_loading_dir = str(Path(lazy_loading_dir.rstrip('/\\')))
    if not os.path.exists(lazy_loading_dir):
        os.makedirs(lazy_loading_dir)
        
    return lazy_loading_dir


def initiate_joblib_tmp_dir(lazy_loading_dir):
    joblib_tmp_dir = os.path.join(lazy_loading_dir, 'joblib_' + str(generate_random_saving_code()))
    if not os.path.exists(joblib_tmp_dir):
        os.makedirs(joblib_tmp_dir)
    return joblib_tmp_dir


    