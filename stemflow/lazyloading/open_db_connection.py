import duckdb
import pandas as pd
import os
from contextlib import contextmanager

@contextmanager
def open_db_connection(X_train, duckdb_config):
    """open the connection to databse using duckdb
    
    Args:
        X_train: Training variables. Can be either a pd.DataFrame object or a string that indicate the path to the database (.duckdb or .parquet).
        self: The AdaSTEM/STEM model instance
    
    return:
        The name of the table in the open connection to the dataframe
    """
    con = None
    try:
        if isinstance(X_train, pd.DataFrame):
            con = duckdb.connect(config=duckdb_config)
            the_only_table_name = X_train
        elif isinstance(X_train, str):
            if X_train.endswith('.duckdb'):
                con = duckdb.connect(X_train, read_only=True, config=duckdb_config)
                the_only_table_name = con.sql("""SHOW TABLES""").fetchall()[0][0]
                the_only_table_name = con.sql(f"select * from {the_only_table_name};")
            elif X_train.endswith('.parquet'):
                con = duckdb.connect(config=duckdb_config)
                the_only_table_name = con.read_parquet(X_train, hive_partitioning=False)
                
        yield the_only_table_name, con
        
    finally:
        if con is not None:
            con.close()


def duckdb_config(max_mem, joblib_tmp_dir):
    return {
        "threads": "1",
        "memory_limit": max_mem,
        "temp_directory": os.path.join(joblib_tmp_dir, 'duckdb'),
    }
    