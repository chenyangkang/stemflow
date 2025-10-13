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
    
    
    
from contextlib import contextmanager
import duckdb
import pandas as pd

def _as_relation(con, obj, view_name, attach_alias):
    """Normalize obj into a relation visible in `con` under `view_name`."""
    if isinstance(obj, pd.DataFrame):
        # keep pandas behavior same as before: return the DF
        return obj
    if isinstance(obj, str) and obj.endswith(".duckdb"):
        # attach the DB file under its own alias, then expose its (first) table as a view
        con.execute(f"ATTACH '{obj}' AS {attach_alias} (READ_ONLY)")
        tbl = con.sql(
            f"""
            SELECT table_name FROM {attach_alias}.information_schema.tables
            WHERE table_schema='main' LIMIT 1
            """
        ).fetchone()[0]
        rel = con.sql(f"SELECT * FROM {attach_alias}.main.{tbl}")
        rel.create_view(view_name)
        return rel
    if isinstance(obj, str) and obj.endswith(".parquet"):
        rel = con.read_parquet(obj, hive_partitioning=False)
        rel.create_view(view_name)
        return rel
    raise TypeError("Input must be a pandas DataFrame, .duckdb, or .parquet path.")


@contextmanager
def open_both_Xy_db_connection(X_train, y_train, duckdb_config):
    """
    Open a DuckDB connection. With one source (X_train), behaves like before and yields (X_obj, con).
    With two sources (X_train, Y_train), yields (X_rel_or_df, Y_rel_or_df, con) sharing the SAME connection.
    """
    con = None
    try:
        # one shared connection for both cases
        con = duckdb.connect(config=duckdb_config)
        # dual-source mode: expose BOTH in the SAME connection
        X_obj = _as_relation(con, X_train, "X_df", "xdb")
        Y_obj = _as_relation(con, y_train, "y_df", "ydb")
        yield X_obj, Y_obj, con

    finally:
        if con is not None:
            con.close()
            