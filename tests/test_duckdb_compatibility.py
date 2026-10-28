"""Exercise supported relation APIs against the installed DuckDB version."""

import duckdb
import pytest

from stemflow.utils.duckdb_compat import arrow_reader, fetch_one


@pytest.mark.parametrize("query, expected", [
    ("SELECT 42", (42,)),
    ("SELECT NULL", (None,)),
    ("SELECT i FROM range(5) AS t(i) ORDER BY i", (0,)),
    ("SELECT 42 WHERE FALSE", None),
])
def test_fetch_one_preserves_first_row_and_empty_result(query, expected):
    with duckdb.connect() as con:
        relation = con.sql(query)
        assert fetch_one(relation) == expected
        assert fetch_one(relation) == expected


@pytest.mark.parametrize("rows", [0, 5])
def test_arrow_reader_retains_schema_and_bounded_batches(rows):
    with duckdb.connect() as con:
        relation = con.sql(f"SELECT i, (i + 0.25)::FLOAT AS value FROM range({rows}) AS t(i) ORDER BY i")
        reader = arrow_reader(relation, batch_size=2)
        assert reader.schema.names == ["i", "value"]
        batches = list(reader)
        assert all(len(batch) <= 2 for batch in batches)
        assert sum(len(batch) for batch in batches) == rows
        assert [row for batch in batches for row in batch.to_pylist()] == [
            dict(i=i, value=i + .25) for i in range(rows)
        ]
