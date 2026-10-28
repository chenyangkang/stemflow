"""Read-only relation adapters for DuckDB 1.x and 2.x."""


def fetch_one(relation):
    """Return the first row, or None, without a stateful relation cursor."""
    rows = relation.limit(1).fetchall()
    return rows[0] if rows else None


def arrow_reader(relation, batch_size):
    """Stream Arrow batches, retaining support for older DuckDB releases."""
    for name in ("to_arrow_reader", "fetch_record_batch", "record_batch"):
        if callable(getattr(type(relation), name, None)):
            return getattr(relation, name)(batch_size)
    raise TypeError("The DuckDB relation has no supported Arrow batch reader.")
