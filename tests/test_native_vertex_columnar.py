"""Native: a vertex type loaded by ``insert_dataframe`` or ``insert_arrow`` crosses as columns (``arcadedb_n_insert_colp``), one crossing
per batch, with no GraphBatch crossing and no JSON text (ArcadeData/arcadedb-python #319). The stored vertices are compared with the old
route in ``tests/test_vertex_load_columnar.py``, which runs on both wheels. Needs the native library, so conftest.py collects this file on
the native backend only.
"""

import numpy as np
import pandas as pd
import pyarrow as pa

from arcadedb._native import lib  # noqa: E402

COLP, ROWS, MANY = (
    "arcadedb_n_insert_colp",
    "arcadedb_n_insert_rows",
    "arcadedb_n_exec_many",
)


def _spy(monkeypatch):
    calls = []
    real = lib.call

    def call(name, *args):
        calls.append(name)
        return real(name, *args)

    monkeypatch.setattr(lib, "call", call)
    return calls


def _vertex_type(db, name):
    db.command("sql", f"CREATE VERTEX TYPE {name}")
    db.command("sql", f"CREATE PROPERTY {name}.id LONG")


def test_every_batch_of_a_vertex_load_is_one_columnar_crossing(temp_db, monkeypatch):
    _vertex_type(temp_db, "V")
    calls = _spy(monkeypatch)
    frame = pd.DataFrame(
        {
            "id": np.arange(25, dtype=np.int64),
            "name": [f"p{i}" for i in range(25)],
            "score": [None if i % 5 == 0 else i * 0.5 for i in range(25)],
            "born": pd.to_datetime(["2000-01-01"] * 25),
        }
    )
    assert temp_db.insert_dataframe("V", frame, batch_size=10) == 25
    assert calls.count(COLP) == 3  # batches of 10, 10 and 5
    assert calls.count(ROWS) == 0 and calls.count(MANY) == 0
    assert not [c for c in calls if "gb" in c or "graph" in c], calls
    assert temp_db.query("sql", "SELECT count(*) AS n FROM V").first().get("n") == 25
    calls.clear()
    assert temp_db.insert_arrow("V", pa.Table.from_pandas(frame)) == 25
    assert calls.count(COLP) == 1
