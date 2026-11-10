"""The native columnar fast path with a rows-dropped fault in it (ArcadeData/arcadedb-python#336).

``arcadedb._native.bulk.insert_columns_fast`` is the one entry that ``insert_columns``, ``insert_arrow`` and ``insert_dataframe``
share on the native build. Each test puts a fault into it and runs the same checks as ``test_ingest_integrity.py``: they must
report it. Needs the native library, so conftest.py collects this file on the native backend only.
"""

import pytest

from arcadedb._native import bulk
from tests.test_ingest_integrity import (
    BY_ID,
    FOUND,
    MUTATION_SIZE,
    IntegrityError,
    run_cases,
)


@pytest.mark.parametrize("fault", ["drop_the_last_row", "skip_the_call"])
@pytest.mark.parametrize("family", ["cols", "arrow", "df"])
def test_a_fault_in_the_columnar_fast_path_is_reported(
    store, small_chunks, monkeypatch, family, fault
):
    real = bulk.insert_columns_fast
    seen = []

    def faulty(db, type_name, columns, *args, **kwargs):
        seen.append(type_name)
        if fault == "skip_the_call":
            return len(next(iter(columns.values())))
        short = {name: column[:-1] for name, column in columns.items()}
        return real(db, type_name, short, *args, **kwargs)

    monkeypatch.setattr(bulk, "insert_columns_fast", faulty)
    with pytest.raises(IntegrityError) as found:
        run_cases(store, BY_ID[family], [MUTATION_SIZE])
    assert FOUND in str(found.value), found.value
    assert seen, "the fault was not in the path"
