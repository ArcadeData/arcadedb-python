"""Columnar bulk insert through one crossing (``arcadedb_n_insert_colp``): whole columns go over by address, with no JSON text. Numpy
ints, floats and bools, lists of one value type (None is a null), strings (UTF-8 bytes plus offsets, as Arrow lays them out), float32
vector columns, date-times, dates, decimals, bytes and Arrow arrays are laid out by :mod:`.columnar`; anything else (nested values,
mixed-type columns, integers past 64 bits) returns None and the caller uses the per-row JSON path. The documents are built image-side as
``DocumentBatcher.insertColumns`` builds them on JPype; with ``parallel`` they are handed to the async writers instead of a transaction
(``DocumentBatcher.insertColumnsParallel``). ``ARCADEDB_NATIVE_BULK=0`` turns the columnar path off.
"""

import os

from ..exceptions import ArcadeDBError
from . import columnar, lib

_ENABLED = os.environ.get("ARCADEDB_NATIVE_BULK", "1") != "0"


def _usable():
    return (
        _ENABLED
        and columnar._np is not None
        and lib._lib is not None
        and hasattr(lib._lib, "arcadedb_n_insert_colp")
    )


def _send(db, type_name, packed, rows, commit_every, prefix, fid=None):
    from .database import _call

    spec = columnar.spec_of(packed)
    return _call(
        prefix,
        "arcadedb_n_insert_colp",
        db._h,
        lib.b(type_name),
        lib.b(columnar.header_of(rows, commit_every, packed, fid)),
        spec,
        len(spec),
    )  # `packed` holds the arrays the addresses point into until the call returns


def insert_columns_fast(
    db, type_name, columns, commit_every, declared=None, prefix="Failed to bulk-insert"
):
    """Insert ``columns`` (name -> column) in one crossing and return the row count, or None when a column cannot be sent this way.

    ``declared``: a dict of declared property types when the caller already knows them (the DataFrame path), else they are read from the
    schema if a column needs them.
    """
    if not _usable():
        return None
    packed = columnar.pack(columns, columnar.Declared(db, type_name, declared))
    if packed is None:
        return None
    rows, cols = packed
    return int(_send(db, type_name, cols, rows, commit_every, prefix)["n"])


def insert_columns_parallel(db, type_name, columns, declared=None):
    """Hand ``columns`` (name -> column) to the async writers in one crossing, wait for them and return the row count, or None when a
    column cannot be sent this way (nothing was handed over then). A record that a writer rejects raises
    (``Database._finish_parallel``), as for ``insert_many(parallel=True)``, and so does an interrupt, after the writers have finished:
    the failure token is made before the crossing, so the call always has the id to wait for and to read.
    """
    if not _usable():
        return None
    packed = columnar.pack(columns, columnar.Declared(db, type_name, declared))
    if packed is None:
        return None
    rows, cols = packed
    from . import extras

    fid = extras.failure_token(db)
    try:
        _send(db, type_name, cols, rows, 0, "Failed to bulk-insert", fid=fid)
    except ArcadeDBError as e:
        raise db._parallel_load_failed(type_name, e, [fid]) from e
    except BaseException:
        db._abandon_parallel([fid])  # a ^C: the writers still run, so wait for them
        raise
    return db._finish_parallel(type_name, [fid], rows)


def insert_rows_columnar(db, type_name, chunk):
    """One ``insert_many`` chunk (a list of dicts with the same keys) through the columnar entry inside the caller's transaction (no
    commit here). Returns True when it was sent, False when the chunk needs the JSON path.
    """
    if not _usable():
        return False
    columns = columnar.rows_to_columns(chunk)
    if columns is None:
        return False
    packed = columnar.pack(columns, columnar.Declared(db, type_name))
    if packed is None:
        return False
    rows, cols = packed
    _send(db, type_name, cols, rows, 0, f"Failed to bulk-insert into '{type_name}'")
    return True


def rids_of(out):
    """The RID texts (``#bucket:position``, the ``RID.toString`` text) of an int64 array of ``(bucket, position)`` pairs, in row order."""
    pairs = out.reshape(-1, 2)
    return list(map("#{}:{}".format, pairs[:, 0].tolist(), pairs[:, 1].tolist()))


def create_vertices_columns(gb, type_name, rows, declared, prefix):
    """One ``GraphBatch.create_vertices`` chunk (a list of dicts) through ``arcadedb_n_gb_colp``: the rows cross as columns by address and
    the RIDs come back in an int64 buffer, one crossing and one transaction, as the JSON route has it. Returns the list of RID texts, or None
    when the chunk needs the JSON route (nothing was sent then): rows that do not all have the same keys in the same order, a value no column
    holds (a nested list or map, a mixed-type column, a set, a NaT, a numpy scalar, an integer past 64 bits, a lone surrogate) or an empty chunk.

    ``declared``: a :class:`columnar.Declared` of the type, read once by the caller for all chunks of a call.
    """
    if not (lib.GB_COLP and _usable()):
        return None
    columns = columnar.rows_to_columns(rows, same_order=True)
    if columns is None:
        return None
    packed = columnar.pack(columns, declared)
    if packed is None:
        return None
    count, cols = packed
    spec = columnar.spec_of(cols)
    out = columnar._np.empty(2 * count, dtype=columnar._np.int64)
    from .database import _call

    reply = _call(
        prefix,
        "arcadedb_n_gb_colp",
        gb._gb,
        lib.b(type_name),
        lib.b(columnar.header_of(count, 0, cols)),
        spec,
        len(spec),
        out.ctypes.data,
        out.size,
    )  # `cols` and `out` hold the memory the addresses point into until the call returns
    if reply["n"] != count:
        raise ArcadeDBError(
            f"{prefix}: the library created {reply['n']} vertices for {count} rows"
        )
    return rids_of(out)
