"""Example 29 at test size: the one-statement hybrid retrieval returns only the
requested tenant's facts, where a global top-k followed by a tenant test returns
none, and the tenant guard keeps the graph hops inside the tenant.

Both pre-filter forms run: ``filter`` (a RID subquery) and ``where`` (ArcadeDB
26.11.1, the version this package is built against)."""

import importlib.util
from pathlib import Path

import pytest

import arcadedb

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "29_hybrid_memory_retrieval.py"
)


@pytest.fixture(scope="module")
def example29():
    spec = importlib.util.spec_from_file_location("example29", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def memory(example29, tmp_path_factory):
    path = tmp_path_factory.mktemp("hybrid") / "db"
    with arcadedb.create_database(str(path)) as db:
        centroids = example29.build(db)
        yield db, centroids


@pytest.fixture(scope="module")
def has_where(example29, memory):
    """True when the engine takes the ``where`` option of ``vector.neighbors`` (ArcadeDB 26.11.1).

    The native build links the newest engine RELEASE, which predates it until 26.11.1 ships: the form is then refused with the engine's
    own message, and the tests below assert that answer instead of the new one. When the native build links 26.11.1, delete this
    fixture and the old-answer branches.
    """
    db, centroids = memory
    try:
        example29.run(db, centroids, "where")
    except arcadedb.ArcadeDBError as e:
        assert "Unknown option 'where'" in str(e), e
        return False
    return True


@pytest.mark.parametrize("form", ["filter", "where"])
def test_hybrid_rows_and_hops_stay_in_the_tenant(example29, memory, form, has_where):
    db, centroids = memory
    if form == "where" and not has_where:
        with pytest.raises(arcadedb.ArcadeDBError, match="Unknown option 'where'"):
            example29.run(db, centroids, form)
        return
    out = example29.run(db, centroids, form)
    tenant = example29.REQUESTED_TENANT

    assert len(out["pre_filtered"]) == example29.K
    assert len(out["hybrid"]) == example29.TOP
    scores = [r["score"] for r in out["hybrid"]]
    assert scores == sorted(scores, reverse=True)
    assert example29.tenants_of(db, [r["fid"] for r in out["pre_filtered"]]) == {tenant}
    assert example29.tenants_of(db, [r["fid"] for r in out["hybrid"]]) == {tenant}
    related = [f for r in out["hybrid"] for f in r["related_fids"]]
    assert related and example29.tenants_of(db, related) == {tenant}
    assert all(r["related_entities"] for r in out["hybrid"])


def test_filtering_after_the_top_k_loses_the_rows(example29, memory):
    db, centroids = memory
    out = example29.run(db, centroids, "filter")
    assert len(out["post_filtered"]) < example29.K  # the pre-filter returns K


def test_the_forms_agree(example29, memory, has_where):
    db, centroids = memory
    by_filter = example29.run(db, centroids, "filter")
    if (
        not has_where
    ):  # the engine predates the option: the other form is refused (see has_where)
        with pytest.raises(arcadedb.ArcadeDBError, match="Unknown option 'where'"):
            example29.run(db, centroids, "where")
        return
    by_where = example29.run(db, centroids, "where")
    assert [r["fid"] for r in by_filter["hybrid"]] == [
        r["fid"] for r in by_where["hybrid"]
    ]


def test_the_hop_guard_matters(example29, memory):
    """The memory has one stray cross-tenant RELATES edge; without the guard the
    hops of the acme facts reach globex facts."""
    db, centroids = memory
    out = example29.run(db, centroids, "filter")
    leaked = example29.tenants_of(db, out["unguarded_fids"])
    assert leaked - {example29.REQUESTED_TENANT}
