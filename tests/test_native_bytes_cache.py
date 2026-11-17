"""The bytes cache of the native glue (``lib.bc``) keeps statement texts and language names, never the value of a write (#412).

``lib.bc`` remembers the UTF-8 of a text so the few statement texts that cross a million times in a load are encoded once. It used to
remember the JSON text of the bound parameters too, and for a write of a large property that text holds the whole value: 300 writes of
1 MiB kept 600 MiB in the cache. The cache is now bounded in entries and in the length of a text, and the parameter text does not go
through it. These tests read the cache content, not the resident set and not a time.
"""

import os

import pytest

import arcadedb
from arcadedb._native import lib

# the documented bounds of lib.bc, written out so that a change of either is a decision of this file too
MAX_CHARS = 1024
MAX_ENTRIES = 4096


@pytest.fixture
def cache(monkeypatch):
    """An empty cache for the test, so what is in it afterwards is what the test put there."""
    fresh = {}
    monkeypatch.setattr(lib, "_bytes_cache", fresh)
    return fresh


def test_a_small_text_is_cached_and_the_hit_returns_the_same_bytes(cache):
    text = "SELECT FROM Person WHERE name = ?"
    first = lib.bc(text)
    assert first == text.encode("utf-8")
    # the hit path: the same bytes object, not a new encode
    assert lib.bc(text) is first
    same_text = text[:-1] + text[-1]  # a different str object with the same text
    assert same_text is not text
    assert lib.bc(same_text) is first
    assert list(cache) == [text]


def test_a_text_at_the_bound_is_cached_and_one_character_more_is_not(cache):
    at_bound = "a" * MAX_CHARS
    over_bound = "a" * (MAX_CHARS + 1)
    assert lib.bc(at_bound) is lib.bc(at_bound)
    assert at_bound in cache
    assert lib.bc(over_bound) == over_bound.encode("utf-8")
    assert over_bound not in cache
    assert lib.bc(over_bound) is not lib.bc(
        over_bound
    )  # encoded again, nothing is kept


def test_a_large_text_is_encoded_and_not_stored(cache):
    text = os.urandom(128 * 1024).hex()  # 256 KiB
    assert lib.bc(text) == text.encode("utf-8")
    # compared by length: a failed assert on a 256 KiB key would make pytest diff the two dicts, which takes minutes
    assert len(cache) == 0


def test_a_non_ascii_text_is_encoded_as_utf8_on_both_sides_of_the_bound(cache):
    small = "café 日本語 \U0001f600"
    large = "é日" * MAX_CHARS
    assert lib.bc(small) == small.encode("utf-8")
    assert lib.bc(large) == large.encode("utf-8")
    assert list(cache) == [small]


def test_the_cache_stops_at_its_entry_limit_and_still_answers(cache):
    for i in range(MAX_ENTRIES + 10):
        assert lib.bc(f"SELECT {i}") == f"SELECT {i}".encode("utf-8")
    assert len(cache) == MAX_ENTRIES
    assert "SELECT 0" in cache and f"SELECT {MAX_ENTRIES + 5}" not in cache


def test_writes_of_a_large_property_keep_only_the_statement_texts(temp_db_path, cache):
    create = "CREATE VERTEX TYPE WatchState"
    prop = "CREATE PROPERTY WatchState.state_json STRING"
    key = "CREATE PROPERTY WatchState.key STRING"
    index = "CREATE INDEX ON WatchState (key) UNIQUE"  # an UPSERT needs a unique index
    upsert = "UPDATE WatchState SET key = ?, state_json = ? UPSERT WHERE key = ?"
    named = "UPDATE WatchState SET state_json = :value WHERE key = :key"
    read = "SELECT state_json FROM WatchState"
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", create)
        for statement in (prop, key, index):
            db.command("sql", statement)
        with db.transaction():
            db.command("sql", upsert, "k", "first", "k")
        sizes = 0
        for i in range(100):
            text = os.urandom(128 * 1024).hex()  # 256 KiB, new content every write
            sizes += len(text)
            with db.transaction():
                if i % 2:
                    db.command("sql", upsert, "k", text, "k")
                else:
                    db.command("sql", named, {"value": text, "key": "k"})
        stored = db.query("sql", read).first()
        stored_is_last = stored.get("state_json") == text  # the last write is stored
        assert stored_is_last
    assert sizes >= 25 * 1024 * 1024
    sent = {"sql", create, prop, key, index, upsert, named, read}
    # what the cache holds is reported by length, not by content (a failed assert would print every value)
    not_sent = [len(key) for key in cache if key not in sent]  # no parameter text
    assert not_sent == []
    assert max(len(key) for key in cache) <= MAX_CHARS
    kept = sum(len(key) for key in cache) + sum(len(v) for v in cache.values())
    assert kept < 64 * 1024
