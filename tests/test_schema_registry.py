"""Tests for memory/schema_registry.py – schema drift detection.

Covers:
    1. Loading an expected schema from JSON
    2. Missing schema file → empty dict
    3. Getting actual schema via PRAGMA table_info
    4. No-drift detection
    5. New-column drift detection
    6. Deleted-column drift detection
    7. No expected schema → no drift
    8. Snapshot persistence to schema_snapshots table
    9. Multiple simultaneous drift types
"""

import json
import os
import sqlite3

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Canonical expected schema matching schemas/orders.json
ORDERS_SCHEMA = {
    "table": "orders",
    "version": "1.0",
    "columns": [
        {"name": "order_id",     "type": "TEXT",     "nullable": False, "primary_key": True},
        {"name": "created_at",   "type": "DATETIME", "nullable": False, "primary_key": False},
        {"name": "order_amount", "type": "REAL",     "nullable": True,  "primary_key": False},
        {"name": "source_db",    "type": "TEXT",     "nullable": False, "primary_key": False},
    ],
}

# DDL that exactly matches the expected schema above
ORDERS_DDL = """\
CREATE TABLE orders (
    order_id     TEXT    NOT NULL PRIMARY KEY,
    created_at   DATETIME NOT NULL,
    order_amount REAL,
    source_db    TEXT    NOT NULL
);
"""

SCHEMA_SNAPSHOTS_DDL = """\
CREATE TABLE IF NOT EXISTS schema_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name   TEXT NOT NULL,
    schema_json  TEXT NOT NULL,
    snapshot_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def _write_schema_json(base_dir: str, table: str, schema: dict) -> str:
    """Write a schema JSON file under ``base_dir/schemas/{table}.json``."""
    schemas_dir = os.path.join(base_dir, "schemas")
    os.makedirs(schemas_dir, exist_ok=True)
    path = os.path.join(schemas_dir, f"{table}.json")
    with open(path, "w") as f:
        json.dump(schema, f)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def project_root(tmp_path, monkeypatch):
    """Point ``_PROJECT_ROOT`` at a temp directory and write orders.json."""
    import memory.schema_registry as sr

    _write_schema_json(str(tmp_path), "orders", ORDERS_SCHEMA)
    monkeypatch.setattr(sr, "_PROJECT_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture()
def db_conn(tmp_path):
    """Return a fresh SQLite connection with orders + schema_snapshots tables."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(ORDERS_DDL)
    conn.execute(SCHEMA_SNAPSHOTS_DDL)
    conn.commit()
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# 1. test_load_expected_schema
# ---------------------------------------------------------------------------

def test_load_expected_schema(project_root):
    """Loading orders.json returns the correct structure."""
    from memory.schema_registry import load_expected_schema

    schema = load_expected_schema("orders")

    assert schema["table"] == "orders"
    assert schema["version"] == "1.0"
    assert len(schema["columns"]) == 4

    col_names = [c["name"] for c in schema["columns"]]
    assert col_names == ["order_id", "created_at", "order_amount", "source_db"]

    # Spot-check column details
    order_id_col = schema["columns"][0]
    assert order_id_col["type"] == "TEXT"
    assert order_id_col["nullable"] is False
    assert order_id_col["primary_key"] is True


# ---------------------------------------------------------------------------
# 2. test_load_expected_schema_missing_file
# ---------------------------------------------------------------------------

def test_load_expected_schema_missing_file(project_root):
    """A missing JSON file returns an empty dict."""
    from memory.schema_registry import load_expected_schema

    result = load_expected_schema("nonexistent_table")
    assert result == {}


# ---------------------------------------------------------------------------
# 3. test_get_actual_schema
# ---------------------------------------------------------------------------

def test_get_actual_schema(db_conn):
    """PRAGMA table_info returns correct column info."""
    from memory.schema_registry import get_actual_schema

    columns = get_actual_schema(db_conn, "orders")

    assert len(columns) == 4

    by_name = {c["name"]: c for c in columns}

    assert by_name["order_id"]["type"] == "TEXT"
    assert by_name["order_id"]["notnull"] is True

    assert by_name["created_at"]["type"] == "DATETIME"
    assert by_name["created_at"]["notnull"] is True

    assert by_name["order_amount"]["type"] == "REAL"
    assert by_name["order_amount"]["notnull"] is False

    assert by_name["source_db"]["type"] == "TEXT"
    assert by_name["source_db"]["notnull"] is True


# ---------------------------------------------------------------------------
# 4. test_check_drift_no_drift
# ---------------------------------------------------------------------------

def test_check_drift_no_drift(project_root, db_conn):
    """When expected and actual schemas match, drift_detected is False."""
    from memory.schema_registry import check_drift

    result = check_drift(db_conn, "orders")

    assert result["drift_detected"] is False
    assert result["new_columns"] == []
    assert result["deleted_columns"] == []
    assert result["type_changes"] == []
    assert result["nullable_changes"] == []


# ---------------------------------------------------------------------------
# 5. test_check_drift_new_column
# ---------------------------------------------------------------------------

def test_check_drift_new_column(project_root, db_conn):
    """Adding a column to the DB triggers new_columns detection."""
    from memory.schema_registry import check_drift

    db_conn.execute("ALTER TABLE orders ADD COLUMN discount REAL")
    db_conn.commit()

    result = check_drift(db_conn, "orders")

    assert result["drift_detected"] is True
    assert "discount" in result["new_columns"]
    assert result["deleted_columns"] == []


# ---------------------------------------------------------------------------
# 6. test_check_drift_deleted_column
# ---------------------------------------------------------------------------

def test_check_drift_deleted_column(tmp_path, db_conn, monkeypatch):
    """An extra column in the JSON (not in DB) triggers deleted_columns."""
    import memory.schema_registry as sr

    # Build a schema with an extra column that does NOT exist in the DB table
    extended_schema = {
        "table": "orders",
        "version": "1.1",
        "columns": ORDERS_SCHEMA["columns"] + [
            {"name": "customer_id", "type": "TEXT", "nullable": True, "primary_key": False},
        ],
    }
    _write_schema_json(str(tmp_path), "orders", extended_schema)
    monkeypatch.setattr(sr, "_PROJECT_ROOT", str(tmp_path))

    result = sr.check_drift(db_conn, "orders")

    assert result["drift_detected"] is True
    assert "customer_id" in result["deleted_columns"]
    assert result["new_columns"] == []


# ---------------------------------------------------------------------------
# 7. test_check_drift_no_expected_schema
# ---------------------------------------------------------------------------

def test_check_drift_no_expected_schema(tmp_path, db_conn, monkeypatch):
    """When no JSON file exists, drift check returns drift_detected=False."""
    import memory.schema_registry as sr

    # Point _PROJECT_ROOT at a dir with no schemas/ sub-directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(sr, "_PROJECT_ROOT", str(empty_dir))

    result = sr.check_drift(db_conn, "orders")

    assert result["drift_detected"] is False
    assert result["new_columns"] == []
    assert result["deleted_columns"] == []
    assert result["type_changes"] == []
    assert result["nullable_changes"] == []


# ---------------------------------------------------------------------------
# 8. test_save_schema_snapshot
# ---------------------------------------------------------------------------

def test_save_schema_snapshot(db_conn):
    """Snapshot row is persisted to schema_snapshots with correct data."""
    from memory.schema_registry import save_schema_snapshot

    snapshot_data = {"columns": [{"name": "order_id", "type": "TEXT"}]}
    save_schema_snapshot(db_conn, "orders", snapshot_data)

    row = db_conn.execute(
        "SELECT table_name, schema_json, snapshot_at "
        "FROM schema_snapshots WHERE table_name = 'orders'"
    ).fetchone()

    assert row is not None
    assert row[0] == "orders"

    stored_schema = json.loads(row[1])
    assert stored_schema == snapshot_data

    # snapshot_at should be populated (ISO-format string)
    assert row[2] is not None and len(row[2]) > 0


# ---------------------------------------------------------------------------
# 9. test_multiple_drift_types
# ---------------------------------------------------------------------------

def test_multiple_drift_types(tmp_path, db_conn, monkeypatch):
    """Detect new columns, deleted columns, type and nullable changes at once."""
    import memory.schema_registry as sr

    # Craft an expected schema that differs from the actual DB in every way:
    #   - "order_id"     → change type to INTEGER (type drift)
    #   - "created_at"   → set nullable=True (actual is NOT NULL → nullable drift)
    #   - "order_amount" → keep identical (no drift)
    #   - "source_db"    → OMIT from expected → should appear as new_columns
    #   - "region"       → extra col in expected, not in DB → deleted_columns
    modified_schema = {
        "table": "orders",
        "version": "2.0",
        "columns": [
            {"name": "order_id",     "type": "INTEGER",  "nullable": False, "primary_key": True},
            {"name": "created_at",   "type": "DATETIME", "nullable": True,  "primary_key": False},
            {"name": "order_amount", "type": "REAL",     "nullable": True,  "primary_key": False},
            {"name": "region",       "type": "TEXT",     "nullable": True,  "primary_key": False},
        ],
    }

    _write_schema_json(str(tmp_path), "orders", modified_schema)
    monkeypatch.setattr(sr, "_PROJECT_ROOT", str(tmp_path))

    result = sr.check_drift(db_conn, "orders")

    assert result["drift_detected"] is True

    # source_db is in DB but NOT in expected → new column
    assert "source_db" in result["new_columns"]

    # region is in expected but NOT in DB → deleted column
    assert "region" in result["deleted_columns"]

    # order_id type changed TEXT → INTEGER
    type_change_cols = [tc["column"] for tc in result["type_changes"]]
    assert "order_id" in type_change_cols

    # created_at nullable mismatch (expected True, actual False)
    nullable_change_cols = [nc["column"] for nc in result["nullable_changes"]]
    assert "created_at" in nullable_change_cols
