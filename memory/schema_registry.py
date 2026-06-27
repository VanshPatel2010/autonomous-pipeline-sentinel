"""Schema Registry: semantic LTM for schema drift detection.

Compares the actual DB schema against a JSON baseline stored in
schemas/{table}.json. Part of the Phase 5 semantic memory layer.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

from logging_config import logger

# Project root: two levels up from this file (memory/ -> project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_expected_schema(table: str) -> dict:
    """Load the expected schema definition from schemas/{table}.json.

    Args:
        table: Table name (used to locate the JSON file).

    Returns:
        Parsed JSON dict with table schema, or empty dict if not found.
    """
    schema_path = os.path.join(_PROJECT_ROOT, 'schemas', f'{table}.json')
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        logger.debug(f'Loaded expected schema for "{table}" from {schema_path}')
        return schema
    except FileNotFoundError:
        logger.warning(
            f'Schema file not found for table "{table}": {schema_path}'
        )
        return {}


def get_actual_schema(conn: sqlite3.Connection, table: str) -> List[dict]:
    """Retrieve the actual column schema from SQLite via PRAGMA table_info.

    Args:
        conn: Active SQLite connection.
        table: Table name to inspect.

    Returns:
        List of dicts with keys: name (str), type (str), notnull (bool).
    """
    cursor = conn.execute(f'PRAGMA table_info({table})')
    rows = cursor.fetchall()

    columns: List[dict] = []
    for row in rows:
        # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
        # Note: SQLite reports notnull=0 for TEXT PRIMARY KEY columns even
        # though they are effectively NOT NULL. We check the pk flag and
        # treat any primary key column as notnull=True.
        is_pk = bool(row[5])
        columns.append({
            'name': row[1],
            'type': row[2],
            'notnull': bool(row[3]) or is_pk,
        })

    logger.debug(
        f'Actual schema for "{table}": {len(columns)} columns'
    )
    return columns


def check_drift(conn: sqlite3.Connection, table: str) -> dict:
    """Compare expected schema (JSON) vs actual schema (DB) and report drift.

    Args:
        conn: Active SQLite connection.
        table: Table name to check.

    Returns:
        Dict with keys:
            - drift_detected (bool)
            - new_columns (list[str]): columns in DB but not in expected
            - deleted_columns (list[str]): columns in expected but not in DB
            - type_changes (list[dict]): columns where type differs
            - nullable_changes (list[dict]): columns where nullable differs
    """
    result: Dict[str, Any] = {
        'drift_detected': False,
        'new_columns': [],
        'deleted_columns': [],
        'type_changes': [],
        'nullable_changes': [],
    }

    expected_schema = load_expected_schema(table)
    if not expected_schema or 'columns' not in expected_schema:
        logger.warning(
            f'No expected schema found for "{table}", skipping drift check'
        )
        return result

    actual_columns = get_actual_schema(conn, table)

    # Build lookup maps
    expected_map: Dict[str, dict] = {
        col['name']: col for col in expected_schema['columns']
    }
    actual_map: Dict[str, dict] = {
        col['name']: col for col in actual_columns
    }

    expected_names = set(expected_map.keys())
    actual_names = set(actual_map.keys())

    # New columns (in DB but not in expected schema)
    new_cols = sorted(actual_names - expected_names)
    if new_cols:
        result['new_columns'] = new_cols

    # Deleted columns (in expected schema but not in DB)
    deleted_cols = sorted(expected_names - actual_names)
    if deleted_cols:
        result['deleted_columns'] = deleted_cols

    # Check type and nullable changes for columns present in both
    common_cols = expected_names & actual_names
    for col_name in sorted(common_cols):
        exp = expected_map[col_name]
        act = actual_map[col_name]

        # Type comparison (case-insensitive)
        if exp['type'].upper() != act['type'].upper():
            result['type_changes'].append({
                'column': col_name,
                'expected': exp['type'],
                'actual': act['type'],
            })

        # Nullable comparison
        # expected: nullable=True means NULLs allowed → notnull=False
        # actual: notnull=True means NULLs NOT allowed
        expected_nullable = exp.get('nullable', True)
        actual_nullable = not act['notnull']

        if expected_nullable != actual_nullable:
            result['nullable_changes'].append({
                'column': col_name,
                'expected_nullable': expected_nullable,
                'actual_nullable': actual_nullable,
            })

    # Set drift flag if any changes detected
    if (result['new_columns'] or result['deleted_columns']
            or result['type_changes'] or result['nullable_changes']):
        result['drift_detected'] = True
        logger.warning(f'Schema drift detected for "{table}": {result}')
    else:
        logger.info(f'No schema drift for "{table}"')

    return result


def save_schema_snapshot(
    conn: sqlite3.Connection, table: str, schema_dict: dict
) -> None:
    """Persist a schema snapshot to the schema_snapshots table.

    Args:
        conn: Active SQLite connection.
        table: Table name being snapshotted.
        schema_dict: Schema dict to serialize as JSON.
    """
    snapshot_at = datetime.now(timezone.utc).isoformat()
    schema_json = json.dumps(schema_dict)

    conn.execute(
        'INSERT INTO schema_snapshots (table_name, schema_json, snapshot_at) '
        'VALUES (?, ?, ?)',
        (table, schema_json, snapshot_at),
    )
    conn.commit()
    logger.info(f'Saved schema snapshot for "{table}" at {snapshot_at}')
