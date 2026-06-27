"""Incident store: episodic long-term memory (LTM).

Stores every pipeline incident in SQLite. The Diagnoser queries past
incidents of the same anomaly_type to improve its root-cause reasoning.
This is the episodic memory — it grows across runs.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import DB_PATH
from logging_config import logger


def _get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Get a SQLite connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize the incidents table from schema.sql.

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.

    Args:
        db_path: Path to the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

    with open(schema_path, "r") as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()
    logger.info("Incident store initialized (episodic LTM)")


def insert_incident(incident: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """Insert a completed incident into the episodic memory.

    Args:
        incident: Dict with keys matching the incidents table columns.
                  'affected_tables' can be a list (will be JSON-encoded).
        db_path: Path to the SQLite database.
    """
    conn = sqlite3.connect(db_path)

    # Serialize list fields to JSON
    affected_tables = incident.get("affected_tables", [])
    if isinstance(affected_tables, list):
        affected_tables = json.dumps(affected_tables)

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO incidents 
            (run_id, timestamp, anomaly_type, severity, gap_minutes,
             root_cause, affected_tables, fix_taken, resolved, resolved_at, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                incident.get("run_id", ""),
                incident.get("timestamp", datetime.now(timezone.utc).isoformat()),
                incident.get("anomaly_type", ""),
                incident.get("severity", "NONE"),
                incident.get("gap_minutes", 0.0),
                incident.get("root_cause", ""),
                affected_tables,
                incident.get("fix_taken", ""),
                incident.get("resolved", 0),
                incident.get("resolved_at", ""),
                incident.get("confidence", 0.0),
            ),
        )
        conn.commit()
        logger.info(
            f"[{incident.get('run_id', '?')}] Incident saved to episodic LTM: "
            f"{incident.get('anomaly_type', 'unknown')} ({incident.get('severity', '?')})"
        )
    except sqlite3.Error as e:
        logger.error(f"Failed to insert incident: {e}")
    finally:
        conn.close()


def get_similar_incidents(
    anomaly_type: str, limit: int = 5, db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieve recent incidents of the same anomaly type.

    Used by the Diagnoser to build context from episodic memory.

    Args:
        anomaly_type: The type of anomaly to search for.
        limit: Maximum number of incidents to return.
        db_path: Path to the SQLite database.

    Returns:
        List of incident dicts, most recent first.
    """
    conn = _get_connection(db_path)

    try:
        rows = conn.execute(
            """
            SELECT run_id, timestamp, anomaly_type, severity, gap_minutes,
                   root_cause, affected_tables, fix_taken, resolved, confidence
            FROM incidents
            WHERE anomaly_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (anomaly_type, limit),
        ).fetchall()

        incidents = []
        for row in rows:
            incident = dict(row)
            # Deserialize JSON fields
            try:
                incident["affected_tables"] = json.loads(
                    incident.get("affected_tables", "[]")
                )
            except (json.JSONDecodeError, TypeError):
                incident["affected_tables"] = []
            incidents.append(incident)

        return incidents

    except sqlite3.Error as e:
        logger.error(f"Failed to query similar incidents: {e}")
        return []
    finally:
        conn.close()


def get_all_incidents(
    limit: int = 50, db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieve the most recent incidents regardless of type.

    Args:
        limit: Maximum number of incidents to return.
        db_path: Path to the SQLite database.

    Returns:
        List of incident dicts, most recent first.
    """
    conn = _get_connection(db_path)

    try:
        rows = conn.execute(
            """
            SELECT run_id, timestamp, anomaly_type, severity, gap_minutes,
                   root_cause, affected_tables, fix_taken, resolved, confidence
            FROM incidents
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    except sqlite3.Error as e:
        logger.error(f"Failed to query incidents: {e}")
        return []
    finally:
        conn.close()


def auto_resolve_incident(
    run_id: str, db_path: str = DB_PATH
) -> bool:
    """Mark an incident as resolved and record the resolution timestamp.

    Sets ``resolved = 1`` and ``resolved_at`` to the current UTC time.
    Used by the Repairer Agent after a successful fix and by the
    dashboard's manual "Resolve Incident" button.

    Args:
        run_id: The run_id of the incident to resolve.
        db_path: Path to the SQLite database.

    Returns:
        True if the incident was found and updated, False otherwise.
    """
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).isoformat()

    try:
        cursor = conn.execute(
            """
            UPDATE incidents
            SET resolved = 1, resolved_at = ?
            WHERE run_id = ? AND resolved = 0
            """,
            (now, run_id),
        )
        conn.commit()

        if cursor.rowcount > 0:
            logger.info(
                f"[{run_id}] Incident auto-resolved at {now}"
            )
            return True
        else:
            logger.warning(
                f"[{run_id}] auto_resolve_incident: "
                f"no unresolved incident found with this run_id"
            )
            return False

    except sqlite3.Error as e:
        logger.error(f"[{run_id}] Failed to auto-resolve incident: {e}")
        return False
    finally:
        conn.close()
