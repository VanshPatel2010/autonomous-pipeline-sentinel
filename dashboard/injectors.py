"""Anomaly injectors for the dashboard simulation tab.

These functions inject synthetic anomalies into the pipeline database
so the user can test the monitoring system without touching the terminal.

All functions operate directly on SQLite — no subprocess calls.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from logging_config import logger


def inject_missing_data(db_path: str, gap_minutes: int = 30) -> Dict[str, Any]:
    """Delete recent rows to simulate a pipeline outage.

    Args:
        db_path: Path to the SQLite database.
        gap_minutes: How many minutes of data to delete.

    Returns:
        Dict with injection details: rows_deleted, cutoff, gap_minutes.
    """
    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = (now - timedelta(minutes=gap_minutes)).isoformat()

        before = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.execute("DELETE FROM orders WHERE created_at > ?", (cutoff,))
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

        deleted = before - after
        logger.info(
            f"Injected missing_data: deleted {deleted} rows "
            f"(last {gap_minutes} minutes)"
        )
        return {
            "success": True,
            "type": "missing_data",
            "rows_deleted": deleted,
            "cutoff": cutoff,
            "gap_minutes": gap_minutes,
        }
    except Exception as e:
        logger.error(f"Failed to inject missing_data: {e}")
        return {"success": False, "type": "missing_data", "error": str(e)}
    finally:
        conn.close()


def inject_null_spike(db_path: str, null_pct: float = 100.0) -> Dict[str, Any]:
    """Set order_amount to NULL on recent rows to simulate quality degradation.

    Args:
        db_path: Path to the SQLite database.
        null_pct: Percentage of recent rows to nullify (0-100).

    Returns:
        Dict with injection details: rows_affected, null_pct.
    """
    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = (now - timedelta(minutes=5)).isoformat()

        total_recent = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE created_at > ?", (cutoff,)
        ).fetchone()[0]

        if total_recent == 0:
            return {
                "success": False,
                "type": "data_quality",
                "error": "No recent rows to nullify. Seed the database first.",
            }

        # Calculate how many rows to nullify
        rows_to_null = max(1, int(total_recent * null_pct / 100))

        # Use ORDER BY RANDOM() + LIMIT to select a subset
        conn.execute(
            f"""
            UPDATE orders SET order_amount = NULL
            WHERE rowid IN (
                SELECT rowid FROM orders
                WHERE created_at > ?
                ORDER BY RANDOM()
                LIMIT ?
            )
            """,
            (cutoff, rows_to_null),
        )
        conn.commit()

        actual_nulls = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE created_at > ? AND order_amount IS NULL",
            (cutoff,),
        ).fetchone()[0]

        logger.info(
            f"Injected null_spike: {actual_nulls}/{total_recent} rows "
            f"({null_pct:.0f}% target)"
        )
        return {
            "success": True,
            "type": "data_quality",
            "rows_affected": actual_nulls,
            "total_recent": total_recent,
            "null_pct": null_pct,
        }
    except Exception as e:
        logger.error(f"Failed to inject null_spike: {e}")
        return {"success": False, "type": "data_quality", "error": str(e)}
    finally:
        conn.close()


def inject_schema_drift(
    db_path: str, column_name: str = "upi_transaction_id"
) -> Dict[str, Any]:
    """Add an unexpected column to the orders table to trigger schema drift.

    Args:
        db_path: Path to the SQLite database.
        column_name: Name of the column to add.

    Returns:
        Dict with injection details: column_name added.
    """
    conn = sqlite3.connect(db_path)
    try:
        # Check if column already exists
        cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(orders)").fetchall()
        ]
        if column_name in cols:
            return {
                "success": False,
                "type": "schema_drift",
                "error": f"Column '{column_name}' already exists.",
            }

        conn.execute(f"ALTER TABLE orders ADD COLUMN {column_name} TEXT")
        conn.commit()

        logger.info(f"Injected schema_drift: added column '{column_name}'")
        return {
            "success": True,
            "type": "schema_drift",
            "column_name": column_name,
        }
    except Exception as e:
        logger.error(f"Failed to inject schema_drift: {e}")
        return {"success": False, "type": "schema_drift", "error": str(e)}
    finally:
        conn.close()


def restore_healthy_state(db_path: str) -> Dict[str, Any]:
    """Re-seed the database to restore a clean, healthy pipeline state.

    This drops the orders table, recreates it, and re-seeds with 30 days
    of synthetic data. Useful after injecting anomalies.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Dict with seed results.
    """
    try:
        from seed_db import seed_database
        result = seed_database(db_path)
        logger.info("Healthy state restored via full re-seed")
        return result
    except Exception as e:
        logger.error(f"Failed to restore healthy state: {e}")
        return {"success": False, "error": str(e)}
