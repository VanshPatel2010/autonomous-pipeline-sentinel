"""Predefined test scenarios for LOW / MEDIUM / HIGH-CRITICAL severity.

Each scenario injects a specific anomaly and sets up the expected
pipeline behavior, demonstrating failover at different severity levels.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

from db.database_manager import get_active_db_path
from logging_config import logger


def run_scenario_low() -> dict:
    """LOW severity: Small data gap (15 minutes).

    Expected: Monitor detects gap. Repairer waits and retries. NO failover.
    Simulates: brief network hiccup, auto-recovers.
    """
    db_path = get_active_db_path()
    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = (now - timedelta(minutes=15)).isoformat()
        cursor = conn.execute("DELETE FROM orders WHERE created_at > ?", (cutoff,))
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"Scenario LOW: deleted {deleted} rows (last 15 min)")
    except Exception as e:
        logger.error(f"Scenario LOW failed: {e}")
        deleted = 0
    finally:
        conn.close()

    return {
        "scenario": "LOW",
        "title": "Brief Network Hiccup",
        "description": (
            "Deleted last 15 minutes of orders data. Gap is small — "
            "Repairer will wait and retry without switching nodes."
        ),
        "expected_action": "wait_and_retry",
        "expected_failover": False,
        "rows_deleted": deleted,
        "gap_minutes": 15,
        "icon": "🟡",
    }


def run_scenario_medium() -> dict:
    """MEDIUM severity: Large data gap (2 hours) + null spike.

    Expected: Monitor detects compound anomaly. Repairer switches to REPLICA-1.
    Simulates: Mumbai DB connection lost for 2 hours.
    """
    db_path = get_active_db_path()
    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff_2h = (now - timedelta(hours=2)).isoformat()
        cutoff_4h = (now - timedelta(hours=4)).isoformat()

        # Delete last 2 hours of data
        cursor = conn.execute("DELETE FROM orders WHERE created_at > ?", (cutoff_2h,))
        deleted = cursor.rowcount

        # Inject nulls in order_amount for recent remaining rows
        cursor2 = conn.execute(
            """
            UPDATE orders SET order_amount = NULL
            WHERE rowid IN (
                SELECT rowid FROM orders
                WHERE created_at > ? AND order_amount IS NOT NULL
                ORDER BY RANDOM()
                LIMIT 50
            )
            """,
            (cutoff_4h,),
        )
        nulled = cursor2.rowcount
        conn.commit()
        logger.info(
            f"Scenario MEDIUM: deleted {deleted} rows, nulled {nulled} rows"
        )
    except Exception as e:
        logger.error(f"Scenario MEDIUM failed: {e}")
        deleted, nulled = 0, 0
    finally:
        conn.close()

    return {
        "scenario": "MEDIUM",
        "title": "Mumbai DB Connection Lost (2hrs)",
        "description": (
            "2-hour data gap + null spike in order_amount. "
            "Repairer will switch pipeline to REPLICA-1."
        ),
        "expected_action": "switch_to_backup",
        "expected_failover": True,
        "target_node": "REPLICA-1",
        "rows_deleted": deleted,
        "rows_nulled": nulled,
        "gap_hours": 2,
        "icon": "🟠",
    }


def run_scenario_high() -> dict:
    """HIGH/CRITICAL severity: Massive gap + null storm + schema drift.

    Expected: Full failover PRIMARY → REPLICA-1 → REPLICA-2.
    Slack fires. Incident escalated.
    Simulates: Primary DB corruption, cascading failure.
    """
    db_path = get_active_db_path()
    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff_4h = (now - timedelta(hours=4)).isoformat()
        cutoff_6h = (now - timedelta(hours=6)).isoformat()

        # Delete last 4 hours
        cursor = conn.execute("DELETE FROM orders WHERE created_at > ?", (cutoff_4h,))
        deleted = cursor.rowcount

        # Null storm: nullify order_amount on all recent remaining rows
        cursor2 = conn.execute(
            "UPDATE orders SET order_amount = NULL WHERE created_at > ?",
            (cutoff_6h,),
        )
        nulled = cursor2.rowcount

        # Schema drift: add rogue column
        try:
            conn.execute(
                "ALTER TABLE orders ADD COLUMN upi_transaction_id TEXT"
            )
        except Exception:
            pass  # column may already exist from previous run

        conn.commit()
        logger.info(
            f"Scenario HIGH: deleted {deleted} rows, nulled {nulled} rows, "
            f"schema drift injected"
        )
    except Exception as e:
        logger.error(f"Scenario HIGH failed: {e}")
        deleted, nulled = 0, 0
    finally:
        conn.close()

    return {
        "scenario": "HIGH",
        "title": "PRIMARY DB Corruption — Cascading Failure",
        "description": (
            "4-hour gap + null storm + schema drift detected. "
            "Full cluster failover. PRIMARY marked failed, "
            "pipeline moves to REPLICA-2."
        ),
        "expected_action": "escalate_with_failover",
        "expected_failover": True,
        "target_node": "REPLICA-2",
        "rows_deleted": deleted,
        "rows_nulled": nulled,
        "schema_drift": True,
        "rogue_column": "upi_transaction_id",
        "gap_hours": 4,
        "icon": "🔴",
    }
