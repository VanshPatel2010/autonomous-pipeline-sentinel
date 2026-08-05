"""Playbook Store: procedural long-term memory (LTM).

Stores repair playbooks — which actions work for which anomaly types.
The Repairer queries playbooks before choosing a strategy,
and updates success/failure counts after each attempt.
This is how the system learns what fixes work over time.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from db.client import get_db_connection
from config import DB_PATH
from logging_config import logger


def get_best_playbook(
    anomaly_type: str,
    severity: str,
    db_path: str = DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Find the most successful playbook for the given anomaly+severity.

    Returns the playbook with the highest success rate (success_count
    relative to total attempts). Ties are broken by most recent usage.

    Args:
        anomaly_type: Type of anomaly (e.g., 'missing_data', 'data_quality').
        severity: Severity level (e.g., 'LOW', 'MEDIUM', 'HIGH').
        db_path: Path to the SQLite database.

    Returns:
        Best matching playbook dict, or None if no playbooks exist.
    """
    conn = get_db_connection()

    try:
        row = conn.execute(
            """
            SELECT id, anomaly_type, severity, action_taken,
                   success_count, failure_count, last_used
            FROM playbooks
            WHERE anomaly_type = %s AND severity = %s
            ORDER BY
                CAST(success_count AS REAL) / GREATEST(success_count + failure_count, 1) DESC,
                last_used DESC
            LIMIT 1
            """ if conn.is_postgres else
            """
            SELECT id, anomaly_type, severity, action_taken,
                   success_count, failure_count, last_used
            FROM playbooks
            WHERE anomaly_type = %s AND severity = %s
            ORDER BY
                CAST(success_count AS REAL) / MAX(success_count + failure_count, 1) DESC,
                last_used DESC
            LIMIT 1
            """,
            (anomaly_type, severity),
        ).fetchone()

        if row:
            playbook = dict(row)
            total = playbook["success_count"] + playbook["failure_count"]
            playbook["success_rate"] = (
                playbook["success_count"] / total if total > 0 else 0.0
            )
            logger.info(
                f"Playbook found: {playbook['action_taken']} "
                f"(success_rate={playbook['success_rate']:.0%}, "
                f"used={total} times)"
            )
            return playbook

        logger.info(
            f"No playbook found for {anomaly_type}/{severity} — "
            f"will use default strategy"
        )
        return None

    except Exception as e:
        logger.error(f"Failed to query playbooks: {e}")
        return None
    finally:
        conn.close()


def record_outcome(
    anomaly_type: str,
    severity: str,
    action_taken: str,
    success: bool,
    db_path: str = DB_PATH,
) -> None:
    """Record the outcome of a repair action in the playbook.

    Uses INSERT OR IGNORE + UPDATE pattern to handle both new and
    existing playbook entries. Increments success_count or failure_count.

    Args:
        anomaly_type: Type of anomaly that was repaired.
        severity: Severity level of the anomaly.
        action_taken: Description of the repair action performed.
        success: Whether the repair was successful.
        db_path: Path to the SQLite database.
    """
    conn = get_db_connection()
    now = datetime.now(timezone.utc).isoformat()

    try:
        # Ensure the playbook entry exists
        conn.execute(
            """
            INSERT INTO playbooks
                (anomaly_type, severity, action_taken, success_count, failure_count, last_used)
            VALUES (%s, %s, %s, 0, 0, %s)
            ON CONFLICT DO NOTHING
            """,
            (anomaly_type, severity, action_taken, now),
        )

        # Update counts
        if success:
            conn.execute(
                """
                UPDATE playbooks
                SET success_count = success_count + 1, last_used = %s
                WHERE anomaly_type = %s AND severity = %s AND action_taken = %s
                """,
                (now, anomaly_type, severity, action_taken),
            )
        else:
            conn.execute(
                """
                UPDATE playbooks
                SET failure_count = failure_count + 1, last_used = %s
                WHERE anomaly_type = %s AND severity = %s AND action_taken = %s
                """,
                (now, anomaly_type, severity, action_taken),
            )

        conn.commit()
        outcome = "SUCCESS" if success else "FAILURE"
        logger.info(
            f"Playbook updated: {action_taken} | "
            f"{anomaly_type}/{severity} → {outcome}"
        )

    except Exception as e:
        logger.error(f"Failed to record playbook outcome: {e}")
    finally:
        conn.close()


def get_all_playbooks(
    limit: int = 50, db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieve all playbook entries, ordered by success rate.

    Args:
        limit: Maximum number of playbooks to return.
        db_path: Path to the SQLite database.

    Returns:
        List of playbook dicts.
    """
    conn = get_db_connection()

    try:
        rows = conn.execute(
            """
            SELECT id, anomaly_type, severity, action_taken,
                   success_count, failure_count, last_used
            FROM playbooks
            ORDER BY
                CAST(success_count AS REAL) / GREATEST(success_count + failure_count, 1) DESC
            LIMIT %s
            """ if conn.is_postgres else
            """
            SELECT id, anomaly_type, severity, action_taken,
                   success_count, failure_count, last_used
            FROM playbooks
            ORDER BY
                CAST(success_count AS REAL) / MAX(success_count + failure_count, 1) DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    except Exception as e:
        logger.error(f"Failed to query playbooks: {e}")
        return []
    finally:
        conn.close()
