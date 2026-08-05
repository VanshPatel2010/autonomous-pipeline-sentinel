"""Gap Tracker: records and manages data gaps discovered during monitoring.

When the Monitor detects missing data, the Repairer logs the gap here.
After a failover to a backup source, the gap tracker can mark gaps as
reconciled once the backfill is complete.
"""

from db.client import get_db_connection
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config import DB_PATH
from logging_config import logger


def record_gap(
    run_id: str,
    gap_minutes: float,
    source_db: str = "mumbai",
    estimated_rows: int = 0,
    db_path: str = DB_PATH,
) -> str:
    """Record a newly discovered data gap.

    Args:
        run_id: Run ID that discovered the gap.
        gap_minutes: Duration of the gap in minutes.
        source_db: Source database that experienced the outage.
        estimated_rows: Estimated number of missing rows.
        db_path: Path to the SQLite database.

    Returns:
        The generated gap_id for tracking.
    """
    conn = get_db_connection()
    gap_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc)
    start_time = (now - timedelta(minutes=gap_minutes)).isoformat()
    end_time = now.isoformat()

    try:
        conn.execute(
            """
            INSERT INTO data_gaps
                (gap_id, run_id, start_time, end_time, estimated_rows,
                 source_db, reconciled, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, 0, %s)
            """ if conn.is_postgres else
            """
            INSERT INTO data_gaps
                (gap_id, run_id, start_time, end_time, estimated_rows,
                 source_db, reconciled, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (gap_id, run_id, start_time, end_time, estimated_rows,
             source_db, now.isoformat()),
        )
        conn.commit()
        logger.info(
            f"[{run_id}] Gap recorded: {gap_id} | "
            f"{gap_minutes:.0f}min | ~{estimated_rows} rows | "
            f"source={source_db}"
        )
        return gap_id

    except Exception as e:
        logger.error(f"Failed to record gap: {e}")
        return ""
    finally:
        conn.close()


def mark_reconciled(gap_id: str, db_path: str = DB_PATH) -> bool:
    """Mark a data gap as reconciled (backfill complete).

    Args:
        gap_id: The gap ID to mark.
        db_path: Path to the SQLite database.

    Returns:
        True if the gap was found and updated, False otherwise.
    """
    conn = get_db_connection()

    try:
        cursor = conn.execute(
            "UPDATE data_gaps SET reconciled = 1 WHERE gap_id = %s" if conn.is_postgres else
            "UPDATE data_gaps SET reconciled = 1 WHERE gap_id = ?",
            (gap_id,),
        )
        conn.commit()

        if cursor.rowcount > 0:
            logger.info(f"Gap {gap_id} marked as reconciled")
            return True

        logger.warning(f"Gap {gap_id} not found for reconciliation")
        return False

    except Exception as e:
        logger.error(f"Failed to mark gap as reconciled: {e}")
        return False
    finally:
        conn.close()


def get_unreconciled_gaps(
    limit: int = 20, db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieve all unreconciled (pending) data gaps.

    Args:
        limit: Maximum number of gaps to return.
        db_path: Path to the SQLite database.

    Returns:
        List of gap dicts, most recent first.
    """
    conn = get_db_connection()

    try:
        rows = conn.execute(
            """
            SELECT gap_id, run_id, start_time, end_time,
                   estimated_rows, source_db, reconciled, created_at
            FROM data_gaps
            WHERE reconciled = 0
            ORDER BY created_at DESC
            LIMIT %s
            """ if conn.is_postgres else
            """
            SELECT gap_id, run_id, start_time, end_time,
                   estimated_rows, source_db, reconciled, created_at
            FROM data_gaps
            WHERE reconciled = 0
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    except Exception as e:
        logger.error(f"Failed to query gaps: {e}")
        return []
    finally:
        conn.close()


def get_all_gaps(
    limit: int = 50, db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieve all data gaps (both reconciled and pending).

    Args:
        limit: Maximum number of gaps to return.
        db_path: Path to the SQLite database.

    Returns:
        List of gap dicts, most recent first.
    """
    conn = get_db_connection()

    try:
        rows = conn.execute(
            """
            SELECT gap_id, run_id, start_time, end_time,
                   estimated_rows, source_db, reconciled, created_at
            FROM data_gaps
            ORDER BY created_at DESC
            LIMIT %s
            """ if conn.is_postgres else
            """
            SELECT gap_id, run_id, start_time, end_time,
                   estimated_rows, source_db, reconciled, created_at
            FROM data_gaps
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [dict(row) for row in rows]

    except Exception as e:
        logger.error(f"Failed to query gaps: {e}")
        return []
    finally:
        conn.close()
