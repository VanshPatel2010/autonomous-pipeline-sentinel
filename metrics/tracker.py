"""Metrics Tracker: MTTR, detection rate, severity distribution.

Computes operational metrics from the incidents table for the
Streamlit dashboard and monitoring reports.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from logging_config import logger


def compute_mttr(conn: sqlite3.Connection) -> float:
    """Compute Mean Time To Resolution (MTTR) in minutes.

    Averages (resolved_at - timestamp) for all resolved incidents
    in the last 30 days.

    Args:
        conn: Active SQLite connection.

    Returns:
        MTTR in minutes, or 0.0 if no resolved incidents found.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    cursor = conn.execute(
        """
        SELECT AVG(
            ABS(julianday(resolved_at) - julianday(timestamp)) * 24 * 60
        ) as avg_mttr
        FROM incidents
        WHERE resolved = 1
          AND timestamp > ?
          AND resolved_at IS NOT NULL
          AND resolved_at != ''
        """,
        (cutoff,),
    )
    row = cursor.fetchone()

    if row and row[0] is not None:
        mttr = float(row[0])
        logger.info(f'MTTR (last 30 days): {mttr:.2f} minutes')
        return mttr

    logger.info('No resolved incidents in last 30 days, MTTR = 0.0')
    return 0.0


def compute_detection_rate(conn: sqlite3.Connection) -> float:
    """Compute anomaly detection rate over the last 7 days.

    Ratio of incidents with a non-empty anomaly_type to total incidents.

    Args:
        conn: Active SQLite connection.

    Returns:
        Detection rate as a float (0.0 to 1.0), or 0.0 if no incidents.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN anomaly_type IS NOT NULL AND anomaly_type != ''
                     THEN 1 ELSE 0 END) as detected
        FROM incidents
        WHERE timestamp > ?
        """,
        (cutoff,),
    )
    row = cursor.fetchone()

    if row and row[0] and row[0] > 0:
        rate = float(row[1]) / float(row[0])
        logger.info(
            f'Detection rate (last 7 days): {rate:.2%} '
            f'({row[1]}/{row[0]})'
        )
        return rate

    logger.info('No incidents in last 7 days, detection rate = 0.0')
    return 0.0


def get_severity_distribution(conn: sqlite3.Connection) -> Dict[str, int]:
    """Get incident count grouped by severity for the last 30 days.

    Args:
        conn: Active SQLite connection.

    Returns:
        Dict mapping severity level to count, e.g. {'HIGH': 5, 'MEDIUM': 3}.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    cursor = conn.execute(
        """
        SELECT severity, COUNT(*) as cnt
        FROM incidents
        WHERE timestamp > ?
          AND severity IS NOT NULL
          AND severity != ''
        GROUP BY severity
        ORDER BY cnt DESC
        """,
        (cutoff,),
    )

    distribution: Dict[str, int] = {}
    for row in cursor.fetchall():
        distribution[row[0]] = row[1]

    logger.info(f'Severity distribution (last 30 days): {distribution}')
    return distribution


def get_recent_incidents(
    conn: sqlite3.Connection, limit: int = 50
) -> List[Dict[str, Any]]:
    """Fetch the most recent incidents with all columns.

    Args:
        conn: Active SQLite connection.
        limit: Maximum number of incidents to return.

    Returns:
        List of incident dicts with all column values.
    """
    cursor = conn.execute(
        """
        SELECT run_id, timestamp, anomaly_type, severity,
               gap_minutes, root_cause, affected_tables,
               fix_taken, resolved, resolved_at, confidence
        FROM incidents
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )

    incidents: List[Dict[str, Any]] = []
    for row in cursor.fetchall():
        incidents.append({
            'run_id': row[0],
            'timestamp': row[1],
            'anomaly_type': row[2],
            'severity': row[3],
            'gap_minutes': row[4],
            'root_cause': row[5],
            'affected_tables': row[6],
            'fix_taken': row[7],
            'resolved': row[8],
            'resolved_at': row[9],
            'confidence': row[10],
        })

    logger.info(f'Fetched {len(incidents)} recent incidents')
    return incidents


def get_daily_incident_counts(
    conn: sqlite3.Connection, days: int = 30
) -> List[Dict[str, Any]]:
    """Get daily incident counts for the specified number of past days.

    Args:
        conn: Active SQLite connection.
        days: Number of days to look back.

    Returns:
        List of dicts with 'date' and 'count' keys, sorted chronologically.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    cursor = conn.execute(
        """
        SELECT DATE(timestamp) as day, COUNT(*) as cnt
        FROM incidents
        WHERE timestamp > ?
        GROUP BY DATE(timestamp)
        ORDER BY day ASC
        """,
        (cutoff,),
    )

    counts: List[Dict[str, Any]] = []
    for row in cursor.fetchall():
        counts.append({
            'date': row[0],
            'count': row[1],
        })

    logger.info(f'Daily incident counts: {len(counts)} days with incidents')
    return counts
