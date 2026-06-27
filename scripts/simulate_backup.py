"""Delhi Replica Simulator: simulates a backup database for failover.

When the Mumbai source DB goes down, the Repairer Agent switches to
the Delhi replica. This script creates a `backup_orders` table with
synthetic data to simulate the failover source.

The simulator copies a sample of existing orders and re-tags them as
coming from 'delhi', allowing the Repairer to demonstrate real failover
without needing a second database.

Usage:
    python scripts/simulate_backup.py       # Populate backup_orders table
"""

import random
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from config import BACKUP_TABLE, DB_PATH, ROWS_PER_WINDOW
from logging_config import logger


def create_backup_table(conn: sqlite3.Connection) -> None:
    """Create the backup_orders table if it doesn't exist.

    This table mirrors the orders schema but stores data from the
    Delhi replica.

    Args:
        conn: SQLite connection.
    """
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS {BACKUP_TABLE} (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL DEFAULT 'delhi'
        );

        CREATE INDEX IF NOT EXISTS idx_backup_orders_created_at
        ON {BACKUP_TABLE}(created_at);
    """)
    conn.commit()


def populate_backup(
    hours: int = 6,
    db_path: str = DB_PATH,
) -> int:
    """Populate the backup_orders table with Delhi replica data.

    Generates synthetic orders for the specified number of recent hours,
    simulating what the Delhi replica would have captured during a
    Mumbai outage.

    Args:
        hours: Number of hours of backup data to generate.
        db_path: Path to the SQLite database.

    Returns:
        Number of backup orders inserted.
    """
    conn = sqlite3.connect(db_path)
    create_backup_table(conn)

    # Clear existing backup data
    conn.execute(f"DELETE FROM {BACKUP_TABLE}")
    conn.commit()

    # Generate backup data for the last N hours
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    start = now - timedelta(hours=hours)
    window = timedelta(minutes=5)
    current = start

    orders: List[Tuple[str, str, Optional[float], str]] = []

    while current < now:
        window_end = current + window
        # Generate fewer rows than primary (80% capacity — Delhi is smaller)
        num_rows = random.randint(
            int(ROWS_PER_WINDOW * 0.6),
            int(ROWS_PER_WINDOW * 0.9),
        )

        for _ in range(num_rows):
            order_id = f"del-{str(uuid.uuid4())[:8]}"
            offset_seconds = random.randint(0, 299)
            created_at = current + timedelta(seconds=offset_seconds)
            order_amount: Optional[float] = round(random.uniform(5.0, 500.0), 2)
            orders.append((order_id, created_at.isoformat(), order_amount, "delhi"))

        current = window_end

    # Batch insert
    conn.executemany(
        f"INSERT OR IGNORE INTO {BACKUP_TABLE} "
        "(order_id, created_at, order_amount, source_db) VALUES (?, ?, ?, ?)",
        orders,
    )
    conn.commit()

    total = conn.execute(f"SELECT COUNT(*) FROM {BACKUP_TABLE}").fetchone()[0]
    logger.info(f"Delhi replica populated: {total:,} backup orders ({hours}h window)")

    conn.close()
    return total


def failover_from_backup(
    gap_start: str,
    gap_end: str,
    db_path: str = DB_PATH,
) -> int:
    """Copy backup orders into the primary orders table for the gap window.

    This simulates a failover: orders from the Delhi replica are injected
    into the primary table for the duration of the outage gap.

    Args:
        gap_start: ISO timestamp for the start of the gap.
        gap_end: ISO timestamp for the end of the gap.
        db_path: Path to the SQLite database.

    Returns:
        Number of rows copied from backup to primary.
    """
    conn = sqlite3.connect(db_path)
    create_backup_table(conn)

    try:
        cursor = conn.execute(
            f"""
            INSERT OR IGNORE INTO orders (order_id, created_at, order_amount, source_db)
            SELECT order_id, created_at, order_amount, source_db
            FROM {BACKUP_TABLE}
            WHERE created_at >= ? AND created_at <= ?
            """,
            (gap_start, gap_end),
        )
        conn.commit()

        rows_copied = cursor.rowcount
        logger.info(
            f"Failover complete: {rows_copied} rows copied from "
            f"Delhi replica ({gap_start} to {gap_end})"
        )
        return rows_copied

    except sqlite3.Error as e:
        logger.error(f"Failover failed: {e}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    total = populate_backup()
    print(f"Delhi replica ready: {total:,} backup orders")
