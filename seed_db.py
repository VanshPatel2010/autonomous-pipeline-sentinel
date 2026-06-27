"""Data simulator: seeds SQLite with 30 days of synthetic order data.

Generates ~200 orders per 5-minute window with:
- One configurable 4-hour gap (simulates Mumbai DB outage)
- 3% random nulls on order_amount
- Realistic order amounts between $5 and $500

Usage:
    python seed_db.py
"""

import random
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from config import (
    DB_PATH,
    GAP_DURATION_HOURS,
    NULL_RATE,
    ROWS_PER_WINDOW,
    SIMULATION_DAYS,
)
from logging_config import logger


def create_tables(conn: sqlite3.Connection) -> None:
    """Create the orders table and all supporting tables.

    Tables created:
    - orders: Primary data table for pipeline monitoring
    - quarantine_orders: Holds rows quarantined by Repairer (Phase 3)
    - data_gaps: Tracks data gaps created by failover (Phase 3)
    - schema_snapshots: Audit trail for schema changes (Phase 5)
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL DEFAULT 'mumbai'
        );

        CREATE INDEX IF NOT EXISTS idx_orders_created_at
        ON orders(created_at);

        CREATE TABLE IF NOT EXISTS quarantine_orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL,
            quarantine_reason TEXT,
            quarantined_at DATETIME,
            run_id TEXT
        );

        CREATE TABLE IF NOT EXISTS data_gaps (
            gap_id TEXT PRIMARY KEY,
            run_id TEXT,
            start_time TEXT,
            end_time TEXT,
            estimated_rows INTEGER,
            source_db TEXT,
            reconciled INTEGER DEFAULT 0,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS schema_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            schema_json TEXT NOT NULL,
            snapshot_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()


def generate_orders(
    start_time: datetime,
    end_time: datetime,
    gap_start: datetime,
    gap_end: datetime,
) -> List[Tuple[str, str, Optional[float], str]]:
    """Generate synthetic order rows for the given time range.

    Args:
        start_time: Beginning of the simulation period.
        end_time: End of the simulation period.
        gap_start: Start of the simulated outage gap.
        gap_end: End of the simulated outage gap.

    Returns:
        List of (order_id, created_at, order_amount, source_db) tuples.
    """
    orders: List[Tuple[str, str, Optional[float], str]] = []
    current = start_time
    window = timedelta(minutes=5)

    while current < end_time:
        window_end = current + window

        # Skip this window if it falls within the gap
        if current >= gap_start and current < gap_end:
            current = window_end
            continue

        # Generate ~ROWS_PER_WINDOW orders for this 5-min window
        # Add some natural variation (±20%)
        num_rows = random.randint(
            int(ROWS_PER_WINDOW * 0.8),
            int(ROWS_PER_WINDOW * 1.2),
        )

        for _ in range(num_rows):
            order_id = str(uuid.uuid4())[:12]
            # Random timestamp within this 5-min window
            offset_seconds = random.randint(0, 299)
            created_at = current + timedelta(seconds=offset_seconds)

            # Order amount: $5 to $500, with 3% chance of NULL
            if random.random() < NULL_RATE:
                order_amount: Optional[float] = None
            else:
                order_amount = round(random.uniform(5.0, 500.0), 2)

            source_db = "mumbai"
            orders.append((order_id, created_at.isoformat(), order_amount, source_db))

        current = window_end

    return orders


def seed_database(db_path: str = None, days: int = SIMULATION_DAYS, gap_hours: int = GAP_DURATION_HOURS) -> dict:
    """Seed the SQLite database with synthetic order data.

    When no db_path is provided, seeds the PRIMARY cluster node and
    syncs all replicas via reset_all_nodes(). When a specific db_path
    is given, seeds only that single database file.

    Args:
        db_path: Path to the SQLite database file. If None, uses the
                 cluster PRIMARY node and syncs replicas.
        days: Number of days of historical data to generate.
        gap_hours: Duration of the simulated outage gap in hours.

    Returns:
        Dict with seeding results: orders_inserted, null_count,
        gap_start, gap_end, success.
    """
    if db_path is not None:
        # Direct path specified — seed only this DB (legacy / dashboard sidebar)
        return _seed_single_db(db_path, days, gap_hours)

    # No path specified — seed PRIMARY and sync to replicas
    try:
        from db.database_manager import DB_NODES, reset_all_nodes
        import os

        primary_path = DB_NODES[0]["path"]
        os.makedirs(os.path.dirname(primary_path), exist_ok=True)
        result = _seed_single_db(primary_path, days, gap_hours)
        # Sync PRIMARY to all replicas and reset active node
        reset_all_nodes()
        result["nodes_seeded"] = len(DB_NODES)
        logger.info(f"Cluster seeded: {len(DB_NODES)} nodes synced from PRIMARY")
        return result
    except ImportError:
        # Fallback: database_manager not available, use config DB_PATH
        return _seed_single_db(DB_PATH, days, gap_hours)


def _seed_single_db(db_path: str, days: int = SIMULATION_DAYS, gap_hours: int = GAP_DURATION_HOURS) -> dict:
    """Seed a single SQLite database file with synthetic order data.

    Args:
        db_path: Path to the SQLite database file.
        days: Number of days of historical data to generate.
        gap_hours: Duration of the simulated outage gap in hours.

    Returns:
        Dict with seeding results.
    """
    conn = sqlite3.connect(db_path)

    # Create tables
    create_tables(conn)

    # Clear existing data
    conn.execute("DELETE FROM orders")
    conn.commit()

    # Time range: last N days
    end_time = datetime.now(timezone.utc).replace(tzinfo=None)
    start_time = end_time - timedelta(days=days)

    # Place the gap ~2 days ago at 3:14 AM (matching the Mumbai outage scenario)
    gap_day = end_time - timedelta(days=2)
    gap_start = gap_day.replace(hour=3, minute=14, second=0, microsecond=0)
    gap_end = gap_start + timedelta(hours=gap_hours)

    logger.info(f"Generating {days} days of synthetic data...")
    logger.info(f"Gap window: {gap_start.isoformat()} to {gap_end.isoformat()}")

    # Generate orders
    orders = generate_orders(start_time, end_time, gap_start, gap_end)

    # Batch insert
    conn.executemany(
        "INSERT OR IGNORE INTO orders (order_id, created_at, order_amount, source_db) VALUES (?, ?, ?, ?)",
        orders,
    )
    conn.commit()

    # Stats
    total = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    nulls = conn.execute("SELECT COUNT(*) FROM orders WHERE order_amount IS NULL").fetchone()[0]
    null_pct = (nulls / total * 100) if total > 0 else 0

    logger.info(f"Seeded {total:,} orders into {db_path}")
    logger.info(f"Null order_amounts: {nulls:,} ({null_pct:.1f}%)")
    logger.info(f"Gap: {gap_hours}h starting at {gap_start}")

    conn.close()

    return {
        "success": True,
        "orders_inserted": total,
        "null_count": nulls,
        "null_pct": null_pct,
        "gap_start": gap_start.isoformat(),
        "gap_end": gap_end.isoformat(),
        "days": days,
    }


if __name__ == "__main__":
    result = seed_database()
    print(f"Seeded {result['orders_inserted']:,} orders")
    print(f"Null order_amounts: {result['null_count']:,} ({result['null_pct']:.1f}%)")
    print(f"Gap: {result['gap_start']} to {result['gap_end']}")

