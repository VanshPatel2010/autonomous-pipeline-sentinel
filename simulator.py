"""Live Data Simulator: Continuously streams synthetic data to the database.

Simulates a real-world event-driven architecture by pushing 1-5 orders every
few seconds. Supports CLI flags to intentionally break the data pipeline
for testing and demonstration purposes.

Usage:
    python simulator.py
    python simulator.py --inject-gap
    python simulator.py --inject-nulls
"""

import argparse
import random
import time
import uuid
from datetime import datetime, timezone
import logging

from db.client import get_db_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def setup_tables():
    """Ensure the orders table exists in the target DB."""
    conn = get_db_connection()
    logger.info(f"Connected to DB. Postgres mode: {conn.is_postgres}")
    
    # Postgres syntax vs SQLite
    id_type = "TEXT PRIMARY KEY"
    datetime_type = "TIMESTAMP" if conn.is_postgres else "DATETIME"
    real_type = "REAL"
    
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS orders (
            order_id {id_type},
            created_at {datetime_type} NOT NULL,
            order_amount {real_type},
            source_db TEXT NOT NULL DEFAULT 'mumbai'
        );
        CREATE TABLE IF NOT EXISTS quarantine_orders (
            order_id {id_type},
            created_at {datetime_type} NOT NULL,
            order_amount {real_type},
            source_db TEXT NOT NULL,
            quarantine_reason TEXT,
            quarantined_at {datetime_type},
            run_id TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Tables verified.")


def generate_batch(inject_nulls: bool = False):
    """Generate a batch of 1 to 5 synthetic orders."""
    orders = []
    num_rows = random.randint(1, 5)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    for _ in range(num_rows):
        order_id = str(uuid.uuid4())[:12]
        created_at = now.isoformat()
        
        if inject_nulls or (random.random() < 0.03):
            # 3% natural nulls, or 100% if injecting
            order_amount = None
        else:
            order_amount = round(random.uniform(5.0, 500.0), 2)
            
        orders.append((order_id, created_at, order_amount, "mumbai"))
    return orders


import json
import os
import requests

def run_simulator(inject_gap=False, inject_nulls=False):
    """Run continuous data generation.

    Args:
        inject_gap: If True, pauses generation to simulate an outage.
        inject_nulls: If True, generates some records with null order_amount.
    """
    setup_tables()
    logger.info("Data Simulator started.")
    logger.info(f"Initial State - Gap: {inject_gap}, Nulls: {inject_nulls}")
    
    # State flags can be modified at runtime via Postgres table 'simulator_config'
    current_inject_gap = inject_gap
    current_inject_nulls = inject_nulls
    current_inject_duplicate = False
    
    try:
        while True:
            # Check state from PostgreSQL
            try:
                state_conn = get_db_connection()
                row = state_conn.execute("SELECT inject_gap, inject_nulls, inject_duplicate FROM simulator_config WHERE id = 1").fetchone()
                if row:
                    current_inject_gap = bool(row["inject_gap"])
                    current_inject_nulls = bool(row["inject_nulls"])
                    current_inject_duplicate = bool(row["inject_duplicate"])
                state_conn.close()
            except Exception as e:
                logger.warning(f"Could not read state from Postgres: {e}")
                
            if current_inject_gap:
                logger.warning("SIMULATING OUTAGE: Paused data generation")
                time.sleep(15)
                continue
                
            orders = generate_batch(current_inject_nulls)
            
            if current_inject_duplicate and orders:
                # Duplicate the first order in the batch
                orders.append(orders[0])
                
            # Retry logic for Supabase connection drops
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    conn = get_db_connection()
                    
                    # Using INSERT OR IGNORE style syntax handled by db.client.py
                    conn.executemany(
                        "INSERT INTO orders (order_id, created_at, order_amount, source_db) VALUES (?, ?, ?, ?)",
                        orders
                    )
                    conn.commit()
                    conn.close()
                    break
                except Exception as db_err:
                    if attempt < max_retries - 1:
                        logger.warning(f"DB connection dropped, retrying... ({db_err})")
                        time.sleep(2)
                    else:
                        logger.error(f"Failed to insert batch after {max_retries} attempts: {db_err}")
            
            # Fire Webhooks to simulate pg_net
            for order in orders:
                try:
                    payload = {
                        "type": "INSERT",
                        "table": "orders",
                        "record": {
                            "order_id": order[0],
                            "created_at": order[1],
                            "order_amount": order[2],
                            "source_db": order[3]
                        }
                    }
                    requests.post("http://localhost:8000/api/webhook/orders", json=payload, timeout=1)
                except Exception as e:
                    logger.debug(f"Webhook failed: {e}")
            
            logger.info(f"Streamed {len(orders)} orders into database and fired webhooks.")
            # Sleep 1-3 seconds to simulate continuous stream
            time.sleep(random.uniform(1.0, 3.0))
            
    except KeyboardInterrupt:
        logger.info("Simulator stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Data Simulator")
    parser.add_argument("--inject-gap", action="store_true", help="Simulate a database outage by halting data insertion.")
    parser.add_argument("--inject-nulls", action="store_true", help="Simulate data corruption by inserting NULL order amounts.")
    args = parser.parse_args()
    
    run_simulator(args.inject_gap, args.inject_nulls)
