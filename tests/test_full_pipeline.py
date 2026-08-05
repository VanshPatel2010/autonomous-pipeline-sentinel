"""
Integration tests for the full Autonomous Data Pipeline.
Tests cover: healthy pipeline, missing data triggering repairs,
schema drift detection, graph structure, and the run_pipeline entry point.
"""

import sys
sys.path.insert(0, '/Users/vanshpatel/documents/Autonomous Data Pipeline Agent')

import sqlite3
import uuid
import random
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helper – seed synthetic orders
# ---------------------------------------------------------------------------

def seed_orders(db_path, days=7, rows_per_window=200):
    """
    Generate synthetic orders across *days* using 5-minute windows.
    Each order gets a random UUID, a created_at inside the window,
    a random amount between 5-500, and source_db='mumbai'.
    Rows are batch-inserted for speed.
    """
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    start = now - timedelta(days=days)

    rows = []
    current = start
    while current < now:
        window_end = current + timedelta(minutes=5)
        for _ in range(rows_per_window):
            order_id = str(uuid.uuid4())
            # random timestamp within the 5-min window
            offset_seconds = random.uniform(0, 300)
            created_at = current + timedelta(seconds=offset_seconds)
            order_amount = round(random.uniform(5, 500), 2)
            rows.append((order_id, created_at.isoformat(), order_amount, 'mumbai'))
        current = window_end

    conn.executemany(
        'INSERT INTO orders (order_id, created_at, order_amount, source_db) VALUES (?, ?, ?, ?)',
        rows,
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixture – temporary pipeline database
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_db(tmp_path, monkeypatch):
    """
    Create a throw-away SQLite database with ALL required tables/indexes,
    patch config.DB_PATH and config.MOCK_MODE, then yield the db path.
    """
    db_path = str(tmp_path / 'test_pipeline.db')
    conn = sqlite3.connect(db_path)

    # -- orders
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL DEFAULT 'mumbai'
        );
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);"
    )

    # -- quarantine_orders
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quarantine_orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL,
            quarantine_reason TEXT,
            quarantined_at DATETIME,
            run_id TEXT
        );
    """)

    # -- backup_orders
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backup_orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL DEFAULT 'delhi'
        );
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_backup_orders_created_at ON backup_orders(created_at);"
    )

    # -- incidents
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            anomaly_type TEXT,
            severity TEXT,
            gap_minutes REAL DEFAULT 0,
            root_cause TEXT,
            affected_tables TEXT,
            fix_taken TEXT,
            resolved INTEGER DEFAULT 0,
            resolved_at TEXT,
            confidence REAL DEFAULT 0.0
        );
    """)

    # -- playbooks
    conn.execute("""
        CREATE TABLE IF NOT EXISTS playbooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            anomaly_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            action_taken TEXT NOT NULL,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            last_used TEXT,
            UNIQUE(anomaly_type, severity, action_taken)
        );
    """)

    # -- data_gaps
    conn.execute("""
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
    """)

    # -- schema_snapshots
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            schema_json TEXT NOT NULL,
            snapshot_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()

    # Patch config so every agent/module picks up the temp DB
    import config
    monkeypatch.setattr(config, 'DB_PATH', db_path)
    monkeypatch.setattr(config, 'MOCK_MODE', True)

    yield db_path


# ---------------------------------------------------------------------------
# 1. Healthy pipeline – monitor sees no anomaly, pipeline stops early
# ---------------------------------------------------------------------------

def test_healthy_pipeline_ends_after_monitor(pipeline_db):
    seed_orders(pipeline_db, days=7, rows_per_window=200)

    from graph import build_graph
    from state import create_initial_state

    graph = build_graph()
    initial_state = create_initial_state(
        run_id='test-healthy',
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    result = graph.invoke(initial_state)

    assert result['anomaly_detected'] is False
    assert result['diagnoser_output'] == {}
    assert result['repairer_output'] == {}
    assert result.get('slack_sent') is False


# ---------------------------------------------------------------------------
# 2. Missing data triggers the full pipeline
# ---------------------------------------------------------------------------

def test_missing_data_triggers_full_pipeline(pipeline_db):
    seed_orders(pipeline_db, days=7, rows_per_window=200)

    # Remove all recent data to simulate a pipeline outage.
    # Use timezone-naive datetime to match the format used by seed_orders.
    conn = sqlite3.connect(pipeline_db)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = (now - timedelta(minutes=30)).isoformat()
    conn.execute('DELETE FROM orders WHERE created_at > ?', (cutoff,))
    conn.commit()
    conn.close()

    from graph import build_graph
    from state import create_initial_state

    graph = build_graph()
    initial_state = create_initial_state(
        run_id='test-missing',
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    result = graph.invoke(initial_state)

    assert result['anomaly_detected'] is True
    assert result['anomaly_type'] in ('missing_data', 'schema_drift')
    assert result['diagnoser_output'] != {}
    assert result['slack_sent'] is True


# ---------------------------------------------------------------------------
# 3. Schema drift detected
# ---------------------------------------------------------------------------

def test_schema_drift_detected(pipeline_db):
    seed_orders(pipeline_db, days=7, rows_per_window=200)

    # Introduce an unexpected column → schema drift
    conn = sqlite3.connect(pipeline_db)
    conn.execute('ALTER TABLE orders ADD COLUMN extra_col TEXT')
    conn.commit()
    conn.close()

    from graph import build_graph
    from state import create_initial_state

    graph = build_graph()
    initial_state = create_initial_state(
        run_id='test-drift',
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    result = graph.invoke(initial_state)

    assert result['anomaly_detected'] is True
    assert result['anomaly_type'] == 'schema_drift'


# ---------------------------------------------------------------------------
# 4. Graph structure validation (no DB needed)
# ---------------------------------------------------------------------------

def test_graph_structure():
    from graph import build_graph

    graph = build_graph()

    assert graph is not None
    assert callable(getattr(graph, 'invoke', None))


# ---------------------------------------------------------------------------
# 5. run_pipeline entry-point with mocked graph + incident store
# ---------------------------------------------------------------------------

def test_run_pipeline_function(pipeline_db):
    fake_result = {
        'run_id': 'test-0001',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'anomaly_detected': False,
        'anomaly_type': '',
        'severity': 'NONE',
        'gap_minutes': 0.0,
        'affected_tables': [],
        'raw_count': 200,
        'expected_avg': 200.0,
        'null_rate': 0.0,
        'diagnoser_output': {},
        'repairer_output': {},
        'slack_sent': False,
    }

    with patch('main.build_graph') as mock_build_graph, \
         patch('main.insert_incident') as mock_insert, \
         patch('main.get_checkpointer'):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = fake_result
        mock_build_graph.return_value = mock_graph
        
        from main import run_pipeline
        run_pipeline()

        mock_graph.invoke.assert_called()
        mock_insert.assert_called()
