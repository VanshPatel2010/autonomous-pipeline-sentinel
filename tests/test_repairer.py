"""Tests for the Repairer Agent (Phase 3).

Verifies:
- Skipping repair when no anomaly detected
- Skipping repair when diagnoser confidence is below threshold
- Wait-and-retry for LOW severity missing_data
- Switch-to-backup for MEDIUM severity missing_data
- Quarantine bad data for HIGH severity data_quality
- Escalation to human for CRITICAL severity
- Playbook lookup overrides default strategy
- Outcome recording in playbooks (procedural LTM)
- Backup failover copies rows into primary table
- Quarantine moves null rows out of primary table
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agents.repairer import RepairerAgent
from state import create_initial_state


# ── Helpers ───────────────────────────────────────────────────────────


def _init_db(db_path: str) -> None:
    """Create all tables needed by RepairerAgent in a temp database."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL DEFAULT 'mumbai'
        );

        CREATE INDEX IF NOT EXISTS idx_orders_created_at
        ON orders(created_at);

        CREATE TABLE IF NOT EXISTS backup_orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL DEFAULT 'delhi'
        );

        CREATE INDEX IF NOT EXISTS idx_backup_orders_created_at
        ON backup_orders(created_at);

        CREATE TABLE IF NOT EXISTS quarantine_orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL,
            quarantine_reason TEXT,
            quarantined_at DATETIME,
            run_id TEXT
        );

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
    conn.commit()
    conn.close()


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary SQLite database with all required tables."""
    path = str(tmp_path / "test_repairer.db")
    _init_db(path)
    return path


@pytest.fixture
def agent(db_path):
    """Create a RepairerAgent wired to the temp database."""
    return RepairerAgent(db_path=db_path)


@pytest.fixture
def healthy_state():
    """Create a state dict with no anomaly detected."""
    return create_initial_state(
        run_id="repair-healthy-001",
        timestamp="2026-06-25T08:00:00+00:00",
    )


@pytest.fixture
def low_missing_state():
    """LOW severity missing_data anomaly with sufficient confidence."""
    state = create_initial_state(
        run_id="repair-low-001",
        timestamp="2026-06-25T08:00:00+00:00",
    )
    state["anomaly_detected"] = True
    state["anomaly_type"] = "missing_data"
    state["severity"] = "LOW"
    state["gap_minutes"] = 15.0
    state["raw_count"] = 80
    state["expected_avg"] = 200.0
    state["affected_tables"] = ["orders"]
    state["diagnoser_output"] = {
        "root_cause": "Brief network hiccup",
        "confidence": 0.75,
        "estimated_missing_rows": 120,
        "affected_tables": ["orders"],
        "maintenance_window_likely": False,
    }
    return state


@pytest.fixture
def medium_missing_state():
    """MEDIUM severity missing_data anomaly with sufficient confidence."""
    state = create_initial_state(
        run_id="repair-med-001",
        timestamp="2026-06-25T08:00:00+00:00",
    )
    state["anomaly_detected"] = True
    state["anomaly_type"] = "missing_data"
    state["severity"] = "MEDIUM"
    state["gap_minutes"] = 60.0
    state["raw_count"] = 0
    state["expected_avg"] = 200.0
    state["affected_tables"] = ["orders"]
    state["diagnoser_output"] = {
        "root_cause": "Mumbai source DB outage",
        "confidence": 0.85,
        "estimated_missing_rows": 2400,
        "affected_tables": ["orders"],
        "maintenance_window_likely": False,
    }
    return state


@pytest.fixture
def high_quality_state():
    """HIGH severity data_quality anomaly with sufficient confidence."""
    state = create_initial_state(
        run_id="repair-high-001",
        timestamp="2026-06-25T08:00:00+00:00",
    )
    state["anomaly_detected"] = True
    state["anomaly_type"] = "data_quality"
    state["severity"] = "HIGH"
    state["gap_minutes"] = 0.0
    state["raw_count"] = 200
    state["expected_avg"] = 200.0
    state["null_rate"] = 0.25
    state["affected_tables"] = ["orders"]
    state["diagnoser_output"] = {
        "root_cause": "Upstream validation failure",
        "confidence": 0.90,
        "estimated_missing_rows": 0,
        "affected_tables": ["orders"],
        "maintenance_window_likely": False,
    }
    return state


@pytest.fixture
def critical_state():
    """CRITICAL severity anomaly with sufficient confidence."""
    state = create_initial_state(
        run_id="repair-crit-001",
        timestamp="2026-06-25T08:00:00+00:00",
    )
    state["anomaly_detected"] = True
    state["anomaly_type"] = "missing_data"
    state["severity"] = "CRITICAL"
    state["gap_minutes"] = 500.0
    state["raw_count"] = 0
    state["expected_avg"] = 200.0
    state["affected_tables"] = ["orders"]
    state["diagnoser_output"] = {
        "root_cause": "Complete DB failure",
        "confidence": 0.95,
        "estimated_missing_rows": 20000,
        "affected_tables": ["orders"],
        "maintenance_window_likely": False,
    }
    return state


# ── Skip-path Tests ───────────────────────────────────────────────────


class TestSkipPaths:
    """Tests for conditions that cause the repairer to skip repair."""

    def test_no_anomaly_skips_repair(self, agent, healthy_state):
        """When anomaly_detected=False, repairer_output should be empty dict."""
        result = agent.run(healthy_state)

        assert result["repairer_output"] == {}

    def test_low_confidence_skips_repair(self, agent):
        """When confidence < CONFIDENCE_MIN (0.6), should skip with 'skipped_low_confidence'."""
        state = create_initial_state(
            run_id="repair-lowconf-001",
            timestamp="2026-06-25T08:00:00+00:00",
        )
        state["anomaly_detected"] = True
        state["anomaly_type"] = "missing_data"
        state["severity"] = "MEDIUM"
        state["gap_minutes"] = 60.0
        state["affected_tables"] = ["orders"]
        state["diagnoser_output"] = {
            "root_cause": "Uncertain cause",
            "confidence": 0.45,  # Below CONFIDENCE_MIN (0.6)
            "estimated_missing_rows": 500,
            "affected_tables": ["orders"],
            "maintenance_window_likely": False,
        }

        result = agent.run(state)

        output = result["repairer_output"]
        assert output["action_taken"] == "skipped_low_confidence"
        assert output["success"] is False
        assert output["rows_affected"] == 0
        assert "0.45" in output["details"]


# ── Strategy Tests ────────────────────────────────────────────────────


class TestRepairStrategies:
    """Tests for each graduated repair strategy."""

    def test_wait_and_retry_for_low_severity(self, agent, low_missing_state):
        """LOW severity missing_data → wait_and_retry action."""
        result = agent.run(low_missing_state)

        output = result["repairer_output"]
        assert output["action_taken"] == "wait_and_retry"
        assert output["success"] is True
        assert output["rows_affected"] == 0

    def test_switch_to_backup_for_medium_missing_data(self, agent, medium_missing_state):
        """MEDIUM severity missing_data → switch_to_backup action."""
        result = agent.run(medium_missing_state)

        output = result["repairer_output"]
        assert output["action_taken"] == "switch_to_backup"
        # Success depends on whether backup rows fall in the gap window;
        # the key assertion is that the strategy was selected.
        assert "switch_to_backup" == output["action_taken"]
        assert "rows_affected" in output

    def test_quarantine_bad_data_for_high_data_quality(self, agent, high_quality_state, db_path):
        """HIGH severity data_quality → quarantine_bad_data action."""
        # Insert some rows with NULL order_amount to be quarantined
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        conn = sqlite3.connect(db_path)
        for i in range(5):
            created = (now - timedelta(minutes=10 - i)).isoformat()
            conn.execute(
                "INSERT INTO orders (order_id, created_at, order_amount, source_db) "
                "VALUES (?, ?, NULL, 'mumbai')",
                (f"null-{i:03d}", created),
            )
        conn.commit()
        conn.close()

        result = agent.run(high_quality_state)

        output = result["repairer_output"]
        assert output["action_taken"] == "quarantine_bad_data"
        assert output["success"] is True
        assert output["rows_affected"] == 5

    def test_escalate_to_human_for_critical(self, agent, critical_state):
        """CRITICAL severity → escalate_to_human action, success=False."""
        result = agent.run(critical_state)

        output = result["repairer_output"]
        assert output["action_taken"] == "escalate_to_human"
        assert output["success"] is False
        assert output["rows_affected"] == 0
        assert "CRITICAL" in output["details"]


# ── Playbook Tests ────────────────────────────────────────────────────


class TestPlaybookIntegration:
    """Tests for procedural LTM (playbook) lookup and recording."""

    def test_playbook_lookup_used(self, db_path):
        """When playbook has high success rate, that strategy is used."""
        # Pre-seed a playbook entry with high success rate for missing_data/LOW.
        # Default strategy would be wait_and_retry; playbook says quarantine_bad_data.
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO playbooks "
            "(anomaly_type, severity, action_taken, success_count, failure_count, last_used) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("missing_data", "LOW", "quarantine_bad_data", 9, 1,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

        agent = RepairerAgent(db_path=db_path)
        state = create_initial_state(
            run_id="repair-playbook-001",
            timestamp="2026-06-25T08:00:00+00:00",
        )
        state["anomaly_detected"] = True
        state["anomaly_type"] = "missing_data"
        state["severity"] = "LOW"
        state["gap_minutes"] = 15.0
        state["affected_tables"] = ["orders"]
        state["diagnoser_output"] = {
            "root_cause": "Brief network hiccup",
            "confidence": 0.75,
            "estimated_missing_rows": 100,
            "affected_tables": ["orders"],
            "maintenance_window_likely": False,
        }

        result = agent.run(state)

        # Playbook success_rate = 9/10 = 0.9 > 0.6 → use playbook strategy
        assert result["repairer_output"]["action_taken"] == "quarantine_bad_data"

    def test_outcome_recorded_in_playbooks(self, agent, db_path, low_missing_state):
        """After repair, outcome is recorded in playbook_store."""
        agent.run(low_missing_state)

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT anomaly_type, severity, action_taken, success_count, failure_count "
            "FROM playbooks WHERE anomaly_type = ? AND severity = ?",
            ("missing_data", "LOW"),
        ).fetchone()
        conn.close()

        assert row is not None
        anomaly_type, severity, action_taken, success_count, failure_count = row
        assert anomaly_type == "missing_data"
        assert severity == "LOW"
        assert action_taken == "wait_and_retry"
        assert success_count == 1
        assert failure_count == 0


# ── Database-level Verification Tests ─────────────────────────────────


class TestDatabaseOperations:
    """Tests that verify actual database mutations by repair strategies."""

    def test_switch_to_backup_copies_rows(self, db_path):
        """Verify backup table failover actually copies rows into orders."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        conn = sqlite3.connect(db_path)

        # Seed backup_orders with rows in the gap window
        for i in range(10):
            created = (now - timedelta(minutes=30 - i)).isoformat()
            conn.execute(
                "INSERT INTO backup_orders (order_id, created_at, order_amount, source_db) "
                "VALUES (?, ?, ?, 'delhi')",
                (f"del-bk-{i:03d}", created, 100.0 + i),
            )
        conn.commit()
        conn.close()

        agent = RepairerAgent(db_path=db_path)
        state = create_initial_state(
            run_id="repair-backup-001",
            timestamp="2026-06-25T08:00:00+00:00",
        )
        state["anomaly_detected"] = True
        state["anomaly_type"] = "missing_data"
        state["severity"] = "MEDIUM"
        state["gap_minutes"] = 60.0
        state["affected_tables"] = ["orders"]
        state["diagnoser_output"] = {
            "root_cause": "Mumbai DB down",
            "confidence": 0.85,
            "estimated_missing_rows": 2000,
            "affected_tables": ["orders"],
            "maintenance_window_likely": False,
        }

        result = agent.run(state)

        output = result["repairer_output"]
        assert output["action_taken"] == "switch_to_backup"

        # Verify rows actually landed in the orders table
        conn = sqlite3.connect(db_path)
        rows_in_orders = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE source_db = 'delhi'"
        ).fetchone()[0]
        conn.close()

        assert rows_in_orders > 0
        assert output["rows_affected"] > 0

    def test_quarantine_moves_null_rows(self, db_path):
        """Verify null rows are moved from orders to quarantine_orders."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        conn = sqlite3.connect(db_path)

        # Insert 3 good rows and 4 rows with NULL order_amount
        for i in range(3):
            created = (now - timedelta(minutes=10 - i)).isoformat()
            conn.execute(
                "INSERT INTO orders (order_id, created_at, order_amount, source_db) "
                "VALUES (?, ?, ?, 'mumbai')",
                (f"good-{i:03d}", created, 99.99),
            )
        for i in range(4):
            created = (now - timedelta(minutes=10 - i)).isoformat()
            conn.execute(
                "INSERT INTO orders (order_id, created_at, order_amount, source_db) "
                "VALUES (?, ?, NULL, 'mumbai')",
                (f"bad-{i:03d}", created),
            )
        conn.commit()
        conn.close()

        agent = RepairerAgent(db_path=db_path)
        state = create_initial_state(
            run_id="repair-quarantine-001",
            timestamp="2026-06-25T08:00:00+00:00",
        )
        state["anomaly_detected"] = True
        state["anomaly_type"] = "data_quality"
        state["severity"] = "HIGH"
        state["null_rate"] = 0.25
        state["affected_tables"] = ["orders"]
        state["diagnoser_output"] = {
            "root_cause": "Upstream validation failure",
            "confidence": 0.90,
            "estimated_missing_rows": 0,
            "affected_tables": ["orders"],
            "maintenance_window_likely": False,
        }

        result = agent.run(state)

        output = result["repairer_output"]
        assert output["action_taken"] == "quarantine_bad_data"
        assert output["rows_affected"] == 4

        # Verify: null rows removed from orders
        conn = sqlite3.connect(db_path)
        nulls_remaining = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE order_amount IS NULL"
        ).fetchone()[0]

        # Verify: null rows present in quarantine_orders
        quarantined = conn.execute(
            "SELECT COUNT(*) FROM quarantine_orders"
        ).fetchone()[0]

        # Verify: good rows untouched
        good_rows = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE order_amount IS NOT NULL"
        ).fetchone()[0]
        conn.close()

        assert nulls_remaining == 0
        assert quarantined == 4
        assert good_rows == 3
