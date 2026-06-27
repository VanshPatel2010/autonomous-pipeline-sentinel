"""Tests for the Monitor Agent (Phase 1).

Verifies:
- Row counting in 5-minute windows
- 7-day rolling baseline computation
- Null rate detection on order_amount
- Gap duration estimation
- Severity assignment logic
- Full run() with healthy and anomalous data
"""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from agents.monitor import MonitorAgent
from state import create_initial_state


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary SQLite database with the orders table."""
    path = str(tmp_path / "test_pipeline.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL DEFAULT 'mumbai'
        );
        CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
    """)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def agent(db_path):
    """Create a MonitorAgent pointing at the temp database."""
    return MonitorAgent(db_path=db_path)


def _insert_orders(db_path, orders):
    """Helper: insert a list of (order_id, created_at, order_amount, source_db) tuples."""
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO orders (order_id, created_at, order_amount, source_db) VALUES (?, ?, ?, ?)",
        orders,
    )
    conn.commit()
    conn.close()


def _now():
    """Current UTC time without tzinfo (matching the agent's behavior)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── Row Count Tests ───────────────────────────────────────────────────


class TestGetCurrentCount:
    """Tests for MonitorAgent._get_current_count."""

    def test_with_data(self, agent, db_path):
        """Should count rows within the last 5-minute window."""
        now = _now()
        orders = [
            (str(uuid.uuid4())[:12], (now - timedelta(minutes=2)).isoformat(), 100.0, "mumbai"),
            (str(uuid.uuid4())[:12], (now - timedelta(minutes=3)).isoformat(), 200.0, "mumbai"),
            (str(uuid.uuid4())[:12], (now - timedelta(minutes=4)).isoformat(), 150.0, "mumbai"),
        ]
        _insert_orders(db_path, orders)

        conn = agent._get_connection()
        count = agent._get_current_count(conn, now)
        conn.close()

        assert count == 3

    def test_empty(self, agent):
        """Should return 0 when no rows exist in the window."""
        conn = agent._get_connection()
        count = agent._get_current_count(conn, _now())
        conn.close()

        assert count == 0

    def test_excludes_old_rows(self, agent, db_path):
        """Should not count rows older than the 5-min window."""
        now = _now()
        old_order = (str(uuid.uuid4())[:12], (now - timedelta(minutes=10)).isoformat(), 50.0, "mumbai")
        recent_order = (str(uuid.uuid4())[:12], (now - timedelta(minutes=2)).isoformat(), 100.0, "mumbai")
        _insert_orders(db_path, [old_order, recent_order])

        conn = agent._get_connection()
        count = agent._get_current_count(conn, now)
        conn.close()

        assert count == 1


# ── Baseline Tests ────────────────────────────────────────────────────


class TestComputeBaseline:
    """Tests for MonitorAgent._compute_baseline."""

    def test_with_data(self, agent, db_path):
        """Should compute a positive baseline from historical data."""
        now = _now()
        orders = []
        # Insert 100 rows per 5-min window for the last 3 days (enough for baseline)
        for day_offset in range(3):
            for hour in range(24):
                for window in range(12):  # 12 windows per hour
                    ts = now - timedelta(days=day_offset, hours=hour, minutes=window * 5 + 2)
                    orders.append(
                        (str(uuid.uuid4())[:12], ts.isoformat(), 100.0, "mumbai")
                    )
        _insert_orders(db_path, orders)

        conn = agent._get_connection()
        baseline = agent._compute_baseline(conn, now)
        conn.close()

        assert baseline > 0

    def test_empty_database(self, agent):
        """Should return 0.0 when no historical data exists."""
        conn = agent._get_connection()
        baseline = agent._compute_baseline(conn, _now())
        conn.close()

        assert baseline == 0.0


# ── Null Rate Tests ───────────────────────────────────────────────────


class TestCheckNullRate:
    """Tests for MonitorAgent._check_null_rate."""

    def test_with_nulls(self, agent, db_path):
        """Should correctly compute null rate when some rows have null order_amount."""
        now = _now()
        orders = [
            (str(uuid.uuid4())[:12], (now - timedelta(minutes=2)).isoformat(), None, "mumbai"),
            (str(uuid.uuid4())[:12], (now - timedelta(minutes=2)).isoformat(), 100.0, "mumbai"),
            (str(uuid.uuid4())[:12], (now - timedelta(minutes=3)).isoformat(), None, "mumbai"),
            (str(uuid.uuid4())[:12], (now - timedelta(minutes=3)).isoformat(), 200.0, "mumbai"),
        ]
        _insert_orders(db_path, orders)

        conn = agent._get_connection()
        null_rate = agent._check_null_rate(conn, now)
        conn.close()

        assert null_rate == pytest.approx(0.5, abs=0.01)

    def test_no_data(self, agent):
        """Should return 0.0 when no rows exist in the window."""
        conn = agent._get_connection()
        null_rate = agent._check_null_rate(conn, _now())
        conn.close()

        assert null_rate == 0.0

    def test_no_nulls(self, agent, db_path):
        """Should return 0.0 when all rows have valid order_amount."""
        now = _now()
        orders = [
            (str(uuid.uuid4())[:12], (now - timedelta(minutes=2)).isoformat(), 100.0, "mumbai"),
            (str(uuid.uuid4())[:12], (now - timedelta(minutes=3)).isoformat(), 200.0, "mumbai"),
        ]
        _insert_orders(db_path, orders)

        conn = agent._get_connection()
        null_rate = agent._check_null_rate(conn, now)
        conn.close()

        assert null_rate == 0.0


# ── Gap Estimation Tests ──────────────────────────────────────────────


class TestEstimateGapMinutes:
    """Tests for MonitorAgent._estimate_gap_minutes."""

    def test_with_gap(self, agent, db_path):
        """Should detect a gap when the most recent record is old."""
        now = _now()
        old_order = (str(uuid.uuid4())[:12], (now - timedelta(hours=2)).isoformat(), 100.0, "mumbai")
        _insert_orders(db_path, [old_order])

        conn = agent._get_connection()
        gap = agent._estimate_gap_minutes(conn, now)
        conn.close()

        # gap should be ~120 minutes
        assert gap > 100

    def test_no_gap(self, agent, db_path):
        """Should return 0.0 when the most recent record is within 2x polling interval."""
        now = _now()
        recent_order = (str(uuid.uuid4())[:12], (now - timedelta(minutes=3)).isoformat(), 100.0, "mumbai")
        _insert_orders(db_path, [recent_order])

        conn = agent._get_connection()
        gap = agent._estimate_gap_minutes(conn, now)
        conn.close()

        assert gap == 0.0

    def test_empty_table(self, agent):
        """Should return 0.0 when no records exist."""
        conn = agent._get_connection()
        gap = agent._estimate_gap_minutes(conn, _now())
        conn.close()

        assert gap == 0.0


# ── Severity Assignment Tests ─────────────────────────────────────────


class TestAssignSeverity:
    """Tests for MonitorAgent._assign_severity."""

    def test_none_no_issue(self, agent):
        """No gap and no null rate → NONE."""
        assert agent._assign_severity(0.0, 0.0) == "NONE"

    def test_low_small_gap(self, agent):
        """Gap < 30 min → LOW."""
        assert agent._assign_severity(15.0, 0.0) == "LOW"

    def test_low_small_null_rate(self, agent):
        """Small null rate (< 5%) → LOW."""
        assert agent._assign_severity(0.0, 0.01) == "LOW"

    def test_medium_gap(self, agent):
        """Gap 30-360 min → MEDIUM."""
        assert agent._assign_severity(120.0, 0.0) == "MEDIUM"

    def test_high_large_gap(self, agent):
        """Gap > 360 min → HIGH."""
        assert agent._assign_severity(400.0, 0.0) == "HIGH"

    def test_high_null_rate(self, agent):
        """Null rate > 5% → HIGH (overrides gap)."""
        assert agent._assign_severity(0.0, 0.10) == "HIGH"


# ── Full Run Tests ────────────────────────────────────────────────────


class TestMonitorRun:
    """Tests for MonitorAgent.run() end-to-end."""

    def test_healthy_pipeline(self, agent, db_path):
        """Should detect no anomaly when data is within baseline."""
        now = _now()
        orders = []
        # Insert 200 rows per window for the last 7 days
        for day in range(7):
            for window_idx in range(288):  # 288 windows per day
                ts = now - timedelta(days=day, minutes=window_idx * 5 + 2)
                for _ in range(2):  # 2 rows per window (for speed)
                    orders.append(
                        (str(uuid.uuid4())[:12], ts.isoformat(), 100.0, "mumbai")
                    )
        _insert_orders(db_path, orders)

        state = create_initial_state(run_id="test-001", timestamp=now.isoformat())
        result = agent.run(state)

        assert result["anomaly_detected"] is False
        assert result["anomaly_type"] == ""
        assert result["severity"] == "NONE"

    def test_missing_data_anomaly(self, agent, db_path):
        """Should detect missing_data when current window is below threshold."""
        now = _now()
        orders = []
        # Insert baseline data for past 7 days (but NOT for the last 5 minutes)
        for day in range(1, 7):
            for window_idx in range(288):
                ts = now - timedelta(days=day, minutes=window_idx * 5 + 2)
                for _ in range(10):
                    orders.append(
                        (str(uuid.uuid4())[:12], ts.isoformat(), 100.0, "mumbai")
                    )
        _insert_orders(db_path, orders)

        state = create_initial_state(run_id="test-002", timestamp=now.isoformat())
        result = agent.run(state)

        assert result["anomaly_detected"] is True
        assert result["anomaly_type"] == "missing_data"
        assert result["raw_count"] == 0

    def test_data_quality_anomaly(self, agent, db_path):
        """Should detect data_quality when null rate exceeds threshold."""
        now = _now()
        orders = []
        # Insert baseline data for past days
        for day in range(1, 7):
            for window_idx in range(288):
                ts = now - timedelta(days=day, minutes=window_idx * 5 + 2)
                for _ in range(2):
                    orders.append(
                        (str(uuid.uuid4())[:12], ts.isoformat(), 100.0, "mumbai")
                    )

        # Insert recent rows with ALL nulls (100% null rate)
        for _ in range(10):
            ts = (now - timedelta(minutes=2)).isoformat()
            orders.append(
                (str(uuid.uuid4())[:12], ts, None, "mumbai")
            )
        _insert_orders(db_path, orders)

        state = create_initial_state(run_id="test-003", timestamp=now.isoformat())
        result = agent.run(state)

        assert result["anomaly_detected"] is True
        assert result["anomaly_type"] == "data_quality"
        assert result["null_rate"] > 0.05
