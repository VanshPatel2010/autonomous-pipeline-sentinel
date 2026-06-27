"""Tests for metrics.tracker module: MTTR, detection rate, severity distribution,
recent incidents, and daily incident counts."""

import sys
sys.path.insert(0, '/Users/vanshpatel/documents/Autonomous Data Pipeline Agent')

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from metrics.tracker import (
    compute_detection_rate,
    compute_mttr,
    get_daily_incident_counts,
    get_recent_incidents,
    get_severity_distribution,
)

INCIDENTS_DDL = """
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
"""


@pytest.fixture
def db_conn(tmp_path):
    """Create a temporary SQLite database with the incidents table."""
    db_path = tmp_path / "test_metrics.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(INCIDENTS_DDL)
    conn.commit()
    yield conn
    conn.close()


# ── helpers ──────────────────────────────────────────────────────────────────

def _insert_incident(conn, run_id, timestamp, anomaly_type="", severity="",
                     gap_minutes=0.0, root_cause="", affected_tables="",
                     fix_taken="", resolved=0, resolved_at=None,
                     confidence=0.0):
    conn.execute(
        """INSERT INTO incidents
           (run_id, timestamp, anomaly_type, severity, gap_minutes, root_cause,
            affected_tables, fix_taken, resolved, resolved_at, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, timestamp, anomaly_type, severity, gap_minutes, root_cause,
         affected_tables, fix_taken, resolved, resolved_at, confidence),
    )
    conn.commit()


# ── 1. test_compute_mttr_with_resolved_incidents ────────────────────────────

def test_compute_mttr_with_resolved_incidents(db_conn):
    now = datetime.now(timezone.utc)

    # Incident 1: resolved in 60 minutes
    ts1 = (now - timedelta(days=1)).isoformat()
    ra1 = (now - timedelta(days=1) + timedelta(minutes=60)).isoformat()

    # Incident 2: resolved in 120 minutes
    ts2 = (now - timedelta(days=2)).isoformat()
    ra2 = (now - timedelta(days=2) + timedelta(minutes=120)).isoformat()

    # Incident 3: resolved in 30 minutes
    ts3 = (now - timedelta(days=3)).isoformat()
    ra3 = (now - timedelta(days=3) + timedelta(minutes=30)).isoformat()

    _insert_incident(db_conn, "r1", ts1, resolved=1, resolved_at=ra1)
    _insert_incident(db_conn, "r2", ts2, resolved=1, resolved_at=ra2)
    _insert_incident(db_conn, "r3", ts3, resolved=1, resolved_at=ra3)

    expected_avg = (60 + 120 + 30) / 3.0  # 70.0 minutes
    result = compute_mttr(db_conn)
    assert result == pytest.approx(expected_avg, abs=0.5)


# ── 2. test_compute_mttr_no_resolved ────────────────────────────────────────

def test_compute_mttr_no_resolved(db_conn):
    assert compute_mttr(db_conn) == 0.0


# ── 3. test_compute_detection_rate ──────────────────────────────────────────

def test_compute_detection_rate(db_conn):
    now = datetime.now(timezone.utc)

    _insert_incident(db_conn, "d1", (now - timedelta(days=1)).isoformat(),
                     anomaly_type="missing_data")
    _insert_incident(db_conn, "d2", (now - timedelta(days=2)).isoformat(),
                     anomaly_type="schema_drift")
    _insert_incident(db_conn, "d3", (now - timedelta(days=3)).isoformat(),
                     anomaly_type="late_arrival")
    # This one has empty anomaly_type → not detected
    _insert_incident(db_conn, "d4", (now - timedelta(days=4)).isoformat(),
                     anomaly_type="")

    rate = compute_detection_rate(db_conn)
    assert rate == pytest.approx(0.75)


# ── 4. test_compute_detection_rate_no_data ──────────────────────────────────

def test_compute_detection_rate_no_data(db_conn):
    assert compute_detection_rate(db_conn) == 0.0


# ── 5. test_get_severity_distribution ───────────────────────────────────────

def test_get_severity_distribution(db_conn):
    now = datetime.now(timezone.utc)

    for i in range(3):
        _insert_incident(db_conn, f"h{i}", (now - timedelta(days=i + 1)).isoformat(),
                         severity="HIGH")
    for i in range(2):
        _insert_incident(db_conn, f"m{i}", (now - timedelta(days=i + 1, hours=1)).isoformat(),
                         severity="MEDIUM")
    _insert_incident(db_conn, "l0", (now - timedelta(days=1, hours=2)).isoformat(),
                     severity="LOW")

    dist = get_severity_distribution(db_conn)
    assert dist == {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


# ── 6. test_get_severity_distribution_empty ─────────────────────────────────

def test_get_severity_distribution_empty(db_conn):
    assert get_severity_distribution(db_conn) == {}


# ── 7. test_get_recent_incidents ────────────────────────────────────────────

def test_get_recent_incidents(db_conn):
    now = datetime.now(timezone.utc)
    timestamps = []
    for i in range(5):
        ts = (now - timedelta(days=i)).isoformat()
        timestamps.append(ts)
        _insert_incident(db_conn, f"ri{i}", ts, anomaly_type="test",
                         severity="LOW")

    results = get_recent_incidents(db_conn)
    assert len(results) == 5
    # Verify descending order by timestamp
    result_timestamps = [r["timestamp"] for r in results]
    assert result_timestamps == sorted(result_timestamps, reverse=True)


# ── 8. test_get_recent_incidents_limit ──────────────────────────────────────

def test_get_recent_incidents_limit(db_conn):
    now = datetime.now(timezone.utc)
    for i in range(5):
        _insert_incident(db_conn, f"lim{i}",
                         (now - timedelta(days=i)).isoformat())

    results = get_recent_incidents(db_conn, limit=2)
    assert len(results) == 2


# ── 9. test_get_daily_incident_counts ───────────────────────────────────────

def test_get_daily_incident_counts(db_conn):
    now = datetime.now(timezone.utc)

    day1 = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    day2 = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    day3 = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    # 2 incidents on day1
    _insert_incident(db_conn, "dc1", f"{day1}T10:00:00+00:00")
    _insert_incident(db_conn, "dc2", f"{day1}T14:00:00+00:00")
    # 1 incident on day2
    _insert_incident(db_conn, "dc3", f"{day2}T09:00:00+00:00")
    # 3 incidents on day3
    _insert_incident(db_conn, "dc4", f"{day3}T08:00:00+00:00")
    _insert_incident(db_conn, "dc5", f"{day3}T12:00:00+00:00")
    _insert_incident(db_conn, "dc6", f"{day3}T18:00:00+00:00")

    counts = get_daily_incident_counts(db_conn, days=30)
    assert len(counts) == 3
    # Should be sorted ascending by date
    dates = [c["date"] for c in counts]
    assert dates == sorted(dates)
    # Verify per-day counts
    count_map = {c["date"]: c["count"] for c in counts}
    assert count_map[day1] == 2
    assert count_map[day2] == 1
    assert count_map[day3] == 3


# ── 10. test_get_daily_incident_counts_empty ────────────────────────────────

def test_get_daily_incident_counts_empty(db_conn):
    assert get_daily_incident_counts(db_conn) == []
