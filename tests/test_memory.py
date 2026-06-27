"""Tests for the Incident Store — Episodic Long-Term Memory (Phase 2).

Verifies:
- Database initialization creates tables
- Incident insertion and retrieval
- Similar incident filtering by anomaly_type
- JSON serialization of affected_tables
- Edge cases (empty DB, no matches)
"""

import json
import sqlite3

import pytest

from memory.incident_store import (
    get_all_incidents,
    get_similar_incidents,
    init_db,
    insert_incident,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    """Create and initialize a temporary database for incident storage."""
    path = str(tmp_path / "test_incidents.db")
    init_db(path)
    return path


@pytest.fixture
def sample_incident():
    """A sample incident dict for testing."""
    return {
        "run_id": "test-001",
        "timestamp": "2026-06-25T08:00:00+00:00",
        "anomaly_type": "missing_data",
        "severity": "HIGH",
        "gap_minutes": 72.0,
        "root_cause": "Mumbai source DB outage",
        "affected_tables": ["orders"],
        "fix_taken": "switched to Delhi replica",
        "resolved": 1,
        "confidence": 0.85,
    }


# ── Init Tests ────────────────────────────────────────────────────────


class TestInitDb:
    """Tests for init_db."""

    def test_creates_incidents_table(self, db_path):
        """Should create the incidents table."""
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='incidents'"
        ).fetchall()
        conn.close()

        assert len(tables) == 1
        assert tables[0][0] == "incidents"

    def test_creates_playbooks_table(self, db_path):
        """Should create the playbooks table."""
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='playbooks'"
        ).fetchall()
        conn.close()

        assert len(tables) == 1
        assert tables[0][0] == "playbooks"

    def test_idempotent(self, tmp_path):
        """Should be safe to call multiple times (IF NOT EXISTS)."""
        path = str(tmp_path / "test_idempotent.db")
        init_db(path)
        init_db(path)  # Second call should not raise

        conn = sqlite3.connect(path)
        tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()
        conn.close()

        assert tables[0] >= 2  # incidents + playbooks


# ── Insert Tests ──────────────────────────────────────────────────────


class TestInsertIncident:
    """Tests for insert_incident."""

    def test_insert_incident(self, db_path, sample_incident):
        """Should insert an incident and be retrievable."""
        insert_incident(sample_incident, db_path=db_path)

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT run_id, anomaly_type, severity FROM incidents WHERE run_id = ?",
            ("test-001",),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "test-001"
        assert row[1] == "missing_data"
        assert row[2] == "HIGH"

    def test_insert_serializes_affected_tables(self, db_path, sample_incident):
        """Should JSON-serialize the affected_tables list."""
        insert_incident(sample_incident, db_path=db_path)

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT affected_tables FROM incidents WHERE run_id = ?",
            ("test-001",),
        ).fetchone()
        conn.close()

        # Should be stored as JSON string
        tables = json.loads(row[0])
        assert tables == ["orders"]

    def test_insert_replaces_on_duplicate(self, db_path, sample_incident):
        """Should replace incident with same run_id (INSERT OR REPLACE)."""
        insert_incident(sample_incident, db_path=db_path)

        # Update severity and re-insert
        sample_incident["severity"] = "CRITICAL"
        insert_incident(sample_incident, db_path=db_path)

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT severity FROM incidents WHERE run_id = ?",
            ("test-001",),
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "CRITICAL"


# ── Query Tests ───────────────────────────────────────────────────────


class TestGetSimilarIncidents:
    """Tests for get_similar_incidents."""

    def test_returns_matching_type(self, db_path, sample_incident):
        """Should return incidents with matching anomaly_type."""
        insert_incident(sample_incident, db_path=db_path)

        # Insert a different type
        other = sample_incident.copy()
        other["run_id"] = "test-002"
        other["anomaly_type"] = "data_quality"
        insert_incident(other, db_path=db_path)

        results = get_similar_incidents("missing_data", db_path=db_path)

        assert len(results) == 1
        assert results[0]["anomaly_type"] == "missing_data"

    def test_empty_results(self, db_path):
        """Should return empty list when no matching incidents."""
        results = get_similar_incidents("schema_drift", db_path=db_path)
        assert results == []

    def test_respects_limit(self, db_path, sample_incident):
        """Should limit the number of returned incidents."""
        for i in range(10):
            incident = sample_incident.copy()
            incident["run_id"] = f"test-{i:03d}"
            insert_incident(incident, db_path=db_path)

        results = get_similar_incidents("missing_data", limit=3, db_path=db_path)

        assert len(results) == 3

    def test_deserializes_affected_tables(self, db_path, sample_incident):
        """Should deserialize affected_tables from JSON string to list."""
        insert_incident(sample_incident, db_path=db_path)

        results = get_similar_incidents("missing_data", db_path=db_path)

        assert isinstance(results[0]["affected_tables"], list)
        assert results[0]["affected_tables"] == ["orders"]

    def test_ordered_by_timestamp_desc(self, db_path, sample_incident):
        """Should return most recent incidents first."""
        for i, ts in enumerate(["2026-06-23T00:00:00", "2026-06-25T00:00:00", "2026-06-24T00:00:00"]):
            incident = sample_incident.copy()
            incident["run_id"] = f"test-{i:03d}"
            incident["timestamp"] = ts
            insert_incident(incident, db_path=db_path)

        results = get_similar_incidents("missing_data", db_path=db_path)

        # Should be: 25th, 24th, 23rd
        assert results[0]["timestamp"] == "2026-06-25T00:00:00"
        assert results[2]["timestamp"] == "2026-06-23T00:00:00"


class TestGetAllIncidents:
    """Tests for get_all_incidents."""

    def test_returns_all_types(self, db_path, sample_incident):
        """Should return incidents of all anomaly types."""
        insert_incident(sample_incident, db_path=db_path)

        other = sample_incident.copy()
        other["run_id"] = "test-002"
        other["anomaly_type"] = "data_quality"
        insert_incident(other, db_path=db_path)

        results = get_all_incidents(db_path=db_path)

        assert len(results) == 2

    def test_empty_database(self, db_path):
        """Should return empty list when no incidents exist."""
        results = get_all_incidents(db_path=db_path)
        assert results == []

    def test_respects_limit(self, db_path, sample_incident):
        """Should limit the number of returned incidents."""
        for i in range(10):
            incident = sample_incident.copy()
            incident["run_id"] = f"test-{i:03d}"
            insert_incident(incident, db_path=db_path)

        results = get_all_incidents(limit=5, db_path=db_path)

        assert len(results) == 5
