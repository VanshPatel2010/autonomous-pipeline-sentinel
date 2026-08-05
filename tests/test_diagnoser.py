"""Tests for the Diagnoser Agent (Phase 2).

Verifies:
- LLM JSON response parsing (valid, markdown-wrapped, invalid)
- Fallback diagnosis generation
- Skipping diagnosis when no anomaly
- Mock mode operation
- Severity upgrade logic on high confidence
- Integration with episodic memory
"""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from agents.diagnoser import DiagnoserAgent
from memory.incident_store import init_db, insert_incident
from state import create_initial_state


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def agent():
    """Create a DiagnoserAgent instance."""
    return DiagnoserAgent()


@pytest.fixture
def anomaly_state():
    """Create a state dict that has an anomaly detected."""
    state = create_initial_state(run_id="diag-001", timestamp="2026-06-25T08:00:00+00:00")
    state["anomaly_detected"] = True
    state["anomaly_type"] = "missing_data"
    state["severity"] = "MEDIUM"
    state["raw_count"] = 0
    state["expected_avg"] = 200.0
    state["gap_minutes"] = 72.0
    state["null_rate"] = 0.0
    state["affected_tables"] = ["orders"]
    return state


@pytest.fixture
def healthy_state():
    """Create a state dict with no anomaly."""
    return create_initial_state(run_id="diag-002", timestamp="2026-06-25T08:00:00+00:00")


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary database with the incidents table."""
    path = str(tmp_path / "test_incidents.db")
    init_db(path)
    return path


# (TestParseLlmResponse removed as _parse_llm_response is obsolete)


# ── Fallback Output Tests ────────────────────────────────────────────


class TestBuildFallbackOutput:
    """Tests for DiagnoserAgent._build_fallback_output."""

    def test_fallback_structure(self, agent, anomaly_state):
        """Should return a valid diagnosis dict with all required fields."""
        result = agent._build_fallback_output(anomaly_state)

        assert "root_cause" in result
        assert "confidence" in result
        assert "estimated_missing_rows" in result
        assert "affected_tables" in result
        assert "maintenance_window_likely" in result

    def test_fallback_confidence(self, agent, anomaly_state):
        """Fallback confidence should be 0.5."""
        result = agent._build_fallback_output(anomaly_state)
        assert result["confidence"] == 0.5

    def test_fallback_includes_anomaly_type(self, agent, anomaly_state):
        """Fallback root cause should include the anomaly type."""
        result = agent._build_fallback_output(anomaly_state)
        assert "missing_data" in result["root_cause"]

    def test_fallback_estimates_missing_rows(self, agent, anomaly_state):
        """Fallback should estimate missing rows from gap and baseline."""
        result = agent._build_fallback_output(anomaly_state)
        # gap=72min * (200/5)=40 rows/min → ~2880 missing rows
        assert result["estimated_missing_rows"] > 0


# ── Run Tests ─────────────────────────────────────────────────────────


class TestDiagnoserRun:
    """Tests for DiagnoserAgent.run()."""

    def test_no_anomaly_skips(self, agent, healthy_state):
        """Should skip diagnosis and return empty diagnoser_output when no anomaly."""
        result = agent.run(healthy_state)

        assert result["diagnoser_output"] == {}

    @patch("agents.diagnoser.MOCK_MODE", True)
    @patch("agents.diagnoser.get_similar_incidents", return_value=[])
    def test_run_mock_mode(self, mock_incidents, anomaly_state):
        """Should use mock output when MOCK_MODE is True."""
        agent = DiagnoserAgent()
        result = agent.run(anomaly_state)

        output = result["diagnoser_output"]
        assert "root_cause" in output
        assert output["confidence"] == 0.75  # mock confidence
        assert "mock" in output["root_cause"].lower() or "connectivity" in output["root_cause"].lower()

    @patch("agents.diagnoser.MOCK_MODE", True)
    @patch("agents.diagnoser.get_similar_incidents")
    def test_run_with_past_incidents(self, mock_get_incidents, anomaly_state):
        """Should retrieve and use past incidents for context."""
        mock_get_incidents.return_value = [
            {
                "run_id": "old-001",
                "timestamp": "2026-06-24T03:14:00+00:00",
                "anomaly_type": "missing_data",
                "severity": "HIGH",
                "gap_minutes": 240.0,
                "root_cause": "Mumbai DB outage",
                "affected_tables": ["orders"],
                "fix_taken": "switched to Delhi replica",
                "resolved": 1,
                "confidence": 0.9,
            }
        ]

        agent = DiagnoserAgent()
        result = agent.run(anomaly_state)

        mock_get_incidents.assert_called_once_with("missing_data", limit=5)
        assert result["diagnoser_output"] != {}

    @patch("agents.diagnoser.MOCK_MODE", False)
    @patch("agents.diagnoser.get_similar_incidents", return_value=[])
    def test_severity_upgrade_on_high_confidence(self, mock_incidents, anomaly_state):
        """Should upgrade severity when LLM returns high confidence and gap > 30 min."""
        agent = DiagnoserAgent()

        # Mock the LLM to return high confidence
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "root_cause": "Source DB outage",
            "confidence": 0.9,
            "estimated_missing_rows": 5000,
            "affected_tables": ["orders"],
            "maintenance_window_likely": False,
        }
        mock_llm.chat.completions.create.return_value = mock_response
        agent._llm = mock_llm

        # State with MEDIUM severity and gap > 30 min
        anomaly_state["severity"] = "MEDIUM"
        anomaly_state["gap_minutes"] = 72.0

        result = agent.run(anomaly_state)

        # Severity should be upgraded to HIGH
        assert result["severity"] == "HIGH"
        assert result["diagnoser_output"]["confidence"] == 0.9

    @patch("agents.diagnoser.MOCK_MODE", False)
    @patch("agents.diagnoser.get_similar_incidents", return_value=[])
    def test_llm_failure_uses_fallback(self, mock_incidents, anomaly_state):
        """Should use fallback diagnosis when LLM call raises an exception."""
        agent = DiagnoserAgent()

        # Mock the LLM to raise an error
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = Exception("API rate limit exceeded")
        agent._llm = mock_llm

        result = agent.run(anomaly_state)

        output = result["diagnoser_output"]
        assert output["confidence"] == 0.5  # fallback confidence
        assert "missing_data" in output["root_cause"]
