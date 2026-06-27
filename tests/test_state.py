"""Tests for the PipelineState TypedDict (Phase 1).

Verifies:
- create_initial_state returns all expected keys
- Default values are correct
- Custom run_id and timestamp are set
"""

import pytest

from state import PipelineState, create_initial_state


class TestCreateInitialState:
    """Tests for create_initial_state factory."""

    def test_returns_dict(self):
        """Should return a dict (TypedDict is a dict at runtime)."""
        state = create_initial_state(run_id="test-001", timestamp="2026-06-25T08:00:00")
        assert isinstance(state, dict)

    def test_custom_values(self):
        """Should set run_id and timestamp from arguments."""
        state = create_initial_state(run_id="abc123", timestamp="2026-06-25T10:00:00")
        assert state["run_id"] == "abc123"
        assert state["timestamp"] == "2026-06-25T10:00:00"

    def test_default_anomaly_detected(self):
        """Should default anomaly_detected to False."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["anomaly_detected"] is False

    def test_default_anomaly_type(self):
        """Should default anomaly_type to empty string."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["anomaly_type"] == ""

    def test_default_severity(self):
        """Should default severity to 'NONE'."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["severity"] == "NONE"

    def test_default_gap_minutes(self):
        """Should default gap_minutes to 0.0."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["gap_minutes"] == 0.0

    def test_default_affected_tables(self):
        """Should default affected_tables to empty list."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["affected_tables"] == []

    def test_default_raw_count(self):
        """Should default raw_count to 0."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["raw_count"] == 0

    def test_default_expected_avg(self):
        """Should default expected_avg to 0.0."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["expected_avg"] == 0.0

    def test_default_null_rate(self):
        """Should default null_rate to 0.0."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["null_rate"] == 0.0

    def test_default_diagnoser_output(self):
        """Should default diagnoser_output to empty dict."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["diagnoser_output"] == {}

    def test_default_repairer_output(self):
        """Should default repairer_output to empty dict."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["repairer_output"] == {}

    def test_default_slack_sent(self):
        """Should default slack_sent to False."""
        state = create_initial_state(run_id="test", timestamp="now")
        assert state["slack_sent"] is False

    def test_has_all_keys(self):
        """Should have all keys defined in PipelineState."""
        state = create_initial_state(run_id="test", timestamp="now")
        expected_keys = {
            "run_id", "timestamp", "anomaly_detected", "anomaly_type",
            "severity", "gap_minutes", "affected_tables", "raw_count",
            "expected_avg", "null_rate", "diagnoser_output", "repairer_output",
            "slack_sent",
        }
        assert set(state.keys()) == expected_keys
