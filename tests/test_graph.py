"""Tests for the LangGraph State Machine (Phase 3).

Verifies:
- Conditional routing after monitor (anomaly → diagnose, no anomaly → END)
- Graph structure (all three nodes exist)
- Full pipeline invocation with mocked agents (including repair)
"""

from unittest.mock import patch

import pytest

from graph import build_graph, route_after_monitor
from state import create_initial_state


# ── Routing Tests ─────────────────────────────────────────────────────


class TestRouteAfterMonitor:
    """Tests for the conditional routing function."""

    def test_routes_to_diagnose_on_anomaly(self):
        """Should route to 'diagnose' when anomaly is detected."""
        state = create_initial_state(run_id="test-001", timestamp="2026-06-25T08:00:00")
        state["anomaly_detected"] = True

        result = route_after_monitor(state)

        assert result == "diagnose"

    def test_routes_to_end_no_anomaly(self):
        """Should route to 'end' when no anomaly detected."""
        state = create_initial_state(run_id="test-002", timestamp="2026-06-25T08:00:00")
        state["anomaly_detected"] = False

        result = route_after_monitor(state)

        assert result == "end"

    def test_routes_to_end_missing_key(self):
        """Should default to 'end' when anomaly_detected key is missing."""
        state = {"run_id": "test-003", "timestamp": "2026-06-25T08:00:00"}

        result = route_after_monitor(state)

        assert result == "end"


# ── Graph Structure Tests ────────────────────────────────────────────


class TestGraphStructure:
    """Tests for the graph topology."""

    def test_graph_compiles(self):
        """Should compile without errors."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_monitor_node(self):
        """Graph should contain the monitor_node."""
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "monitor_node" in node_names

    def test_graph_has_diagnose_node(self):
        """Graph should contain the diagnose_node."""
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "diagnose_node" in node_names

    def test_graph_has_repair_node(self):
        """Graph should contain the repair_node (Phase 3)."""
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "repair_node" in node_names


# ── Integration Tests ─────────────────────────────────────────────────


class TestFullPipelineInvocation:
    """Tests for full pipeline invocation with mocking."""

    @patch("graph.RepairerAgent")
    @patch("graph.DiagnoserAgent")
    @patch("graph.MonitorAgent")
    def test_no_anomaly_skips_diagnose_and_repair(self, MockMonitor, MockDiagnoser, MockRepairer):
        """When monitor detects no anomaly, diagnoser and repairer should not run."""
        mock_monitor_instance = MockMonitor.return_value
        def mock_monitor_run(state):
            state["anomaly_detected"] = False
            state["anomaly_type"] = ""
            state["severity"] = "NONE"
            state["raw_count"] = 200
            state["expected_avg"] = 200.0
            state["null_rate"] = 0.02
            state["gap_minutes"] = 0.0
            state["affected_tables"] = []
            return state
        mock_monitor_instance.run.side_effect = mock_monitor_run

        graph = build_graph()
        initial_state = create_initial_state(run_id="test-int-001", timestamp="2026-06-25T08:00:00")
        final_state = graph.invoke(initial_state)

        mock_monitor_instance.run.assert_called_once()
        MockDiagnoser.return_value.run.assert_not_called()
        MockRepairer.return_value.run.assert_not_called()
        assert final_state["anomaly_detected"] is False

    @patch("graph.RepairerAgent")
    @patch("graph.DiagnoserAgent")
    @patch("graph.MonitorAgent")
    def test_anomaly_triggers_full_pipeline(self, MockMonitor, MockDiagnoser, MockRepairer):
        """When monitor detects anomaly, all three agents should run."""
        # Configure mock monitor
        mock_monitor_instance = MockMonitor.return_value
        def mock_monitor_run(state):
            state["anomaly_detected"] = True
            state["anomaly_type"] = "missing_data"
            state["severity"] = "MEDIUM"
            state["raw_count"] = 0
            state["expected_avg"] = 200.0
            state["null_rate"] = 0.0
            state["gap_minutes"] = 72.0
            state["affected_tables"] = ["orders"]
            return state
        mock_monitor_instance.run.side_effect = mock_monitor_run

        # Configure mock diagnoser
        mock_diagnoser_instance = MockDiagnoser.return_value
        def mock_diagnoser_run(state):
            state["diagnoser_output"] = {
                "root_cause": "Test root cause",
                "confidence": 0.85,
                "estimated_missing_rows": 5000,
                "affected_tables": ["orders"],
                "maintenance_window_likely": False,
            }
            return state
        mock_diagnoser_instance.run.side_effect = mock_diagnoser_run

        # Configure mock repairer
        mock_repairer_instance = MockRepairer.return_value
        def mock_repairer_run(state):
            state["repairer_output"] = {
                "action_taken": "switch_to_backup",
                "success": True,
                "details": "Switched to Delhi replica",
                "rows_affected": 5000,
            }
            return state
        mock_repairer_instance.run.side_effect = mock_repairer_run

        graph = build_graph()
        initial_state = create_initial_state(run_id="test-int-002", timestamp="2026-06-25T08:00:00")
        final_state = graph.invoke(initial_state)

        # All three should have run
        mock_monitor_instance.run.assert_called_once()
        mock_diagnoser_instance.run.assert_called_once()
        mock_repairer_instance.run.assert_called_once()

        # State should have all outputs
        assert final_state["anomaly_detected"] is True
        assert final_state["diagnoser_output"]["root_cause"] == "Test root cause"
        assert final_state["repairer_output"]["action_taken"] == "switch_to_backup"
        assert final_state["repairer_output"]["success"] is True
