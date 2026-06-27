"""Tests for the Diagnoser Prompt Templates (Phase 2).

Verifies:
- System prompt exists and contains key instructions
- User prompt building with and without past incidents
- Mock diagnoser output has all required fields
"""

import pytest

from prompts.diagnoser_prompt import (
    MOCK_DIAGNOSER_OUTPUT,
    SYSTEM_PROMPT,
    build_user_prompt,
)


# ── System Prompt Tests ───────────────────────────────────────────────


class TestSystemPrompt:
    """Tests for the system prompt constant."""

    def test_system_prompt_exists(self):
        """System prompt should be a non-empty string."""
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_requires_json(self):
        """System prompt should instruct the LLM to respond in JSON."""
        assert "JSON" in SYSTEM_PROMPT

    def test_system_prompt_mentions_response_format(self):
        """System prompt should include the expected response fields."""
        assert "root_cause" in SYSTEM_PROMPT
        assert "confidence" in SYSTEM_PROMPT
        assert "estimated_missing_rows" in SYSTEM_PROMPT
        assert "affected_tables" in SYSTEM_PROMPT
        assert "maintenance_window_likely" in SYSTEM_PROMPT

    def test_system_prompt_mentions_data_context(self):
        """System prompt should provide context about the pipeline domain."""
        assert "pipeline" in SYSTEM_PROMPT.lower()
        assert "data" in SYSTEM_PROMPT.lower()


# ── Build User Prompt Tests ──────────────────────────────────────────


class TestBuildUserPrompt:
    """Tests for build_user_prompt."""

    @pytest.fixture
    def state(self):
        """Sample state dict for prompt building."""
        return {
            "anomaly_type": "missing_data",
            "severity": "HIGH",
            "raw_count": 0,
            "expected_avg": 200.0,
            "gap_minutes": 72.0,
            "null_rate": 0.01,
            "affected_tables": ["orders"],
            "timestamp": "2026-06-25T03:14:00+00:00",
        }

    def test_with_incidents(self, state):
        """Should include past incident details in the prompt."""
        past_incidents = [
            {
                "timestamp": "2026-06-24T03:14:00+00:00",
                "severity": "HIGH",
                "root_cause": "Mumbai DB outage",
                "gap_minutes": 240.0,
                "fix_taken": "switched to Delhi replica",
                "resolved": True,
            },
            {
                "timestamp": "2026-06-22T10:00:00+00:00",
                "severity": "MEDIUM",
                "root_cause": "Network partition",
                "gap_minutes": 45.0,
                "fix_taken": "waited for auto-recovery",
                "resolved": True,
            },
        ]

        prompt = build_user_prompt(state, past_incidents)

        assert "CURRENT ANOMALY DETECTED" in prompt
        assert "missing_data" in prompt
        assert "SIMILAR PAST INCIDENTS (2 found)" in prompt
        assert "Mumbai DB outage" in prompt
        assert "Network partition" in prompt
        assert "Incident #1" in prompt
        assert "Incident #2" in prompt

    def test_without_incidents(self, state):
        """Should indicate no past incidents found."""
        prompt = build_user_prompt(state, [])

        assert "CURRENT ANOMALY DETECTED" in prompt
        assert "No similar past incidents found" in prompt
        assert "new type of failure" in prompt

    def test_includes_state_values(self, state):
        """Should include all state values in the prompt."""
        prompt = build_user_prompt(state, [])

        assert "missing_data" in prompt
        assert "HIGH" in prompt
        assert "200.0" in prompt
        assert "72.0" in prompt
        assert "orders" in prompt

    def test_includes_null_rate(self, state):
        """Should include the null rate in the prompt."""
        prompt = build_user_prompt(state, [])

        assert "Null rate" in prompt or "null" in prompt.lower()

    def test_ends_with_instruction(self, state):
        """Should end with instruction to analyze and respond."""
        prompt = build_user_prompt(state, [])

        assert "Analyze this anomaly" in prompt


# ── Mock Output Tests ─────────────────────────────────────────────────


class TestMockDiagnoserOutput:
    """Tests for the MOCK_DIAGNOSER_OUTPUT constant."""

    def test_structure(self):
        """Mock output should have all required fields."""
        required_fields = [
            "root_cause",
            "confidence",
            "estimated_missing_rows",
            "affected_tables",
            "maintenance_window_likely",
        ]
        for field in required_fields:
            assert field in MOCK_DIAGNOSER_OUTPUT, f"Missing field: {field}"

    def test_types(self):
        """Mock output fields should have correct types."""
        assert isinstance(MOCK_DIAGNOSER_OUTPUT["root_cause"], str)
        assert isinstance(MOCK_DIAGNOSER_OUTPUT["confidence"], (int, float))
        assert isinstance(MOCK_DIAGNOSER_OUTPUT["estimated_missing_rows"], int)
        assert isinstance(MOCK_DIAGNOSER_OUTPUT["affected_tables"], list)
        assert isinstance(MOCK_DIAGNOSER_OUTPUT["maintenance_window_likely"], bool)

    def test_confidence_range(self):
        """Mock confidence should be between 0 and 1."""
        assert 0.0 <= MOCK_DIAGNOSER_OUTPUT["confidence"] <= 1.0

    def test_is_copyable(self):
        """Mock output should be safely copyable (no shared references)."""
        copy1 = MOCK_DIAGNOSER_OUTPUT.copy()
        copy2 = MOCK_DIAGNOSER_OUTPUT.copy()
        copy1["confidence"] = 0.0
        assert copy2["confidence"] == MOCK_DIAGNOSER_OUTPUT["confidence"]
