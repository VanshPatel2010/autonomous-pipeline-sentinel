"""Tests for agents.slack_agent.SlackAgent.

Covers:
  - Skip when no anomaly detected
  - MOCK_MODE payload logging
  - Template selection (standard vs. escalation)
  - Webhook POST success / failure / timeout / connection error
  - Missing webhook URL
  - Slack message format validation (Block Kit structure)
  - Escalation message @channel mention
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from state import create_initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anomaly_state(**overrides) -> dict:
    """Return a PipelineState with an active anomaly and sensible defaults.

    Accepts arbitrary keyword overrides applied on top of defaults.
    """
    state = create_initial_state(run_id="test-run-001", timestamp="2026-06-25T11:00:00")
    state["anomaly_detected"] = True
    state["anomaly_type"] = "missing_data"
    state["severity"] = "HIGH"
    state["gap_minutes"] = 120.0
    state["affected_tables"] = ["orders"]
    state["raw_count"] = 10
    state["expected_avg"] = 200.0
    state["null_rate"] = 0.02
    state["diagnoser_output"] = {
        "root_cause": "Upstream API outage",
        "confidence": 0.85,
    }
    state["repairer_output"] = {
        "status": "partial",
        "actions_taken": ["Retried ingestion", "Backfilled 50 rows"],
    }
    state.update(overrides)
    return state


# ========================== 1. No anomaly ================================

def test_no_anomaly_skips_slack(monkeypatch):
    """When anomaly_detected=False the agent should skip and set slack_sent=False."""
    monkeypatch.setattr("agents.slack_agent.MOCK_MODE", False)
    monkeypatch.setattr("agents.slack_agent.SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

    from agents.slack_agent import SlackAgent

    state = create_initial_state(run_id="run-skip", timestamp="2026-06-25T11:00:00")
    assert state["anomaly_detected"] is False  # sanity

    result = SlackAgent().run(state)

    assert result["slack_sent"] is False


# ========================== 2. Mock mode =================================

def test_mock_mode_logs_payload(monkeypatch, caplog):
    """In MOCK_MODE the payload is logged (not POSTed) and slack_sent=True."""
    monkeypatch.setattr("agents.slack_agent.MOCK_MODE", True)

    from agents.slack_agent import SlackAgent

    state = _anomaly_state()

    with patch("agents.slack_agent.requests.post") as mock_post:
        result = SlackAgent().run(state)

        # Must NOT have called requests.post
        mock_post.assert_not_called()

    assert result["slack_sent"] is True


# ========================== 3. Standard template =========================

def test_standard_template_used_for_non_critical(monkeypatch):
    """For HIGH severity, format_slack_message (standard template) should be used."""
    monkeypatch.setattr("agents.slack_agent.MOCK_MODE", True)

    from agents.slack_agent import SlackAgent

    state = _anomaly_state(severity="HIGH")

    with patch("agents.slack_agent.format_slack_message", wraps=__import__(
        "prompts.slack_template", fromlist=["format_slack_message"]
    ).format_slack_message) as mock_std, \
         patch("agents.slack_agent.format_escalation_message") as mock_esc:

        SlackAgent().run(state)

        mock_std.assert_called_once_with(state)
        mock_esc.assert_not_called()


# ========================== 4. Escalation template =======================

def test_escalation_template_used_for_critical(monkeypatch):
    """For CRITICAL severity, format_escalation_message should be used."""
    monkeypatch.setattr("agents.slack_agent.MOCK_MODE", True)

    from agents.slack_agent import SlackAgent

    state = _anomaly_state(severity="CRITICAL")

    with patch("agents.slack_agent.format_slack_message") as mock_std, \
         patch("agents.slack_agent.format_escalation_message", wraps=__import__(
             "prompts.slack_template", fromlist=["format_escalation_message"]
         ).format_escalation_message) as mock_esc:

        SlackAgent().run(state)

        mock_esc.assert_called_once_with(state)
        mock_std.assert_not_called()


# ========================== 5. Webhook POST success ======================

def test_webhook_post_success(monkeypatch):
    """A 200/'ok' response from the webhook should set slack_sent=True."""
    monkeypatch.setattr("agents.slack_agent.MOCK_MODE", False)
    monkeypatch.setattr("agents.slack_agent.SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")

    from agents.slack_agent import SlackAgent

    state = _anomaly_state()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "ok"

    with patch("agents.slack_agent.requests.post", return_value=mock_response) as mock_post:
        agent = SlackAgent(webhook_url="https://hooks.slack.com/test")
        result = agent.run(state)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "https://hooks.slack.com/test"  # URL arg
        assert "json" in call_kwargs[1]  # payload passed as json kwarg

    assert result["slack_sent"] is True


# ========================== 6. Webhook POST failure ======================

def test_webhook_post_failure(monkeypatch):
    """A 500 response from the webhook should set slack_sent=False."""
    monkeypatch.setattr("agents.slack_agent.MOCK_MODE", False)

    from agents.slack_agent import SlackAgent

    state = _anomaly_state()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "server_error"

    with patch("agents.slack_agent.requests.post", return_value=mock_response):
        agent = SlackAgent(webhook_url="https://hooks.slack.com/test")
        result = agent.run(state)

    assert result["slack_sent"] is False


# ========================== 7. Webhook timeout ===========================

def test_webhook_timeout(monkeypatch):
    """A Timeout exception should set slack_sent=False."""
    monkeypatch.setattr("agents.slack_agent.MOCK_MODE", False)

    from agents.slack_agent import SlackAgent

    state = _anomaly_state()

    with patch(
        "agents.slack_agent.requests.post",
        side_effect=requests.exceptions.Timeout("Connection timed out"),
    ):
        agent = SlackAgent(webhook_url="https://hooks.slack.com/test")
        result = agent.run(state)

    assert result["slack_sent"] is False


# ========================== 8. Webhook connection error ==================

def test_webhook_connection_error(monkeypatch):
    """A ConnectionError should set slack_sent=False."""
    monkeypatch.setattr("agents.slack_agent.MOCK_MODE", False)

    from agents.slack_agent import SlackAgent

    state = _anomaly_state()

    with patch(
        "agents.slack_agent.requests.post",
        side_effect=requests.exceptions.ConnectionError("DNS resolution failed"),
    ):
        agent = SlackAgent(webhook_url="https://hooks.slack.com/test")
        result = agent.run(state)

    assert result["slack_sent"] is False


# ========================== 9. No webhook URL ============================

def test_no_webhook_url(monkeypatch):
    """When webhook_url is empty the agent should refuse to POST and set slack_sent=False."""
    monkeypatch.setattr("agents.slack_agent.MOCK_MODE", False)
    monkeypatch.setattr("agents.slack_agent.SLACK_WEBHOOK_URL", "")

    from agents.slack_agent import SlackAgent

    state = _anomaly_state()

    with patch("agents.slack_agent.requests.post") as mock_post:
        agent = SlackAgent(webhook_url="")
        result = agent.run(state)

        mock_post.assert_not_called()

    assert result["slack_sent"] is False


# ========================== 10. Slack message format =====================

def test_slack_message_format():
    """format_slack_message should return a valid Block Kit structure."""
    from prompts.slack_template import format_slack_message

    state = _anomaly_state(severity="HIGH")
    payload = format_slack_message(state)

    # Top-level key
    assert "blocks" in payload
    blocks = payload["blocks"]
    assert isinstance(blocks, list)
    assert len(blocks) > 0

    # First block must be a header
    header = blocks[0]
    assert header["type"] == "header"
    assert header["text"]["type"] == "plain_text"
    assert "HIGH" in header["text"]["text"]

    # Must contain a divider
    block_types = [b["type"] for b in blocks]
    assert "divider" in block_types
    assert "section" in block_types
    assert "context" in block_types

    # Context block should mention the run_id
    context_blocks = [b for b in blocks if b["type"] == "context"]
    assert any(
        "test-run-001" in elem["text"]
        for ctx in context_blocks
        for elem in ctx["elements"]
    )


# ========================== 11. Escalation message format ================

def test_escalation_message_format():
    """format_escalation_message should include @channel mention and Block Kit blocks."""
    from prompts.slack_template import format_escalation_message

    state = _anomaly_state(severity="CRITICAL")
    payload = format_escalation_message(state)

    # Must have both 'text' (fallback) and 'blocks'
    assert "text" in payload
    assert "blocks" in payload

    # Fallback text must contain @channel
    assert "@channel" in payload["text"]

    blocks = payload["blocks"]
    assert isinstance(blocks, list)
    assert len(blocks) > 0

    # Header block must signal CRITICAL
    header = blocks[0]
    assert header["type"] == "header"
    assert "CRITICAL" in header["text"]["text"]

    # One of the section blocks must contain <!channel>
    section_texts = []
    for block in blocks:
        if block["type"] == "section":
            text_obj = block.get("text", {})
            if isinstance(text_obj, dict) and "text" in text_obj:
                section_texts.append(text_obj["text"])
            for field in block.get("fields", []):
                section_texts.append(field.get("text", ""))
    assert any("<!channel>" in t for t in section_texts), (
        "Expected <!channel> mention inside a section block"
    )

    # Context block should mention the run_id
    context_blocks = [b for b in blocks if b["type"] == "context"]
    assert any(
        "test-run-001" in elem["text"]
        for ctx in context_blocks
        for elem in ctx["elements"]
    )
