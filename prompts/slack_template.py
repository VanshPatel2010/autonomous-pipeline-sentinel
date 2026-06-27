"""Slack message templates for pipeline anomaly notifications.

Builds Slack Block Kit payloads for webhook delivery.
Two templates are provided:
  - format_slack_message: Standard alert for LOW/MEDIUM/HIGH anomalies.
  - format_escalation_message: Urgent alert for CRITICAL severity with @channel mention.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict


# IST offset: UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))

# Severity → emoji mapping
_SEVERITY_EMOJI: Dict[str, str] = {
    "CRITICAL": "🚨",
    "HIGH": "🚨",
    "MEDIUM": "⚠️",
    "LOW": "ℹ️",
    "NONE": "✅",
}

_PIPELINE_NAME = "orders_ingestion"


def _timestamp_ist(iso_timestamp: str) -> str:
    """Convert an ISO-format UTC timestamp to IST display string.

    Args:
        iso_timestamp: ISO-8601 timestamp string (e.g. '2026-06-25T11:00:00').

    Returns:
        Human-readable timestamp in IST, e.g. '2026-06-25 16:30:00 IST'.
    """
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_ist = dt.astimezone(_IST)
        return dt_ist.strftime("%Y-%m-%d %H:%M:%S IST")
    except (ValueError, TypeError):
        return iso_timestamp or "N/A"


def _extract_fields(state: dict) -> dict:
    """Pull relevant fields from pipeline state with safe defaults.

    Args:
        state: Current PipelineState dict.

    Returns:
        Dict of extracted and formatted fields.
    """
    severity = state.get("severity", "NONE")
    emoji = _SEVERITY_EMOJI.get(severity, "❓")
    diagnoser = state.get("diagnoser_output", {}) or {}
    repairer = state.get("repairer_output", {}) or {}

    root_cause = diagnoser.get("root_cause", "Unknown")
    confidence = diagnoser.get("confidence", 0.0)
    actions_taken = repairer.get("actions_taken", [])
    repair_status = repairer.get("status", "unknown")
    affected_tables = state.get("affected_tables", []) or []

    return {
        "severity": severity,
        "emoji": emoji,
        "timestamp_ist": _timestamp_ist(state.get("timestamp", "")),
        "anomaly_type": state.get("anomaly_type", "unknown"),
        "root_cause": root_cause,
        "confidence": confidence,
        "actions_taken": actions_taken,
        "repair_status": repair_status,
        "affected_tables": affected_tables,
        "run_id": state.get("run_id", "N/A"),
    }


def format_slack_message(state: dict) -> dict:
    """Build a Slack webhook payload using Block Kit for rich formatting.

    Includes severity emoji, IST timestamp, pipeline name, issue description,
    actions taken, affected tables, confidence score, and run ID.

    Args:
        state: Current PipelineState dict.

    Returns:
        Slack webhook JSON payload with 'blocks' list.
    """
    f = _extract_fields(state)

    actions_text = (
        "\n".join(f"• {a}" for a in f["actions_taken"])
        if f["actions_taken"]
        else "• No automated actions taken"
    )
    tables_text = ", ".join(f"`{t}`" for t in f["affected_tables"]) or "None"

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{f['emoji']} Pipeline Alert — {f['severity']}",
                "emoji": True,
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Pipeline:*\n`{_PIPELINE_NAME}`",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Detected At:*\n{f['timestamp_ist']}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Anomaly Type:*\n{f['anomaly_type']}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Severity:*\n{f['emoji']} {f['severity']}",
                },
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🔍 Root Cause:*\n{f['root_cause']}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🔧 Actions Taken* (status: `{f['repair_status']}`):\n{actions_text}",
            },
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Affected Tables:*\n{tables_text}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Confidence:*\n{f['confidence']:.0%}",
                },
            ],
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Run ID: `{f['run_id']}`",
                }
            ],
        },
    ]

    return {"blocks": blocks}


def format_escalation_message(state: dict) -> dict:
    """Build an escalation Slack payload for CRITICAL severity anomalies.

    Adds @channel mention, urgent header language, and a warning callout
    on top of the standard alert blocks.

    Args:
        state: Current PipelineState dict.

    Returns:
        Slack webhook JSON payload with 'text' (fallback) and 'blocks'.
    """
    f = _extract_fields(state)

    actions_text = (
        "\n".join(f"• {a}" for a in f["actions_taken"])
        if f["actions_taken"]
        else "• No automated actions taken"
    )
    tables_text = ", ".join(f"`{t}`" for t in f["affected_tables"]) or "None"

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"🚨 CRITICAL PIPELINE FAILURE — IMMEDIATE ACTION REQUIRED",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "<!channel> *The `orders_ingestion` pipeline has a CRITICAL failure.* "
                    "Automated repair may be insufficient — manual intervention is required."
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Pipeline:*\n`{_PIPELINE_NAME}`",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Detected At:*\n{f['timestamp_ist']}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Anomaly Type:*\n{f['anomaly_type']}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Severity:*\n🚨 CRITICAL",
                },
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🔍 Root Cause:*\n{f['root_cause']}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🔧 Actions Taken* (status: `{f['repair_status']}`):\n{actions_text}",
            },
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Affected Tables:*\n{tables_text}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Confidence:*\n{f['confidence']:.0%}",
                },
            ],
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Run ID: `{f['run_id']}` | ⏰ Escalated at {f['timestamp_ist']}",
                }
            ],
        },
    ]

    return {
        "text": f"🚨 CRITICAL: {_PIPELINE_NAME} pipeline failure — @channel",
        "blocks": blocks,
    }
