"""Slack notification agent for the pipeline monitor.

Phase 4: Sends formatted Slack alerts when anomalies are detected and
diagnosed/repaired. Supports MOCK_MODE for testing without a real webhook.
"""

import json

import requests

from config import MOCK_MODE, SLACK_TIMEOUT, SLACK_WEBHOOK_URL
from logging_config import logger
from prompts.slack_template import format_escalation_message, format_slack_message
from state import PipelineState


class SlackAgent:
    """Sends pipeline anomaly notifications to Slack.

    In MOCK_MODE the payload is logged instead of being POSTed.
    For CRITICAL severity, the escalation template is used.

    Attributes:
        webhook_url: Slack incoming-webhook URL.
    """

    def __init__(self, webhook_url: str = None) -> None:
        """Initialise the Slack agent.

        Args:
            webhook_url: Override webhook URL (defaults to config value).
        """
        self.webhook_url: str = webhook_url or SLACK_WEBHOOK_URL

    def run(self, state: PipelineState) -> PipelineState:
        """Execute Slack notification logic.

        Skips sending if no anomaly was detected.  Chooses the escalation
        template for CRITICAL severity and the standard template otherwise.

        Args:
            state: Current pipeline state after repair.

        Returns:
            Updated PipelineState with ``slack_sent`` flag set.
        """
        if not state.get("anomaly_detected", False):
            logger.info("SlackAgent: No anomaly detected — skipping notification.")
            return {**state, "slack_sent": False}

        severity = state.get("severity", "NONE")

        # Choose template
        if severity == "CRITICAL":
            payload = format_escalation_message(state)
            logger.info("SlackAgent: Using CRITICAL escalation template.")
        else:
            payload = format_slack_message(state)
            logger.info(f"SlackAgent: Using standard alert template (severity={severity}).")

        # --- Mock mode: log and return ---
        if MOCK_MODE:
            logger.info(
                f"SlackAgent [MOCK]: Would send payload:\n"
                f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
            )
            return {**state, "slack_sent": True}

        # --- Live mode: POST to webhook ---
        return self._post_webhook(state, payload)

    def _post_webhook(self, state: PipelineState, payload: dict) -> PipelineState:
        """POST the payload to the Slack webhook URL.

        Args:
            state: Current pipeline state.
            payload: Slack Block Kit payload dict.

        Returns:
            Updated PipelineState with ``slack_sent`` result.
        """
        if not self.webhook_url:
            logger.error("SlackAgent: SLACK_WEBHOOK_URL is not configured.")
            return {**state, "slack_sent": False}

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=SLACK_TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 200 and response.text == "ok":
                logger.info("SlackAgent: Notification sent successfully.")
                return {**state, "slack_sent": True}
            else:
                logger.error(
                    f"SlackAgent: Webhook returned {response.status_code} — {response.text}"
                )
                return {**state, "slack_sent": False}

        except requests.exceptions.Timeout:
            logger.error(f"SlackAgent: Webhook request timed out ({SLACK_TIMEOUT}s).")
            return {**state, "slack_sent": False}

        except requests.exceptions.RequestException as exc:
            logger.error(f"SlackAgent: Webhook request failed — {exc}")
            return {**state, "slack_sent": False}


def force_send_slack(state: dict, webhook_url: str) -> bool:
    """Send a Slack notification bypassing MOCK_MODE.

    This function ignores the MOCK_MODE config flag entirely and sends
    the Slack message directly via webhook. Used as a fallback for
    HIGH/CRITICAL incidents that weren't sent during the pipeline run.

    Args:
        state: Pipeline state dict with anomaly details.
        webhook_url: Slack incoming webhook URL.

    Returns:
        True if the message was sent successfully (HTTP 200), False otherwise.
    """
    run_id = state.get("run_id", "unknown")
    severity = state.get("severity", "NONE")

    if not webhook_url:
        logger.error(f"[{run_id}] force_send_slack: No webhook URL provided")
        return False

    try:
        from prompts.slack_template import format_escalation_message, format_slack_message

        if severity == "CRITICAL":
            payload = format_escalation_message(state)
        else:
            payload = format_slack_message(state)

        response = requests.post(
            webhook_url,
            json=payload,
            timeout=SLACK_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200 and response.text == "ok":
            logger.info(f"[{run_id}] force_send_slack: Sent successfully (severity={severity})")
            return True
        else:
            logger.error(
                f"[{run_id}] force_send_slack: Webhook returned "
                f"{response.status_code} — {response.text}"
            )
            return False

    except requests.exceptions.RequestException as exc:
        logger.error(f"[{run_id}] force_send_slack: Request failed — {exc}")
        return False
