"""Entry point for the Autonomous Data Pipeline Monitor.

Runs the LangGraph pipeline on a 5-minute schedule.
Each run generates a unique run_id for traceability.

Usage:
    python main.py              # Start the scheduled monitor
    python main.py --once       # Run a single check and exit
"""

import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone

import schedule

from config import DB_PATH, MOCK_MODE, POLLING_INTERVAL_MINUTES
from graph import pipeline_graph
from logging_config import logger
from memory.incident_store import init_db as init_incident_db, insert_incident
from seed_db import create_tables
from state import create_initial_state


def run_pipeline() -> None:
    """Execute a single pipeline monitoring run.

    Generates a unique run_id, creates initial state,
    invokes the LangGraph pipeline, and logs results.
    """
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(f"{'=' * 60}")
    logger.info(f"Pipeline run starting | run_id={run_id} | time={timestamp}")
    logger.info(f"{'=' * 60}")

    # Create fresh state for this run
    initial_state = create_initial_state(run_id=run_id, timestamp=timestamp)

    try:
        # Invoke the LangGraph pipeline
        final_state = pipeline_graph.invoke(initial_state)

        # Log results
        diagnoser = final_state.get("diagnoser_output", {})
        logger.info(
            f"[{run_id}] Run complete | "
            f"anomaly={final_state['anomaly_detected']} | "
            f"type={final_state['anomaly_type'] or 'none'} | "
            f"severity={final_state['severity']} | "
            f"count={final_state['raw_count']} | "
            f"baseline={final_state['expected_avg']:.1f} | "
            f"null_rate={final_state.get('null_rate', 0):.2%}"
        )

        if final_state["anomaly_detected"]:
            logger.warning(
                f"[{run_id}] ⚠️  ANOMALY SUMMARY: "
                f"{final_state['anomaly_type']} ({final_state['severity']}) | "
                f"gap={final_state['gap_minutes']:.0f}min | "
                f"tables={final_state['affected_tables']} | "
                f"root_cause={diagnoser.get('root_cause', 'N/A')}"
            )
        else:
            logger.info(f"[{run_id}] ✅ Pipeline healthy. No issues detected.")

        # Persist incident to episodic LTM
        insert_incident({
            "run_id": run_id,
            "timestamp": timestamp,
            "anomaly_type": final_state.get("anomaly_type", ""),
            "severity": final_state.get("severity", "NONE"),
            "gap_minutes": final_state.get("gap_minutes", 0.0),
            "root_cause": diagnoser.get("root_cause", ""),
            "affected_tables": final_state.get("affected_tables", []),
            "fix_taken": final_state.get("repairer_output", {}).get("action_taken", ""),
            "resolved": int(final_state.get("repairer_output", {}).get("success", False)),
            "confidence": diagnoser.get("confidence", 0.0),
        })

        # Fallback: force-send Slack for HIGH/CRITICAL if it wasn't sent
        from config import SLACK_WEBHOOK_URL
        if (
            not final_state.get("slack_sent")
            and final_state.get("severity") in ("HIGH", "CRITICAL")
            and SLACK_WEBHOOK_URL
        ):
            try:
                from agents.slack_agent import force_send_slack
                force_send_slack(final_state, SLACK_WEBHOOK_URL)
            except Exception as slack_err:
                logger.error(f"[{run_id}] force_send_slack fallback failed: {slack_err}")

    except Exception as e:
        logger.error(f"[{run_id}] Pipeline run failed: {e}", exc_info=True)


def init() -> None:
    """Initialize database tables on startup."""
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    conn.close()
    init_incident_db(DB_PATH)
    logger.info(f"Database initialized at {DB_PATH}")


def main() -> None:
    """Main entry point."""
    logger.info("Autonomous Data Pipeline Monitor starting...")
    logger.info(f"Mock mode: {MOCK_MODE}")
    logger.info(f"Polling interval: {POLLING_INTERVAL_MINUTES} minutes")
    logger.info(f"Database: {DB_PATH}")

    # Initialize DB
    init()

    # Check for --once flag
    if "--once" in sys.argv:
        logger.info("Running single check (--once mode)")
        run_pipeline()
        return

    # Schedule recurring runs
    schedule.every(POLLING_INTERVAL_MINUTES).minutes.do(run_pipeline)

    # Run immediately on startup
    logger.info("Running initial check...")
    run_pipeline()

    # Enter scheduling loop
    logger.info(
        f"Entering schedule loop (every {POLLING_INTERVAL_MINUTES} min). "
        f"Press Ctrl+C to stop."
    )
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user.")


if __name__ == "__main__":
    main()
