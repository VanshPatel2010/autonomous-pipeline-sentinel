"""Repairer Agent: autonomous repair strategies for pipeline anomalies.

Applies graduated repair actions based on anomaly type and severity:

| Severity | missing_data                | data_quality               |
|----------|----------------------------|----------------------------|
| LOW      | Wait & retry next cycle     | Log & monitor               |
| MEDIUM   | Switch to Delhi replica     | Quarantine bad rows         |
| HIGH     | Failover + backfill + log   | Quarantine + flag for review|
| CRITICAL | Escalate to human           | Escalate to human           |

Uses procedural LTM (playbooks) to learn which fixes work over time.
Tracks data gaps for reconciliation.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from config import (
    BACKUP_TABLE,
    CONFIDENCE_MIN,
    DB_PATH,
    MAX_RETRY_ATTEMPTS,
    ORDERS_TABLE,
)
from logging_config import logger
from memory.gap_tracker import record_gap
from memory.playbook_store import get_best_playbook, record_outcome
from scripts.simulate_backup import create_backup_table, failover_from_backup, populate_backup
from state import PipelineState


class RepairerAgent:
    """Applies autonomous repairs to pipeline anomalies.

    The Repairer:
    1. Checks diagnoser confidence (skips if below threshold)
    2. Consults procedural LTM for known-good fixes
    3. Applies the appropriate repair strategy
    4. Records outcome in playbooks (procedural LTM)
    5. Tracks data gaps for later reconciliation
    6. Writes results to state

    Repair strategies are graduated by severity.
    """

    def __init__(self, db_path: str = None) -> None:
        """Initialize RepairerAgent.

        Args:
            db_path: Path to the SQLite database. Defaults to config.DB_PATH.
        """
        if db_path is None:
            import config as _cfg
            db_path = _cfg.DB_PATH
        self.db_path = db_path

    def _wait_and_retry(self, state: PipelineState) -> Dict[str, Any]:
        """LOW severity: log the issue and wait for next polling cycle.

        Args:
            state: Current pipeline state.

        Returns:
            Repair output dict.
        """
        run_id = state["run_id"]
        logger.info(
            f"[{run_id}] Repair strategy: WAIT_AND_RETRY | "
            f"Severity LOW — scheduling retry next cycle. "
            f"No failover — waiting for PRIMARY to recover."
        )

        return {
            "action_taken": "wait_and_retry",
            "success": True,
            "details": (
                f"Low severity {state['anomaly_type']} detected. "
                f"Monitoring for auto-recovery in next polling cycle."
            ),
            "rows_affected": 0,
        }

    def _switch_to_backup(self, state: PipelineState) -> Dict[str, Any]:
        """MEDIUM severity missing_data: failover to Delhi replica.

        Populates the backup table (if needed) and copies backup orders
        into the primary orders table for the gap window.

        Args:
            state: Current pipeline state.

        Returns:
            Repair output dict.
        """
        run_id = state["run_id"]
        gap_minutes = state.get("gap_minutes", 0)
        diagnoser = state.get("diagnoser_output", {})
        estimated_missing = diagnoser.get("estimated_missing_rows", 0)

        logger.info(
            f"[{run_id}] Repair strategy: SWITCH_TO_BACKUP | "
            f"Failover to Delhi replica"
        )

        try:
            # Ensure backup table has data
            populate_backup(hours=max(int(gap_minutes / 60) + 1, 6), db_path=self.db_path)

            # Calculate gap window
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            gap_start = (now - timedelta(minutes=gap_minutes)).isoformat()
            gap_end = now.isoformat()

            # Failover: copy backup rows into primary
            rows_copied = failover_from_backup(gap_start, gap_end, db_path=self.db_path)

            # Record the gap in gap tracker
            gap_id = record_gap(
                run_id=run_id,
                gap_minutes=gap_minutes,
                source_db="mumbai",
                estimated_rows=estimated_missing,
                db_path=self.db_path,
            )

            success = rows_copied > 0

            # Trigger database node failover
            failover_result = None
            try:
                from db.database_manager import trigger_failover
                root_cause = state.get("diagnoser_output", {}).get("root_cause", "unknown")
                failover_result = trigger_failover(
                    reason=root_cause,
                    severity=state.get("severity", "MEDIUM"),
                )
                if failover_result.get("success"):
                    logger.info(
                        f"[{run_id}] Node failover: "
                        f"{failover_result['from_node']['label']} → "
                        f"{failover_result['to_node']['label']}"
                    )
            except Exception as fo_err:
                logger.warning(f"[{run_id}] Node failover skipped: {fo_err}")

            logger.info(
                f"[{run_id}] Failover {'SUCCESS' if success else 'FAILED'}: "
                f"{rows_copied} rows from Delhi replica | gap_id={gap_id}"
            )

            result = {
                "action_taken": "switch_to_backup",
                "success": success,
                "details": (
                    f"Switched to Delhi replica. Copied {rows_copied} rows "
                    f"for {gap_minutes:.0f}min gap. Gap ID: {gap_id}"
                ),
                "rows_affected": rows_copied,
                "gap_id": gap_id,
            }
            if failover_result:
                result["failover_result"] = failover_result
            return result

        except Exception as e:
            logger.error(f"[{run_id}] Failover failed: {e}")
            return {
                "action_taken": "switch_to_backup",
                "success": False,
                "details": f"Failover to Delhi replica failed: {e}",
                "rows_affected": 0,
            }

    def _quarantine_bad_data(self, state: PipelineState) -> Dict[str, Any]:
        """HIGH severity data_quality: move bad rows to quarantine table.

        Moves rows with NULL order_amount from the orders table to the
        quarantine_orders table.

        Args:
            state: Current pipeline state.

        Returns:
            Repair output dict.
        """
        run_id = state["run_id"]

        logger.info(
            f"[{run_id}] Repair strategy: QUARANTINE_BAD_DATA | "
            f"Moving null-amount rows to quarantine"
        )

        conn = sqlite3.connect(self.db_path)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        window_start = (now - timedelta(minutes=30)).isoformat()

        try:
            # Copy bad rows to quarantine
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO quarantine_orders
                    (order_id, created_at, order_amount, source_db,
                     quarantine_reason, quarantined_at, run_id)
                SELECT order_id, created_at, order_amount, source_db,
                       'null_order_amount', ?, ?
                FROM orders
                WHERE order_amount IS NULL AND created_at > ?
                """,
                (now.isoformat(), run_id, window_start),
            )
            quarantined = cursor.rowcount

            # Delete quarantined rows from primary table
            conn.execute(
                """
                DELETE FROM orders
                WHERE order_amount IS NULL AND created_at > ?
                """,
                (window_start,),
            )
            conn.commit()

            success = quarantined > 0

            logger.info(
                f"[{run_id}] Quarantine {'SUCCESS' if success else 'NO ROWS'}: "
                f"{quarantined} rows moved to quarantine_orders"
            )

            return {
                "action_taken": "quarantine_bad_data",
                "success": success,
                "details": (
                    f"Quarantined {quarantined} rows with null order_amount. "
                    f"Rows moved to quarantine_orders table for review."
                ),
                "rows_affected": quarantined,
            }

        except sqlite3.Error as e:
            logger.error(f"[{run_id}] Quarantine failed: {e}")
            return {
                "action_taken": "quarantine_bad_data",
                "success": False,
                "details": f"Quarantine operation failed: {e}",
                "rows_affected": 0,
            }
        finally:
            conn.close()

    def _escalate_to_human(self, state: PipelineState) -> Dict[str, Any]:
        """CRITICAL severity: escalate to human operator.

        The system cannot safely auto-repair CRITICAL issues.
        Logs a prominent warning and marks the state for Slack escalation.

        Args:
            state: Current pipeline state.

        Returns:
            Repair output dict.
        """
        run_id = state["run_id"]
        anomaly_type = state.get("anomaly_type", "unknown")
        severity = state.get("severity", "CRITICAL")
        diagnoser = state.get("diagnoser_output", {})

        # Trigger failover first — move off corrupted node even for CRITICAL
        failover_result = None
        try:
            from db.database_manager import trigger_failover
            failover_result = trigger_failover(
                reason=diagnoser.get("root_cause", "CRITICAL escalation"),
                severity=severity,
            )
            if failover_result.get("success"):
                logger.warning(
                    f"[{run_id}] Emergency failover: "
                    f"{failover_result['from_node']['label']} → "
                    f"{failover_result['to_node']['label']}"
                )
            elif failover_result.get("error") == "ALL_NODES_FAILED":
                logger.critical(f"[{run_id}] ALL_NODES_FAILED — no replicas left")
        except Exception as fo_err:
            logger.warning(f"[{run_id}] Emergency failover skipped: {fo_err}")

        logger.critical(
            f"[{run_id}] 🚨 ESCALATION REQUIRED | "
            f"{anomaly_type} ({severity}) | "
            f"Root cause: {diagnoser.get('root_cause', 'Unknown')} | "
            f"Confidence: {diagnoser.get('confidence', 0):.2f} | "
            f"Auto-repair NOT safe — requires human intervention"
        )

        result = {
            "action_taken": "escalate_to_human",
            "success": False,
            "details": (
                f"CRITICAL: {anomaly_type} anomaly escalated to human operator. "
                f"Root cause: {diagnoser.get('root_cause', 'Unknown')}. "
                f"Auto-repair was not attempted due to severity level."
            ),
            "rows_affected": 0,
        }
        if failover_result:
            result["failover_result"] = failover_result
        return result

    def _choose_strategy(self, state: PipelineState) -> str:
        """Choose the repair strategy based on anomaly type and severity.

        First checks procedural LTM (playbooks) for a known-good fix.
        Falls back to default strategy based on severity matrix.

        Args:
            state: Current pipeline state.

        Returns:
            Strategy name string.
        """
        anomaly_type = state.get("anomaly_type", "")
        severity = state.get("severity", "NONE")

        # Check playbooks for a known-good strategy
        playbook = get_best_playbook(anomaly_type, severity, db_path=self.db_path)
        if playbook and playbook.get("success_rate", 0) > 0.6:
            logger.info(
                f"[{state['run_id']}] Using playbook strategy: "
                f"{playbook['action_taken']} "
                f"(success_rate={playbook['success_rate']:.0%})"
            )
            return playbook["action_taken"]

        # Default strategy matrix
        if severity == "CRITICAL":
            return "escalate_to_human"

        if anomaly_type == "missing_data":
            if severity == "LOW":
                return "wait_and_retry"
            elif severity in ("MEDIUM", "HIGH"):
                return "switch_to_backup"

        if anomaly_type == "data_quality":
            if severity == "LOW":
                return "wait_and_retry"
            elif severity in ("MEDIUM", "HIGH"):
                return "quarantine_bad_data"

        # Fallback
        return "wait_and_retry"

    def run(self, state: PipelineState) -> PipelineState:
        """Execute the repair action for the detected anomaly.

        Steps:
        1. Skip if no anomaly or confidence too low
        2. Choose repair strategy (playbook or default)
        3. Execute the repair
        4. Record outcome in procedural LTM
        5. Write results to state

        Args:
            state: Current pipeline state from Diagnoser Agent.

        Returns:
            Updated state with repairer_output populated.
        """
        run_id = state["run_id"]

        # Skip if no anomaly
        if not state.get("anomaly_detected", False):
            logger.info(f"[{run_id}] Repairer: No anomaly — skipping repair")
            state["repairer_output"] = {}
            return state

        # Check confidence threshold
        diagnoser = state.get("diagnoser_output", {})
        confidence = diagnoser.get("confidence", 0)

        if confidence < CONFIDENCE_MIN:
            logger.warning(
                f"[{run_id}] Repairer: Confidence {confidence:.2f} < "
                f"threshold {CONFIDENCE_MIN}. Skipping auto-repair."
            )
            state["repairer_output"] = {
                "action_taken": "skipped_low_confidence",
                "success": False,
                "details": (
                    f"Diagnosis confidence ({confidence:.2f}) below minimum "
                    f"threshold ({CONFIDENCE_MIN}). Manual review recommended."
                ),
                "rows_affected": 0,
            }
            return state

        logger.info(
            f"[{run_id}] Repairer Agent starting | "
            f"{state['anomaly_type']} ({state['severity']}) | "
            f"confidence={confidence:.2f}"
        )

        # Choose strategy
        strategy = self._choose_strategy(state)
        logger.info(f"[{run_id}] Selected strategy: {strategy}")

        # Execute strategy
        strategy_map = {
            "wait_and_retry": self._wait_and_retry,
            "switch_to_backup": self._switch_to_backup,
            "quarantine_bad_data": self._quarantine_bad_data,
            "escalate_to_human": self._escalate_to_human,
        }

        repair_fn = strategy_map.get(strategy, self._wait_and_retry)
        repairer_output = repair_fn(state)

        # Record outcome in procedural LTM (playbooks)
        record_outcome(
            anomaly_type=state.get("anomaly_type", ""),
            severity=state.get("severity", "NONE"),
            action_taken=repairer_output["action_taken"],
            success=repairer_output["success"],
            db_path=self.db_path,
        )

        # Write to state
        state["repairer_output"] = repairer_output

        # Propagate failover result to top-level state for dashboard
        if repairer_output.get("failover_result"):
            state["failover_result"] = repairer_output["failover_result"]

        # Auto-resolve the incident if repair succeeded
        if repairer_output.get("success"):
            try:
                from memory.incident_store import auto_resolve_incident
                auto_resolve_incident(run_id, db_path=self.db_path)
            except Exception as e:
                logger.warning(f"[{run_id}] Could not auto-resolve incident: {e}")

        logger.info(
            f"[{run_id}] Repair complete: "
            f"action={repairer_output['action_taken']} | "
            f"success={repairer_output['success']} | "
            f"rows_affected={repairer_output['rows_affected']}"
        )

        return state
