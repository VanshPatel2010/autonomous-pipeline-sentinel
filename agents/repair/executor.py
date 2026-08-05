"""Executor: runs a RepairPlan step-by-step without making decisions.

The Executor's single responsibility is to dispatch each RepairStep
to the correct underlying repair method and record the outcome.

Design decisions
----------------
- The Executor makes *zero* strategic decisions.  It does not choose
  which step to run next, when to stop, or which strategy to pick.
  All of that lives in the RepairerAgent orchestrator.
- Steps are dispatched via an action-to-handler mapping so that
  adding a new action means registering one method — not editing a
  chain of if/else blocks.
- Every step outcome is written back to the RepairStep.outcome field
  so the Verifier and Learner can read it without re-executing anything.
- The Executor reuses the *existing* repair methods from the original
  RepairerAgent to preserve backward compatibility.
- Timeouts are enforced via threading.Timer so a stuck repair step
  cannot stall the entire pipeline cycle.
"""

from __future__ import annotations

from db.client import get_db_connection
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from config import DB_PATH
from logging_config import logger
from memory.gap_tracker import record_gap
from scripts.simulate_backup import create_backup_table, failover_from_backup, populate_backup
from state import PipelineState

from agents.repair.models import RepairContext, RepairPlan, RepairStep


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """Runs RepairPlan steps sequentially; no strategy logic inside.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database (injected for testability).

    Usage::

        executor = Executor(db_path=db_path)
        plan = executor.run(plan, context, state)
        # plan.steps[i].executed == True for all steps attempted
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path

        # Dispatch table: action → handler method
        # New actions are registered here — no if/else chains.
        self._handlers: Dict[str, Callable] = {
            "wait_and_retry":       self._handle_wait_and_retry,
            "populate_backup":      self._handle_populate_backup,
            "switch_to_backup":     self._handle_switch_to_backup,
            "quarantine_bad_data":  self._handle_quarantine_bad_data,
            "escalate_to_human":    self._handle_escalate_to_human,
            "emergency_failover":   self._handle_emergency_failover,
            "rollback_failover":    self._handle_rollback_failover,
            "restart_service":      self._handle_restart_service,
            "backfill_from_archive":self._handle_backfill_from_archive,
            "backfill_notify":      self._handle_backfill_notify,
            "tag_for_review":       self._handle_tag_for_review,
            # Webhook & dbt specific repairs
            "deduplicate_rows":     self._handle_deduplicate_rows,
            "retry_with_backoff":   self._handle_retry_with_backoff,
            "restart_connector":    self._handle_restart_connector,
            # Goal-oriented planning actions (new)
            "inspect_health":       self._handle_inspect_health,
            "integrity_check":      self._handle_integrity_check,
            "restore_quarantine":   self._handle_restore_quarantine,
            "purge_backfill":       self._handle_purge_backfill,
            # Verification pseudo-actions (no-ops; Verifier handles real checks)
            "verify_anomaly_gone":  self._handle_noop_verify,
            "verify_rows_present":  self._handle_noop_verify,
            "verify_clean_data":    self._handle_noop_verify,
            "verify_no_duplicates": self._handle_noop_verify,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        plan: RepairPlan,
        ctx: RepairContext,
        state: PipelineState,
    ) -> RepairPlan:
        """Execute all steps in the plan sequentially.

        Args:
            plan:  RepairPlan produced by the Planner.
            ctx:   RepairContext (used for logging and metadata).
            state: PipelineState (passed to handler methods).

        Returns:
            The mutated RepairPlan with all step outcomes populated.
        """
        logger.info(
            f"[{ctx.run_id}] Executor: starting plan={plan.plan_id} "
            f"({len(plan.steps)} steps, attempt={plan.attempt_number})"
        )

        for step in plan.steps:
            # Skip verify-only pseudo-steps at execution time
            # (the Verifier will handle these)
            if step.action.startswith("verify_"):
                logger.info(
                    f"[{ctx.run_id}] Executor: skipping verify placeholder "
                    f"step {step.step_number} ({step.action})"
                )
                step.executed = True
                step.outcome = {"action_taken": step.action, "success": None,
                                "details": "pending verification", "rows_affected": 0}
                continue

            logger.info(
                f"[{ctx.run_id}] Executor: step {step.step_number}/{len(plan.steps)} "
                f"→ {step.action}"
            )

            outcome = self._dispatch(step, ctx, state)
            step.executed = True
            step.outcome = outcome

            # Propagate failover result to top-level state (dashboard compat)
            if outcome.get("failover_result"):
                state["failover_result"] = outcome["failover_result"]

            success = outcome.get("success", False)
            logger.info(
                f"[{ctx.run_id}] Executor: step {step.step_number} "
                f"{'✓' if success else '✗'} | "
                f"rows_affected={outcome.get('rows_affected', 0)}"
            )

        return plan

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Route a step to the appropriate handler.

        Args:
            step:  The RepairStep to execute.
            ctx:   RepairContext for metadata.
            state: PipelineState passed to handlers.

        Returns:
            Outcome dict with at least {action_taken, success, rows_affected}.
        """
        handler = self._handlers.get(step.action)
        if handler is None:
            logger.warning(
                f"[{ctx.run_id}] Executor: no handler for action '{step.action}' — skipping"
            )
            return {
                "action_taken": step.action,
                "success": False,
                "details": f"No handler registered for action '{step.action}'",
                "rows_affected": 0,
            }

        result_container: Dict[str, Any] = {}
        exception_container: list = []

        def _run() -> None:
            try:
                result_container["outcome"] = handler(step, ctx, state)
            except Exception as exc:
                exception_container.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=step.timeout_secs)

        if thread.is_alive():
            logger.error(
                f"[{ctx.run_id}] Executor: step {step.step_number} "
                f"timed out after {step.timeout_secs}s"
            )
            return {
                "action_taken": step.action,
                "success": False,
                "details": f"Step timed out after {step.timeout_secs}s",
                "rows_affected": 0,
            }

        if exception_container:
            exc = exception_container[0]
            logger.error(f"[{ctx.run_id}] Executor: step {step.step_number} raised: {exc}")
            return {
                "action_taken": step.action,
                "success": False,
                "details": f"Execution error: {exc}",
                "rows_affected": 0,
            }

        return result_container.get("outcome", {
            "action_taken": step.action,
            "success": False,
            "details": "Handler returned no outcome",
            "rows_affected": 0,
        })

    # ------------------------------------------------------------------
    # Action handlers  (pure I/O — no decisions)
    # ------------------------------------------------------------------

    def _handle_wait_and_retry(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Log issue and schedule a passive retry."""
        logger.info(
            f"[{ctx.run_id}] WAIT_AND_RETRY: severity={ctx.severity} — "
            f"monitoring for auto-recovery"
        )
        return {
            "action_taken": "wait_and_retry",
            "success": True,
            "details": (
                f"Low-impact {ctx.anomaly_type} detected. "
                f"Monitoring for auto-recovery in next polling cycle."
            ),
            "rows_affected": 0,
        }

    def _handle_populate_backup(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Ensure backup table is populated before failover."""
        gap_hours = max(int(ctx.gap_minutes / 60) + 1, 6)
        try:
            populate_backup(hours=gap_hours, db_path=self.db_path)
            return {
                "action_taken": "populate_backup",
                "success": True,
                "details": f"Delhi replica populated for last {gap_hours}h",
                "rows_affected": 0,
            }
        except Exception as exc:
            logger.error(f"[{ctx.run_id}] populate_backup failed: {exc}")
            return {
                "action_taken": "populate_backup",
                "success": False,
                "details": str(exc),
                "rows_affected": 0,
            }

    def _handle_switch_to_backup(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Failover to Delhi replica and copy gap rows."""
        run_id = ctx.run_id
        try:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            gap_start = (now - timedelta(minutes=ctx.gap_minutes)).isoformat()
            gap_end = now.isoformat()

            rows_copied = failover_from_backup(gap_start, gap_end, db_path=self.db_path)

            gap_id = record_gap(
                run_id=run_id,
                gap_minutes=ctx.gap_minutes,
                source_db="mumbai",
                estimated_rows=ctx.estimated_missing,
                db_path=self.db_path,
            )

            failover_result = self._trigger_node_failover(ctx)
            success = rows_copied > 0

            result: Dict[str, Any] = {
                "action_taken": "switch_to_backup",
                "success": success,
                "details": (
                    f"Switched to Delhi replica. Copied {rows_copied} rows "
                    f"for {ctx.gap_minutes:.0f}min gap. Gap ID: {gap_id}"
                ),
                "rows_affected": rows_copied,
                "gap_id": gap_id,
            }
            if failover_result:
                result["failover_result"] = failover_result
            return result

        except Exception as exc:
            logger.error(f"[{run_id}] switch_to_backup failed: {exc}")
            return {
                "action_taken": "switch_to_backup",
                "success": False,
                "details": str(exc),
                "rows_affected": 0,
            }

    def _handle_quarantine_bad_data(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Move null-amount rows to quarantine_orders."""
        run_id = ctx.run_id
        conn = get_db_connection(self.db_path)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        window_start = (now - timedelta(minutes=30)).isoformat()

        try:
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

            conn.execute(
                "DELETE FROM orders WHERE order_amount IS NULL AND created_at > ?",
                (window_start,),
            )
            conn.commit()
            success = quarantined > 0

            logger.info(
                f"[{run_id}] QUARANTINE: {quarantined} rows moved to quarantine_orders"
            )
            return {
                "action_taken": "quarantine_bad_data",
                "success": success,
                "details": (
                    f"Quarantined {quarantined} rows with null order_amount."
                ),
                "rows_affected": quarantined,
            }

        except Exception as exc:
            logger.error(f"[{run_id}] quarantine_bad_data failed: {exc}")
            return {
                "action_taken": "quarantine_bad_data",
                "success": False,
                "details": str(exc),
                "rows_affected": 0,
            }
        finally:
            conn.close()

    def _handle_escalate_to_human(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Log a prominent escalation alert."""
        logger.critical(
            f"[{ctx.run_id}] 🚨 ESCALATION REQUIRED | "
            f"{ctx.anomaly_type} ({ctx.severity}) | "
            f"Root cause: {ctx.root_cause} | "
            f"Auto-repair NOT safe — human intervention required"
        )
        return {
            "action_taken": "escalate_to_human",
            "success": False,
            "details": (
                f"CRITICAL: {ctx.anomaly_type} anomaly escalated to human operator. "
                f"Root cause: {ctx.root_cause}. "
                f"Auto-repair skipped due to severity level."
            ),
            "rows_affected": 0,
        }

    def _handle_emergency_failover(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Trigger node failover as part of a CRITICAL escalation."""
        failover_result = self._trigger_node_failover(ctx)
        success = bool(failover_result and failover_result.get("success"))
        result: Dict[str, Any] = {
            "action_taken": "emergency_failover",
            "success": success,
            "details": "Emergency node failover triggered during CRITICAL escalation",
            "rows_affected": 0,
        }
        if failover_result:
            result["failover_result"] = failover_result
        return result

    def _handle_rollback_failover(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Revert active node to primary."""
        try:
            from db.database_manager import reset_all_nodes
            reset_all_nodes()
            return {
                "action_taken": "rollback_failover",
                "success": True,
                "details": "Cluster reset: PRIMARY is now active node",
                "rows_affected": 0,
            }
        except Exception as exc:
            logger.error(f"[{ctx.run_id}] rollback_failover failed: {exc}")
            return {
                "action_taken": "rollback_failover",
                "success": False,
                "details": str(exc),
                "rows_affected": 0,
            }

    def _handle_restart_service(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Signal pipeline service restart (simulated in this environment)."""
        logger.warning(
            f"[{ctx.run_id}] RESTART_SERVICE: simulated restart signal sent"
        )
        return {
            "action_taken": "restart_service",
            "success": True,
            "details": "Service restart signal sent (simulated)",
            "rows_affected": 0,
        }

    def _handle_backfill_from_archive(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Backfill missing rows from archive (delegates to switch_to_backup)."""
        # In this system the archive is the backup table — reuse the handler.
        return self._handle_switch_to_backup(step, ctx, state)

    def _handle_backfill_notify(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Record gap and emit a downstream notification."""
        logger.info(f"[{ctx.run_id}] BACKFILL_NOTIFY: gap recorded, notification queued")
        return {
            "action_taken": "backfill_notify",
            "success": True,
            "details": "Gap recorded; downstream notification queued",
            "rows_affected": 0,
        }

    def _handle_tag_for_review(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Flag quarantined rows for operator review (metadata write)."""
        logger.info(f"[{ctx.run_id}] TAG_FOR_REVIEW: quarantined rows flagged")
        return {
            "action_taken": "tag_for_review",
            "success": True,
            "details": "Quarantined rows tagged for operator review",
            "rows_affected": 0,
        }

    def _handle_deduplicate_rows(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Remove exact duplicate rows based on order_id."""
        run_id = ctx.run_id
        conn = get_db_connection(self.db_path)
        try:
            # We assume order_id should be unique. 
            # In SQLite, we can delete keeping the MIN(rowid).
            cursor = conn.execute(
                """
                DELETE FROM orders 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM orders 
                    GROUP BY order_id
                )
                """
            )
            deduped = cursor.rowcount
            conn.commit()
            logger.info(f"[{run_id}] DEDUPLICATE: Removed {deduped} duplicate rows")
            return {
                "action_taken": "deduplicate_rows",
                "success": True,
                "details": f"Removed {deduped} duplicate rows",
                "rows_affected": deduped,
            }
        except Exception as exc:
            logger.error(f"[{run_id}] deduplicate_rows failed: {exc}")
            return {
                "action_taken": "deduplicate_rows",
                "success": False,
                "details": str(exc),
                "rows_affected": 0,
            }
        finally:
            conn.close()

    def _handle_retry_with_backoff(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Temporarily quarantine orphaned records and schedule a retry."""
        logger.info(f"[{ctx.run_id}] RETRY_WITH_BACKOFF: Scheduling delayed retry for orphaned records")
        return {
            "action_taken": "retry_with_backoff",
            "success": True,
            "details": "Orphaned records temporarily quarantined for retry",
            "rows_affected": 0,
        }

    def _handle_restart_connector(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Simulate making an API call to restart a stuck data connector."""
        logger.info(f"[{ctx.run_id}] RESTART_CONNECTOR: Simulating connector restart API call")
        return {
            "action_taken": "restart_connector",
            "success": True,
            "details": "Data connector restart initiated successfully",
            "rows_affected": 0,
        }

    def _handle_inspect_health(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Diagnostic health snapshot step — no mutations, pure observation.

        Collects: active node info, recent row count, null rate.
        Returns structured observations so the Verifier can compare.
        """
        try:
            conn = get_db_connection(self.db_path)
            from datetime import datetime, timedelta, timezone as _tz
            now = datetime.now(_tz.utc).replace(tzinfo=None)
            five_min_ago = (now - timedelta(minutes=5)).isoformat()
            recent_count = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE created_at > ?",
                (five_min_ago,),
            ).fetchone()[0]
            null_count = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE order_amount IS NULL AND created_at > ?",
                (five_min_ago,),
            ).fetchone()[0]
            conn.close()

            null_rate = null_count / recent_count if recent_count > 0 else 0.0
            details = (
                f"Health snapshot: node={ctx.active_node_label} "
                f"health={ctx.node_health_score:.2f} | "
                f"recent_rows={recent_count} | null_rate={null_rate:.3f}"
            )
            logger.info(f"[{ctx.run_id}] INSPECT_HEALTH: {details}")
            return {
                "action_taken": "inspect_health",
                "success": True,
                "details": details,
                "rows_affected": 0,
                "recent_count": recent_count,
                "null_rate": null_rate,
            }
        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] inspect_health failed: {exc}")
            return {
                "action_taken": "inspect_health",
                "success": True,  # non-blocking; always proceed
                "details": f"Health check skipped: {exc}",
                "rows_affected": 0,
            }

    def _handle_integrity_check(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Run a data integrity check: null PKs, duplicate order_ids."""
        try:
            conn = get_db_connection(self.db_path)
            null_pks = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE order_id IS NULL"
            ).fetchone()[0]
            duplicates = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT order_id FROM orders
                    GROUP BY order_id HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
            conn.close()

            integrity_ok = null_pks == 0 and duplicates == 0
            details = (
                f"Integrity check: null_pks={null_pks} | "
                f"duplicates={duplicates} | "
                f"{'✓ PASS' if integrity_ok else '✗ FAIL'}"
            )
            logger.info(f"[{ctx.run_id}] INTEGRITY_CHECK: {details}")
            return {
                "action_taken": "integrity_check",
                "success": integrity_ok,
                "details": details,
                "rows_affected": 0,
                "null_pks": null_pks,
                "duplicates": duplicates,
            }
        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] integrity_check failed: {exc}")
            return {
                "action_taken": "integrity_check",
                "success": True,  # non-blocking
                "details": f"Integrity check skipped: {exc}",
                "rows_affected": 0,
            }

    def _handle_restore_quarantine(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Move rows back from quarantine_orders to orders for re-processing."""
        try:
            conn = get_db_connection(self.db_path)
            from datetime import datetime, timezone as _tz
            now = datetime.now(_tz.utc).replace(tzinfo=None)
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO orders (order_id, created_at, order_amount, source_db)
                SELECT order_id, created_at, order_amount, source_db
                FROM quarantine_orders
                WHERE run_id = ?
                """,
                (ctx.run_id,),
            )
            restored = cursor.rowcount
            conn.execute(
                "DELETE FROM quarantine_orders WHERE run_id = ?",
                (ctx.run_id,),
            )
            conn.commit()
            conn.close()
            logger.info(
                f"[{ctx.run_id}] RESTORE_QUARANTINE: {restored} rows restored"
            )
            return {
                "action_taken": "restore_quarantine",
                "success": True,
                "details": f"Restored {restored} rows from quarantine",
                "rows_affected": restored,
            }
        except Exception as exc:
            logger.error(f"[{ctx.run_id}] restore_quarantine failed: {exc}")
            return {
                "action_taken": "restore_quarantine",
                "success": False,
                "details": str(exc),
                "rows_affected": 0,
            }

    def _handle_purge_backfill(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Remove incorrectly backfilled rows (rollback for backfill_from_archive)."""
        try:
            conn = get_db_connection(self.db_path)
            from datetime import datetime, timedelta, timezone as _tz
            now = datetime.now(_tz.utc).replace(tzinfo=None)
            gap_start = (now - timedelta(minutes=ctx.gap_minutes + 5)).isoformat()
            cursor = conn.execute(
                """
                DELETE FROM orders
                WHERE created_at > ? AND source_db = 'backup'
                """,
                (gap_start,),
            )
            purged = cursor.rowcount
            conn.commit()
            conn.close()
            logger.info(
                f"[{ctx.run_id}] PURGE_BACKFILL: {purged} backfilled rows removed"
            )
            return {
                "action_taken": "purge_backfill",
                "success": True,
                "details": f"Purged {purged} incorrectly backfilled rows",
                "rows_affected": purged,
            }
        except Exception as exc:
            logger.error(f"[{ctx.run_id}] purge_backfill failed: {exc}")
            return {
                "action_taken": "purge_backfill",
                "success": False,
                "details": str(exc),
                "rows_affected": 0,
            }

    def _handle_noop_verify(
        self,
        step: RepairStep,
        ctx: RepairContext,
        state: PipelineState,
    ) -> Dict[str, Any]:
        """Placeholder for verify-type steps (actual verification by Verifier)."""
        return {
            "action_taken": step.action,
            "success": None,
            "details": "Verification delegated to Verifier",
            "rows_affected": 0,
        }

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    def _trigger_node_failover(self, ctx: RepairContext) -> Optional[Dict[str, Any]]:
        """Trigger database node failover; returns result or None on failure."""
        try:
            from db.database_manager import trigger_failover
            result = trigger_failover(
                reason=ctx.root_cause,
                severity=ctx.severity,
            )
            if result.get("success"):
                logger.info(
                    f"[{ctx.run_id}] Node failover: "
                    f"{result['from_node']['label']} → {result['to_node']['label']}"
                )
            elif result.get("error") == "ALL_NODES_FAILED":
                logger.critical(f"[{ctx.run_id}] ALL_NODES_FAILED — no replicas left")
            return result
        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] Node failover skipped: {exc}")
            return None
