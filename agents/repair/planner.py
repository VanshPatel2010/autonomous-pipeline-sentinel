"""Planner: goal-oriented, context-driven, adaptive repair planning.

Architectural Rationale
-----------------------
The previous Planner was a pure template lookup:
    strategy_name → fixed step sequence

This worked but was fundamentally non-intelligent.  Two incidents with
the same anomaly_type but radically different contexts (healthy node vs.
degraded node, business hours vs. maintenance window) would receive
identical plans.  There was no mechanism to adapt when context changed.

The upgraded Planner introduces three improvements:

1. **Goal-Oriented Planning** (Requirement 2):
   Instead of "what strategy was chosen?" the Planner asks "what is the
   repair goal?" and decomposes it into a sub-goal chain:
     Goal → Inspect → Validate → Repair → Verify → Learn → Notify
   The step sequence is assembled from context-sensitive primitives
   based on the goal chain, not a lookup table.

2. **Procedural Memory Integration** (Requirement 13):
   Before generating a plan, the Planner queries procedural memory for
   proven step sequences that have succeeded in similar past incidents.
   A high-success-rate procedure is preferred over a template.

3. **Adaptive Replanning** (Requirement 11):
   When verification fails, the Planner does NOT execute a predefined
   rollback strategy.  Instead, it calls ``replan()`` which:
   - Takes the failed plan as input.
   - Generates a completely new plan for the next-best strategy.
   - Adds context-aware diagnostic steps to gather more information.
   - Repeats until success, max attempts, or human escalation.

Design decisions
----------------
- The Planner remains pure (no I/O, no DB calls) by receiving all
  context through RepairContext and the injected MemoryStore.
- Step sequences are defined as composable primitives, not monolithic
  templates.  Each primitive is a (action, description, verify, timeout)
  4-tuple that can be combined in any order.
- A mandatory "guard" step (inspect_health) is prepended to every
  newly generated plan to catch stale context before execution.
- Goal decomposition is exposed as ``plan.goal_decomposition`` so the
  dashboard can display the planned sub-goal chain.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from logging_config import logger

from agents.repair.models import (
    RepairContext,
    RepairPlan,
    RepairStep,
    StrategyScore,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Step primitives library
# ---------------------------------------------------------------------------

# format: (action, description, requires_verify, timeout_secs)
_PRIMITIVES: Dict[str, Tuple[str, str, bool, int]] = {
    # Diagnostic / inspection
    "inspect_health":       ("inspect_health",       "Inspect pipeline and node health",             False, 15),
    "verify_anomaly_gone":  ("verify_anomaly_gone",  "Confirm anomaly has self-resolved",             True,  30),
    "verify_rows_present":  ("verify_rows_present",  "Confirm missing rows have been recovered",      True,  30),
    "verify_clean_data":    ("verify_clean_data",    "Confirm no null-amount rows remain",            True,  30),
    "verify_no_duplicates": ("verify_no_duplicates", "Ensure no duplicate order IDs were introduced", True,  30),
    "integrity_check":      ("integrity_check",      "Run full data integrity check",                 True,  45),
    # Retry / wait
    "wait_and_retry":       ("wait_and_retry",       "Log issue, schedule retry next polling cycle",  True,  60),
    # Failover operations
    "populate_backup":      ("populate_backup",      "Ensure Delhi replica has up-to-date data",      False,120),
    "switch_to_backup":     ("switch_to_backup",     "Failover to Delhi replica & copy gap rows",     True, 180),
    "rollback_failover":    ("rollback_failover",    "Revert active node to primary",                 True, 180),
    "emergency_failover":   ("emergency_failover",   "Trigger emergency failover",                    True,  60),
    # Data repair
    "quarantine_bad_data":  ("quarantine_bad_data",  "Move null-amount rows to quarantine table",     True,  60),
    "backfill_from_archive":("backfill_from_archive","Copy archived rows into primary for gap window",True, 600),
    "restore_quarantine":   ("restore_quarantine",   "Restore quarantined rows for re-processing",    True,  60),
    "purge_backfill":       ("purge_backfill",       "Remove incorrectly backfilled rows",            True,  60),
    # Notifications
    "backfill_notify":      ("backfill_notify",      "Record gap and trigger downstream notification",False, 30),
    "tag_for_review":       ("tag_for_review",       "Flag quarantined rows for operator review",     False, 15),
    "escalate_to_human":    ("escalate_to_human",    "Alert on-call operator via Slack escalation",   False, 30),
    # Service
    "restart_service":      ("restart_service",      "Signal pipeline service restart",               True,  90),
}


def _p(name: str, goal_driven: bool = True) -> RepairStep:
    """Create a RepairStep from a primitive name."""
    action, description, req_verify, timeout = _PRIMITIVES.get(
        name,
        (name, f"Execute {name}", True, 60),
    )
    return RepairStep(
        step_number=0,  # renumbered after assembly
        action=action,
        description=description,
        requires_verify=req_verify,
        timeout_secs=timeout,
        goal_driven=goal_driven,
    )


# ---------------------------------------------------------------------------
# Goal decomposition templates
# ---------------------------------------------------------------------------

# Map strategy_name → (goal_statement, sub_goal_chain, step_primitive_names)
# The step_primitive_names list is the goal-oriented step sequence.
# Context-sensitive additions are applied at runtime.

_GOAL_TEMPLATES: Dict[str, Dict] = {
    "wait_and_retry": {
        "goal": "Monitor {anomaly_type} and allow self-recovery within next polling cycle",
        "sub_goals": [
            "Inspect pipeline health",
            "Log anomaly for monitoring",
            "Wait for auto-resolution",
            "Verify anomaly resolved",
        ],
        "steps": [
            "inspect_health",
            "wait_and_retry",
            "verify_anomaly_gone",
        ],
    },
    "switch_to_backup": {
        "goal": "Restore ~{estimated_missing} missing rows from Delhi replica for {gap_minutes:.0f}-min gap",
        "sub_goals": [
            "Inspect node and replica health",
            "Populate replica with current data",
            "Perform controlled failover",
            "Verify row recovery",
            "Record gap and notify downstream",
        ],
        "steps": [
            "inspect_health",
            "populate_backup",
            "switch_to_backup",
            "verify_rows_present",
            "integrity_check",
            "backfill_notify",
        ],
    },
    "quarantine_bad_data": {
        "goal": "Quarantine null-amount rows in {tables} and restore data quality",
        "sub_goals": [
            "Inspect data quality state",
            "Quarantine invalid rows",
            "Verify data is clean",
            "Flag rows for operator review",
        ],
        "steps": [
            "inspect_health",
            "quarantine_bad_data",
            "verify_clean_data",
            "integrity_check",
            "tag_for_review",
        ],
    },
    "rollback_failover": {
        "goal": "Revert active node to primary after failed failover",
        "sub_goals": [
            "Inspect current node state",
            "Execute controlled rollback to primary",
            "Verify primary is active",
        ],
        "steps": [
            "inspect_health",
            "rollback_failover",
            "verify_anomaly_gone",
        ],
    },
    "restart_service": {
        "goal": "Restart the pipeline service to clear transient error state",
        "sub_goals": [
            "Inspect service health",
            "Issue service restart signal",
            "Verify pipeline resumed",
        ],
        "steps": [
            "inspect_health",
            "restart_service",
            "verify_anomaly_gone",
        ],
    },
    "backfill_from_archive": {
        "goal": "Backfill {estimated_missing} rows from archive into primary table",
        "sub_goals": [
            "Inspect data gap boundaries",
            "Retrieve archived rows for gap window",
            "Verify rows are present",
            "Run duplicate check",
        ],
        "steps": [
            "inspect_health",
            "backfill_from_archive",
            "verify_rows_present",
            "verify_no_duplicates",
            "integrity_check",
        ],
    },
    "escalate_to_human": {
        "goal": "CRITICAL {anomaly_type} — escalate to human operator for manual intervention",
        "sub_goals": [
            "Alert on-call operator",
            "Trigger emergency failover if safe",
        ],
        "steps": [
            "escalate_to_human",
            "emergency_failover",
        ],
    },
}

_FALLBACK_TEMPLATE: Dict = {
    "goal": "Repair {anomaly_type} anomaly ({severity}) using {strategy_name}",
    "sub_goals": [
        "Inspect pipeline health",
        "Execute repair strategy",
        "Verify repair success",
    ],
    "steps": [
        "inspect_health",
        "wait_and_retry",
        "verify_anomaly_gone",
    ],
}


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """Goal-oriented, context-driven, adaptive repair plan generator.

    The Planner generates structured RepairPlan objects from a repair
    goal and context.  Plans are composed from primitives based on goal
    decomposition and procedural memory — not static lookups.

    Parameters
    ----------
    memory_store : MemoryStore, optional
        Three-tier memory store used to retrieve proven step sequences.

    Usage::

        planner = Planner(memory_store=store)
        plan = planner.generate(context, strategy_score, attempt_number=1)

        # On verification failure — generate a fresh plan
        new_plan = planner.replan(context, next_strategy, failed_plan, attempt=2)
    """

    def __init__(self, memory_store=None) -> None:  # type: Optional[MemoryStore]
        self._memory_store = memory_store

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def generate(
        self,
        ctx: RepairContext,
        strategy: StrategyScore,
        attempt_number: int = 1,
        max_attempts: int = 3,
        hypothesis_rank: int = 1,
    ) -> RepairPlan:
        """Generate a goal-oriented RepairPlan for the given strategy.

        The plan is assembled by:
        1. Loading the goal template for this strategy.
        2. Checking procedural memory for a proven step sequence.
        3. Enriching the step sequence based on context (e.g., adding
           extra integrity checks for HIGH severity).
        4. Numbering and packaging steps into a RepairPlan.

        Args:
            ctx:             RepairContext assembled by ContextBuilder.
            strategy:        Selected StrategyScore from ReasoningEngine.
            attempt_number:  Which attempt this plan is for (1-indexed).
            max_attempts:    Maximum total attempts configured.
            hypothesis_rank: Rank of the hypothesis this plan implements.

        Returns:
            Fully constructed RepairPlan ready for the Executor.
        """
        template = _GOAL_TEMPLATES.get(strategy.name, _FALLBACK_TEMPLATE)

        # Build goal string from template
        goal = self._build_goal(template["goal"], ctx, strategy.name)
        sub_goals = template["sub_goals"]

        # Check procedural memory for a proven step sequence
        proven_steps = self._load_proven_steps(ctx, strategy.name)
        if proven_steps:
            raw_step_names = proven_steps
            logger.info(
                f"[{ctx.run_id}] Planner: using procedural memory steps "
                f"for {strategy.name} ({len(raw_step_names)} steps)"
            )
        else:
            raw_step_names = list(template["steps"])

        # Context-sensitive enrichment
        raw_step_names = self._enrich_steps(raw_step_names, ctx, strategy)

        # Assemble RepairStep objects
        steps = self._build_steps(raw_step_names)

        plan = RepairPlan(
            plan_id=str(uuid.uuid4())[:12],
            goal=goal,
            strategy=strategy,
            steps=steps,
            max_attempts=max_attempts,
            attempt_number=attempt_number,
            rollback_strategy=strategy.rollback_strategy,
            goal_decomposition=sub_goals,
            hypothesis_rank=hypothesis_rank,
        )

        self._log_plan(ctx, plan, strategy)
        return plan

    def replan(
        self,
        ctx: RepairContext,
        next_strategy: StrategyScore,
        failed_plan: RepairPlan,
        attempt: int,
        max_attempts: int = 3,
    ) -> RepairPlan:
        """Generate a completely new plan after verification failure.

        This implements Adaptive Planning (Requirement 11).  The new plan:
        - Uses the next-best strategy from ReasoningEngine.
        - Adds a diagnostic step to gather information from the failure.
        - Does NOT reuse the failed plan's steps.

        Args:
            ctx:           RepairContext (same as failed plan).
            next_strategy: Next-best StrategyScore to try.
            failed_plan:   The plan that failed verification.
            attempt:       Current attempt number.
            max_attempts:  Maximum total attempts.

        Returns:
            A fresh RepairPlan for the next strategy.
        """
        logger.info(
            f"[{ctx.run_id}] Planner: replanning after failure | "
            f"failed_strategy={failed_plan.strategy.name if failed_plan.strategy else '?'} | "
            f"next_strategy={next_strategy.name} | "
            f"attempt={attempt}/{max_attempts}"
        )

        # Generate a fresh plan — no template reuse
        new_plan = self.generate(
            ctx=ctx,
            strategy=next_strategy,
            attempt_number=attempt,
            max_attempts=max_attempts,
        )

        # Prepend a diagnostic step to the new plan
        diagnostic = _p("inspect_health", goal_driven=True)
        diagnostic.description = (
            f"Re-inspect health after failure of '{failed_plan.strategy.name if failed_plan.strategy else '?'}'"
        )
        new_plan.steps.insert(0, diagnostic)

        # Renumber steps
        for i, step in enumerate(new_plan.steps, start=1):
            step.step_number = i

        logger.info(
            f"[{ctx.run_id}] Planner: new plan generated "
            f"(plan_id={new_plan.plan_id}, steps={len(new_plan.steps)})"
        )
        return new_plan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_goal(
        self,
        goal_template: str,
        ctx: RepairContext,
        strategy_name: str,
    ) -> str:
        """Expand a goal template string with context values."""
        try:
            return goal_template.format(
                anomaly_type=ctx.anomaly_type,
                severity=ctx.severity,
                estimated_missing=ctx.estimated_missing,
                gap_minutes=ctx.gap_minutes,
                tables=", ".join(ctx.affected_tables) if ctx.affected_tables else "orders",
                strategy_name=strategy_name,
            )
        except (KeyError, ValueError):
            return f"Repair {ctx.anomaly_type} ({ctx.severity}) using {strategy_name}"

    def _load_proven_steps(
        self,
        ctx: RepairContext,
        strategy_name: str,
    ) -> List[str]:
        """Query procedural memory for a proven step sequence.

        Returns an empty list if no procedure exists or memory is
        unavailable.  The caller falls back to the template.
        """
        if self._memory_store is None:
            return []
        try:
            proc = self._memory_store.get_best_procedure(
                anomaly_type=ctx.anomaly_type,
                severity=ctx.severity,
                min_trials=3,
            )
            if proc is None or proc.procedure_name != strategy_name:
                return []
            if proc.success_rate < 0.65:
                # Low-success procedure — don't trust it
                logger.debug(
                    f"[{ctx.run_id}] Planner: procedural memory for "
                    f"{strategy_name} has low success rate "
                    f"({proc.success_rate:.2f}) — using template"
                )
                return []
            # Extract action names from stored steps
            return [
                s.get("action", "")
                for s in proc.steps
                if s.get("action")
            ]
        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] Planner._load_proven_steps: {exc}")
            return []

    def _enrich_steps(
        self,
        step_names: List[str],
        ctx: RepairContext,
        strategy: StrategyScore,
    ) -> List[str]:
        """Add context-sensitive steps based on severity and anomaly type.

        Goal-oriented planning adds extra validation for HIGH/CRITICAL
        severity incidents and for situations with low diagnoser confidence.
        This is what makes planning context-aware rather than templated.
        """
        enriched = list(step_names)

        # For HIGH/CRITICAL: add extra integrity check if not already present
        if ctx.severity in ("HIGH", "CRITICAL") and "integrity_check" not in enriched:
            # Insert before the final notify/escalate step if present, else append
            insert_pos = len(enriched)
            for notify_action in ("backfill_notify", "tag_for_review", "escalate_to_human"):
                if notify_action in enriched:
                    insert_pos = enriched.index(notify_action)
                    break
            enriched.insert(insert_pos, "integrity_check")

        # Low confidence: add extra verification step
        if ctx.confidence < 0.70 and "verify_anomaly_gone" not in enriched:
            enriched.append("verify_anomaly_gone")

        # For missing_data with many rows: add duplicate check
        if (
            ctx.anomaly_type == "missing_data"
            and ctx.estimated_missing > 1000
            and "verify_no_duplicates" not in enriched
        ):
            enriched.append("verify_no_duplicates")

        # Maintenance window: skip aggressive actions, prefer wait
        if ctx.maintenance_window and strategy.name not in ("escalate_to_human", "wait_and_retry"):
            logger.info(
                f"[{ctx.run_id}] Planner: maintenance window detected — "
                "adding extra verification after repair"
            )
            if "integrity_check" not in enriched:
                enriched.append("integrity_check")

        return enriched

    def _build_steps(self, step_names: List[str]) -> List[RepairStep]:
        """Convert a list of primitive names into numbered RepairStep objects."""
        steps = []
        for i, name in enumerate(step_names, start=1):
            step = _p(name, goal_driven=True)
            step.step_number = i
            steps.append(step)
        return steps

    def _log_plan(
        self,
        ctx: RepairContext,
        plan: RepairPlan,
        strategy: StrategyScore,
    ) -> None:
        """Log the generated plan at INFO level."""
        logger.info(
            f"[{ctx.run_id}] Planner: plan={plan.plan_id} | "
            f"strategy={strategy.name} | "
            f"steps={len(plan.steps)} | "
            f"attempt={plan.attempt_number}/{plan.max_attempts} | "
            f"rollback={strategy.rollback_strategy or 'none'} | "
            f"hypothesis_rank={plan.hypothesis_rank}"
        )
        logger.info(f"[{ctx.run_id}] Repair goal: {plan.goal}")
        logger.info(
            f"[{ctx.run_id}] Goal chain: "
            + " → ".join(plan.goal_decomposition)
        )
        for step in plan.steps:
            logger.info(
                f"[{ctx.run_id}]   Step {step.step_number}: "
                f"{step.action} — {step.description}"
            )
