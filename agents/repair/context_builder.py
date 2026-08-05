"""ContextBuilder: assembles a RepairContext from diverse data sources.

The ContextBuilder is the *first* stage in the autonomous repair pipeline.
It aggregates information from:

  1. PipelineState          — current anomaly data (STM)
  2. DiagnoserAgent output  — root cause, confidence, estimated rows
  3. DatabaseManager        — active node identity and health score
  4. PlaybookStore           — historical repair outcomes for this anomaly type
  5. IncidentStore           — similar past incidents for contextual retrieval
  6. Business heuristics     — derived impact / SLA scores

Design decisions
----------------
- ContextBuilder has *no* side effects: it only reads, never writes.
- All external I/O is wrapped in try/except so a flaky DB call never
  kills the repair cycle.
- Business-impact, customer-impact, and SLA scores are currently
  heuristic functions of severity + gap_minutes.  They are designed
  as replaceable callables so a real ML model can be injected later
  via the ``impact_scorer`` dependency.
- Node health is normalised to [0.0, 1.0]: 1.0 = active primary,
  0.5 = standby replica, 0.0 = no nodes available.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from logging_config import logger
from memory.incident_store import get_similar_incidents
from memory.playbook_store import get_all_playbooks
from state import PipelineState

from agents.repair.models import RepairContext


# ---------------------------------------------------------------------------
# Default business-impact heuristics
# ---------------------------------------------------------------------------

_SEVERITY_IMPACT: Dict[str, float] = {
    "CRITICAL": 1.0,
    "HIGH":     0.75,
    "MEDIUM":   0.45,
    "LOW":      0.20,
    "NONE":     0.0,
}

_SEVERITY_CUSTOMER: Dict[str, float] = {
    "CRITICAL": 0.95,
    "HIGH":     0.70,
    "MEDIUM":   0.40,
    "LOW":      0.15,
    "NONE":     0.0,
}

_SEVERITY_SLA: Dict[str, float] = {
    "CRITICAL": 1.0,
    "HIGH":     0.85,
    "MEDIUM":   0.55,
    "LOW":      0.25,
    "NONE":     0.0,
}


def _default_business_impact(severity: str, gap_minutes: float) -> float:
    """Heuristic business-impact score normalised to [0.0, 1.0].

    Gap duration beyond the base severity contributes a small boost so
    that a 6-hour MEDIUM gap scores higher than a 15-min MEDIUM gap.
    """
    base = _SEVERITY_IMPACT.get(severity, 0.3)
    gap_boost = min(gap_minutes / 720.0, 0.25)   # cap at +0.25 for 12h gap
    return min(base + gap_boost, 1.0)


def _default_customer_impact(severity: str, null_rate: float) -> float:
    """Heuristic customer-impact score.

    Null rate amplifies the customer impact for data-quality issues.
    """
    base = _SEVERITY_CUSTOMER.get(severity, 0.3)
    quality_boost = min(null_rate * 2.0, 0.20)
    return min(base + quality_boost, 1.0)


def _default_sla_importance(severity: str) -> float:
    """Heuristic SLA importance — currently severity-only."""
    return _SEVERITY_SLA.get(severity, 0.4)


# ---------------------------------------------------------------------------
# Node-health helper
# ---------------------------------------------------------------------------

def _get_node_health() -> tuple[str, str, float]:
    """Return (active_node_id, active_node_label, health_score).

    Health score:
      1.0  — active primary node reachable
      0.7  — replica node active (degraded but operational)
      0.0  — no node available (should not happen in normal operation)
    """
    try:
        from db.database_manager import get_active_node, get_all_nodes_status
        node = get_active_node()
        all_nodes = get_all_nodes_status()
        active_count = sum(1 for n in all_nodes if n["status"] == "active")
        standby_count = sum(1 for n in all_nodes if n["status"] == "standby")

        # Score: primary active = 1.0, replica active = 0.7, degraded = 0.4
        if node["role"] == "primary":
            health = 1.0
        elif active_count > 0 and standby_count > 0:
            health = 0.7
        elif active_count > 0:
            health = 0.5
        else:
            health = 0.2

        return node["id"], node["label"], health

    except Exception as exc:
        logger.warning(f"ContextBuilder: could not query node health: {exc}")
        return "unknown", "UNKNOWN", 0.5


# ---------------------------------------------------------------------------
# System-load helper  (lightweight — avoids psutil dependency)
# ---------------------------------------------------------------------------

def _get_system_load() -> float:
    """Return a normalised system-load estimate in [0.0, 1.0].

    Uses ``os.getloadavg()`` (Unix) if available; falls back to 0.5.
    """
    try:
        import os
        load_1min = os.getloadavg()[0]
        # Normalise: assume 4 CPUs, load > 4 = 1.0
        import os as _os
        cpu_count = _os.cpu_count() or 4
        return min(load_1min / cpu_count, 1.0)
    except (AttributeError, OSError):
        return 0.5


# ---------------------------------------------------------------------------
# Historical-repairs helper
# ---------------------------------------------------------------------------

def _get_historical_repairs(
    anomaly_type: str,
    severity: str,
    db_path: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return recent playbook entries filtered to this anomaly+severity."""
    try:
        all_books = get_all_playbooks(limit=50, db_path=db_path)
        filtered = [
            p for p in all_books
            if p.get("anomaly_type") == anomaly_type and p.get("severity") == severity
        ]
        return filtered[:limit]
    except Exception as exc:
        logger.warning(f"ContextBuilder: could not load historical repairs: {exc}")
        return []


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """Assembles RepairContext from PipelineState and external sources.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    impact_scorer : callable, optional
        Injected function ``(severity, gap_minutes) -> float`` for
        overriding the default business-impact heuristic.
    """

    def __init__(
        self,
        db_path: str,
        impact_scorer: Optional[Callable[[str, float], float]] = None,
    ) -> None:
        self.db_path = db_path
        self._impact_scorer = impact_scorer or _default_business_impact

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, state: PipelineState) -> RepairContext:
        """Build and return a RepairContext from the given PipelineState.

        This method is the only public entry point.  It is deliberately
        free of side effects so callers can call it multiple times safely.

        Args:
            state: Current pipeline state (output from DiagnoserAgent).

        Returns:
            Fully populated RepairContext snapshot.
        """
        run_id = state["run_id"]
        anomaly_type = state.get("anomaly_type", "unknown")
        severity = state.get("severity", "NONE")
        gap_minutes = state.get("gap_minutes", 0.0)
        null_rate = state.get("null_rate", 0.0)

        diagnoser = state.get("diagnoser_output", {})
        confidence = diagnoser.get("confidence", 0.5)
        root_cause = diagnoser.get("root_cause", "Unknown")
        estimated_missing = int(diagnoser.get("estimated_missing_rows", 0))
        affected_tables = list(diagnoser.get("affected_tables", state.get("affected_tables", [])))
        maintenance_window = bool(diagnoser.get("maintenance_window_likely", False))

        # Node health
        node_id, node_label, node_health = _get_node_health()

        # Memory retrieval
        historical_repairs = _get_historical_repairs(
            anomaly_type, severity, self.db_path
        )
        similar_incidents = self._get_similar_incidents(anomaly_type)

        # Impact scores
        business_impact = self._impact_scorer(severity, gap_minutes)
        customer_impact = _default_customer_impact(severity, null_rate)
        sla_importance = _default_sla_importance(severity)

        # System context
        system_load = _get_system_load()
        hour_of_day = datetime.now(timezone.utc).hour

        # Pipeline metrics pass-through
        pipeline_metrics = {
            "raw_count": state.get("raw_count", 0),
            "expected_avg": state.get("expected_avg", 0.0),
            "null_rate": null_rate,
            "gap_minutes": gap_minutes,
        }

        ctx = RepairContext(
            run_id=run_id,
            anomaly_type=anomaly_type,
            severity=severity,
            confidence=confidence,
            root_cause=root_cause,
            gap_minutes=gap_minutes,
            estimated_missing=estimated_missing,
            null_rate=null_rate,
            affected_tables=affected_tables,
            active_node_id=node_id,
            active_node_label=node_label,
            node_health_score=node_health,
            historical_repairs=historical_repairs,
            similar_incidents=similar_incidents,
            business_impact=business_impact,
            customer_impact=customer_impact,
            sla_importance=sla_importance,
            current_system_load=system_load,
            hour_of_day=hour_of_day,
            maintenance_window=maintenance_window,
            pipeline_metrics=pipeline_metrics,
        )

        logger.info(
            f"[{run_id}] ContextBuilder: context assembled | "
            f"node={node_label} health={node_health:.2f} | "
            f"business_impact={business_impact:.2f} | "
            f"similar_incidents={len(similar_incidents)} | "
            f"historical_repairs={len(historical_repairs)}"
        )

        return ctx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_similar_incidents(self, anomaly_type: str) -> List[Dict[str, Any]]:
        """Retrieve semantically similar past incidents.

        Currently uses anomaly_type-based retrieval from IncidentStore.
        This method is designed for future embedding/vector DB support:
        simply swap the implementation here without touching callers.

        Args:
            anomaly_type: Anomaly type for similarity search.

        Returns:
            List of similar incident dicts, most recent first.
        """
        try:
            incidents = get_similar_incidents(
                anomaly_type,
                limit=10,
                db_path=self.db_path,
            )
            # Enrich: add a computed "similarity_score" field so callers
            # can filter — for now set to 1.0 (exact type match).
            for inc in incidents:
                inc["similarity_score"] = 1.0
            return incidents
        except Exception as exc:
            logger.warning(f"ContextBuilder: could not retrieve similar incidents: {exc}")
            return []
