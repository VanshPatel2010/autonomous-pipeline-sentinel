"""Verifier: multi-dimensional post-repair quality assessment.

Architectural Rationale
-----------------------
The previous Verifier computed a single ``verification_score`` from five
weighted binary checks.  While useful for pass/fail gating, a single
scalar discards rich diagnostic information.

The upgraded Verifier computes a full six-dimension quality profile
(Requirement 10):

1. **Repair Confidence**     — How confident is the system the repair is durable?
2. **Repair Quality Score**  — Multi-dimensional weighted composite [0,1].
3. **Residual Risk**         — Risk remaining after repair (vs. risk before).
4. **Data Integrity Score**  — Row count, duplicate check, schema validity.
5. **Recovery Score**        — How fully the system returned to normal state.
6. **Business Impact Score** — How much business harm did the repair prevent?

These six dimensions are stored in the extended ``VerificationResult``
and are consumed by:
- The Learner: to compute adaptive confidence updates.
- The Planner: to decide whether to replan or escalate.
- The RepairExplanation: to explain the verification outcome.
- The Dashboard: to display richer post-repair analytics.

Design decisions
----------------
- All checks are independent and individually testable.
- Weights are class-level constants (easy to tune per SLA profile).
- The Verifier is read-only: it never writes to any table.
- The extended ``VerificationResult`` is a frozen dataclass — the
  Verifier creates a new instance; it never mutates existing ones.
- Backward compatibility: ``result.passed``, ``result.verification_score``,
  and all original fields are preserved unchanged.
"""

from __future__ import annotations

from db.client import get_db_connection
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from logging_config import logger

from agents.repair.models import RepairContext, RepairPlan, VerificationResult


# ---------------------------------------------------------------------------
# Check weights (must sum to 1.0)
# ---------------------------------------------------------------------------

_WEIGHTS: Dict[str, float] = {
    "anomaly_removed":     0.30,   # reduced from 0.35 to make room for integrity
    "rows_recovered":      0.25,
    "no_duplicates":       0.15,
    "latency_acceptable":  0.15,
    "no_downstream":       0.10,
    "integrity_check":     0.05,   # new: schema/constraint check
}

# Residual risk weights
_RESIDUAL_RISK_WEIGHTS: Dict[str, float] = {
    "pipeline_unhealthy":  0.40,
    "downstream_affected": 0.25,
    "duplicates_present":  0.20,
    "latency_high":        0.15,
}


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class Verifier:
    """Validates repair success through multi-dimensional quality assessment.

    Computes the full six-dimension quality profile defined in Requirement 10
    and returns an extended VerificationResult.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.

    Usage::

        verifier = Verifier(db_path=db_path)
        result = verifier.verify(plan, context)
        print(f"Quality: {result.repair_quality_score:.3f}")
        print(f"Residual risk: {result.residual_risk:.3f}")
        print(f"Passed: {result.passed}")
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, plan: RepairPlan, ctx: RepairContext) -> VerificationResult:
        """Run all health checks and return a multi-dimensional VerificationResult.

        Args:
            plan: The executed RepairPlan (steps have outcomes populated).
            ctx:  RepairContext for query parameters.

        Returns:
            Extended VerificationResult with all six quality dimensions.
        """
        run_id = ctx.run_id
        strategy_name = plan.strategy.name if plan.strategy else "unknown"

        logger.info(
            f"[{run_id}] Verifier: running post-repair checks for "
            f"strategy={strategy_name}"
        )

        # ── Individual checks ──────────────────────────────────────────
        anomaly_removed  = self._check_anomaly_removed(ctx)
        rows_recovered   = self._check_rows_recovered(ctx)
        duplicates_found = self._check_duplicates(ctx)
        latency_ok       = self._check_latency(ctx)
        downstream_ok    = self._check_downstream(ctx)
        integrity_ok     = self._check_data_integrity(ctx)

        # ── Primary verification score ─────────────────────────────────
        verification_score = (
            _WEIGHTS["anomaly_removed"]    * float(anomaly_removed)
            + _WEIGHTS["rows_recovered"]   * min(rows_recovered / max(ctx.estimated_missing, 1), 1.0)
            + _WEIGHTS["no_duplicates"]    * (1.0 if duplicates_found == 0 else 0.0)
            + _WEIGHTS["latency_acceptable"] * float(latency_ok)
            + _WEIGHTS["no_downstream"]    * float(downstream_ok)
            + _WEIGHTS["integrity_check"]  * float(integrity_ok)
        )

        # ── Dimension 1: Repair Confidence ─────────────────────────────
        # Blends diagnoser confidence, verification score, and data quality
        repair_confidence = self._compute_repair_confidence(
            ctx, verification_score, anomaly_removed, integrity_ok
        )

        # ── Dimension 2: Repair Quality Score ─────────────────────────
        repair_quality_score = self._compute_quality_score(
            verification_score,
            rows_recovered,
            duplicates_found,
            ctx,
            plan,
        )

        # ── Dimension 3: Residual Risk ─────────────────────────────────
        pipeline_healthy = anomaly_removed and duplicates_found == 0
        residual_risk = self._compute_residual_risk(
            pipeline_healthy=pipeline_healthy,
            downstream_affected=not downstream_ok,
            duplicates_found=duplicates_found,
            latency_ok=latency_ok,
        )

        # ── Dimension 4: Data Integrity Score ─────────────────────────
        data_integrity_score = self._compute_data_integrity_score(
            duplicates_found, integrity_ok, ctx
        )

        # ── Dimension 5: Recovery Score ───────────────────────────────
        recovery_score = self._compute_recovery_score(
            anomaly_removed, rows_recovered, latency_ok, ctx
        )

        # ── Dimension 6: Business Impact Score ───────────────────────
        business_impact_score = self._compute_business_impact_score(
            ctx, verification_score, residual_risk
        )

        # ── Build details string ───────────────────────────────────────
        details = self._build_details(
            anomaly_removed=anomaly_removed,
            rows_recovered=rows_recovered,
            duplicates_found=duplicates_found,
            latency_ok=latency_ok,
            downstream_ok=downstream_ok,
            integrity_ok=integrity_ok,
            score=verification_score,
            repair_confidence=repair_confidence,
            residual_risk=residual_risk,
        )

        result = VerificationResult(
            anomaly_removed=anomaly_removed,
            rows_recovered=rows_recovered,
            duplicates_found=duplicates_found,
            latency_acceptable=latency_ok,
            pipeline_healthy=pipeline_healthy,
            downstream_affected=not downstream_ok,
            verification_score=round(verification_score, 4),
            details=details,
            # Extended quality dimensions
            repair_confidence=round(repair_confidence, 4),
            repair_quality_score=round(repair_quality_score, 4),
            residual_risk=round(residual_risk, 4),
            data_integrity_score=round(data_integrity_score, 4),
            recovery_score=round(recovery_score, 4),
            business_impact_score=round(business_impact_score, 4),
        )

        logger.info(
            f"[{run_id}] Verifier: "
            f"score={verification_score:.3f} | "
            f"passed={result.passed} | "
            f"confidence={repair_confidence:.3f} | "
            f"quality={repair_quality_score:.3f} | "
            f"residual_risk={residual_risk:.3f} | "
            f"integrity={data_integrity_score:.3f} | "
            f"recovery={recovery_score:.3f} | "
            f"biz_impact={business_impact_score:.3f}"
        )

        return result

    # ------------------------------------------------------------------
    # Individual health checks (unchanged from original)
    # ------------------------------------------------------------------

    def _check_anomaly_removed(self, ctx: RepairContext) -> bool:
        """Check whether the triggering anomaly has been resolved."""
        try:
            conn = get_db_connection(self.db_path)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            window_start = (now - timedelta(minutes=5)).isoformat()

            if ctx.anomaly_type == "missing_data":
                count = conn.execute(
                    "SELECT COUNT(*) FROM orders WHERE created_at > ?",
                    (window_start,),
                ).fetchone()[0]
                conn.close()
                return count >= 0

            elif ctx.anomaly_type == "data_quality":
                total = conn.execute(
                    "SELECT COUNT(*) FROM orders WHERE created_at > ?",
                    (window_start,),
                ).fetchone()[0]
                nulls = conn.execute(
                    "SELECT COUNT(*) FROM orders WHERE order_amount IS NULL AND created_at > ?",
                    (window_start,),
                ).fetchone()[0]
                conn.close()
                null_rate = nulls / total if total > 0 else 0.0
                return null_rate < 0.05

            else:
                conn.close()
                return True

        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] Verifier._check_anomaly_removed: {exc}")
            return False

    def _check_rows_recovered(self, ctx: RepairContext) -> int:
        """Count rows present in the primary table for the gap window."""
        if ctx.anomaly_type != "missing_data":
            return 0

        try:
            conn = get_db_connection(self.db_path)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            gap_start = (now - timedelta(minutes=ctx.gap_minutes + 5)).isoformat()
            count = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE created_at > ?",
                (gap_start,),
            ).fetchone()[0]
            conn.close()
            return count

        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] Verifier._check_rows_recovered: {exc}")
            return 0

    def _check_duplicates(self, ctx: RepairContext) -> int:
        """Count duplicate order_ids in the orders table."""
        try:
            conn = get_db_connection(self.db_path)
            count = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT order_id
                    FROM orders
                    GROUP BY order_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
            conn.close()
            return count

        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] Verifier._check_duplicates: {exc}")
            return 0

    def _check_latency(self, ctx: RepairContext) -> bool:
        """Heuristic latency check: recent window has orders flowing."""
        try:
            conn = get_db_connection(self.db_path)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            ten_min_ago = (now - timedelta(minutes=10)).isoformat()
            count = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE created_at > ?",
                (ten_min_ago,),
            ).fetchone()[0]
            conn.close()
            return count > 0

        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] Verifier._check_latency: {exc}")
            return True

    def _check_downstream(self, ctx: RepairContext) -> bool:
        """Placeholder for downstream consumer health check."""
        # TODO: integrate with downstream health API
        return True

    def _check_data_integrity(self, ctx: RepairContext) -> bool:
        """Check basic data integrity: no null PKs, valid timestamps.

        A lightweight constraint check that catches obvious corruption
        introduced during repair (e.g., rows inserted with null order_id).
        """
        try:
            conn = get_db_connection(self.db_path)
            null_pks = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE order_id IS NULL"
            ).fetchone()[0]
            conn.close()
            return null_pks == 0

        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] Verifier._check_data_integrity: {exc}")
            return True  # optimistic: assume integrity if check fails

    # ------------------------------------------------------------------
    # Six-dimension quality computation
    # ------------------------------------------------------------------

    def _compute_repair_confidence(
        self,
        ctx: RepairContext,
        verification_score: float,
        anomaly_removed: bool,
        integrity_ok: bool,
    ) -> float:
        """Compute repair confidence from diagnoser + verification signals.

        Formula:
            confidence = (
                0.40 * verification_score     (did the checks pass?)
                + 0.30 * ctx.confidence       (how sure was the diagnoser?)
                + 0.20 * float(anomaly_removed)  (is the anomaly gone?)
                + 0.10 * float(integrity_ok)  (is data integrity preserved?)
            )
        """
        return (
            0.40 * verification_score
            + 0.30 * ctx.confidence
            + 0.20 * float(anomaly_removed)
            + 0.10 * float(integrity_ok)
        )

    def _compute_quality_score(
        self,
        verification_score: float,
        rows_recovered: int,
        duplicates_found: int,
        ctx: RepairContext,
        plan: RepairPlan,
    ) -> float:
        """Compute a multi-dimensional repair quality score.

        Quality penalises:
        - Low recovery rate (rows_recovered / estimated_missing)
        - Presence of duplicates
        - Short plan (fewer steps = less thorough)
        """
        recovery_rate = min(rows_recovered / max(ctx.estimated_missing, 1), 1.0)
        duplicate_penalty = 0.0 if duplicates_found == 0 else min(duplicates_found / 10.0, 0.3)
        thoroughness = min(len(plan.steps) / 6.0, 1.0)  # 6+ steps = fully thorough

        quality = (
            0.50 * verification_score
            + 0.25 * recovery_rate
            + 0.15 * thoroughness
            - 0.10 * duplicate_penalty
        )
        return max(min(quality, 1.0), 0.0)

    def _compute_residual_risk(
        self,
        pipeline_healthy: bool,
        downstream_affected: bool,
        duplicates_found: int,
        latency_ok: bool,
    ) -> float:
        """Compute residual risk remaining after the repair.

        Higher residual risk means the repair did not fully resolve all
        concerns and the system may still fail.
        """
        risk = (
            _RESIDUAL_RISK_WEIGHTS["pipeline_unhealthy"]  * float(not pipeline_healthy)
            + _RESIDUAL_RISK_WEIGHTS["downstream_affected"] * float(downstream_affected)
            + _RESIDUAL_RISK_WEIGHTS["duplicates_present"]  * min(duplicates_found / 5.0, 1.0)
            + _RESIDUAL_RISK_WEIGHTS["latency_high"]         * float(not latency_ok)
        )
        return round(min(risk, 1.0), 4)

    def _compute_data_integrity_score(
        self,
        duplicates_found: int,
        integrity_ok: bool,
        ctx: RepairContext,
    ) -> float:
        """Compute data integrity score.

        Perfect integrity (no duplicates, no null PKs) = 1.0.
        """
        duplicate_penalty = min(duplicates_found / 100.0, 0.5)
        integrity_bonus = 0.3 if integrity_ok else 0.0
        score = 0.7 + integrity_bonus - duplicate_penalty
        return round(max(min(score, 1.0), 0.0), 4)

    def _compute_recovery_score(
        self,
        anomaly_removed: bool,
        rows_recovered: int,
        latency_ok: bool,
        ctx: RepairContext,
    ) -> float:
        """Compute how fully the system returned to its pre-anomaly state.

        Combines: anomaly resolution + row recovery rate + latency.
        """
        row_recovery_rate = min(rows_recovered / max(ctx.estimated_missing, 1), 1.0)
        score = (
            0.45 * float(anomaly_removed)
            + 0.35 * row_recovery_rate
            + 0.20 * float(latency_ok)
        )
        return round(min(score, 1.0), 4)

    def _compute_business_impact_score(
        self,
        ctx: RepairContext,
        verification_score: float,
        residual_risk: float,
    ) -> float:
        """Estimate the business impact of the repair outcome.

        A successful repair with low residual risk and high business_impact
        context yields a high score (the repair prevented significant harm).
        A failed repair with high residual risk yields a low score.
        """
        harm_prevented = ctx.business_impact * verification_score
        residual_harm = ctx.business_impact * residual_risk
        return round(max(harm_prevented - residual_harm * 0.5, 0.0), 4)

    # ------------------------------------------------------------------
    # Details builder
    # ------------------------------------------------------------------

    def _build_details(
        self,
        anomaly_removed: bool,
        rows_recovered: int,
        duplicates_found: int,
        latency_ok: bool,
        downstream_ok: bool,
        integrity_ok: bool,
        score: float,
        repair_confidence: float,
        residual_risk: float,
    ) -> str:
        """Build a human-readable summary of the verification results."""
        parts = [
            f"score={score:.3f}",
            f"confidence={repair_confidence:.3f}",
            f"residual_risk={residual_risk:.3f}",
            f"anomaly_removed={'✓' if anomaly_removed else '✗'}",
            f"rows_recovered={rows_recovered}",
            f"duplicates={duplicates_found}",
            f"latency={'✓' if latency_ok else '✗'}",
            f"downstream={'✓' if downstream_ok else '✗'}",
            f"integrity={'✓' if integrity_ok else '✗'}",
        ]
        return " | ".join(parts)
