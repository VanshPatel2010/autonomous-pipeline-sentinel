"""RiskScorer: computes a dynamic, multi-factor risk score.

Design
------
The risk score replaces the old LOW / MEDIUM / HIGH string-based
decision.  It produces a continuous float in [0.0, 1.0] where 1.0
represents the highest possible risk.

Formula
-------
    risk = (
        business_impact
        * row_pressure           (normalised estimated_missing)
        * customer_impact
        * confidence_weight      (high confidence → known risk)
        * node_health_penalty    (unhealthy node → higher risk)
        * historical_failure_rate
        * time_of_day_weight
        * sla_importance
    ) ** (1/8)                    ← geometric mean across 8 dimensions

Taking the geometric mean (8th root) keeps the result in [0.0, 1.0]
without any dimension dominating the others.

Each dimension is described as a standalone function so it can be
individually unit-tested and replaced.
"""

from __future__ import annotations

import math
from typing import List

from logging_config import logger

from agents.repair.models import RepairContext


# ---------------------------------------------------------------------------
# Individual dimension scorers
# ---------------------------------------------------------------------------

def _row_pressure(estimated_missing: int, scale: int = 10_000) -> float:
    """Normalise estimated missing rows to [0.0, 1.0].

    Args:
        estimated_missing: Rows estimated to be missing.
        scale:             Reference point — 10 000 rows = 1.0.

    Returns:
        Normalised pressure [0.0, 1.0].
    """
    return min(estimated_missing / scale, 1.0)


def _confidence_weight(confidence: float) -> float:
    """Map confidence to a risk-amplification weight.

    High confidence in a bad situation means we *know* the risk is real.
    Low confidence means we are uncertain — risk is discounted slightly.
    Weight range: [0.5, 1.0].
    """
    return 0.5 + confidence * 0.5


def _node_health_penalty(node_health_score: float) -> float:
    """Convert node health to a risk factor.

    Healthy node (1.0) → low risk (0.1).
    Dead node    (0.0) → high risk (1.0).
    """
    return 1.0 - node_health_score * 0.9


def _historical_failure_rate(historical_repairs: list) -> float:
    """Compute historical failure rate from playbook entries.

    Args:
        historical_repairs: List of playbook dicts with success_count /
                            failure_count.

    Returns:
        Failure rate [0.0, 1.0]; defaults to 0.5 if no history.
    """
    if not historical_repairs:
        return 0.5   # neutral / unknown

    total_success = sum(r.get("success_count", 0) for r in historical_repairs)
    total_failure = sum(r.get("failure_count", 0) for r in historical_repairs)
    total = total_success + total_failure

    if total == 0:
        return 0.5

    return total_failure / total


def _time_of_day_weight(hour_utc: int) -> float:
    """Weight risk by time-of-day.

    Business hours (08-18 UTC) → higher risk because impact is immediate.
    Off-hours (nights/weekends) → slightly lower weight.

    Returns:
        Weight in [0.6, 1.0].
    """
    if 8 <= hour_utc < 18:
        return 1.0    # peak hours — full risk weight
    elif 6 <= hour_utc < 8 or 18 <= hour_utc < 22:
        return 0.8    # shoulder hours
    else:
        return 0.6    # overnight


# ---------------------------------------------------------------------------
# Main RiskScorer class
# ---------------------------------------------------------------------------

class RiskScorer:
    """Computes a continuous multi-factor risk score for a repair context.

    The risk score is used by:
    - ReasoningEngine: to penalise high-risk strategies.
    - Planner: to decide whether human escalation is mandatory.
    - Predictor: to classify trend severity.

    Usage::

        scorer = RiskScorer()
        score = scorer.compute(context)
        # score in [0.0, 1.0]; higher = riskier
    """

    # Minimum risk score — even a trivially healthy situation has some risk.
    MIN_RISK: float = 0.05

    def compute(self, ctx: RepairContext) -> float:
        """Compute and return the risk score for this repair context.

        Args:
            ctx: RepairContext snapshot from the ContextBuilder.

        Returns:
            Risk score in [0.0, 1.0].
        """
        dims = [
            ctx.business_impact,
            _row_pressure(ctx.estimated_missing),
            ctx.customer_impact,
            _confidence_weight(ctx.confidence),
            _node_health_penalty(ctx.node_health_score),
            _historical_failure_rate(ctx.historical_repairs),
            _time_of_day_weight(ctx.hour_of_day),
            ctx.sla_importance,
        ]

        # Geometric mean (8th root of the product)
        product = 1.0
        for d in dims:
            product *= max(d, 1e-6)    # guard against exact zero

        risk = product ** (1.0 / len(dims))
        risk = max(risk, self.MIN_RISK)

        logger.debug(
            f"[{ctx.run_id}] RiskScorer dimensions: "
            f"business={ctx.business_impact:.2f}, "
            f"row_pressure={_row_pressure(ctx.estimated_missing):.2f}, "
            f"customer={ctx.customer_impact:.2f}, "
            f"confidence_weight={_confidence_weight(ctx.confidence):.2f}, "
            f"node_penalty={_node_health_penalty(ctx.node_health_score):.2f}, "
            f"hist_failure={_historical_failure_rate(ctx.historical_repairs):.2f}, "
            f"time_weight={_time_of_day_weight(ctx.hour_of_day):.2f}, "
            f"sla={ctx.sla_importance:.2f} → risk={risk:.4f}"
        )

        return risk
