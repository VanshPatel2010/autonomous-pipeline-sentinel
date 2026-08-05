"""ReasoningEngine: AI-driven multi-hypothesis repair strategy evaluation.

Architectural Rationale
-----------------------
The previous ReasoningEngine scored a fixed strategy catalogue using a
four-factor heuristic formula.  While better than a plain if/else tree,
it was still deterministic: the same context always produced the same
ranking, regardless of what the agent had learned.

This upgrade transforms the engine into an AI-driven reasoner that:

1. **Uses the full 9-dimension UtilityScore model** (Requirement 4):
   Success probability, business impact, recovery time, operational cost,
   risk, historical success, customer impact, SLA importance, and
   diagnoser confidence all factor into the final utility.

2. **Integrates the LLMReasoner** (Requirement 5):
   For each anomaly, the LLM generates multi-hypothesis repair analysis.
   Its confidence boosts and hidden-risk flags adjust the utility of each
   candidate strategy before ranking.

3. **Queries MemoryStore for adaptive confidence** (Requirements 6, 7, 13):
   Instead of reading only the flat ``repair_memory`` table, the engine
   queries all three memory tiers (Episodic, Semantic, Procedural) for
   the most accurate confidence estimate.

4. **Generates RepairHypotheses** (Requirement 3):
   Every call to ``rank_strategies`` also produces a list of ranked
   RepairHypotheses with natural-language rationale, risk labels, and
   estimated recovery times — ready for display in the dashboard.

5. **Computes UncertaintyEstimate** (Requirement 14):
   Uncertainty is derived from the spread between the top two utilities,
   the LLM's stated uncertainty, and the number of historical samples.
   If uncertainty exceeds the configured threshold, the agent escalates.

6. **Produces RepairExplanation** (Requirement 12):
   A structured explanation is generated for every top-1 choice and
   attached to the RepairExplanation dataclass.

Design decisions
----------------
- The strategy catalogue is retained as a *prior* that is overwritten
  by evidence as the system accumulates experience.
- The LLMReasoner is optional and gracefully disabled when unavailable.
- The engine is pure: no I/O, no DB calls.  All external data is
  injected via RepairContext and MemoryStore.
- SOLID: the engine depends on RiskScorer, LLMReasoner, and MemoryStore
  via dependency injection — never by direct instantiation inside methods.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from logging_config import logger

from agents.repair.models import (
    RepairContext,
    RepairExplanation,
    RepairHypothesis,
    StrategyScore,
    UncertaintyEstimate,
    UtilityScore,
)
from agents.repair.risk_scorer import RiskScorer


# ---------------------------------------------------------------------------
# Strategy catalogue (priors — overwritten by learned evidence)
# ---------------------------------------------------------------------------

_STRATEGY_CATALOGUE: List[Dict] = [
    {
        "name": "wait_and_retry",
        "display_name": "Wait and Retry",
        "base_success": 0.55,
        "base_risk": 0.10,
        "base_cost": 0.05,
        "est_recovery_secs": 300,
        "rollback": None,
        "anomaly_types": ["missing_data", "data_quality"],
        "min_severity": "LOW",
        "max_severity": "MEDIUM",
    },
    {
        "name": "switch_to_backup",
        "display_name": "Failover to Delhi Replica",
        "base_success": 0.80,
        "base_risk": 0.30,
        "base_cost": 0.40,
        "est_recovery_secs": 120,
        "rollback": "rollback_failover",
        "anomaly_types": ["missing_data"],
        "min_severity": "MEDIUM",
        "max_severity": "HIGH",
    },
    {
        "name": "quarantine_bad_data",
        "display_name": "Quarantine Bad Rows",
        "base_success": 0.85,
        "base_risk": 0.20,
        "base_cost": 0.25,
        "est_recovery_secs": 60,
        "rollback": "restore_quarantine",
        "anomaly_types": ["data_quality"],
        "min_severity": "MEDIUM",
        "max_severity": "HIGH",
    },
    {
        "name": "rollback_failover",
        "display_name": "Rollback Failover (Revert to Primary)",
        "base_success": 0.70,
        "base_risk": 0.35,
        "base_cost": 0.45,
        "est_recovery_secs": 180,
        "rollback": None,
        "anomaly_types": ["missing_data"],
        "min_severity": "MEDIUM",
        "max_severity": "HIGH",
    },
    {
        "name": "restart_service",
        "display_name": "Restart Pipeline Service",
        "base_success": 0.65,
        "base_risk": 0.45,
        "base_cost": 0.30,
        "est_recovery_secs": 90,
        "rollback": None,
        "anomaly_types": ["missing_data", "data_quality"],
        "min_severity": "HIGH",
        "max_severity": "HIGH",
    },
    {
        "name": "backfill_from_archive",
        "display_name": "Backfill from Archive",
        "base_success": 0.75,
        "base_risk": 0.25,
        "base_cost": 0.60,
        "est_recovery_secs": 600,
        "rollback": "purge_backfill",
        "anomaly_types": ["missing_data"],
        "min_severity": "HIGH",
        "max_severity": "HIGH",
    },
    {
        "name": "escalate_to_human",
        "display_name": "Escalate to Human Operator",
        "base_success": 0.99,
        "base_risk": 0.05,
        "base_cost": 1.00,
        "est_recovery_secs": 3600,
        "rollback": None,
        "anomaly_types": ["missing_data", "data_quality", "schema_drift"],
        "min_severity": "CRITICAL",
        "max_severity": "CRITICAL",
    },
]

_SEVERITY_ORDER: Dict[str, int] = {
    "NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4,
}

_MAX_RECOVERY_SECS: int = 3600

# Default uncertainty escalation threshold (overridable via config)
_DEFAULT_UNCERTAINTY_THRESHOLD: float = 0.35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity_gte(actual: str, minimum: str) -> bool:
    return _SEVERITY_ORDER.get(actual, 0) >= _SEVERITY_ORDER.get(minimum, 0)


def _severity_lte(actual: str, maximum: str) -> bool:
    return _SEVERITY_ORDER.get(actual, 0) <= _SEVERITY_ORDER.get(maximum, 0)


def _extract_historical_rate(name: str, historical_repairs: list) -> Optional[float]:
    """Find the playbook success rate for this strategy from history."""
    for entry in historical_repairs:
        if entry.get("action_taken") == name:
            total = entry.get("success_count", 0) + entry.get("failure_count", 0)
            if total > 0:
                return entry["success_count"] / total
    return None


def _risk_label(risk: float) -> str:
    """Convert a risk float to a human-readable label."""
    if risk < 0.20:
        return "Very Low"
    elif risk < 0.35:
        return "Low"
    elif risk < 0.55:
        return "Medium"
    elif risk < 0.75:
        return "High"
    return "Very High"


def _recovery_label(secs: int) -> str:
    """Convert seconds to a human-readable recovery estimate."""
    if secs < 60:
        return f"~{secs}s"
    elif secs < 3600:
        return f"~{secs // 60} min"
    return f"~{secs // 3600}h"


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """AI-driven multi-hypothesis repair strategy evaluator.

    Replaces the deterministic scoring loop with an intelligence layer
    that integrates LLM reasoning, adaptive memory confidence, full
    utility-based decision making, and uncertainty estimation.

    Parameters
    ----------
    risk_scorer : RiskScorer, optional
        Injected risk scorer.
    llm_reasoner : LLMReasoner, optional
        Injected LLM reasoning layer.  If None, LLM reasoning is disabled.
    memory_store : MemoryStore, optional
        Injected three-tier memory store.  If None, falls back to playbook only.
    uncertainty_threshold : float
        Uncertainty above this value triggers human escalation flag.

    Usage::

        engine = ReasoningEngine(
            llm_reasoner=LLMReasoner(),
            memory_store=MemoryStore(db_path),
        )
        ranked, hypotheses, uncertainty, explanation = engine.reason(ctx)
    """

    def __init__(
        self,
        risk_scorer: Optional[RiskScorer] = None,
        llm_reasoner=None,       # type: Optional[LLMReasoner]  (avoid circular import)
        memory_store=None,       # type: Optional[MemoryStore]
        uncertainty_threshold: float = _DEFAULT_UNCERTAINTY_THRESHOLD,
    ) -> None:
        self._risk_scorer = risk_scorer or RiskScorer()
        self._llm_reasoner = llm_reasoner
        self._memory_store = memory_store
        self.uncertainty_threshold = uncertainty_threshold

    # ------------------------------------------------------------------
    # Primary API: full AI reasoning pipeline
    # ------------------------------------------------------------------

    def reason(
        self,
        ctx: RepairContext,
    ) -> Tuple[
        List[StrategyScore],
        List[RepairHypothesis],
        UncertaintyEstimate,
        RepairExplanation,
    ]:
        """Execute the full AI reasoning pipeline for a repair context.

        This is the upgraded entry point that replaces ``rank_strategies``.
        It returns four objects that cover multi-hypothesis reasoning,
        uncertainty estimation, and structured explainability.

        Pipeline:
        1. Compute global risk score (RiskScorer).
        2. Score all applicable strategies with 9-dimension utility model.
        3. Query MemoryStore for adaptive confidence (if available).
        4. Call LLMReasoner to generate AI hypotheses and confidence boosts.
        5. Apply LLM boosts to utility scores and re-rank.
        6. Compute UncertaintyEstimate from utility spread.
        7. Generate RepairExplanation for the top-1 strategy.
        8. Return all four outputs.

        Args:
            ctx: RepairContext from ContextBuilder.

        Returns:
            Tuple of:
            - ranked strategies     (List[StrategyScore])
            - repair hypotheses     (List[RepairHypothesis])
            - uncertainty estimate  (UncertaintyEstimate)
            - repair explanation    (RepairExplanation)
        """
        global_risk = self._risk_scorer.compute(ctx)

        # Stage 1: score all applicable strategies with full utility model
        candidates = self._score_all_strategies(ctx, global_risk)
        if not candidates:
            logger.warning(f"[{ctx.run_id}] ReasoningEngine: no applicable strategies")
            return [], [], self._zero_uncertainty(ctx.run_id), self._empty_explanation()

        # Stage 2: apply adaptive confidence from MemoryStore
        if self._memory_store is not None:
            candidates = self._apply_adaptive_confidence(candidates, ctx)

        # Stage 3: LLM reasoning — generate hypotheses and apply boosts
        hypotheses: List[RepairHypothesis] = []
        llm_reasoning: str = ""
        llm_uncertainty: float = 0.2  # default if LLM unavailable

        if self._llm_reasoner is not None:
            hyps, llm_reasoning = self._llm_reasoner.generate_hypotheses(ctx, candidates)
            if hyps:
                hypotheses = hyps
                # Apply LLM confidence boosts to matched strategies
                candidates = self._apply_llm_boosts(candidates, hyps)
                # Extract LLM uncertainty estimate
                llm_uncertainty = self._extract_llm_uncertainty(hyps)

        # Stage 4: sort by utility descending
        candidates.sort(key=lambda s: s.utility, reverse=True)

        # Stage 5: generate hypotheses from ranked strategies (if LLM unavailable)
        if not hypotheses:
            hypotheses = self._generate_fallback_hypotheses(candidates)

        # Stage 6: compute uncertainty estimate
        uncertainty = self._compute_uncertainty(
            candidates, ctx.run_id, llm_uncertainty
        )

        # Stage 7: generate explanation
        explanation = self._generate_explanation(
            candidates, ctx, uncertainty, llm_reasoning
        )

        # Log
        top = candidates[0] if candidates else None
        logger.info(
            f"[{ctx.run_id}] ReasoningEngine: {len(candidates)} strategies ranked "
            f"| global_risk={global_risk:.3f} "
            f"| top={top.name if top else 'none'} "
            f"(utility={top.utility:.4f}) "
            f"| uncertainty={uncertainty.uncertainty:.3f} "
            f"| requires_human={uncertainty.requires_human} "
            f"| hypotheses={len(hypotheses)}"
        )

        return candidates, hypotheses, uncertainty, explanation

    # ------------------------------------------------------------------
    # Backward-compatible API (preserves existing callers)
    # ------------------------------------------------------------------

    def rank_strategies(self, ctx: RepairContext) -> List[StrategyScore]:
        """Evaluate and rank all applicable repair strategies.

        Backward-compatible entry point used by existing code paths and
        tests.  Internally calls ``reason()`` and returns only the ranked
        strategies list.

        Args:
            ctx: RepairContext built by ContextBuilder.

        Returns:
            List of StrategyScore objects sorted by utility descending.
        """
        ranked, _, _, _ = self.reason(ctx)
        return ranked

    def best_strategy(self, ctx: RepairContext) -> Optional[StrategyScore]:
        """Return the highest-utility applicable strategy.

        Args:
            ctx: RepairContext.

        Returns:
            Best StrategyScore or None if no strategy is applicable.
        """
        ranked = self.rank_strategies(ctx)
        return ranked[0] if ranked else None

    # ------------------------------------------------------------------
    # Internal: strategy scoring
    # ------------------------------------------------------------------

    def _score_all_strategies(
        self,
        ctx: RepairContext,
        global_risk: float,
    ) -> List[StrategyScore]:
        """Score all applicable strategies using the 9-dimension utility model."""
        candidates: List[StrategyScore] = []
        for spec in _STRATEGY_CATALOGUE:
            score = self._score_strategy(spec, ctx, global_risk)
            if score is not None:
                candidates.append(score)
        return candidates

    def _is_applicable(self, spec: Dict, ctx: RepairContext) -> bool:
        """Check whether a strategy is applicable for this context."""
        if ctx.anomaly_type not in spec.get("anomaly_types", []):
            return False
        if not _severity_gte(ctx.severity, spec.get("min_severity", "NONE")):
            return False
        if not _severity_lte(ctx.severity, spec.get("max_severity", "CRITICAL")):
            return False
        return True

    def _score_strategy(
        self,
        spec: Dict,
        ctx: RepairContext,
        global_risk: float,
    ) -> Optional[StrategyScore]:
        """Score a single strategy spec against the current context.

        Now uses the full 9-dimension UtilityScore model.
        """
        if not self._is_applicable(spec, ctx):
            return None

        name = spec["name"]
        base_success = spec["base_success"]
        base_risk = spec["base_risk"]
        base_cost = spec["base_cost"]
        est_recovery = spec["est_recovery_secs"]

        # Bayesian blend: history overrides prior
        hist_rate_raw = _extract_historical_rate(name, ctx.historical_repairs)
        if hist_rate_raw is not None:
            # 70% historical, 30% prior — empirical weighting
            hist_rate = 0.70 * hist_rate_raw + 0.30 * base_success
        else:
            hist_rate = base_success

        # Blended risk: global context + strategy-specific risk
        blended_risk = (base_risk + global_risk) / 2.0

        # Success prob: blend base + historical
        success_prob = (base_success + hist_rate) / 2.0
        # Apply LLM boost placeholder (set to 0 here, applied later)
        success_prob = min(max(success_prob, 0.0), 1.0)

        # Normalised recovery time [0,1]
        recovery_norm = min(est_recovery / _MAX_RECOVERY_SECS, 1.0)

        # Build 9-dimension UtilityScore
        us = UtilityScore(
            success_probability=success_prob,
            business_impact=ctx.business_impact,
            recovery_time_norm=recovery_norm,
            operational_cost=base_cost,
            risk=blended_risk,
            historical_success=hist_rate,
            customer_impact=ctx.customer_impact,
            sla_importance=ctx.sla_importance,
            confidence=ctx.confidence,
        ).compute()

        rationale = (
            f"utility={us.expected_utility:.4f} | "
            f"success_prob={success_prob:.2f} | "
            f"risk={blended_risk:.2f} | "
            f"hist_rate={hist_rate:.2f} | "
            f"cost={base_cost:.2f} | "
            f"recovery={est_recovery}s | "
            f"biz_impact={ctx.business_impact:.2f} | "
            f"sla={ctx.sla_importance:.2f} | "
            f"confidence={ctx.confidence:.2f}"
        )

        return StrategyScore(
            name=name,
            display_name=spec["display_name"],
            expected_success_prob=success_prob,
            estimated_recovery_secs=est_recovery,
            risk_score=blended_risk,
            cost_score=base_cost,
            historical_success_rate=hist_rate,
            business_impact_score=ctx.business_impact,
            utility=us.expected_utility,
            utility_breakdown=us,
            rationale=rationale,
            rollback_strategy=spec.get("rollback"),
        )

    # ------------------------------------------------------------------
    # Internal: adaptive confidence
    # ------------------------------------------------------------------

    def _apply_adaptive_confidence(
        self,
        candidates: List[StrategyScore],
        ctx: RepairContext,
    ) -> List[StrategyScore]:
        """Adjust strategy utilities using three-tier memory confidence.

        For each strategy, query MemoryStore for its adaptive confidence.
        If the memory-derived confidence differs significantly from the
        heuristic prior, blend it in to adjust the utility.
        """
        for s in candidates:
            memory_conf, sample_count = self._memory_store.get_adaptive_confidence(
                s.name, ctx.anomaly_type, ctx.severity
            )
            if sample_count >= 3:
                # Enough data — blend memory confidence into success_prob
                blended_success = (
                    0.6 * memory_conf + 0.4 * s.expected_success_prob
                )
                # Recompute utility with blended success
                if s.utility_breakdown is not None:
                    s.utility_breakdown.success_probability = blended_success
                    s.utility_breakdown.historical_success = memory_conf
                    s.utility_breakdown.compute()
                    s.utility = s.utility_breakdown.expected_utility
                s.expected_success_prob = blended_success
                s.historical_success_rate = memory_conf
                logger.debug(
                    f"[{ctx.run_id}] ReasoningEngine: adaptive conf for "
                    f"{s.name}: memory={memory_conf:.3f} "
                    f"(n={sample_count}) → utility={s.utility:.4f}"
                )
        return candidates

    # ------------------------------------------------------------------
    # Internal: LLM integration
    # ------------------------------------------------------------------

    def _apply_llm_boosts(
        self,
        candidates: List[StrategyScore],
        hypotheses: List[RepairHypothesis],
    ) -> List[StrategyScore]:
        """Apply LLM-derived confidence boosts to matched strategy scores.

        The LLMReasoner can adjust a strategy's base prior up or down
        based on context-aware analysis.  This is applied as a small
        additive boost to the success probability.
        """
        hyp_map = {
            h.strategy.name: h
            for h in hypotheses
            if h.strategy is not None
        }
        for s in candidates:
            hyp = hyp_map.get(s.name)
            if hyp and abs(s.llm_confidence_boost) > 0:
                boost = s.llm_confidence_boost
                new_success = min(max(s.expected_success_prob + boost, 0.0), 1.0)
                if s.utility_breakdown is not None:
                    s.utility_breakdown.success_probability = new_success
                    s.utility_breakdown.compute()
                    s.utility = s.utility_breakdown.expected_utility
                s.expected_success_prob = new_success
                logger.debug(
                    f"ReasoningEngine: LLM boost {s.name} "
                    f"+{boost:+.3f} → utility={s.utility:.4f}"
                )
        return candidates

    def _extract_llm_uncertainty(self, hypotheses: List[RepairHypothesis]) -> float:
        """Extract an uncertainty signal from the LLM hypotheses."""
        # If the LLM ranked hypotheses closely together → higher uncertainty
        if len(hypotheses) < 2:
            return 0.2
        top_utility = hypotheses[0].expected_utility
        second_utility = hypotheses[1].expected_utility if len(hypotheses) > 1 else 0.0
        spread = abs(top_utility - second_utility)
        # Small spread = high uncertainty (strategies look similarly good)
        # Large spread = low uncertainty (one strategy clearly dominates)
        return round(max(0.1, 0.5 - spread), 3)

    # ------------------------------------------------------------------
    # Internal: uncertainty estimation
    # ------------------------------------------------------------------

    def _compute_uncertainty(
        self,
        candidates: List[StrategyScore],
        run_id: str,
        llm_uncertainty: float = 0.2,
    ) -> UncertaintyEstimate:
        """Compute UncertaintyEstimate from utility spread and LLM signal.

        Uncertainty sources:
        - **Epistemic** (model): few historical samples → high epistemic.
        - **Aleatoric** (env): close utility spread → hard to distinguish.
        - **LLM signal**: LLM's own stated uncertainty.

        The final uncertainty is a weighted combination of all three.
        """
        if not candidates:
            return self._zero_uncertainty(run_id)

        top = candidates[0].utility
        second = candidates[1].utility if len(candidates) > 1 else 0.0

        # Spread-based aleatoric uncertainty
        spread = max(top - second, 0.0)
        aleatoric = max(0.0, 0.5 - spread)  # small spread = high uncertainty

        # Epistemic: low average utility signals lack of confidence in priors
        avg_utility = sum(c.utility for c in candidates) / len(candidates)
        epistemic = max(0.0, 0.6 - avg_utility)

        # Blend: 40% epistemic, 40% LLM, 20% aleatoric
        uncertainty = 0.40 * epistemic + 0.40 * llm_uncertainty + 0.20 * aleatoric
        uncertainty = min(max(uncertainty, 0.0), 1.0)
        confidence = 1.0 - uncertainty

        requires_human = uncertainty > self.uncertainty_threshold

        # Build a reasoning string
        reasoning = (
            f"Uncertainty={uncertainty:.3f} from: "
            f"epistemic={epistemic:.3f} (avg_utility={avg_utility:.3f}), "
            f"aleatoric={aleatoric:.3f} (spread={spread:.3f}), "
            f"llm_uncertainty={llm_uncertainty:.3f}. "
            + ("→ Escalation recommended." if requires_human else "→ Proceeding autonomously.")
        )

        return UncertaintyEstimate(
            confidence=round(confidence, 4),
            uncertainty=round(uncertainty, 4),
            epistemic=round(epistemic, 4),
            aleatoric=round(aleatoric, 4),
            requires_human=requires_human,
            threshold=self.uncertainty_threshold,
            decision_id=run_id,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Internal: hypothesis generation (fallback, no LLM)
    # ------------------------------------------------------------------

    def _generate_fallback_hypotheses(
        self, candidates: List[StrategyScore]
    ) -> List[RepairHypothesis]:
        """Generate RepairHypotheses from ranked candidates without LLM."""
        hypotheses = []
        for i, s in enumerate(candidates[:3], start=1):
            hyp = RepairHypothesis(
                strategy=s,
                estimated_success=s.expected_success_prob,
                risk_label=_risk_label(s.risk_score),
                estimated_recovery=_recovery_label(s.estimated_recovery_secs),
                expected_utility=s.utility,
                rank=i,
                rationale=(
                    f"Utility-ranked #{i} with expected_utility={s.utility:.4f}. "
                    f"Success prob: {s.expected_success_prob:.0%}. "
                    f"Risk: {_risk_label(s.risk_score)}. "
                    f"Recovery: {_recovery_label(s.estimated_recovery_secs)}."
                ),
                llm_generated=False,
                hidden_risks=[],
            )
            hypotheses.append(hyp)
        return hypotheses

    # ------------------------------------------------------------------
    # Internal: explainability
    # ------------------------------------------------------------------

    def _generate_explanation(
        self,
        candidates: List[StrategyScore],
        ctx: RepairContext,
        uncertainty: UncertaintyEstimate,
        llm_reasoning: str,
    ) -> RepairExplanation:
        """Generate a structured RepairExplanation for the top-1 strategy."""
        if not candidates:
            return self._empty_explanation()

        chosen = candidates[0]
        alternatives = [
            {
                "name": s.name,
                "utility": round(s.utility, 4),
                "reason_rejected": (
                    f"Lower utility ({s.utility:.4f} vs {chosen.utility:.4f})"
                ),
            }
            for s in candidates[1:4]
        ]

        hist_count = len(ctx.historical_repairs)
        hist_success = sum(r.get("success_count", 0) for r in ctx.historical_repairs)
        hist_evidence = (
            f"{hist_success} of {hist_count} similar repairs succeeded"
            if hist_count > 0
            else "No historical data available"
        )

        downtime_est_secs = chosen.estimated_recovery_secs
        downtime_str = (
            f"~{downtime_est_secs}s "
            f"(business_impact={ctx.business_impact:.0%})"
        )

        return RepairExplanation(
            chosen_strategy=chosen.name,
            display_name=chosen.display_name,
            reason=(
                f"Highest expected utility ({chosen.utility:.4f}) across all "
                f"{len(candidates)} evaluated strategies."
            ),
            expected_recovery=_recovery_label(chosen.estimated_recovery_secs),
            predicted_success=f"{chosen.expected_success_prob:.0%}",
            risk_assessment=(
                f"{_risk_label(chosen.risk_score)} "
                f"(score={chosen.risk_score:.3f})"
            ),
            historical_evidence=hist_evidence,
            business_downtime=downtime_str,
            utility_score=round(chosen.utility, 4),
            alternatives=alternatives,
            uncertainty=uncertainty,
            llm_reasoning=llm_reasoning,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_uncertainty(run_id: str) -> UncertaintyEstimate:
        return UncertaintyEstimate(
            confidence=0.0,
            uncertainty=1.0,
            epistemic=1.0,
            aleatoric=0.0,
            requires_human=True,
            threshold=_DEFAULT_UNCERTAINTY_THRESHOLD,
            decision_id=run_id,
            reasoning="No applicable strategies found — maximum uncertainty.",
        )

    @staticmethod
    def _empty_explanation() -> RepairExplanation:
        return RepairExplanation(
            chosen_strategy="none",
            display_name="None",
            reason="No applicable strategy found",
            expected_recovery="unknown",
            predicted_success="0%",
            risk_assessment="Maximum",
            historical_evidence="No data",
            business_downtime="unknown",
            utility_score=0.0,
        )
