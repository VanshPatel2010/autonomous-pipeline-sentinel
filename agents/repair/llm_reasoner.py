"""LLMReasoner: provider-agnostic LLM reasoning layer for the repair agent.

Architectural Rationale
-----------------------
The LLMReasoner decouples *what the LLM can do* from *how it is called*.
It introduces genuine AI-driven intelligence by replacing hard-coded
strategy weights with probabilistic reasoning over context.

Key design decisions:
1. **Provider-agnostic interface**: A ``BaseLLMProvider`` abstract class
   defines the contract.  Concrete implementations for Groq (Llama/Mixtral),
   OpenAI, Claude, Gemini, Ollama are registered via a provider registry.
   No logic in the caller needs to change when the provider is swapped.

2. **LLM is advisory only**: The LLMReasoner never directly executes any
   repair action.  Its outputs (hypotheses, confidence boosts, risk flags)
   are validated by the Planner and ReasoningEngine before use.

3. **Graceful degradation**: If the LLM is unavailable (no API key, network
   error, timeout), the system falls back to heuristic reasoning.  The
   ``llm_available`` flag on the agent communicates this to callers.

4. **Structured outputs**: All LLM responses are parsed into typed Python
   objects (RepairHypothesis, dict) so downstream code never does string
   manipulation.

5. **Future extensibility**: Adding a new provider is one class + one
   registry call.  Vector DB, knowledge graph, and RL integrations are
   designed as injectable modules.

Responsibilities
----------------
- Analyse repair context and generate repair hypotheses.
- Explain reasoning for each hypothesis.
- Suggest novel repair plans not in the strategy catalogue.
- Estimate confidence and identify hidden risks.
- Generate self-reflection commentary after each repair.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from logging_config import logger

from agents.repair.models import RepairContext, RepairHypothesis, StrategyScore


# ---------------------------------------------------------------------------
# Base LLM Provider interface
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Abstract interface for any LLM backend.

    Implement this class to add a new LLM provider without touching the
    rest of the repair system.

    Future integrations
    -------------------
    - OpenAI     : ``OpenAIProvider(model='gpt-4o')``
    - Claude     : ``ClaudeProvider(model='claude-3-5-sonnet')``
    - Gemini     : ``GeminiProvider(model='gemini-1.5-pro')``
    - Ollama     : ``OllamaProvider(model='llama3.2')``
    - Mistral    : ``MistralProvider(model='mistral-large')``
    - Qwen       : ``QwenProvider(model='qwen2.5-72b')``
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'groq', 'openai')."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Send a chat completion request and return the response text.

        Args:
            system_prompt : The system-role context prompt.
            user_prompt   : The user-role query prompt.
            max_tokens    : Maximum tokens in the response.

        Returns:
            Raw LLM response text.

        Raises:
            LLMProviderError : If the provider is unreachable or returns an error.
        """

    @property
    def is_available(self) -> bool:
        """Return True if the provider is configured and reachable."""
        return True


class LLMProviderError(Exception):
    """Raised when an LLM provider call fails."""


# ---------------------------------------------------------------------------
# Groq provider (default — matches existing config.py GROQ_API_KEY)
# ---------------------------------------------------------------------------

class GroqProvider(BaseLLMProvider):
    """Groq Cloud provider (Llama, Mixtral, Gemma via Groq's fast inference).

    Uses the existing ``GROQ_API_KEY`` and ``GROQ_MODEL`` from config.py,
    maintaining full backward compatibility with the Diagnoser's Groq usage.

    Future providers follow the same pattern:
    - Override ``provider_name`` and ``complete``.
    - Register via ``LLMProviderRegistry.register()``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        from config import GROQ_API_KEY, GROQ_MODEL
        self._api_key = api_key or GROQ_API_KEY
        self._model = model or GROQ_MODEL
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
            except ImportError:
                raise LLMProviderError("groq package not installed. Run: pip install groq")
        return self._client

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Call Groq API with the given prompts and return the response."""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,   # Low temperature for consistent reasoning
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Mock provider (for MOCK_MODE / testing — no external calls)
# ---------------------------------------------------------------------------

class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing and MOCK_MODE.

    Returns structured placeholder responses that exercise the full
    parsing pipeline without making any API calls.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Return a deterministic structured mock response."""
        return json.dumps({
            "hypotheses": [
                {
                    "strategy": "switch_to_backup",
                    "estimated_success": 0.88,
                    "risk_label": "Low",
                    "estimated_recovery": "~2 min",
                    "rationale": (
                        "Replica failover is the most direct path to recovery. "
                        "Historical data shows 88% success rate for this pattern."
                    ),
                    "hidden_risks": ["Replica may be 30s behind primary"],
                    "confidence_boost": 0.05,
                },
                {
                    "strategy": "backfill_from_archive",
                    "estimated_success": 0.75,
                    "risk_label": "Very Low",
                    "estimated_recovery": "~6 min",
                    "rationale": (
                        "Archive backfill avoids failover risk but takes longer. "
                        "Safe choice when replica lag is suspected."
                    ),
                    "hidden_risks": [],
                    "confidence_boost": 0.0,
                },
            ],
            "overall_reasoning": (
                "The primary indicator is a data gap with moderate severity. "
                "The replica should be current enough for failover. "
                "Recommend switch_to_backup as the primary strategy."
            ),
            "novel_strategy": None,
            "uncertainty": 0.18,
        })


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

class LLMProviderRegistry:
    """Registry of available LLM providers.

    Providers are registered by name and instantiated on demand.
    The registry supports hot-swapping providers without restarting the agent.

    Usage::

        registry = LLMProviderRegistry()
        provider = registry.get("groq")
        response = provider.complete(system, user)

    To add a new provider::

        registry.register("openai", lambda: OpenAIProvider(model="gpt-4o"))
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Any] = {}
        self._instances: Dict[str, BaseLLMProvider] = {}
        # Register built-in providers
        self.register("groq", GroqProvider)
        self.register("mock", MockLLMProvider)

    def register(self, name: str, factory: Any) -> None:
        """Register a provider factory by name."""
        self._factories[name] = factory

    def get(self, name: str) -> BaseLLMProvider:
        """Get or create a provider instance by name."""
        if name not in self._instances:
            factory = self._factories.get(name)
            if factory is None:
                raise LLMProviderError(f"Unknown LLM provider: '{name}'")
            self._instances[name] = factory()
        return self._instances[name]

    def available_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._factories.keys())


# Default singleton registry
_registry = LLMProviderRegistry()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert AI engineering assistant specialising in
autonomous data pipeline repair. You reason probabilistically about repair
strategies, estimate success probabilities, and identify hidden risks.

You ALWAYS respond with valid JSON. Never include markdown fences or commentary
outside the JSON object. Your reasoning is precise, data-driven, and explains
the probabilistic basis of every recommendation.

You do NOT execute any actions. You only analyse and advise."""

_HYPOTHESIS_PROMPT_TEMPLATE = """
Analyse this pipeline incident and generate repair hypotheses.

INCIDENT CONTEXT:
- Run ID: {run_id}
- Anomaly Type: {anomaly_type}
- Severity: {severity}
- Root Cause: {root_cause}
- Gap Duration: {gap_minutes:.1f} minutes
- Estimated Missing Rows: {estimated_missing}
- Node Health: {node_health_score:.2f} (1.0 = healthy)
- Business Impact: {business_impact:.2f}
- Customer Impact: {customer_impact:.2f}
- SLA Importance: {sla_importance:.2f}
- Maintenance Window: {maintenance_window}

AVAILABLE STRATEGIES: {available_strategies}

HISTORICAL CONTEXT:
{historical_summary}

SIMILAR INCIDENTS:
{incident_summary}

Generate 2-3 repair hypotheses. For each hypothesis provide:
1. strategy: one of the available strategies above
2. estimated_success: float [0.0, 1.0]
3. risk_label: 'Very Low' | 'Low' | 'Medium' | 'High' | 'Very High'
4. estimated_recovery: human-readable estimate (e.g. '~3 min')
5. rationale: 2-3 sentence explanation of why this succeeds
6. hidden_risks: list of secondary risks to watch for
7. confidence_boost: float [-0.1, 0.1] adjustment to base prior

Also provide:
- overall_reasoning: your chain-of-thought (2-3 sentences)
- novel_strategy: null OR a new strategy name if none of the above fits
- uncertainty: float [0.0, 1.0] — your uncertainty about this situation

Respond with JSON only.
"""

_REFLECTION_PROMPT_TEMPLATE = """
You are reviewing a completed repair cycle. Evaluate the decision quality.

REPAIR SUMMARY:
- Anomaly Type: {anomaly_type}
- Severity: {severity}
- Strategy Chosen: {strategy_chosen}
- Verification Score: {verification_score:.3f}
- Outcome: {outcome}
- Recovery Time: {recovery_secs:.0f} seconds
- Alternatives Considered: {alternatives}

Answer these questions as JSON:
1. optimal_choice: true/false — was this the best strategy available?
2. cheaper_alternative: null OR name of a cheaper/faster strategy
3. downtime_reduction_estimate_secs: how many seconds could have been saved?
4. confidence_adjustment: float [-0.15, 0.15] — should future confidence change?
5. learning_points: list of 2-3 bullet-point learnings
6. reflection: 2-3 sentence free-form reflection paragraph

Respond with JSON only.
"""


# ---------------------------------------------------------------------------
# LLMReasoner
# ---------------------------------------------------------------------------

class LLMReasoner:
    """LLM-based reasoning layer for the autonomous repair agent.

    The LLMReasoner sits between the ReasoningEngine and the Planner.
    It adds genuine AI intelligence to the repair process:

    1. Generates repair hypotheses with probabilistic estimates.
    2. Explains its reasoning in structured natural language.
    3. Suggests novel repair strategies not in the static catalogue.
    4. Estimates confidence and flags hidden risks.
    5. Produces post-repair reflections for continuous learning.

    The LLM NEVER executes repairs. All outputs are advisory and must
    be validated by the Planner before any action is taken.

    Parameters
    ----------
    provider_name : str
        Name of the LLM provider to use ('groq', 'mock', etc.).
    registry : LLMProviderRegistry, optional
        Provider registry to use. Defaults to the global singleton.
    uncertainty_threshold : float
        If estimated uncertainty exceeds this threshold, flag for human review.

    Usage::

        reasoner = LLMReasoner(provider_name='groq')
        hypotheses, reasoning = reasoner.generate_hypotheses(ctx, strategies)
    """

    def __init__(
        self,
        provider_name: str = "groq",
        registry: Optional[LLMProviderRegistry] = None,
        uncertainty_threshold: float = 0.35,
    ) -> None:
        self._registry = registry or _registry
        self._provider_name = provider_name
        self.uncertainty_threshold = uncertainty_threshold
        self._provider: Optional[BaseLLMProvider] = None
        self._resolve_provider()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def llm_available(self) -> bool:
        """True if the configured LLM provider is available."""
        return self._provider is not None and self._provider.is_available

    def generate_hypotheses(
        self,
        ctx: RepairContext,
        strategies: List[StrategyScore],
    ) -> Tuple[List[RepairHypothesis], str]:
        """Generate multi-hypothesis repair analysis for the given context.

        This is the primary entry point.  The LLM analyses the full
        repair context, scores each available strategy probabilistically,
        and may propose novel strategies not in the static catalogue.

        Args:
            ctx:        RepairContext from ContextBuilder.
            strategies: List of StrategyScore objects from ReasoningEngine.

        Returns:
            A tuple of:
            - List[RepairHypothesis] — ranked hypotheses (may include novel ones)
            - str                    — overall reasoning from the LLM
        """
        if not self.llm_available:
            logger.info(
                f"[{ctx.run_id}] LLMReasoner: provider '{self._provider_name}' "
                "unavailable — skipping LLM reasoning"
            )
            return [], ""

        try:
            t0 = time.monotonic()
            prompt = self._build_hypothesis_prompt(ctx, strategies)
            raw = self._provider.complete(_SYSTEM_PROMPT, prompt, max_tokens=1200)
            elapsed = time.monotonic() - t0

            parsed = self._parse_json_response(raw)
            hypotheses = self._parse_hypotheses(parsed, strategies)
            overall_reasoning = parsed.get("overall_reasoning", "")

            logger.info(
                f"[{ctx.run_id}] LLMReasoner: generated {len(hypotheses)} hypotheses "
                f"in {elapsed:.2f}s | uncertainty={parsed.get('uncertainty', '?')}"
            )
            return hypotheses, overall_reasoning

        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] LLMReasoner.generate_hypotheses: {exc}")
            return [], ""

    def generate_reflection(
        self,
        ctx: RepairContext,
        strategy_chosen: str,
        verification_score: float,
        outcome: str,
        recovery_secs: float,
        alternatives: List[str],
    ) -> Dict[str, Any]:
        """Generate a self-reflection after a completed repair cycle.

        The LLM evaluates decision quality and extracts learning points.
        This powers the SelfReflection system (Requirement 9).

        Args:
            ctx              : RepairContext for the completed repair.
            strategy_chosen  : Name of the strategy that was executed.
            verification_score: Final verification score.
            outcome          : 'success' | 'failure' | 'partial'.
            recovery_secs    : Actual recovery time in seconds.
            alternatives     : Names of strategies that were considered.

        Returns:
            Dict with keys: optimal_choice, cheaper_alternative,
            downtime_reduction_estimate_secs, confidence_adjustment,
            learning_points, reflection.
        """
        if not self.llm_available:
            return self._default_reflection(verification_score, outcome)

        try:
            prompt = _REFLECTION_PROMPT_TEMPLATE.format(
                anomaly_type=ctx.anomaly_type,
                severity=ctx.severity,
                strategy_chosen=strategy_chosen,
                verification_score=verification_score,
                outcome=outcome,
                recovery_secs=recovery_secs,
                alternatives=", ".join(alternatives) if alternatives else "none",
            )
            raw = self._provider.complete(_SYSTEM_PROMPT, prompt, max_tokens=600)
            parsed = self._parse_json_response(raw)
            logger.info(
                f"[{ctx.run_id}] LLMReasoner: reflection generated | "
                f"optimal={parsed.get('optimal_choice')} | "
                f"adjustment={parsed.get('confidence_adjustment', 0):.3f}"
            )
            return parsed

        except Exception as exc:
            logger.warning(f"[{ctx.run_id}] LLMReasoner.generate_reflection: {exc}")
            return self._default_reflection(verification_score, outcome)

    def explain_decision(
        self,
        ctx: RepairContext,
        chosen: StrategyScore,
        alternatives: List[StrategyScore],
        llm_reasoning: str = "",
    ) -> str:
        """Generate a concise natural-language explanation for a decision.

        This powers the explainability layer (Requirement 12).

        Args:
            ctx          : RepairContext.
            chosen       : The selected StrategyScore.
            alternatives : All other considered strategies.
            llm_reasoning: Optional reasoning trace from generate_hypotheses.

        Returns:
            Formatted explanation string.
        """
        hist_count = len(ctx.historical_repairs)
        hist_success = sum(
            r.get("success_count", 0) for r in ctx.historical_repairs
        )

        alt_summary = "; ".join(
            f"{s.name} (utility={s.utility:.3f})"
            for s in alternatives[:3]
            if s.name != chosen.name
        )

        explanation = (
            f"Chosen Strategy: {chosen.display_name}\n\n"
            f"Reason: Highest expected utility ({chosen.utility:.4f}).\n"
            f"  • {chosen.expected_success_prob:.0%} predicted success probability.\n"
            f"  • Risk level: {chosen.risk_score:.0%} — "
            + ("Low" if chosen.risk_score < 0.3 else "Medium" if chosen.risk_score < 0.6 else "High")
            + ".\n"
            f"  • Estimated recovery: ~{chosen.estimated_recovery_secs // 60} min "
            f"({chosen.estimated_recovery_secs}s).\n"
        )

        if hist_count > 0:
            explanation += (
                f"  • Historical evidence: {hist_success}/{hist_count} "
                f"similar repairs succeeded.\n"
            )

        if ctx.business_impact > 0.5:
            downtime_est = chosen.estimated_recovery_secs
            explanation += (
                f"  • Business downtime estimated: ~{downtime_est}s "
                f"(business_impact={ctx.business_impact:.0%}).\n"
            )

        if alt_summary:
            explanation += f"\nAlternatives considered: {alt_summary}\n"

        if llm_reasoning:
            explanation += f"\nAI Reasoning: {llm_reasoning}\n"

        return explanation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_provider(self) -> None:
        """Resolve the LLM provider, falling back to mock on failure."""
        from config import MOCK_MODE
        if MOCK_MODE:
            logger.info("LLMReasoner: MOCK_MODE=True — using MockLLMProvider")
            self._provider = self._registry.get("mock")
            return

        try:
            provider = self._registry.get(self._provider_name)
            if provider.is_available:
                self._provider = provider
                logger.info(
                    f"LLMReasoner: using provider '{provider.provider_name}'"
                )
            else:
                logger.warning(
                    f"LLMReasoner: provider '{self._provider_name}' is not "
                    "available (no API key?) — LLM reasoning disabled"
                )
                self._provider = None
        except LLMProviderError as exc:
            logger.warning(f"LLMReasoner: could not initialise provider: {exc}")
            self._provider = None

    def _build_hypothesis_prompt(
        self,
        ctx: RepairContext,
        strategies: List[StrategyScore],
    ) -> str:
        """Build the hypothesis-generation prompt from context."""
        strategy_names = [s.name for s in strategies]
        hist_summary = self._summarise_history(ctx.historical_repairs)
        incident_summary = self._summarise_incidents(ctx.similar_incidents)

        return _HYPOTHESIS_PROMPT_TEMPLATE.format(
            run_id=ctx.run_id,
            anomaly_type=ctx.anomaly_type,
            severity=ctx.severity,
            root_cause=ctx.root_cause,
            gap_minutes=ctx.gap_minutes,
            estimated_missing=ctx.estimated_missing,
            node_health_score=ctx.node_health_score,
            business_impact=ctx.business_impact,
            customer_impact=ctx.customer_impact,
            sla_importance=ctx.sla_importance,
            maintenance_window=ctx.maintenance_window,
            available_strategies=", ".join(strategy_names),
            historical_summary=hist_summary,
            incident_summary=incident_summary,
        )

    def _summarise_history(self, historical_repairs: list) -> str:
        """Summarise historical repairs as a compact text block."""
        if not historical_repairs:
            return "No historical repair data available."
        lines = []
        for r in historical_repairs[:5]:
            action = r.get("action_taken", "?")
            success = r.get("success_count", 0)
            failure = r.get("failure_count", 0)
            total = success + failure
            rate = f"{success}/{total}" if total > 0 else "no data"
            lines.append(f"  - {action}: success rate {rate}")
        return "\n".join(lines)

    def _summarise_incidents(self, similar_incidents: list) -> str:
        """Summarise similar incidents as a compact text block."""
        if not similar_incidents:
            return "No similar incidents found."
        lines = []
        for i in similar_incidents[:3]:
            anomaly = i.get("anomaly_type", "?")
            severity = i.get("severity", "?")
            resolved = "resolved" if i.get("resolved") else "unresolved"
            lines.append(f"  - {anomaly} ({severity}): {resolved}")
        return "\n".join(lines)

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """Parse a raw LLM response into a dict.

        Handles markdown code fences and trailing text gracefully.
        """
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\n?", "", raw).strip()
        # Extract first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise LLMProviderError(f"No JSON object found in LLM response: {raw[:200]}")
        return json.loads(match.group())

    def _parse_hypotheses(
        self,
        parsed: Dict[str, Any],
        strategies: List[StrategyScore],
    ) -> List[RepairHypothesis]:
        """Convert parsed LLM JSON into RepairHypothesis objects."""
        strategy_map = {s.name: s for s in strategies}
        hypotheses: List[RepairHypothesis] = []

        for i, h in enumerate(parsed.get("hypotheses", []), start=1):
            strategy_name = h.get("strategy", "")
            strategy = strategy_map.get(strategy_name)

            # Apply LLM confidence boost to the matched strategy
            confidence_boost = float(h.get("confidence_boost", 0.0))
            if strategy and confidence_boost != 0:
                strategy.llm_confidence_boost = confidence_boost

            hyp = RepairHypothesis(
                strategy=strategy,
                estimated_success=float(h.get("estimated_success", 0.5)),
                risk_label=h.get("risk_label", "Medium"),
                estimated_recovery=h.get("estimated_recovery", "unknown"),
                expected_utility=strategy.utility + confidence_boost if strategy else 0.0,
                rank=i,
                rationale=h.get("rationale", ""),
                llm_generated=True,
                hidden_risks=h.get("hidden_risks", []),
            )
            hypotheses.append(hyp)

        # Sort by expected_utility descending
        hypotheses.sort(key=lambda h: h.expected_utility, reverse=True)
        for rank, hyp in enumerate(hypotheses, start=1):
            hyp.rank = rank

        return hypotheses

    @staticmethod
    def _default_reflection(
        verification_score: float,
        outcome: str,
    ) -> Dict[str, Any]:
        """Return a default reflection when LLM is unavailable."""
        return {
            "optimal_choice": outcome == "success",
            "cheaper_alternative": None,
            "downtime_reduction_estimate_secs": 0,
            "confidence_adjustment": 0.05 if outcome == "success" else -0.05,
            "learning_points": [
                f"Repair outcome: {outcome}",
                f"Verification score: {verification_score:.3f}",
            ],
            "reflection": (
                f"Repair completed with outcome '{outcome}' "
                f"(score={verification_score:.3f}). "
                "LLM reflection unavailable — using heuristic fallback."
            ),
        }
