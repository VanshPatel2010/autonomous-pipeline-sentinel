"""Repair sub-package: autonomous AI-driven repair components.

This package contains the full reasoning → planning → execution →
verification → learning pipeline that powers the refactored RepairerAgent.

Public surface
--------------
RepairContext       - rich context object built from pipeline state
RepairPlan          - executable plan produced by the Planner
RepairStep          - a single step within a plan
StrategyScore       - scored strategy candidate from ReasoningEngine
VerificationResult  - outcome produced by the Verifier

New AI components (added in AI upgrade)
----------------------------------------
LLMReasoner         - optional LLM reasoning layer (provider-agnostic)
MemoryStore         - three-tier long-term memory (Episodic/Semantic/Procedural)
RepairHypothesis    - one ranked repair hypothesis from multi-hypothesis reasoning
UtilityScore        - full 9-dimension expected utility decomposition
UncertaintyEstimate - per-decision uncertainty with human-escalation flag
RepairExplanation   - structured explainability output for every decision
SelfReflection      - post-repair self-evaluation record

Modules
-------
context_builder   - builds RepairContext from PipelineState
reasoning_engine  - AI-driven multi-hypothesis strategy ranking
llm_reasoner      - provider-agnostic LLM reasoning layer (NEW)
memory_store      - three-tier long-term memory store (NEW)
planner           - goal-oriented, context-driven, adaptive planner
executor          - executes RepairPlan steps; no decision logic
verifier          - 6-dimension quality assessment; returns VerificationResult
learner           - adaptive continuous learning across all memory tiers
risk_scorer       - computes dynamic multi-factor risk scores
predictor         - probabilistic failure forecasting
"""

from agents.repair.models import (
    EpisodicMemoryEntry,
    EnrichedPlaybookEntry,
    ProceduralMemoryEntry,
    RepairContext,
    RepairExplanation,
    RepairHypothesis,
    RepairPlan,
    RepairStep,
    SemanticMemoryEntry,
    SelfReflection,
    StrategyScore,
    UncertaintyEstimate,
    UtilityScore,
    VerificationResult,
)

__all__ = [
    # Core models (original)
    "RepairContext",
    "RepairPlan",
    "RepairStep",
    "StrategyScore",
    "VerificationResult",
    "EnrichedPlaybookEntry",
    # New AI models
    "RepairHypothesis",
    "UtilityScore",
    "UncertaintyEstimate",
    "RepairExplanation",
    "SelfReflection",
    "EpisodicMemoryEntry",
    "SemanticMemoryEntry",
    "ProceduralMemoryEntry",
]
