"""Data models for the autonomous AI-driven repair system.

All value objects in the repair pipeline are defined here so that
every module imports from a single source of truth.  No business logic
lives in this file — only dataclasses and type aliases.

Design decisions
----------------
- Frozen dataclasses are used for immutable snapshot objects
  (RepairContext, VerificationResult, UncertaintyEstimate, RepairExplanation).
- Mutable dataclasses (RepairPlan, RepairStep, StrategyScore, SelfReflection)
  are annotated to allow runtime mutation by Executor / Learner / Reflector.
- All monetary/impact values are normalised floats in [0.0, 1.0] or
  counts, making arithmetic across dimensions consistent.
- New models are additive — no existing fields are removed or renamed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# RepairContext
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepairContext:
    """Rich context snapshot assembled before any repair decision is made.

    The ContextBuilder constructs this object from multiple sources:
    PipelineState, DiagnoserOutput, historical incidents, and live
    database-node health.  All downstream modules consume *this* object
    instead of the raw state dict to enforce a clean data boundary.

    Attributes
    ----------
    run_id              : Unique pipeline run identifier.
    anomaly_type        : 'missing_data' | 'data_quality' | 'schema_drift'.
    severity            : 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'.
    confidence          : Diagnoser confidence [0.0, 1.0].
    root_cause          : Free-text root cause from the Diagnoser.
    gap_minutes         : Duration of the detected data gap.
    estimated_missing   : Estimated number of missing rows.
    null_rate           : Fraction of null order_amounts in the window.
    affected_tables     : List of table names flagged by the Diagnoser.
    active_node_id      : ID of the currently active DB node.
    active_node_label   : Human-readable label for the active node.
    node_health_score   : [0.0, 1.0] — 1.0 = fully healthy, 0.0 = dead.
    historical_repairs  : Recent playbook entries for this anomaly type.
    similar_incidents   : Recent incidents with the same anomaly type.
    business_impact     : [0.0, 1.0] — heuristic importance score.
    customer_impact     : [0.0, 1.0] — estimated customer-facing impact.
    sla_importance      : [0.0, 1.0] — how critical SLA compliance is.
    current_system_load : [0.0, 1.0] — system load at repair time.
    hour_of_day         : Hour in UTC (0-23); used for time-of-day weighting.
    maintenance_window  : True if the Diagnoser suspects a maintenance window.
    pipeline_metrics    : Raw pipeline metrics passed through from state.
    created_at          : UTC timestamp when this context was created.
    """

    run_id: str
    anomaly_type: str
    severity: str
    confidence: float
    root_cause: str
    gap_minutes: float
    estimated_missing: int
    null_rate: float
    affected_tables: List[str]

    # Node health
    active_node_id: str
    active_node_label: str
    node_health_score: float  # [0.0, 1.0]

    # Memory
    historical_repairs: List[Dict[str, Any]]
    similar_incidents: List[Dict[str, Any]]

    # Business context
    business_impact: float    # [0.0, 1.0]
    customer_impact: float    # [0.0, 1.0]
    sla_importance: float     # [0.0, 1.0]

    # System context
    current_system_load: float  # [0.0, 1.0]
    hour_of_day: int            # 0-23 UTC

    # Flags
    maintenance_window: bool

    # Pass-through pipeline metrics
    pipeline_metrics: Dict[str, Any]

    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# UtilityScore — full expected-utility model (Requirement 4)
# ---------------------------------------------------------------------------

@dataclass
class UtilityScore:
    """Multi-dimensional expected-utility decomposition for a repair strategy.

    The ReasoningEngine computes this object alongside the legacy
    ``utility`` float on StrategyScore so every dimension is inspectable.

    Dimensions
    ----------
    success_probability  : P(strategy succeeds) — from history + LLM.
    business_impact      : Magnitude of business harm if this fails [0,1].
    recovery_time_norm   : Normalised recovery time [0,1] — lower = better.
    operational_cost     : Relative operational cost [0,1].
    risk                 : Composite risk from RiskScorer [0,1].
    historical_success   : Observed success rate in repair_memory [0,1].
    customer_impact      : Customer-facing disruption if failure [0,1].
    sla_importance       : SLA compliance weight [0,1].
    confidence           : Diagnoser confidence in the anomaly [0,1].
    expected_utility     : Final scalar utility (higher = preferred).
    """

    success_probability: float = 0.5
    business_impact: float = 0.5
    recovery_time_norm: float = 0.5
    operational_cost: float = 0.5
    risk: float = 0.5
    historical_success: float = 0.5
    customer_impact: float = 0.5
    sla_importance: float = 0.5
    confidence: float = 0.5
    expected_utility: float = 0.0

    def compute(self) -> "UtilityScore":
        """Compute and store the expected_utility from all dimensions.

        Formula (9-factor model):
            EU = success_prob
                 * (1 - risk)
                 * historical_success
                 * (1 - operational_cost * 0.3)
                 * (1 - recovery_time_norm * 0.2)
                 * (1 + business_impact * 0.4)
                 * (1 + sla_importance * 0.3)
                 * confidence

        The formula blends expected-value theory with SLA/business boosters.
        """
        self.expected_utility = (
            self.success_probability
            * (1.0 - self.risk)
            * self.historical_success
            * (1.0 - self.operational_cost * 0.3)
            * (1.0 - self.recovery_time_norm * 0.2)
            * (1.0 + self.business_impact * 0.4)
            * (1.0 + self.sla_importance * 0.3)
            * self.confidence
        )
        self.expected_utility = max(self.expected_utility, 0.0)
        return self


# ---------------------------------------------------------------------------
# StrategyScore — one candidate scored by the ReasoningEngine
# ---------------------------------------------------------------------------

@dataclass
class StrategyScore:
    """A repair strategy with computed evaluation scores.

    Attributes
    ----------
    name                    : Strategy identifier (e.g. 'switch_to_backup').
    display_name            : Human-friendly label.
    expected_success_prob   : P(success) estimated from history [0.0, 1.0].
    estimated_recovery_secs : Estimated wall-clock recovery time in seconds.
    risk_score              : Composite risk [0.0, 1.0] — higher = riskier.
    cost_score              : Relative cost [0.0, 1.0] — higher = more expensive.
    historical_success_rate : Observed success rate from playbook memory.
    business_impact_score   : Expected business impact of *this* strategy.
    utility                 : Final utility value used to rank strategies.
    utility_breakdown       : Full UtilityScore object (all dimensions).
    rationale               : One-line explanation of the utility calculation.
    rollback_strategy       : Name of the strategy used to undo this one.
    llm_confidence_boost    : Optional confidence adjustment from LLMReasoner.
    """

    name: str
    display_name: str
    expected_success_prob: float
    estimated_recovery_secs: int
    risk_score: float
    cost_score: float
    historical_success_rate: float
    business_impact_score: float
    utility: float = 0.0
    utility_breakdown: Optional[UtilityScore] = None
    rationale: str = ""
    rollback_strategy: Optional[str] = None
    llm_confidence_boost: float = 0.0


# ---------------------------------------------------------------------------
# RepairHypothesis — multi-hypothesis reasoning (Requirement 3)
# ---------------------------------------------------------------------------

@dataclass
class RepairHypothesis:
    """One repair hypothesis produced during multi-hypothesis reasoning.

    The ReasoningEngine generates N hypotheses, ranks them by expected
    utility, and the Planner receives the top-ranked hypothesis.

    Attributes
    ----------
    hypothesis_id       : Unique short ID for this hypothesis.
    strategy            : StrategyScore backing this hypothesis.
    estimated_success   : P(success) for this hypothesis [0,1].
    risk_label          : Human-readable risk label ('Low', 'Medium', etc.).
    estimated_recovery  : Human-readable recovery estimate ('~3 min').
    expected_utility    : Scalar utility for ranking.
    rank                : 1-indexed rank after sorting (1 = best).
    rationale           : Brief natural-language explanation.
    llm_generated       : True if this hypothesis was proposed by LLMReasoner.
    hidden_risks        : List of hidden/secondary risks identified.
    """

    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy: Optional[StrategyScore] = None
    estimated_success: float = 0.5
    risk_label: str = "Medium"
    estimated_recovery: str = "unknown"
    expected_utility: float = 0.0
    rank: int = 0
    rationale: str = ""
    llm_generated: bool = False
    hidden_risks: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RepairStep — one atomic step in a repair plan
# ---------------------------------------------------------------------------

@dataclass
class RepairStep:
    """One atomic action within a RepairPlan.

    Attributes
    ----------
    step_number     : 1-indexed position in the plan.
    action          : Short action identifier (matches Executor dispatch).
    description     : Human-readable description of this step.
    requires_verify : Whether the Verifier should run after this step.
    timeout_secs    : Maximum execution time for this step.
    executed        : True after the Executor has run this step.
    outcome         : Result dict populated by the Executor.
    verified        : True if the Verifier confirmed success.
    goal_driven     : True if this step was generated by goal-oriented planning.
    """

    step_number: int
    action: str
    description: str
    requires_verify: bool = True
    timeout_secs: int = 60
    executed: bool = False
    outcome: Optional[Dict[str, Any]] = None
    verified: bool = False
    goal_driven: bool = False


# ---------------------------------------------------------------------------
# RepairPlan — the full structured plan for a repair cycle
# ---------------------------------------------------------------------------

@dataclass
class RepairPlan:
    """Executable repair plan produced by the Planner.

    Attributes
    ----------
    plan_id             : Unique plan identifier.
    goal                : Natural-language description of the repair goal.
    strategy            : Selected StrategyScore driving this plan.
    steps               : Ordered list of RepairStep objects.
    max_attempts        : Maximum number of full plan attempts.
    attempt_number      : Current attempt (starts at 1).
    rollback_strategy   : Strategy to execute if verification fails.
    created_at          : UTC timestamp of plan creation.
    completed_at        : UTC timestamp set when the plan finishes.
    success             : Overall plan success flag.
    final_outcome       : Summary dict written by the Executor/Verifier.
    goal_decomposition  : Sub-goals for goal-oriented planning.
    hypothesis_rank     : Which hypothesis this plan implements (1-indexed).
    """

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    goal: str = ""
    strategy: Optional[StrategyScore] = None
    steps: List[RepairStep] = field(default_factory=list)
    max_attempts: int = 3
    attempt_number: int = 1
    rollback_strategy: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    success: bool = False
    final_outcome: Dict[str, Any] = field(default_factory=dict)
    goal_decomposition: List[str] = field(default_factory=list)
    hypothesis_rank: int = 1


# ---------------------------------------------------------------------------
# VerificationResult — post-repair health check (extended, Requirement 10)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerificationResult:
    """Outcome of the Verifier's post-repair health check.

    Extended with AI-computed quality dimensions: repair_confidence,
    repair_quality_score, residual_risk, data_integrity_score,
    recovery_score, business_impact_score.

    Attributes
    ----------
    anomaly_removed         : True if the triggering anomaly is gone.
    rows_recovered          : Number of rows that are now present.
    duplicates_found        : Count of duplicate rows introduced by repair.
    latency_acceptable      : True if pipeline latency is within SLA.
    pipeline_healthy        : Aggregate health flag.
    downstream_affected     : True if downstream consumers are impacted.
    verification_score      : Weighted aggregate [0.0, 1.0]; ≥0.7 = success.
    repair_confidence       : Composite confidence in the repair [0,1].
    repair_quality_score    : Multi-dimensional quality score [0,1].
    residual_risk           : Remaining risk after repair [0,1].
    data_integrity_score    : Data integrity assessment [0,1].
    recovery_score          : How fully the system recovered [0,1].
    business_impact_score   : Business impact of the repair outcome [0,1].
    details                 : Free-text explanation.
    checked_at              : UTC timestamp of the check.
    """

    anomaly_removed: bool
    rows_recovered: int
    duplicates_found: int
    latency_acceptable: bool
    pipeline_healthy: bool
    downstream_affected: bool
    verification_score: float
    details: str
    # Extended quality dimensions (Requirement 10)
    repair_confidence: float = 0.0
    repair_quality_score: float = 0.0
    residual_risk: float = 0.0
    data_integrity_score: float = 0.0
    recovery_score: float = 0.0
    business_impact_score: float = 0.0
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def passed(self) -> bool:
        """Return True if the repair is considered successful."""
        return self.verification_score >= 0.7 and self.pipeline_healthy

    @property
    def uncertainty(self) -> float:
        """Return the uncertainty complement of repair_confidence."""
        return round(1.0 - self.repair_confidence, 4)


# ---------------------------------------------------------------------------
# UncertaintyEstimate — per-decision uncertainty (Requirement 14)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UncertaintyEstimate:
    """Uncertainty estimate attached to every major decision.

    Attributes
    ----------
    confidence          : Agent's confidence in this decision [0,1].
    uncertainty         : 1 - confidence; uncertainty complement [0,1].
    epistemic           : Model uncertainty (not enough data) [0,1].
    aleatoric           : Inherent randomness in the environment [0,1].
    requires_human      : True if uncertainty exceeds escalation threshold.
    threshold           : Configured escalation threshold (from config).
    decision_id         : Identifier of the decision this estimate covers.
    reasoning           : Why this uncertainty level was assigned.
    """

    confidence: float
    uncertainty: float
    epistemic: float
    aleatoric: float
    requires_human: bool
    threshold: float
    decision_id: str
    reasoning: str = ""


# ---------------------------------------------------------------------------
# RepairExplanation — structured explainability (Requirement 12)
# ---------------------------------------------------------------------------

@dataclass
class RepairExplanation:
    """Structured explanation for every repair decision.

    Attributes
    ----------
    chosen_strategy     : Name of the selected strategy.
    display_name        : Human-readable strategy label.
    reason              : Primary reason for selecting this strategy.
    expected_recovery   : Human-readable recovery time estimate.
    predicted_success   : P(success) stated as a percentage string.
    risk_assessment     : Risk label and rationale.
    historical_evidence : Evidence from past repairs ('14 of 16 times').
    business_downtime   : Estimated downtime as a string.
    utility_score       : Final utility value.
    alternatives        : List of {name, utility, reason_rejected} dicts.
    uncertainty         : UncertaintyEstimate for this decision.
    llm_reasoning       : Optional LLM reasoning trace.
    created_at          : UTC timestamp.
    """

    chosen_strategy: str
    display_name: str
    reason: str
    expected_recovery: str
    predicted_success: str
    risk_assessment: str
    historical_evidence: str
    business_downtime: str
    utility_score: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty: Optional[UncertaintyEstimate] = None
    llm_reasoning: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# SelfReflection — post-repair self-evaluation (Requirement 9)
# ---------------------------------------------------------------------------

@dataclass
class SelfReflection:
    """Post-repair self-evaluation record.

    The RepairerAgent reflects on each completed repair cycle to answer:
    - Did I choose the best repair?
    - Was there a cheaper / faster repair?
    - Could downtime have been reduced?
    - What should confidence change to?
    - What can I learn?

    These reflections are persisted in the long-term memory store.

    Attributes
    ----------
    run_id              : Pipeline run this reflection is about.
    plan_id             : RepairPlan that was executed.
    strategy_chosen     : Strategy name that was executed.
    verification_passed : Whether the repair succeeded.
    verification_score  : Final verification score.
    optimal_choice      : Agent's self-assessment of optimality (T/F).
    cheaper_alternative : Name of a cheaper alternative if found.
    downtime_reduction  : Estimated seconds that could have been saved.
    confidence_adjustment : Recommended confidence delta (positive or negative).
    learning_points     : List of extracted learning observations.
    reflection_text     : Free-form reflection paragraph.
    created_at          : UTC timestamp.
    """

    run_id: str
    plan_id: str
    strategy_chosen: str
    verification_passed: bool
    verification_score: float
    optimal_choice: bool = True
    cheaper_alternative: Optional[str] = None
    downtime_reduction: float = 0.0
    confidence_adjustment: float = 0.0
    learning_points: List[str] = field(default_factory=list)
    reflection_text: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# LongTermMemory — episodic / semantic / procedural (Requirement 13)
# ---------------------------------------------------------------------------

@dataclass
class EpisodicMemoryEntry:
    """One specific repair event in episodic memory.

    Episodic memory stores what happened, when, and with what outcome.
    Used to retrieve 'remember when we had this exact failure' evidence.
    """

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    run_id: str = ""
    anomaly_type: str = ""
    severity: str = ""
    root_cause: str = ""
    strategy_used: str = ""
    outcome: str = ""           # 'success' | 'failure' | 'partial'
    verification_score: float = 0.0
    recovery_secs: float = 0.0
    embedding_text: str = ""    # text used to generate semantic embedding
    embedding: List[float] = field(default_factory=list)  # future vector DB
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SemanticMemoryEntry:
    """General repair knowledge in semantic memory.

    Semantic memory stores abstract facts: 'replica lag after failover
    usually resolves in 3-5 minutes if network is stable'.
    """

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    concept: str = ""           # e.g. 'replica_lag_after_failover'
    knowledge: str = ""         # free-text knowledge statement
    confidence: float = 0.5    # how strongly this is believed
    source: str = ""            # 'learner' | 'llm' | 'operator'
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ProceduralMemoryEntry:
    """A successful repair procedure in procedural memory.

    Procedural memory stores how to do things: ordered step sequences
    that have been verified as successful in past repairs.
    """

    memory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    procedure_name: str = ""
    anomaly_type: str = ""
    severity: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)  # serialised steps
    success_count: int = 0
    total_count: int = 0
    avg_verification_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def success_rate(self) -> float:
        """Historical success rate for this procedure."""
        if self.total_count == 0:
            return 0.5
        return self.success_count / self.total_count


# ---------------------------------------------------------------------------
# EnrichedPlaybookEntry — extended record for the learning module
# ---------------------------------------------------------------------------

@dataclass
class EnrichedPlaybookEntry:
    """Extended playbook record written by the Learner after each repair.

    This supersedes the slim ``record_outcome()`` call in the old Repairer.
    It stores enough information to support future reinforcement-learning
    fine-tuning and operator review.

    All fields map to the ``repair_memory`` table added by the schema
    migration in ``memory/repair_memory_schema.sql``.
    """

    run_id: str
    anomaly_type: str
    severity: str
    root_cause: str
    repair_plan_id: str
    strategy_name: str
    execution_time_secs: float
    recovery_time_secs: float
    affected_rows: int
    node_used: str
    repair_confidence: float
    business_impact: float
    failure_reason: str
    verification_score: float
    operator_feedback: str
    final_outcome: str           # 'success' | 'failure' | 'partial'
    # Extended fields for AI learning
    utility_score: float = 0.0
    hypothesis_rank: int = 1
    llm_assisted: bool = False
    reflection_stored: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
