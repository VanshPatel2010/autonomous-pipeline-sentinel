"""Tests for the autonomous RepairerAgent and its sub-components.

Covers:
- ContextBuilder: correct field population from PipelineState
- RiskScorer: score range and dimension sensitivity
- ReasoningEngine: strategy ranking without if/else
- Planner: plan structure, step ordering, rollback field
- Executor: step dispatch, timeout, backward-compat output keys
- Verifier: all health checks, verification_score computation
- Learner: repair_memory schema creation, enriched entry persistence
- Predictor: pattern detection on synthetic incident history
- RepairerAgent.run(): end-to-end with backward-compat assertions
  (preserves all original test contract expectations)
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from agents.repair.context_builder import ContextBuilder
from agents.repair.executor import Executor
from agents.repair.learner import Learner
from agents.repair.models import (
    RepairContext,
    RepairPlan,
    RepairStep,
    StrategyScore,
    VerificationResult,
)
from agents.repair.planner import Planner
from agents.repair.predictor import Predictor
from agents.repair.reasoning_engine import ReasoningEngine
from agents.repair.risk_scorer import RiskScorer
from agents.repair.verifier import Verifier
from agents.repairer import RepairerAgent
from state import create_initial_state


# ═══════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _init_db(db_path: str) -> None:
    """Create all tables needed by the repair sub-system in a temp database."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL DEFAULT 'mumbai'
        );
        CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);

        CREATE TABLE IF NOT EXISTS backup_orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL DEFAULT 'delhi'
        );
        CREATE INDEX IF NOT EXISTS idx_backup_orders_created_at ON backup_orders(created_at);

        CREATE TABLE IF NOT EXISTS quarantine_orders (
            order_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            order_amount REAL,
            source_db TEXT NOT NULL,
            quarantine_reason TEXT,
            quarantined_at DATETIME,
            run_id TEXT
        );

        CREATE TABLE IF NOT EXISTS playbooks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            anomaly_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            action_taken TEXT NOT NULL,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            last_used TEXT,
            UNIQUE(anomaly_type, severity, action_taken)
        );

        CREATE TABLE IF NOT EXISTS data_gaps (
            gap_id TEXT PRIMARY KEY,
            run_id TEXT,
            start_time TEXT,
            end_time TEXT,
            estimated_rows INTEGER,
            source_db TEXT,
            reconciled INTEGER DEFAULT 0,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS incidents (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            anomaly_type TEXT,
            severity TEXT,
            gap_minutes REAL DEFAULT 0,
            root_cause TEXT,
            affected_tables TEXT,
            fix_taken TEXT,
            resolved INTEGER DEFAULT 0,
            resolved_at TEXT,
            confidence REAL DEFAULT 0.0
        );
    """)
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    """Temporary database with all required tables."""
    path = str(tmp_path / "test_autonomous.db")
    _init_db(path)
    return path


@pytest.fixture
def agent(db_path):
    """RepairerAgent wired to temp database."""
    return RepairerAgent(db_path=db_path, max_attempts=2)


def _make_state(
    run_id: str = "test-001",
    anomaly_type: str = "missing_data",
    severity: str = "MEDIUM",
    gap_minutes: float = 60.0,
    confidence: float = 0.85,
    anomaly_detected: bool = True,
) -> dict:
    """Helper: create a fully populated PipelineState for tests."""
    state = create_initial_state(run_id=run_id, timestamp="2026-06-25T08:00:00+00:00")
    state["anomaly_detected"] = anomaly_detected
    state["anomaly_type"] = anomaly_type
    state["severity"] = severity
    state["gap_minutes"] = gap_minutes
    state["raw_count"] = 0
    state["expected_avg"] = 200.0
    state["null_rate"] = 0.02
    state["affected_tables"] = ["orders"]
    state["diagnoser_output"] = {
        "root_cause": "Mumbai source DB outage",
        "confidence": confidence,
        "estimated_missing_rows": int(gap_minutes * 40),
        "affected_tables": ["orders"],
        "maintenance_window_likely": False,
    }
    return state


def _make_context(
    run_id: str = "ctx-001",
    anomaly_type: str = "missing_data",
    severity: str = "MEDIUM",
    confidence: float = 0.85,
    gap_minutes: float = 60.0,
    estimated_missing: int = 2400,
) -> RepairContext:
    """Helper: create a RepairContext without DB I/O."""
    return RepairContext(
        run_id=run_id,
        anomaly_type=anomaly_type,
        severity=severity,
        confidence=confidence,
        root_cause="Mumbai DB outage",
        gap_minutes=gap_minutes,
        estimated_missing=estimated_missing,
        null_rate=0.02,
        affected_tables=["orders"],
        active_node_id="primary",
        active_node_label="PRIMARY",
        node_health_score=1.0,
        historical_repairs=[],
        similar_incidents=[],
        business_impact=0.6,
        customer_impact=0.5,
        sla_importance=0.7,
        current_system_load=0.3,
        hour_of_day=10,
        maintenance_window=False,
        pipeline_metrics={"gap_minutes": gap_minutes},
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. ContextBuilder
# ═══════════════════════════════════════════════════════════════════════════

class TestContextBuilder:
    """ContextBuilder assembles RepairContext from PipelineState."""

    def test_basic_context_populated(self, db_path):
        """All required fields are populated from a standard state."""
        builder = ContextBuilder(db_path=db_path)
        state = _make_state()
        ctx = builder.build(state)

        assert ctx.run_id == "test-001"
        assert ctx.anomaly_type == "missing_data"
        assert ctx.severity == "MEDIUM"
        assert ctx.confidence == pytest.approx(0.85)
        assert ctx.gap_minutes == pytest.approx(60.0)
        assert ctx.estimated_missing == 2400
        assert ctx.affected_tables == ["orders"]
        assert ctx.root_cause != ""

    def test_business_impact_in_range(self, db_path):
        """Business impact is always in [0.0, 1.0]."""
        builder = ContextBuilder(db_path=db_path)
        for sev in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            state = _make_state(severity=sev)
            ctx = builder.build(state)
            assert 0.0 <= ctx.business_impact <= 1.0, f"Out of range for {sev}"

    def test_node_health_in_range(self, db_path):
        """node_health_score is always in [0.0, 1.0]."""
        builder = ContextBuilder(db_path=db_path)
        ctx = builder.build(_make_state())
        assert 0.0 <= ctx.node_health_score <= 1.0

    def test_similar_incidents_list(self, db_path):
        """similar_incidents is a list (possibly empty)."""
        builder = ContextBuilder(db_path=db_path)
        ctx = builder.build(_make_state())
        assert isinstance(ctx.similar_incidents, list)

    def test_maintenance_window_propagated(self, db_path):
        """maintenance_window_likely from diagnoser is propagated."""
        builder = ContextBuilder(db_path=db_path)
        state = _make_state()
        state["diagnoser_output"]["maintenance_window_likely"] = True
        ctx = builder.build(state)
        assert ctx.maintenance_window is True


# ═══════════════════════════════════════════════════════════════════════════
# 2. RiskScorer
# ═══════════════════════════════════════════════════════════════════════════

class TestRiskScorer:
    """RiskScorer produces a continuous risk score."""

    def test_score_in_range(self):
        """Score is always in [0.0, 1.0]."""
        scorer = RiskScorer()
        ctx = _make_context()
        score = scorer.compute(ctx)
        assert 0.0 <= score <= 1.0

    def test_critical_higher_than_low(self):
        """CRITICAL context scores higher risk than LOW context."""
        scorer = RiskScorer()
        ctx_low = _make_context(severity="LOW", estimated_missing=10)
        ctx_crit = _make_context(severity="CRITICAL", estimated_missing=50000)
        assert scorer.compute(ctx_crit) > scorer.compute(ctx_low)

    def test_unhealthy_node_raises_risk(self):
        """Unhealthy node_health_score increases risk."""
        scorer = RiskScorer()
        ctx_healthy = _make_context()
        ctx_unhealthy = RepairContext(
            **{**ctx_healthy.__dict__, "node_health_score": 0.1}
        )
        assert scorer.compute(ctx_unhealthy) > scorer.compute(ctx_healthy)

    def test_minimum_risk_floor(self):
        """Risk is always above the minimum floor (0.05)."""
        scorer = RiskScorer()
        ctx = _make_context(severity="NONE", estimated_missing=0, confidence=0.1)
        assert scorer.compute(ctx) >= RiskScorer.MIN_RISK


# ═══════════════════════════════════════════════════════════════════════════
# 3. ReasoningEngine
# ═══════════════════════════════════════════════════════════════════════════

class TestReasoningEngine:
    """ReasoningEngine ranks strategies by utility without if/else."""

    def test_returns_list_of_strategy_scores(self):
        """rank_strategies returns a non-empty list of StrategyScore objects."""
        engine = ReasoningEngine()
        ctx = _make_context(anomaly_type="missing_data", severity="MEDIUM")
        ranked = engine.rank_strategies(ctx)
        assert isinstance(ranked, list)
        assert len(ranked) > 0
        assert all(isinstance(s, StrategyScore) for s in ranked)

    def test_sorted_by_utility_descending(self):
        """Results are sorted highest utility first."""
        engine = ReasoningEngine()
        ctx = _make_context(anomaly_type="missing_data", severity="MEDIUM")
        ranked = engine.rank_strategies(ctx)
        utilities = [s.utility for s in ranked]
        assert utilities == sorted(utilities, reverse=True)

    def test_critical_severity_includes_escalation(self):
        """CRITICAL context must include escalate_to_human as a candidate."""
        engine = ReasoningEngine()
        ctx = _make_context(severity="CRITICAL")
        ranked = engine.rank_strategies(ctx)
        names = [s.name for s in ranked]
        assert "escalate_to_human" in names

    def test_utility_in_range(self):
        """All utility values are non-negative."""
        engine = ReasoningEngine()
        ctx = _make_context(anomaly_type="data_quality", severity="HIGH")
        for s in engine.rank_strategies(ctx):
            assert s.utility >= 0.0

    def test_wait_and_retry_excluded_for_critical(self):
        """wait_and_retry must NOT appear for CRITICAL severity."""
        engine = ReasoningEngine()
        ctx = _make_context(severity="CRITICAL")
        ranked = engine.rank_strategies(ctx)
        names = [s.name for s in ranked]
        assert "wait_and_retry" not in names

    def test_historical_rate_blended_into_utility(self):
        """High historical success rate raises utility vs zero history."""
        engine = ReasoningEngine()
        ctx_no_hist = _make_context(anomaly_type="missing_data", severity="MEDIUM")
        ctx_with_hist = RepairContext(
            **{
                **ctx_no_hist.__dict__,
                "historical_repairs": [
                    {"action_taken": "switch_to_backup", "success_count": 9, "failure_count": 1}
                ],
            }
        )
        ranked_no = engine.rank_strategies(ctx_no_hist)
        ranked_with = engine.rank_strategies(ctx_with_hist)

        utility_no   = next((s.utility for s in ranked_no   if s.name == "switch_to_backup"), 0)
        utility_with = next((s.utility for s in ranked_with if s.name == "switch_to_backup"), 0)
        assert utility_with >= utility_no

    def test_best_strategy_not_none_for_valid_context(self):
        """best_strategy() never returns None for a valid context."""
        engine = ReasoningEngine()
        ctx = _make_context()
        assert engine.best_strategy(ctx) is not None


# ═══════════════════════════════════════════════════════════════════════════
# 4. Planner
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanner:
    """Planner generates structured RepairPlan objects."""

    def _make_strategy(self, name: str = "switch_to_backup") -> StrategyScore:
        return StrategyScore(
            name=name,
            display_name="Test Strategy",
            expected_success_prob=0.8,
            estimated_recovery_secs=120,
            risk_score=0.3,
            cost_score=0.4,
            historical_success_rate=0.8,
            business_impact_score=0.6,
            utility=0.45,
            rollback_strategy="rollback_failover",
        )

    def test_plan_has_steps(self):
        """Generated plan has at least one RepairStep."""
        planner = Planner()
        ctx = _make_context()
        plan = planner.generate(ctx, self._make_strategy())
        assert len(plan.steps) > 0

    def test_steps_are_numbered_sequentially(self):
        """Step numbers start at 1 and increment."""
        planner = Planner()
        ctx = _make_context()
        plan = planner.generate(ctx, self._make_strategy())
        for i, step in enumerate(plan.steps, start=1):
            assert step.step_number == i

    def test_plan_goal_is_populated(self):
        """Plan goal is a non-empty string."""
        planner = Planner()
        ctx = _make_context()
        plan = planner.generate(ctx, self._make_strategy())
        assert len(plan.goal) > 0

    def test_rollback_strategy_propagated(self):
        """rollback_strategy from StrategyScore is in the plan."""
        planner = Planner()
        ctx = _make_context()
        plan = planner.generate(ctx, self._make_strategy())
        assert plan.rollback_strategy == "rollback_failover"

    def test_attempt_number_recorded(self):
        """attempt_number is set correctly."""
        planner = Planner()
        ctx = _make_context()
        plan = planner.generate(ctx, self._make_strategy(), attempt_number=2, max_attempts=3)
        assert plan.attempt_number == 2
        assert plan.max_attempts == 3

    def test_unknown_strategy_uses_fallback(self):
        """Unknown strategy name falls back to wait_and_retry template."""
        planner = Planner()
        ctx = _make_context()
        strategy = self._make_strategy(name="some_unknown_strategy")
        plan = planner.generate(ctx, strategy)
        # Fallback produces at least one step
        assert len(plan.steps) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. Executor
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutor:
    """Executor runs plan steps and populates outcomes."""

    def _make_plan(self, action: str) -> RepairPlan:
        step = RepairStep(
            step_number=1,
            action=action,
            description="Test step",
            requires_verify=False,
            timeout_secs=5,
        )
        strategy = StrategyScore(
            name=action, display_name=action,
            expected_success_prob=0.8, estimated_recovery_secs=60,
            risk_score=0.2, cost_score=0.2,
            historical_success_rate=0.8, business_impact_score=0.5,
        )
        return RepairPlan(goal="test", strategy=strategy, steps=[step])

    def test_wait_and_retry_outcome(self, db_path):
        """wait_and_retry sets executed=True and success=True."""
        executor = Executor(db_path=db_path)
        ctx = _make_context()
        state = _make_state()
        plan = self._make_plan("wait_and_retry")
        executor.run(plan, ctx, state)
        step = plan.steps[0]
        assert step.executed is True
        assert step.outcome["success"] is True
        assert step.outcome["action_taken"] == "wait_and_retry"

    def test_quarantine_moves_rows(self, db_path):
        """quarantine_bad_data outcome reflects row count."""
        # Seed null rows
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        conn = sqlite3.connect(db_path)
        for i in range(3):
            created = (now - timedelta(minutes=5 - i)).isoformat()
            conn.execute(
                "INSERT INTO orders (order_id, created_at, order_amount, source_db) "
                "VALUES (?, ?, NULL, 'mumbai')",
                (f"null-{i:03d}", created),
            )
        conn.commit()
        conn.close()

        executor = Executor(db_path=db_path)
        ctx = _make_context(anomaly_type="data_quality", severity="HIGH")
        state = _make_state(anomaly_type="data_quality", severity="HIGH")
        plan = self._make_plan("quarantine_bad_data")
        executor.run(plan, ctx, state)

        step = plan.steps[0]
        assert step.executed is True
        assert step.outcome["rows_affected"] == 3

    def test_unregistered_action_returns_failure(self, db_path):
        """Unregistered action produces success=False without crashing."""
        executor = Executor(db_path=db_path)
        ctx = _make_context()
        state = _make_state()
        plan = self._make_plan("totally_unknown_action_xyz")
        executor.run(plan, ctx, state)
        step = plan.steps[0]
        assert step.executed is True
        assert step.outcome["success"] is False

    def test_verify_steps_are_skipped(self, db_path):
        """Verify-prefix steps are marked executed but not dispatched."""
        executor = Executor(db_path=db_path)
        ctx = _make_context()
        state = _make_state()
        plan = self._make_plan("verify_anomaly_gone")
        executor.run(plan, ctx, state)
        step = plan.steps[0]
        assert step.executed is True
        assert step.outcome["success"] is None  # pending

    def test_output_has_required_keys(self, db_path):
        """Every step outcome dict has action_taken, success, rows_affected."""
        executor = Executor(db_path=db_path)
        ctx = _make_context()
        state = _make_state()
        plan = self._make_plan("wait_and_retry")
        executor.run(plan, ctx, state)
        outcome = plan.steps[0].outcome
        assert "action_taken" in outcome
        assert "success" in outcome
        assert "rows_affected" in outcome


# ═══════════════════════════════════════════════════════════════════════════
# 6. Verifier
# ═══════════════════════════════════════════════════════════════════════════

class TestVerifier:
    """Verifier produces VerificationResult with score in [0.0, 1.0]."""

    def _make_plan_for_verifier(self) -> RepairPlan:
        strategy = StrategyScore(
            name="wait_and_retry", display_name="Wait",
            expected_success_prob=0.8, estimated_recovery_secs=60,
            risk_score=0.1, cost_score=0.1,
            historical_success_rate=0.8, business_impact_score=0.5,
        )
        return RepairPlan(goal="verify test", strategy=strategy, steps=[])

    def test_verification_score_in_range(self, db_path):
        """verification_score is always in [0.0, 1.0]."""
        verifier = Verifier(db_path=db_path)
        ctx = _make_context()
        plan = self._make_plan_for_verifier()
        result = verifier.verify(plan, ctx)
        assert 0.0 <= result.verification_score <= 1.0

    def test_verification_result_is_frozen(self, db_path):
        """VerificationResult is frozen (immutable)."""
        verifier = Verifier(db_path=db_path)
        ctx = _make_context()
        plan = self._make_plan_for_verifier()
        result = verifier.verify(plan, ctx)
        with pytest.raises((AttributeError, TypeError)):
            result.verification_score = 999  # should raise

    def test_passed_property_consistent_with_score(self, db_path):
        """passed property is consistent with verification_score >= 0.7."""
        verifier = Verifier(db_path=db_path)
        ctx = _make_context()
        plan = self._make_plan_for_verifier()
        result = verifier.verify(plan, ctx)
        # passed = score >= 0.7 AND pipeline_healthy
        if result.verification_score >= 0.7 and result.pipeline_healthy:
            assert result.passed is True

    def test_no_duplicates_check(self, db_path):
        """With no duplicate rows inserted, duplicates_found should be 0."""
        verifier = Verifier(db_path=db_path)
        ctx = _make_context()
        plan = self._make_plan_for_verifier()
        result = verifier.verify(plan, ctx)
        assert result.duplicates_found == 0

    def test_details_populated(self, db_path):
        """details field is a non-empty string."""
        verifier = Verifier(db_path=db_path)
        ctx = _make_context()
        plan = self._make_plan_for_verifier()
        result = verifier.verify(plan, ctx)
        assert len(result.details) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. Learner
# ═══════════════════════════════════════════════════════════════════════════

class TestLearner:
    """Learner persists enriched entries and updates playbooks."""

    def _make_verification(self, passed: bool) -> VerificationResult:
        score = 0.85 if passed else 0.30
        return VerificationResult(
            anomaly_removed=passed,
            rows_recovered=100 if passed else 0,
            duplicates_found=0,
            latency_acceptable=True,
            pipeline_healthy=passed,
            downstream_affected=False,
            verification_score=score,
            details="test verification",
        )

    def test_repair_memory_table_created(self, db_path):
        """Learner creates repair_memory table on instantiation."""
        Learner(db_path=db_path)
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='repair_memory'"
        ).fetchone()
        conn.close()
        assert tables is not None

    def test_record_writes_entry(self, db_path):
        """record() inserts a row into repair_memory."""
        learner = Learner(db_path=db_path)
        strategy = StrategyScore(
            name="wait_and_retry", display_name="Wait",
            expected_success_prob=0.8, estimated_recovery_secs=60,
            risk_score=0.1, cost_score=0.1,
            historical_success_rate=0.8, business_impact_score=0.5,
        )
        plan = RepairPlan(goal="test", strategy=strategy, steps=[])
        ctx = _make_context()
        verification = self._make_verification(passed=True)
        t0 = time.monotonic()
        learner.record(ctx, plan, verification, t0)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM repair_memory").fetchone()[0]
        conn.close()
        assert count == 1

    def test_final_outcome_success_on_passed(self, db_path):
        """final_outcome = 'success' when verification passes."""
        learner = Learner(db_path=db_path)
        strategy = StrategyScore(
            name="switch_to_backup", display_name="Backup",
            expected_success_prob=0.8, estimated_recovery_secs=120,
            risk_score=0.3, cost_score=0.4,
            historical_success_rate=0.8, business_impact_score=0.6,
        )
        plan = RepairPlan(goal="test", strategy=strategy, steps=[])
        ctx = _make_context()
        verification = self._make_verification(passed=True)
        learner.record(ctx, plan, verification, time.monotonic())

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT final_outcome FROM repair_memory ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert row[0] == "success"

    def test_final_outcome_failure_on_failed(self, db_path):
        """final_outcome = 'failure' when verification fails."""
        learner = Learner(db_path=db_path)
        strategy = StrategyScore(
            name="wait_and_retry", display_name="Wait",
            expected_success_prob=0.5, estimated_recovery_secs=60,
            risk_score=0.1, cost_score=0.1,
            historical_success_rate=0.5, business_impact_score=0.3,
        )
        plan = RepairPlan(goal="test", strategy=strategy, steps=[])
        ctx = _make_context()
        verification = self._make_verification(passed=False)
        learner.record(ctx, plan, verification, time.monotonic())

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT final_outcome FROM repair_memory ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert row[0] == "failure"

    def test_get_strategy_confidence_default(self, db_path):
        """With no history, get_strategy_confidence returns 0.5."""
        learner = Learner(db_path=db_path)
        conf = learner.get_strategy_confidence("missing_data", "MEDIUM", "switch_to_backup")
        assert conf == pytest.approx(0.5)

    def test_playbooks_updated_on_success(self, db_path):
        """Learner calls record_outcome to update the playbooks table."""
        learner = Learner(db_path=db_path)
        strategy = StrategyScore(
            name="wait_and_retry", display_name="Wait",
            expected_success_prob=0.8, estimated_recovery_secs=60,
            risk_score=0.1, cost_score=0.1,
            historical_success_rate=0.8, business_impact_score=0.5,
        )
        plan = RepairPlan(goal="test", strategy=strategy, steps=[])
        ctx = _make_context(anomaly_type="missing_data", severity="LOW")
        verification = self._make_verification(passed=True)
        learner.record(ctx, plan, verification, time.monotonic())

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT success_count FROM playbooks WHERE action_taken = 'wait_and_retry'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 8. Predictor
# ═══════════════════════════════════════════════════════════════════════════

class TestPredictor:
    """Predictor detects recurring failure patterns from incident history."""

    def _seed_incidents(
        self,
        db_path: str,
        count: int,
        anomaly_type: str = "missing_data",
        gap_minutes_start: float = 10.0,
        gap_step: float = 15.0,
    ) -> None:
        """Seed synthetic incident rows for pattern detection."""
        conn = sqlite3.connect(db_path)
        now = datetime.now(timezone.utc)
        for i in range(count):
            ts = (now - timedelta(hours=i * 2)).isoformat()
            gap = gap_minutes_start + i * gap_step
            conn.execute(
                """
                INSERT OR IGNORE INTO incidents
                (run_id, timestamp, anomaly_type, severity, gap_minutes,
                 root_cause, resolved, confidence)
                VALUES (?, ?, ?, 'MEDIUM', ?, 'test', 0, 0.8)
                """,
                (f"inc-{i:03d}", ts, anomaly_type, gap),
            )
        conn.commit()
        conn.close()

    def test_report_has_expected_fields(self, db_path):
        """PredictionReport has run_id, predicted_risk, patterns_detected."""
        predictor = Predictor(db_path=db_path)
        report = predictor.analyse("test-pred-001")
        assert report.run_id == "test-pred-001"
        assert 0.0 <= report.predicted_risk <= 1.0
        assert isinstance(report.patterns_detected, list)

    def test_no_history_produces_zero_risk(self, db_path):
        """With no incident history AND all nodes healthy, predicted_risk is 0.0."""
        # Patch node exhaustion to avoid false-positive from shared failover state
        with patch("agents.repair.predictor.Predictor._detect_node_exhaustion", return_value=None):
            predictor = Predictor(db_path=db_path)
            report = predictor.analyse("no-hist-001")
        assert report.predicted_risk == pytest.approx(0.0)

    def test_increasing_gap_trend_detected(self, db_path):
        """Increasing gap_minutes across incidents triggers gap_trend pattern."""
        self._seed_incidents(db_path, count=6, gap_minutes_start=10, gap_step=20)
        predictor = Predictor(db_path=db_path)
        report = predictor.analyse("trend-001", anomaly_type="missing_data")
        pattern_names = [p["name"] for p in report.patterns_detected]
        assert "increasing_gap_trend" in pattern_names

    def test_high_frequency_detected(self, db_path):
        """Many incidents in a short window triggers db_instability pattern."""
        # 20 incidents across 7 days = ~0.12/h — below threshold
        # Seed 100 incidents to exceed the threshold
        self._seed_incidents(db_path, count=100, gap_minutes_start=5, gap_step=0)
        predictor = Predictor(db_path=db_path)
        report = predictor.analyse("freq-001", anomaly_type="missing_data")
        pattern_names = [p["name"] for p in report.patterns_detected]
        assert "db_instability" in pattern_names

    def test_repeated_outages_detected(self, db_path):
        """Multiple same-type incidents triggers repeated_outages pattern."""
        self._seed_incidents(db_path, count=5, anomaly_type="data_quality")
        predictor = Predictor(db_path=db_path)
        report = predictor.analyse("repeat-001", anomaly_type="data_quality")
        pattern_names = [p["name"] for p in report.patterns_detected]
        assert "repeated_outages" in pattern_names

    def test_is_alert_property(self, db_path):
        """is_alert is True when predicted_risk >= 0.6."""
        self._seed_incidents(db_path, count=100, gap_minutes_start=30, gap_step=30)
        predictor = Predictor(db_path=db_path)
        report = predictor.analyse("alert-001", anomaly_type="missing_data")
        # is_alert must be consistent with predicted_risk
        assert report.is_alert == (report.predicted_risk >= 0.6)


# ═══════════════════════════════════════════════════════════════════════════
# 9. RepairerAgent end-to-end (backward-compatibility contract)
# ═══════════════════════════════════════════════════════════════════════════

class TestRepairerAgentEndToEnd:
    """End-to-end tests; preserve original test contract exactly."""

    # ── Skip paths ───────────────────────────────────────────────────────

    def test_no_anomaly_skips_repair(self, agent):
        """When anomaly_detected=False, repairer_output is empty dict."""
        state = _make_state(anomaly_detected=False)
        result = agent.run(state)
        assert result["repairer_output"] == {}

    def test_low_confidence_skips_repair(self, agent):
        """Confidence < 0.6 → action_taken='skipped_low_confidence'."""
        state = _make_state(confidence=0.45)
        result = agent.run(state)
        out = result["repairer_output"]
        assert out["action_taken"] == "skipped_low_confidence"
        assert out["success"] is False
        assert out["rows_affected"] == 0
        assert "0.45" in out["details"]

    # ── Output shape ─────────────────────────────────────────────────────

    def test_output_has_required_keys(self, agent):
        """repairer_output always has action_taken, success, rows_affected, details."""
        state = _make_state()
        result = agent.run(state)
        out = result["repairer_output"]
        assert "action_taken" in out
        assert "success" in out
        assert "rows_affected" in out
        assert "details" in out

    def test_new_enriched_keys_present_on_success_or_failure(self, agent):
        """New enriched keys are additive: repair_confidence, risk_score, etc."""
        state = _make_state(severity="LOW", gap_minutes=10, confidence=0.75)
        result = agent.run(state)
        out = result["repairer_output"]
        # These keys should be present regardless of success/failure
        if out["action_taken"] not in ("skipped_low_confidence",):
            assert "repair_confidence" in out
            assert "risk_score" in out
            assert "verification_score" in out

    # ── Strategy correctness ──────────────────────────────────────────────

    def test_low_severity_missing_data_uses_wait_or_similar(self, agent, db_path):
        """LOW severity missing_data attempts wait_and_retry as first strategy.

        Even if verification fails and strategies exhaust, wait_and_retry
        must have been the first strategy attempted (recorded in playbooks).
        """
        state = _make_state(severity="LOW", gap_minutes=10, confidence=0.75)
        agent.run(state)

        # Verify wait_and_retry was recorded in playbooks (i.e., it was attempted)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT action_taken FROM playbooks WHERE action_taken = 'wait_and_retry'"
        ).fetchone()
        conn.close()
        assert row is not None, (
            "wait_and_retry should be attempted first for LOW severity missing_data"
        )

    def test_critical_severity_escalates(self, agent):
        """CRITICAL severity should result in escalate_to_human action."""
        state = _make_state(severity="CRITICAL", gap_minutes=500, confidence=0.95)
        result = agent.run(state)
        out = result["repairer_output"]
        assert out["action_taken"] == "escalate_to_human"
        assert out["success"] is False

    def test_quarantine_for_data_quality(self, agent, db_path):
        """HIGH severity data_quality executes quarantine as first strategy.

        The agent always tries quarantine_bad_data first for data_quality/HIGH.
        Even if verification scores below 0.7 and it retries, quarantine rows
        should have been moved — we verify at the DB level.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        conn = sqlite3.connect(db_path)
        for i in range(5):
            created = (now - timedelta(minutes=10 - i)).isoformat()
            conn.execute(
                "INSERT INTO orders (order_id, created_at, order_amount, source_db) "
                "VALUES (?, ?, NULL, 'mumbai')",
                (f"qnull-{i:03d}", created),
            )
        conn.commit()
        conn.close()

        state = _make_state(
            anomaly_type="data_quality",
            severity="HIGH",
            confidence=0.90,
            gap_minutes=0.0,
        )
        state["null_rate"] = 0.25
        with patch("agents.repair.verifier.Verifier.verify") as mock_verify:
            from agents.repair.verifier import VerificationResult
            mock_verify.return_value = VerificationResult(
                anomaly_removed=True,
                rows_recovered=5,
                duplicates_found=0,
                latency_acceptable=True,
                pipeline_healthy=True,
                downstream_affected=False,
                verification_score=0.9,
                details="Mock",
                repair_confidence=0.9,
                repair_quality_score=1.0,
                data_integrity_score=1.0,
                recovery_score=1.0,
                business_impact_score=0.0,
                residual_risk=0.0
            )
            result = agent.run(state)
        out = result["repairer_output"]

        # The agent must have *attempted* quarantine_bad_data (first strategy)
        # regardless of what the final action_taken is after retries.
        # Verify by checking the quarantine_orders table was populated.
        conn = sqlite3.connect(db_path)
        quarantined = conn.execute(
            "SELECT COUNT(*) FROM quarantine_orders"
        ).fetchone()[0]
        conn.close()
        assert quarantined == 5, (
            "quarantine_bad_data should have been executed as first strategy "
            f"but quarantine_orders has {quarantined} rows"
        )
        # Also assert the agent did not escalate to human directly
        assert out["action_taken"] != "escalate_to_human"

    # ── State integrity ──────────────────────────────────────────────────

    def test_state_returned_with_all_original_keys(self, agent):
        """The returned state still contains all original PipelineState keys."""
        state = _make_state()
        result = agent.run(state)
        required_keys = [
            "run_id", "timestamp", "anomaly_detected", "anomaly_type",
            "severity", "gap_minutes", "affected_tables", "raw_count",
            "expected_avg", "null_rate", "diagnoser_output", "repairer_output",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_repairer_output_not_empty_on_anomaly(self, agent):
        """With an anomaly, repairer_output is never an empty dict."""
        state = _make_state()
        result = agent.run(state)
        assert result["repairer_output"] != {}

    # ── Playbook learning ────────────────────────────────────────────────

    def test_outcome_recorded_in_playbooks(self, agent, db_path):
        """After a repair, at least one playbook entry is created."""
        state = _make_state(severity="LOW", gap_minutes=10, confidence=0.75)
        agent.run(state)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM playbooks").fetchone()[0]
        conn.close()
        assert count >= 1

    def test_repair_memory_entry_created(self, agent, db_path):
        """After a repair, at least one repair_memory row is inserted."""
        state = _make_state()
        agent.run(state)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM repair_memory").fetchone()[0]
        conn.close()
        assert count >= 1
