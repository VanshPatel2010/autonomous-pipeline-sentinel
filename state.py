"""LangGraph state TypedDict for the pipeline monitor.

This is the STM (Short-Term Memory) shared across all agents per run.
State dict keys must NEVER be renamed after introduction — all 5 phases
read the same TypedDict.
"""

from typing import Any, Dict, List, TypedDict


class PipelineState(TypedDict):
    """Shared state passed through the LangGraph agent pipeline.

    Every agent reads from and writes to this state dict.
    Reset after each complete run (Slack agent fires).
    """

    # --- Run metadata ---
    run_id: str
    timestamp: str

    # --- Monitor Agent outputs ---
    anomaly_detected: bool
    anomaly_type: str  # 'missing_data', 'data_quality', 'schema_drift', ''
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'NONE'
    gap_minutes: float
    affected_tables: List[str]
    raw_count: int
    expected_avg: float
    null_rate: float

    # --- Diagnoser Agent outputs ---
    diagnoser_output: Dict[str, Any]

    # --- Repairer Agent outputs ---
    repairer_output: Dict[str, Any]

    # --- Slack Agent outputs ---
    slack_sent: bool


def create_initial_state(run_id: str, timestamp: str) -> PipelineState:
    """Create a fresh state dict with all defaults for a new run.

    Args:
        run_id: Unique identifier for this pipeline run.
        timestamp: ISO-format UTC timestamp of run start.

    Returns:
        Initialized PipelineState with default values.
    """
    return PipelineState(
        run_id=run_id,
        timestamp=timestamp,
        anomaly_detected=False,
        anomaly_type="",
        severity="NONE",
        gap_minutes=0.0,
        affected_tables=[],
        raw_count=0,
        expected_avg=0.0,
        null_rate=0.0,
        diagnoser_output={},
        repairer_output={},
        slack_sent=False,
    )
