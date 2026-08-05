"""Diagnoser Agent: LLM-powered root cause analysis.

Uses Groq's free llama3-8b model to reason about pipeline anomalies.
Retrieves similar past incidents from episodic LTM to improve reasoning.
Falls back to mock output if the LLM call fails or MOCK_MODE is enabled.
"""

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field
import instructor

from config import CONFIDENCE_MIN, GROQ_API_KEY, GROQ_MODEL, MOCK_MODE
from logging_config import logger
from memory.incident_store import get_similar_incidents
from prompts.diagnoser_prompt import (
    MOCK_DIAGNOSER_OUTPUT,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from state import PipelineState


LLM_LOG_MAX_CHARS = 4000


class DiagnosisOutput(BaseModel):
    """Structured LLM output for diagnosis."""
    root_cause: str = Field(..., description="The likely root cause of the anomaly")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    estimated_missing_rows: int = Field(..., description="Estimated number of missing rows")
    affected_tables: List[str] = Field(..., description="List of affected tables")
    maintenance_window_likely: bool = Field(..., description="Is this likely a planned maintenance window?")


def _format_llm_log_text(text: str) -> str:
    """Keep LLM request/response logs readable while preserving useful context."""
    if len(text) <= LLM_LOG_MAX_CHARS:
        return text
    return f"{text[:LLM_LOG_MAX_CHARS]}... [truncated {len(text) - LLM_LOG_MAX_CHARS} chars]"


class DiagnoserAgent:
    """Diagnoses root causes of pipeline anomalies using LLM reasoning.

    The Diagnoser:
    1. Retrieves similar past incidents from episodic LTM
    2. Builds a context-rich prompt with current state + history
    3. Calls Groq LLM for structured JSON root-cause analysis
    4. Parses and validates the response
    5. Updates state with diagnosis results

    Falls back to a default diagnosis if LLM is unavailable.
    """

    def __init__(self) -> None:
        """Initialize DiagnoserAgent."""
        self._llm = None

    def _get_llm(self) -> Any:
        """Lazily initialize the Groq client patched with instructor.

        Returns:
            Instructor-patched Groq client.
        """
        if self._llm is None:
            from groq import Groq

            client = Groq(api_key=GROQ_API_KEY)
            self._llm = instructor.from_groq(client, mode=instructor.Mode.TOOLS)
        return self._llm

    def _build_fallback_output(self, state: PipelineState) -> Dict[str, Any]:
        """Build a fallback diagnosis when LLM is unavailable.

        Args:
            state: Current pipeline state.

        Returns:
            Default diagnosis dict.
        """
        gap = state.get("gap_minutes", 0)
        rows_per_min = state.get("expected_avg", 200) / 5  # baseline per minute

        return {
            "root_cause": f"Automated detection: {state.get('anomaly_type', 'unknown')} anomaly",
            "confidence": 0.5,
            "estimated_missing_rows": int(gap * rows_per_min) if gap > 0 else 0,
            "affected_tables": state.get("affected_tables", []),
            "maintenance_window_likely": False,
        }

    def run(self, state: PipelineState) -> PipelineState:
        """Execute root cause diagnosis on the detected anomaly.

        Steps:
        1. Skip if no anomaly detected
        2. Retrieve similar past incidents from episodic LTM
        3. Build prompt with current state + history
        4. Call Groq LLM (or use mock in MOCK_MODE)
        5. Parse and validate JSON response
        6. Update severity if confidence is high
        7. Write diagnosis to state

        Args:
            state: Current pipeline state from Monitor Agent.

        Returns:
            Updated state with diagnoser_output populated.
        """
        run_id = state["run_id"]

        # Skip if no anomaly
        if not state.get("anomaly_detected", False):
            logger.info(f"[{run_id}] Diagnoser: No anomaly — skipping diagnosis")
            state["diagnoser_output"] = {}
            return state

        logger.info(
            f"[{run_id}] Diagnoser Agent starting analysis: "
            f"{state['anomaly_type']} ({state['severity']})"
        )

        # Step 1: Retrieve similar past incidents from episodic LTM
        past_incidents = get_similar_incidents(state["anomaly_type"], limit=5)
        logger.info(
            f"[{run_id}] Found {len(past_incidents)} similar past incidents"
        )

        # Step 2: Build prompt
        user_prompt = build_user_prompt(state, past_incidents)

        # Step 3: Call LLM or use mock
        if MOCK_MODE:
            logger.info(f"[{run_id}] MOCK_MODE: Using mock diagnoser output")
            # Compute realistic mock values from state
            gap = state.get("gap_minutes", 0)
            rows_per_min = state.get("expected_avg", 200) / 5
            diagnoser_output = MOCK_DIAGNOSER_OUTPUT.copy()
            diagnoser_output["estimated_missing_rows"] = int(gap * rows_per_min)
            diagnoser_output["affected_tables"] = state.get("affected_tables", ["orders"])
        else:
            try:
                llm = self._get_llm()
                logger.info(f"[{run_id}] Calling Groq LLM ({GROQ_MODEL}) with Instructor...")

                response = llm.chat.completions.create(
                    model=GROQ_MODEL,
                    response_model=DiagnosisOutput,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_retries=3,
                )

                logger.info(f"[{run_id}] LLM structured response received successfully.")
                diagnoser_output = response.model_dump()

            except Exception as e:
                logger.error(
                    f"[{run_id}] LLM call failed: {e}. Using fallback diagnosis."
                )
                diagnoser_output = self._build_fallback_output(state)

        # Step 4: Re-assign severity if confidence is high
        if diagnoser_output.get("confidence", 0) > 0.8:
            old_severity = state["severity"]
            # High-confidence diagnosis can upgrade severity
            if state["anomaly_type"] == "missing_data" and state["gap_minutes"] > 30:
                state["severity"] = "HIGH"
            if old_severity != state["severity"]:
                logger.info(
                    f"[{run_id}] Severity upgraded: {old_severity} → {state['severity']} "
                    f"(confidence: {diagnoser_output['confidence']:.2f})"
                )

        # Step 5: Write to state
        state["diagnoser_output"] = diagnoser_output

        logger.info(
            f"[{run_id}] Diagnosis complete: "
            f"cause='{diagnoser_output['root_cause']}', "
            f"confidence={diagnoser_output['confidence']:.2f}, "
            f"est_missing={diagnoser_output['estimated_missing_rows']}"
        )

        # Step 6: Pre-save incident to DB so repairer can resolve it later
        try:
            from memory.incident_store import insert_incident
            insert_incident({
                "run_id": run_id,
                "timestamp": state.get("timestamp", ""),
                "anomaly_type": state.get("anomaly_type", ""),
                "severity": state.get("severity", "NONE"),
                "gap_minutes": state.get("gap_minutes", 0.0),
                "root_cause": diagnoser_output.get("root_cause", ""),
                "affected_tables": diagnoser_output.get("affected_tables", state.get("affected_tables", [])),
                "fix_taken": "",
                "resolved": 0,
                "confidence": diagnoser_output.get("confidence", 0.0),
            })
            state["incident_saved"] = True
        except Exception as e:
            logger.warning(f"[{run_id}] Could not pre-save incident: {e}")
            state["incident_saved"] = False

        return state
