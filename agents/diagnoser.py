"""Diagnoser Agent: LLM-powered root cause analysis.

Uses Groq's free llama3-8b model to reason about pipeline anomalies.
Retrieves similar past incidents from episodic LTM to improve reasoning.
Falls back to mock output if the LLM call fails or MOCK_MODE is enabled.
"""

import json
from typing import Any, Dict

from config import CONFIDENCE_MIN, GROQ_API_KEY, GROQ_MODEL, MOCK_MODE
from logging_config import logger
from memory.incident_store import get_similar_incidents
from prompts.diagnoser_prompt import (
    MOCK_DIAGNOSER_OUTPUT,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from state import PipelineState


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
        """Lazily initialize the Groq LLM client.

        Returns:
            ChatGroq instance configured with the model from config.
        """
        if self._llm is None:
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(
                model=GROQ_MODEL,
                temperature=0,
                api_key=GROQ_API_KEY,
            )
        return self._llm

    def _parse_llm_response(self, response_text: str, state: PipelineState) -> Dict[str, Any]:
        """Parse and validate the LLM's JSON response.

        Args:
            response_text: Raw text response from the LLM.
            state: Current pipeline state for fallback values.

        Returns:
            Parsed diagnosis dict with validated fields.
        """
        # Try to extract JSON from the response
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            return self._build_fallback_output(state)

        # Validate and normalize fields
        return {
            "root_cause": str(result.get("root_cause", "Unknown")),
            "confidence": float(min(max(result.get("confidence", 0.5), 0.0), 1.0)),
            "estimated_missing_rows": int(result.get("estimated_missing_rows", 0)),
            "affected_tables": list(result.get("affected_tables", state.get("affected_tables", []))),
            "maintenance_window_likely": bool(result.get("maintenance_window_likely", False)),
        }

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
                logger.info(f"[{run_id}] Calling Groq LLM ({GROQ_MODEL})...")

                from langchain_core.messages import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]

                response = llm.invoke(messages)
                response_text = response.content

                logger.info(f"[{run_id}] LLM response received ({len(response_text)} chars)")
                diagnoser_output = self._parse_llm_response(response_text, state)

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
