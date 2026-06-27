"""Prompt templates for the Diagnoser Agent.

Separates prompt engineering from agent logic for maintainability.
The LLM is instructed to respond ONLY in valid JSON.
"""

from typing import Any, Dict, List


SYSTEM_PROMPT: str = """You are a senior data engineer specializing in pipeline reliability and incident response.

Your task: Analyze the current pipeline anomaly and determine the root cause.

CONTEXT:
- You monitor SQL data pipelines that ingest e-commerce order data
- Pipelines ingest from regional databases (e.g., Mumbai, Delhi) into a data warehouse
- Common failure modes: source DB outage, schema changes, data quality degradation, network issues, maintenance windows

RULES:
1. You MUST respond ONLY in valid JSON — no markdown, no explanation, no preamble
2. Base your analysis on the anomaly details AND any similar past incidents provided
3. If past incidents show a recurring pattern, factor that into your confidence score
4. Be conservative with confidence — only use > 0.8 if evidence is very strong
5. Never hallucinate data or make up specific numbers not supported by the input

RESPONSE FORMAT (strict JSON):
{
    "root_cause": "Brief description of the most likely root cause",
    "confidence": 0.0 to 1.0,
    "estimated_missing_rows": integer,
    "affected_tables": ["table1", "table2"],
    "maintenance_window_likely": true/false
}"""


def build_user_prompt(
    state: Dict[str, Any], past_incidents: List[Dict[str, Any]]
) -> str:
    """Build the user prompt with current anomaly state and past incidents.

    Args:
        state: Current pipeline state dict from the Monitor Agent.
        past_incidents: List of similar past incidents from episodic LTM.

    Returns:
        Formatted prompt string for the LLM.
    """
    # Current anomaly section
    prompt = f"""CURRENT ANOMALY DETECTED:
- Type: {state.get('anomaly_type', 'unknown')}
- Severity: {state.get('severity', 'NONE')}
- Current row count (last 5 min): {state.get('raw_count', 0)}
- Expected baseline average: {state.get('expected_avg', 0):.1f}
- Gap duration: {state.get('gap_minutes', 0):.1f} minutes
- Null rate on order_amount: {state.get('null_rate', 0):.2%}
- Affected tables: {state.get('affected_tables', [])}
- Timestamp: {state.get('timestamp', 'unknown')}
"""

    # Past incidents section
    if past_incidents:
        prompt += f"\nSIMILAR PAST INCIDENTS ({len(past_incidents)} found):\n"
        for i, incident in enumerate(past_incidents, 1):
            prompt += f"""
Incident #{i}:
  - Time: {incident.get('timestamp', 'N/A')}
  - Severity: {incident.get('severity', 'N/A')}
  - Root cause: {incident.get('root_cause', 'N/A')}
  - Gap: {incident.get('gap_minutes', 0):.0f} min
  - Fix applied: {incident.get('fix_taken', 'N/A')}
  - Resolved: {'Yes' if incident.get('resolved') else 'No'}
"""
    else:
        prompt += "\nNo similar past incidents found — this may be a new type of failure.\n"

    prompt += "\nAnalyze this anomaly and respond with the JSON diagnosis."

    return prompt


# Default diagnoser output when LLM is unavailable or in mock mode
MOCK_DIAGNOSER_OUTPUT: Dict[str, Any] = {
    "root_cause": "Source database connectivity issue (mock diagnosis)",
    "confidence": 0.75,
    "estimated_missing_rows": 0,
    "affected_tables": ["orders"],
    "maintenance_window_likely": False,
}
