"""LangGraph state graph definition for the pipeline monitor.

Phase 4: Four-node graph with conditional routing.
- monitor_node → (anomaly?) → diagnose_node → repair_node → slack_node → END
- monitor_node → (no anomaly) → END
"""

from langgraph.graph import END, StateGraph

from agents.diagnoser import DiagnoserAgent
from agents.monitor import MonitorAgent
from agents.repairer import RepairerAgent
from agents.slack_agent import SlackAgent
from logging_config import logger
from state import PipelineState


def monitor_node(state: PipelineState) -> PipelineState:
    """LangGraph node: runs the Monitor Agent.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state after monitoring check.
    """
    agent = MonitorAgent()
    return agent.run(state)


def diagnose_node(state: PipelineState) -> PipelineState:
    """LangGraph node: runs the Diagnoser Agent.

    Args:
        state: Current pipeline state with anomaly detected.

    Returns:
        Updated state with root cause diagnosis.
    """
    agent = DiagnoserAgent()
    return agent.run(state)


def repair_node(state: PipelineState) -> PipelineState:
    """LangGraph node: runs the Repairer Agent.

    Args:
        state: Current pipeline state with diagnosis complete.

    Returns:
        Updated state with repair results.
    """
    agent = RepairerAgent()
    return agent.run(state)


def slack_node(state: PipelineState) -> PipelineState:
    """LangGraph node: runs the Slack Agent.

    Args:
        state: Current pipeline state with repair complete.

    Returns:
        Updated state with Slack notification result.
    """
    agent = SlackAgent()
    return agent.run(state)


def route_after_monitor(state: PipelineState) -> str:
    """Conditional edge: route based on anomaly detection.

    Args:
        state: Current pipeline state after monitoring.

    Returns:
        'diagnose' if anomaly detected, 'end' otherwise.
    """
    if state.get("anomaly_detected", False):
        return "diagnose"
    return "end"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine.

    Phase 4 graph:
        START → monitor_node → [anomaly?] → diagnose_node → repair_node → slack_node → END
                              → [no anomaly] → END

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("monitor_node", monitor_node)
    graph.add_node("diagnose_node", diagnose_node)
    graph.add_node("repair_node", repair_node)
    graph.add_node("slack_node", slack_node)

    # Set entry point
    graph.set_entry_point("monitor_node")

    # Conditional edge after monitor
    graph.add_conditional_edges(
        "monitor_node",
        route_after_monitor,
        {
            "diagnose": "diagnose_node",
            "end": END,
        },
    )

    # Diagnose → Repair → END
    graph.add_edge("diagnose_node", "repair_node")
    graph.add_edge("repair_node", "slack_node")
    graph.add_edge("slack_node", END)

    logger.info(
        "LangGraph built: START → monitor → [anomaly?] → diagnose → repair → slack → END"
    )

    return graph.compile()


# Pre-built graph instance for import
pipeline_graph = build_graph()
