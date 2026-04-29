from langgraph.graph import END, START, StateGraph

from deltaloop.agent import nodes
from deltaloop.agent.state import AgentState


def _route_after_plan(state: AgentState) -> str:
    context_type = state["task"].context_type
    if context_type == "image":
        return "analyze_multimodal"
    if context_type == "document":
        return "retrieve_context"
    return "reason"


def _route_after_context(state: AgentState) -> str:
    return "reason"


_MAX_TOOL_CALLS = 4


def _route_after_reason(state: AgentState) -> str:
    tool_calls = state.get("tool_calls", [])
    completed = sum(1 for tc in tool_calls if not tc.get("pending") and "name" in tc)
    if completed >= _MAX_TOOL_CALLS:
        return "synthesize"
    if tool_calls and tool_calls[-1].get("pending"):
        return "call_tool"
    return "synthesize"


def build_graph() -> StateGraph:
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("plan", nodes.plan)
    graph.add_node("retrieve_context", nodes.retrieve_context)
    graph.add_node("analyze_multimodal", nodes.analyze_multimodal)
    graph.add_node("reason", nodes.reason)
    graph.add_node("call_tool", nodes.call_tool)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("validate", nodes.validate)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _route_after_plan)
    graph.add_edge("retrieve_context", "reason")
    graph.add_edge("analyze_multimodal", "reason")
    graph.add_conditional_edges("reason", _route_after_reason)
    graph.add_edge("call_tool", "reason")
    graph.add_edge("synthesize", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


# Compiled graph singleton — import and call ainvoke() directly
agent_graph = build_graph()


async def run_agent(
    task,  # BenchmarkTask
    iteration: int,
) -> AgentState:
    """Run the agent graph on a single task and return the final state."""
    initial_state: AgentState = {
        "task": task,
        "iteration": iteration,
        "reasoning_steps": [],
        "tool_calls": [],
        "retrieved_context": "",
        "multimodal_output": "",
        "final_answer": "",
        "is_complete": False,
        "error": None,
    }
    result: AgentState = await agent_graph.ainvoke(  # type: ignore[assignment]
        initial_state,
        config={"recursion_limit": 10},
    )
    return result
