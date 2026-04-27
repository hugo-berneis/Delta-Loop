"""Unit tests for Phase 2 agent — all LLM calls mocked."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from deltaloop.agent.graph import _route_after_plan, _route_after_reason, build_graph
from deltaloop.agent.nodes import plan, reason, synthesize, validate
from deltaloop.agent.state import AgentState
from deltaloop.agent.tools import _extract_html_text, search_document
from deltaloop.storage.models import BenchmarkTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task(context_type: str = "html", task_id: str = "webn_001") -> BenchmarkTask:
    return BenchmarkTask(
        id=task_id,
        category="web_navigation",
        question="What is the main heading?",
        context_type=context_type,
        context_path=None,
        ground_truth="Annual Climate Report 2023",
        scoring_rubric="exact_match",
        difficulty="easy",
    )


def make_state(context_type: str = "html", **overrides) -> AgentState:
    base: AgentState = {
        "task": make_task(context_type),
        "iteration": 1,
        "reasoning_steps": [],
        "tool_calls": [],
        "retrieved_context": "",
        "multimodal_output": "",
        "final_answer": "",
        "is_complete": False,
        "error": None,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def test_route_after_plan_image():
    state = make_state(context_type="image")
    assert _route_after_plan(state) == "analyze_multimodal"


def test_route_after_plan_document():
    state = make_state(context_type="document")
    assert _route_after_plan(state) == "retrieve_context"


def test_route_after_plan_html():
    state = make_state(context_type="html")
    assert _route_after_plan(state) == "reason"


def test_route_after_reason_with_pending_tool():
    state = make_state(tool_calls=[{"name": "search_document", "args": {}, "pending": True}])
    assert _route_after_reason(state) == "call_tool"


def test_route_after_reason_ready():
    state = make_state(tool_calls=[{"ready": True}])
    assert _route_after_reason(state) == "synthesize"


def test_route_after_reason_empty_calls():
    state = make_state(tool_calls=[])
    assert _route_after_reason(state) == "synthesize"


def test_route_after_reason_completed_tool():
    state = make_state(tool_calls=[{"name": "search_document", "pending": False, "result": "..."}])
    assert _route_after_reason(state) == "synthesize"


# ---------------------------------------------------------------------------
# plan node
# ---------------------------------------------------------------------------

async def test_plan_node_returns_reasoning_step():
    plan_response = json.dumps({
        "steps": [
            {"step": 1, "action": "Read the HTML page", "rationale": "Need to see content"},
            {"step": 2, "action": "Find the main heading", "rationale": "That is the question"},
        ]
    })
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=plan_response)

    with patch("deltaloop.agent.nodes.get_ollama_client", return_value=mock_client):
        result = await plan(make_state())

    assert "reasoning_steps" in result
    assert len(result["reasoning_steps"]) == 1
    assert "Step 1" in result["reasoning_steps"][0]


async def test_plan_node_handles_malformed_json():
    mock_client = AsyncMock()
    # First call returns garbage, second returns valid JSON
    mock_client.complete = AsyncMock(side_effect=[
        "Here is my plan: step 1 do stuff",
        json.dumps({"steps": [{"step": 1, "action": "Do task", "rationale": "Because"}]}),
    ])

    with patch("deltaloop.agent.nodes.get_ollama_client", return_value=mock_client):
        result = await plan(make_state())

    assert mock_client.complete.call_count == 2
    assert "reasoning_steps" in result


async def test_plan_node_uses_fallback_on_double_failure():
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value="NOT JSON AT ALL")

    with patch("deltaloop.agent.nodes.get_ollama_client", return_value=mock_client):
        result = await plan(make_state())

    assert "reasoning_steps" in result
    assert len(result["reasoning_steps"]) == 1  # fallback plan step


# ---------------------------------------------------------------------------
# reason node
# ---------------------------------------------------------------------------

async def test_reason_node_requests_tool():
    reason_response = json.dumps({
        "next_step": "I need to search the document for the heading",
        "needs_tool": True,
        "tool_name": "search_document",
        "tool_args": {"query": "main heading"},
        "ready_to_answer": False,
    })
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=reason_response)

    with patch("deltaloop.agent.nodes.get_ollama_client", return_value=mock_client):
        result = await reason(make_state())

    assert result["tool_calls"][-1]["pending"] is True
    assert result["tool_calls"][-1]["name"] == "search_document"
    assert "search" in result["reasoning_steps"][-1].lower()


async def test_reason_node_ready_to_answer():
    reason_response = json.dumps({
        "next_step": "I have enough information to answer.",
        "needs_tool": False,
        "tool_name": None,
        "tool_args": {},
        "ready_to_answer": True,
    })
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=reason_response)

    with patch("deltaloop.agent.nodes.get_ollama_client", return_value=mock_client):
        result = await reason(make_state())

    assert result["tool_calls"][-1].get("ready") is True


async def test_reason_node_ignores_unknown_tool():
    reason_response = json.dumps({
        "next_step": "Call a magic tool",
        "needs_tool": True,
        "tool_name": "nonexistent_tool",
        "tool_args": {},
        "ready_to_answer": False,
    })
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=reason_response)

    with patch("deltaloop.agent.nodes.get_ollama_client", return_value=mock_client):
        result = await reason(make_state())

    # Unknown tool → treated as ready (falls through to synthesize)
    assert result["tool_calls"][-1].get("ready") is True


# ---------------------------------------------------------------------------
# synthesize node
# ---------------------------------------------------------------------------

async def test_synthesize_node_writes_final_answer():
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value="Annual Climate Report 2023")

    state = make_state(reasoning_steps=["Step 1: Read the page"])
    with patch("deltaloop.agent.nodes.get_ollama_client", return_value=mock_client):
        result = await synthesize(state)

    assert result["final_answer"] == "Annual Climate Report 2023"


# ---------------------------------------------------------------------------
# validate node
# ---------------------------------------------------------------------------

async def test_validate_accepts_good_answer():
    state = make_state(final_answer="Annual Climate Report 2023")
    result = await validate(state)
    assert result["is_complete"] is True
    assert result["error"] is None


async def test_validate_rejects_empty_answer():
    state = make_state(final_answer="")
    result = await validate(state)
    assert result["is_complete"] is False
    assert result["error"] is not None


async def test_validate_rejects_refusal():
    state = make_state(final_answer="I cannot answer this question.")
    result = await validate(state)
    assert result["is_complete"] is False


async def test_validate_rejects_i_dont_know():
    state = make_state(final_answer="I don't know the answer.")
    result = await validate(state)
    assert result["is_complete"] is False


# ---------------------------------------------------------------------------
# tools — search_document
# ---------------------------------------------------------------------------

async def test_search_document_finds_match():
    state = make_state(
        retrieved_context="The report was published in 2023. Solar energy grew significantly."
    )
    result = await search_document(state, {"query": "solar energy"})
    assert "Solar energy" in result


async def test_search_document_returns_excerpt_on_no_match():
    state = make_state(retrieved_context="A" * 3000)
    result = await search_document(state, {"query": "quantum entanglement"})
    assert len(result) <= 2000


async def test_search_document_empty_context():
    state = make_state(retrieved_context="")
    result = await search_document(state, {"query": "anything"})
    assert "No document context" in result


# ---------------------------------------------------------------------------
# tools — _extract_html_text
# ---------------------------------------------------------------------------

def test_extract_html_text_basic():
    html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
    text = _extract_html_text(html)
    assert "Hello" in text
    assert "World" in text


def test_extract_html_text_strips_scripts():
    html = "<html><body><script>var x = 1;</script><p>Content</p></body></html>"
    text = _extract_html_text(html)
    assert "var x" not in text
    assert "Content" in text


def test_extract_html_text_strips_style():
    html = "<html><head><style>body { color: red; }</style></head><body><p>Text</p></body></html>"
    text = _extract_html_text(html)
    assert "color" not in text
    assert "Text" in text


# ---------------------------------------------------------------------------
# graph structure
# ---------------------------------------------------------------------------

def test_build_graph_compiles_without_error():
    graph = build_graph()
    assert graph is not None
