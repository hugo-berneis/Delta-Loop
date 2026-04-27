"""Integration test: run one task end-to-end through the agent graph.

Skipped automatically if Ollama is not reachable.
Requires: ollama pull llama3.1:8b && ollama pull llava:7b
"""
import json

import httpx
import pytest

from deltaloop.agent.graph import run_agent
from deltaloop.storage.models import BenchmarkTask


async def _ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not pytest.importorskip("httpx"),
    reason="httpx not available",
)


@pytest.fixture
async def ollama_required():
    if not await _ollama_available():
        pytest.skip("Ollama not reachable — skipping integration test")


async def test_agent_runs_html_task_end_to_end(ollama_required):
    """Run a web_navigation task through the full graph."""
    task = BenchmarkTask(
        id="webn_test_001",
        category="web_navigation",
        question="What is the main topic of this page?",
        context_type="html",
        context_path=None,  # no file — agent will gracefully handle missing asset
        ground_truth="Energy research",
        scoring_rubric="partial_match",
        difficulty="easy",
    )

    state = await run_agent(task, iteration=0)

    assert state["final_answer"], "final_answer must be non-empty"
    assert isinstance(state["reasoning_steps"], list)
    assert len(state["reasoning_steps"]) >= 1
    assert isinstance(state["tool_calls"], list)
    # validate node must have run
    assert "is_complete" in state


async def test_agent_graph_terminates_without_infinite_loop(ollama_required):
    """Verify recursion limit prevents infinite tool-calling loops."""
    task = BenchmarkTask(
        id="webn_test_002",
        category="web_navigation",
        question="Count the number of links on the page.",
        context_type="html",
        context_path=None,
        ground_truth="5",
        scoring_rubric="exact_match",
        difficulty="medium",
    )

    # Should complete (not hang), even if it hits recursion limit
    try:
        state = await run_agent(task, iteration=0)
        assert state is not None
    except Exception as exc:
        # GraphRecursionError is acceptable — it means the limit worked
        assert "recursion" in str(exc).lower(), f"Unexpected error: {exc}"
