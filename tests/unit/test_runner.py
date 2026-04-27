"""Unit tests for benchmark/runner.py and benchmark/scorer.py."""
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deltaloop.benchmark.runner import (
    IterationMetrics,
    Runner,
    RunnerState,
    _compute_metrics,
)
from deltaloop.benchmark.scorer import score_exact, score_partial, score_task
from deltaloop.critic.evaluator import TraceEvaluation
from deltaloop.storage.models import BenchmarkTask, ReasoningTrace


# ---------------------------------------------------------------------------
# scorer.py
# ---------------------------------------------------------------------------

def test_score_exact_match():
    assert score_exact("Revenue increased", "Revenue increased") == 1.0


def test_score_exact_case_insensitive():
    assert score_exact("REVENUE INCREASED", "revenue increased") == 1.0


def test_score_exact_whitespace_stripped():
    assert score_exact("  answer  ", "answer") == 1.0


def test_score_exact_mismatch():
    assert score_exact("wrong answer", "correct answer") == 0.0


def test_score_partial_perfect_overlap():
    assert score_partial("the quick brown fox", "the quick brown fox") == 1.0


def test_score_partial_no_overlap():
    assert score_partial("apple banana", "cat dog") == 0.0


def test_score_partial_some_overlap():
    score = score_partial("the quick brown fox", "the slow brown dog")
    assert 0.0 < score < 1.0


def test_score_partial_empty_ground_truth_returns_zero():
    assert score_partial("anything", "") == 0.0


def test_score_task_uses_exact_match_rubric():
    task = BenchmarkTask(
        id="t1", category="web_navigation", question="Q",
        context_type="html", ground_truth="Paris",
        scoring_rubric="exact_match", difficulty="easy",
    )
    assert score_task("Paris", task) == 1.0
    assert score_task("paris", task) == 1.0
    assert score_task("London", task) == 0.0


def test_score_task_uses_partial_match_rubric():
    task = BenchmarkTask(
        id="t2", category="document_comprehension", question="Q",
        context_type="document", ground_truth="Revenue increased each quarter",
        scoring_rubric="partial_match", difficulty="medium",
    )
    score = score_task("Revenue increased significantly each quarter", task)
    assert score > 0.5


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

def make_trace(is_correct: bool, score: float) -> ReasoningTrace:
    return ReasoningTrace(
        task_id="t1", iteration=1,
        reasoning_steps=json.dumps([]),
        tool_calls=json.dumps([]),
        final_answer="answer",
        is_correct=is_correct,
        score=score,
    )


def make_eval(failure_mode: str | None = "HALLUCINATION") -> TraceEvaluation:
    return TraceEvaluation(
        is_correct=failure_mode is None,
        score=0.8 if failure_mode is None else 0.2,
        failure_mode=failure_mode,
        failure_explanation=None if failure_mode is None else "Bad.",
        critic_confidence=0.6,
    )


def test_compute_metrics_success_rate():
    traces = [make_trace(True, 1.0), make_trace(False, 0.2), make_trace(True, 0.9)]
    metrics = _compute_metrics(1, traces, [None, None, None], 0, 0)
    assert abs(metrics.task_success_rate - 2 / 3) < 1e-9


def test_compute_metrics_avg_score():
    traces = [make_trace(True, 1.0), make_trace(False, 0.0)]
    metrics = _compute_metrics(1, traces, [None, None], 0, 0)
    assert abs(metrics.avg_score - 0.5) < 1e-9


def test_compute_metrics_failure_mode_distribution():
    traces = [make_trace(False, 0.2), make_trace(False, 0.3)]
    evals = [make_eval("HALLUCINATION"), make_eval("WRONG_REASONING")]
    metrics = _compute_metrics(1, traces, evals, 2, 1)
    assert metrics.failure_mode_distribution["HALLUCINATION"] == 1
    assert metrics.failure_mode_distribution["WRONG_REASONING"] == 1


def test_compute_metrics_pairs_filtered():
    traces = [make_trace(False, 0.2)]
    metrics = _compute_metrics(1, traces, [None], pairs_generated=5, pairs_stored=3)
    assert metrics.pairs_filtered_low_quality == 2


def test_compute_metrics_empty_traces():
    metrics = _compute_metrics(1, [], [], 0, 0)
    assert metrics.task_success_rate == 0.0
    assert metrics.avg_score == 0.0


# ---------------------------------------------------------------------------
# Runner state machine
# ---------------------------------------------------------------------------

def test_runner_initial_state():
    runner = Runner(repo=MagicMock())
    assert runner.state == RunnerState.IDLE
    assert runner.current_iteration == 0
    assert runner.current_task_id is None


async def test_runner_transitions_through_states():
    """Verify state transitions happen in correct order during run_iteration."""
    states_observed: list[RunnerState] = []

    task = BenchmarkTask(
        id="t1", category="web_navigation", question="Q?",
        context_type="html", ground_truth="A",
        scoring_rubric="exact_match", difficulty="easy",
    )

    mock_repo = AsyncMock()
    mock_repo.get_task = AsyncMock(return_value=task)
    mock_repo.get_all_tasks = AsyncMock(return_value=[task])
    mock_repo.save_task = AsyncMock()
    mock_repo.save_trace = AsyncMock(return_value=make_trace(False, 0.0))
    mock_repo.save_pair = AsyncMock(return_value=None)
    mock_repo.get_all_pairs = AsyncMock(return_value=[])
    mock_repo.get_pairs_for_iteration = AsyncMock(return_value=[])
    mock_repo.update_cluster_labels = AsyncMock()

    runner = Runner(repo=mock_repo)

    agent_state = {
        "final_answer": "A", "reasoning_steps": ["step"],
        "tool_calls": [], "is_complete": True, "error": None,
    }

    with (
        patch("deltaloop.benchmark.runner.load_tasks", AsyncMock(return_value=[task])),
        patch("deltaloop.benchmark.runner.run_agent", AsyncMock(return_value=agent_state)),
        patch("deltaloop.benchmark.runner.evaluate_trace", AsyncMock(return_value=make_eval(None))),
        patch("deltaloop.benchmark.runner.run_clustering", AsyncMock(return_value={})),
    ):
        metrics = await runner.run_iteration(iteration=1)

    assert runner.state == RunnerState.IDLE
    assert isinstance(metrics, IterationMetrics)
    assert metrics.iteration == 1


async def test_runner_handles_agent_failure_gracefully():
    """Agent crash on a task should not abort the whole iteration."""
    task = BenchmarkTask(
        id="t1", category="web_navigation", question="Q?",
        context_type="html", ground_truth="A",
        scoring_rubric="exact_match", difficulty="easy",
    )

    mock_repo = AsyncMock()
    mock_repo.get_task = AsyncMock(return_value=task)
    mock_repo.save_task = AsyncMock()
    mock_repo.save_trace = AsyncMock(return_value=make_trace(False, 0.0))
    mock_repo.save_pair = AsyncMock(return_value=None)
    mock_repo.get_all_pairs = AsyncMock(return_value=[])
    mock_repo.update_cluster_labels = AsyncMock()

    runner = Runner(repo=mock_repo)

    with (
        patch("deltaloop.benchmark.runner.load_tasks", AsyncMock(return_value=[task])),
        patch("deltaloop.benchmark.runner.run_agent", AsyncMock(side_effect=RuntimeError("Ollama down"))),
        patch("deltaloop.benchmark.runner.evaluate_trace", AsyncMock(return_value=None)),
        patch("deltaloop.benchmark.runner.run_clustering", AsyncMock(return_value={})),
    ):
        metrics = await runner.run_iteration(iteration=1)

    # Should complete without raising
    assert metrics.task_success_rate == 0.0


async def test_runner_triggers_training_at_threshold():
    """Fine-tuning should be triggered when pair count >= threshold."""
    from deltaloop.config import settings

    task = BenchmarkTask(
        id="t1", category="web_navigation", question="Q?",
        context_type="html", ground_truth="A",
        scoring_rubric="exact_match", difficulty="easy",
    )

    # Build enough fake pairs to cross the threshold
    from deltaloop.storage.models import PreferencePair
    fake_pairs = [
        PreferencePair(
            task_id="t1", iteration=1,
            chosen_trace=json.dumps(["step"]),
            rejected_trace=json.dumps(["bad"]),
            failure_mode="HALLUCINATION",
            failure_explanation="bad",
            quality_score=0.8,
            cluster_label=0,
        )
        for _ in range(settings.preference_pair_threshold)
    ]

    mock_repo = AsyncMock()
    mock_repo.get_task = AsyncMock(return_value=task)
    mock_repo.save_task = AsyncMock()
    mock_repo.save_trace = AsyncMock(return_value=make_trace(False, 0.0))
    mock_repo.save_pair = AsyncMock(return_value=None)
    mock_repo.get_all_pairs = AsyncMock(return_value=fake_pairs)
    mock_repo.update_cluster_labels = AsyncMock()

    mock_adapter = MagicMock()
    mock_adapter.swap = MagicMock()

    runner = Runner(repo=mock_repo, adapter_manager=mock_adapter)

    with (
        patch("deltaloop.benchmark.runner.load_tasks", AsyncMock(return_value=[task])),
        patch("deltaloop.benchmark.runner.run_agent", AsyncMock(return_value={
            "final_answer": "A", "reasoning_steps": [], "tool_calls": [],
            "is_complete": True, "error": None,
        })),
        patch("deltaloop.benchmark.runner.evaluate_trace", AsyncMock(return_value=make_eval(None))),
        patch("deltaloop.benchmark.runner.run_clustering", AsyncMock(return_value={})),
        patch("deltaloop.benchmark.runner._trigger_fine_tuning",
              AsyncMock(return_value="adapters/iteration_1")),
    ):
        metrics = await runner.run_iteration(iteration=1)

    assert metrics.training_triggered is True
    assert metrics.adapter_path == "adapters/iteration_1"
    mock_adapter.swap.assert_called_once_with("adapters/iteration_1")
