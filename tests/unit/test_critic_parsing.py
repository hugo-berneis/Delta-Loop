"""TDD tests for critic JSON parsing — written BEFORE implementation per spec."""
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

# These imports will fail until the implementation exists — that's intentional TDD
from deltaloop.critic.evaluator import (
    TraceEvaluation,
    compute_quality_score,
    parse_eval_response,
)
from deltaloop.storage.models import BenchmarkTask, PreferencePair, ReasoningTrace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trace(score: float = 0.2, is_correct: bool = False) -> ReasoningTrace:
    return ReasoningTrace(
        task_id="mmqa_001",
        iteration=1,
        reasoning_steps=json.dumps(["Step 1: analyze", "Step 2: conclude"]),
        tool_calls=json.dumps([]),
        final_answer="A wrong answer.",
        is_correct=is_correct,
        score=score,
    )


def make_task() -> BenchmarkTask:
    return BenchmarkTask(
        id="mmqa_001",
        category="multimodal_qa",
        question="What trend does the chart show?",
        context_type="image",
        context_path=None,
        ground_truth="Revenue increased each quarter.",
        scoring_rubric="partial_match",
        difficulty="easy",
    )


# ---------------------------------------------------------------------------
# parse_eval_response — the pure JSON parsing function (TDD required)
# ---------------------------------------------------------------------------

def test_parse_valid_json():
    raw = json.dumps({
        "is_correct": False,
        "score": 0.2,
        "failure_mode": "HALLUCINATION",
        "failure_explanation": "Agent stated facts not in the image.",
    })
    result = parse_eval_response(raw)
    assert result is not None
    assert result.is_correct is False
    assert result.score == 0.2
    assert result.failure_mode == "HALLUCINATION"
    assert result.failure_explanation == "Agent stated facts not in the image."


def test_parse_correct_response_with_null_failure_mode():
    raw = json.dumps({
        "is_correct": True,
        "score": 0.95,
        "failure_mode": None,
        "failure_explanation": None,
    })
    result = parse_eval_response(raw)
    assert result is not None
    assert result.is_correct is True
    assert result.failure_mode is None


def test_parse_malformed_json_returns_none():
    raw = "Here is my evaluation: {is_correct: false, this is not json}"
    result = parse_eval_response(raw)
    assert result is None


def test_parse_empty_string_returns_none():
    result = parse_eval_response("")
    assert result is None


def test_parse_invalid_failure_mode_rejected():
    raw = json.dumps({
        "is_correct": False,
        "score": 0.3,
        "failure_mode": "MADE_UP_MODE",
        "failure_explanation": "Some explanation.",
    })
    result = parse_eval_response(raw)
    assert result is None


def test_parse_all_valid_failure_modes_accepted():
    valid_modes = [
        "WRONG_RETRIEVAL",
        "WRONG_REASONING",
        "INCOMPLETE_ANSWER",
        "HALLUCINATION",
        "TOOL_MISUSE",
    ]
    for mode in valid_modes:
        raw = json.dumps({
            "is_correct": False,
            "score": 0.1,
            "failure_mode": mode,
            "failure_explanation": "Explanation.",
        })
        result = parse_eval_response(raw)
        assert result is not None, f"Valid failure mode '{mode}' was rejected"
        assert result.failure_mode == mode


def test_parse_missing_required_field_returns_none():
    # Missing 'score'
    raw = json.dumps({
        "is_correct": False,
        "failure_mode": "HALLUCINATION",
        "failure_explanation": "Explanation.",
    })
    result = parse_eval_response(raw)
    assert result is None


def test_parse_score_out_of_range_returns_none():
    raw = json.dumps({
        "is_correct": False,
        "score": 1.5,  # invalid — must be 0.0 to 1.0
        "failure_mode": "HALLUCINATION",
        "failure_explanation": "Explanation.",
    })
    result = parse_eval_response(raw)
    assert result is None


def test_parse_critic_confidence_computed_correctly():
    # score=0.1 → confidence = abs(0.1 - 0.5) * 2 = 0.8
    raw = json.dumps({
        "is_correct": False,
        "score": 0.1,
        "failure_mode": "WRONG_REASONING",
        "failure_explanation": "Bad reasoning.",
    })
    result = parse_eval_response(raw)
    assert result is not None
    assert abs(result.critic_confidence - 0.8) < 1e-9


def test_parse_critic_confidence_at_midpoint_is_zero():
    # score=0.5 → confidence = abs(0.5 - 0.5) * 2 = 0.0
    raw = json.dumps({
        "is_correct": False,
        "score": 0.5,
        "failure_mode": "INCOMPLETE_ANSWER",
        "failure_explanation": "Stopped too early.",
    })
    result = parse_eval_response(raw)
    assert result is not None
    assert abs(result.critic_confidence - 0.0) < 1e-9


def test_parse_strips_markdown_code_fences():
    inner = json.dumps({
        "is_correct": False,
        "score": 0.2,
        "failure_mode": "HALLUCINATION",
        "failure_explanation": "Fabricated data.",
    })
    raw = f"```json\n{inner}\n```"
    result = parse_eval_response(raw)
    assert result is not None
    assert result.failure_mode == "HALLUCINATION"


# ---------------------------------------------------------------------------
# compute_quality_score
# ---------------------------------------------------------------------------

def test_quality_score_formula():
    eval_ = TraceEvaluation(
        is_correct=False,
        score=0.2,
        failure_mode="HALLUCINATION",
        failure_explanation="Bad.",
        critic_confidence=0.8,
    )
    trace = make_trace(score=0.1)
    # quality = (1 - 0.1) * 0.8 = 0.72
    qs = compute_quality_score(trace, eval_)
    assert abs(qs - 0.72) < 1e-9


def test_quality_score_filter():
    # From spec: quality_score = (1.0 - 0.9) * 0.3 = 0.03 → should be ≤ 0.4
    eval_ = TraceEvaluation(
        is_correct=False,
        score=0.9,
        failure_mode="WRONG_REASONING",
        failure_explanation="Reasoning was off.",
        critic_confidence=0.3,
    )
    trace = make_trace(score=0.9)
    assert compute_quality_score(trace, eval_) <= 0.4


def test_quality_score_high_for_clear_failure():
    # Agent totally failed (score=0.0), critic very confident (confidence=1.0)
    eval_ = TraceEvaluation(
        is_correct=False,
        score=0.0,
        failure_mode="HALLUCINATION",
        failure_explanation="Complete fabrication.",
        critic_confidence=1.0,
    )
    trace = make_trace(score=0.0)
    assert compute_quality_score(trace, eval_) == 1.0


def test_quality_score_zero_when_agent_was_correct():
    eval_ = TraceEvaluation(
        is_correct=True,
        score=1.0,
        failure_mode=None,
        failure_explanation=None,
        critic_confidence=1.0,
    )
    trace = make_trace(score=1.0)
    assert compute_quality_score(trace, eval_) == 0.0


# ---------------------------------------------------------------------------
# evaluate_trace — async, mocked critic calls
# ---------------------------------------------------------------------------

async def test_evaluate_trace_returns_evaluation_on_valid_response():
    from deltaloop.critic.evaluator import evaluate_trace

    valid_response = json.dumps({
        "is_correct": False,
        "score": 0.15,
        "failure_mode": "WRONG_RETRIEVAL",
        "failure_explanation": "Retrieved wrong context.",
    })
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=valid_response)

    result = await evaluate_trace(make_trace(), make_task(), mock_client)
    assert result is not None
    assert result.failure_mode == "WRONG_RETRIEVAL"
    assert result.is_correct is False


async def test_evaluate_trace_retries_on_malformed_json():
    from deltaloop.critic.evaluator import evaluate_trace

    valid_response = json.dumps({
        "is_correct": False,
        "score": 0.2,
        "failure_mode": "HALLUCINATION",
        "failure_explanation": "Fabricated.",
    })
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(side_effect=[
        "NOT JSON at all lol",
        valid_response,
    ])

    result = await evaluate_trace(make_trace(), make_task(), mock_client)
    assert result is not None
    assert mock_client.complete.call_count == 2


async def test_evaluate_trace_returns_none_on_double_failure():
    from deltaloop.critic.evaluator import evaluate_trace

    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value="still not json")

    result = await evaluate_trace(make_trace(), make_task(), mock_client)
    assert result is None
    assert mock_client.complete.call_count == 2


# ---------------------------------------------------------------------------
# synthesize_preference_pair — async, mocked
# ---------------------------------------------------------------------------

async def test_synthesize_returns_none_for_correct_trace():
    from deltaloop.critic.synthesizer import synthesize_preference_pair

    eval_ = TraceEvaluation(
        is_correct=True,
        score=0.95,
        failure_mode=None,
        failure_explanation=None,
        critic_confidence=0.9,
    )
    mock_client = AsyncMock()
    result = await synthesize_preference_pair(make_trace(), eval_, make_task(), mock_client)
    assert result is None
    mock_client.complete.assert_not_called()


async def test_synthesize_returns_none_for_low_quality():
    from deltaloop.critic.synthesizer import synthesize_preference_pair

    # score=0.95 agent, confidence=0.1 critic → quality = 0.05 * 0.1 = 0.005
    eval_ = TraceEvaluation(
        is_correct=False,
        score=0.95,
        failure_mode="WRONG_REASONING",
        failure_explanation="Minor issue.",
        critic_confidence=0.1,
    )
    mock_client = AsyncMock()
    result = await synthesize_preference_pair(make_trace(score=0.95), eval_, make_task(), mock_client)
    assert result is None
    mock_client.complete.assert_not_called()


async def test_synthesize_returns_pair_for_high_quality():
    from deltaloop.critic.synthesizer import synthesize_preference_pair

    eval_ = TraceEvaluation(
        is_correct=False,
        score=0.1,
        failure_mode="HALLUCINATION",
        failure_explanation="Agent made up data.",
        critic_confidence=0.9,
    )
    corrected_steps = ["Step 1: Look at actual data", "Step 2: Report real trend"]
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=json.dumps(corrected_steps))

    result = await synthesize_preference_pair(make_trace(score=0.1), eval_, make_task(), mock_client)
    assert result is not None
    assert isinstance(result, PreferencePair)
    assert json.loads(result.chosen_trace) == corrected_steps


async def test_synthesize_returns_none_on_invalid_correction_json():
    from deltaloop.critic.synthesizer import synthesize_preference_pair

    eval_ = TraceEvaluation(
        is_correct=False,
        score=0.1,
        failure_mode="HALLUCINATION",
        failure_explanation="Bad.",
        critic_confidence=0.9,
    )
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value="not a json list")

    result = await synthesize_preference_pair(make_trace(score=0.1), eval_, make_task(), mock_client)
    assert result is None


async def test_synthesize_returns_none_on_empty_corrected_steps():
    from deltaloop.critic.synthesizer import synthesize_preference_pair

    eval_ = TraceEvaluation(
        is_correct=False,
        score=0.1,
        failure_mode="HALLUCINATION",
        failure_explanation="Bad.",
        critic_confidence=0.9,
    )
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(return_value=json.dumps([]))  # empty list

    result = await synthesize_preference_pair(make_trace(score=0.1), eval_, make_task(), mock_client)
    assert result is None
