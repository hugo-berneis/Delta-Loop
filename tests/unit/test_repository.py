"""Unit tests for storage/repository.py — insert and query all three table types."""
import json

import pytest

from deltaloop.storage.models import BenchmarkTask, PreferencePair, ReasoningTrace
from deltaloop.storage.repository import Repository


def make_task(task_id: str = "mmqa_001") -> BenchmarkTask:
    return BenchmarkTask(
        id=task_id,
        category="multimodal_qa",
        question="What trend does the chart show?",
        context_type="image",
        context_path="data/benchmark/assets/chart_001.png",
        ground_truth="Revenue increased each quarter.",
        scoring_rubric="partial_match",
        difficulty="easy",
    )


def make_trace(task_id: str = "mmqa_001", is_correct: bool = False, score: float = 0.2) -> ReasoningTrace:
    return ReasoningTrace(
        task_id=task_id,
        iteration=1,
        reasoning_steps=json.dumps(["Step 1: analyze the chart", "Step 2: identify trend"]),
        tool_calls=json.dumps([{"name": "describe_image", "args": {}}]),
        final_answer="Revenue went up.",
        is_correct=is_correct,
        score=score,
    )


def make_pair(task_id: str = "mmqa_001", quality_score: float = 0.7) -> PreferencePair:
    return PreferencePair(
        task_id=task_id,
        iteration=1,
        chosen_trace=json.dumps(["Correct step 1", "Correct step 2"]),
        rejected_trace=json.dumps(["Wrong step 1"]),
        failure_mode="HALLUCINATION",
        failure_explanation="Agent stated facts not present in the image.",
        quality_score=quality_score,
    )


# ------------------------------------------------------------------
# BenchmarkTask
# ------------------------------------------------------------------

async def test_save_and_get_task(repo: Repository) -> None:
    task = make_task()
    await repo.save_task(task)
    retrieved = await repo.get_task("mmqa_001")
    assert retrieved is not None
    assert retrieved.id == "mmqa_001"
    assert retrieved.category == "multimodal_qa"
    assert retrieved.ground_truth == "Revenue increased each quarter."


async def test_get_task_returns_none_for_missing(repo: Repository) -> None:
    result = await repo.get_task("nonexistent_id")
    assert result is None


async def test_get_all_tasks_returns_all(repo: Repository) -> None:
    await repo.save_task(make_task("mmqa_001"))
    await repo.save_task(make_task("mmqa_002"))
    tasks = await repo.get_all_tasks()
    assert len(tasks) == 2
    ids = {t.id for t in tasks}
    assert ids == {"mmqa_001", "mmqa_002"}


# ------------------------------------------------------------------
# ReasoningTrace
# ------------------------------------------------------------------

async def test_save_and_query_trace(repo: Repository) -> None:
    await repo.save_task(make_task())
    trace = make_trace()
    saved = await repo.save_trace(trace)
    assert saved.id is not None

    traces = await repo.get_traces_for_iteration(1)
    assert len(traces) == 1
    assert traces[0].final_answer == "Revenue went up."
    assert traces[0].task_id == "mmqa_001"


async def test_get_failed_traces_filters_correctly(repo: Repository) -> None:
    await repo.save_task(make_task())
    await repo.save_trace(make_trace(is_correct=False, score=0.2))
    await repo.save_trace(make_trace(is_correct=True, score=1.0))

    failed = await repo.get_failed_traces(1)
    assert len(failed) == 1
    assert failed[0].is_correct is False


async def test_get_traces_for_iteration_isolates_iterations(repo: Repository) -> None:
    await repo.save_task(make_task())
    t1 = make_trace()
    t1.iteration = 1
    t2 = make_trace()
    t2.iteration = 2
    await repo.save_trace(t1)
    await repo.save_trace(t2)

    iter1 = await repo.get_traces_for_iteration(1)
    iter2 = await repo.get_traces_for_iteration(2)
    assert len(iter1) == 1
    assert len(iter2) == 1


# ------------------------------------------------------------------
# PreferencePair
# ------------------------------------------------------------------

async def test_save_high_quality_pair(repo: Repository) -> None:
    await repo.save_task(make_task())
    pair = make_pair(quality_score=0.75)
    saved = await repo.save_pair(pair)
    assert saved is not None
    assert saved.id is not None

    all_pairs = await repo.get_all_pairs()
    assert len(all_pairs) == 1


async def test_save_low_quality_pair_filtered(repo: Repository) -> None:
    await repo.save_task(make_task())
    pair = make_pair(quality_score=0.3)
    result = await repo.save_pair(pair)
    assert result is None

    all_pairs = await repo.get_all_pairs()
    assert len(all_pairs) == 0


async def test_save_pair_at_threshold_boundary(repo: Repository) -> None:
    """quality_score == 0.4 should be filtered (must be strictly > 0.4)."""
    await repo.save_task(make_task())
    result = await repo.save_pair(make_pair(quality_score=0.4))
    assert result is None


async def test_get_pairs_for_iteration(repo: Repository) -> None:
    await repo.save_task(make_task())
    p1 = make_pair(quality_score=0.8)
    p1.iteration = 1
    p2 = make_pair(quality_score=0.8)
    p2.iteration = 2
    await repo.save_pair(p1)
    await repo.save_pair(p2)

    pairs_iter1 = await repo.get_pairs_for_iteration(1)
    assert len(pairs_iter1) == 1
    assert pairs_iter1[0].iteration == 1


async def test_update_cluster_labels(repo: Repository) -> None:
    await repo.save_task(make_task())
    saved = await repo.save_pair(make_pair(quality_score=0.8))
    assert saved is not None
    assert saved.cluster_label == -1  # default

    await repo.update_cluster_labels([(saved.id, 2)])

    pairs = await repo.get_all_pairs()
    assert pairs[0].cluster_label == 2
