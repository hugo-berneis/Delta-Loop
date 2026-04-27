"""TDD tests for training/data_builder.py — written BEFORE implementation per spec."""
import json

import pytest

from deltaloop.storage.models import BenchmarkTask, PreferencePair
from deltaloop.training.data_builder import format_pair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task(task_id: str = "mmqa_001") -> BenchmarkTask:
    return BenchmarkTask(
        id=task_id,
        category="multimodal_qa",
        question="What trend does the chart show?",
        context_type="image",
        context_path=None,
        ground_truth="Revenue increased each quarter.",
        scoring_rubric="partial_match",
        difficulty="easy",
    )


def make_pair(
    chosen: list[str],
    rejected: list[str],
    task_id: str = "mmqa_001",
    quality_score: float = 0.8,
) -> PreferencePair:
    return PreferencePair(
        task_id=task_id,
        iteration=1,
        chosen_trace=json.dumps(chosen),
        rejected_trace=json.dumps(rejected),
        failure_mode="HALLUCINATION",
        failure_explanation="Agent fabricated data.",
        quality_score=quality_score,
        cluster_label=0,
    )


# ---------------------------------------------------------------------------
# format_pair — TDD required per spec
# ---------------------------------------------------------------------------

def test_format_pair_produces_required_keys():
    pair = make_pair(chosen=["Step A", "Step B"], rejected=["Bad step"])
    record = format_pair(pair, make_task())
    assert set(record.keys()) == {"prompt", "chosen", "rejected"}


def test_chosen_includes_all_steps():
    pair = make_pair(chosen=["Step A", "Step B", "Step C"], rejected=["Bad step"])
    record = format_pair(pair, make_task())
    assert "Step 1:" in record["chosen"]
    assert "Step 2:" in record["chosen"]
    assert "Step 3:" in record["chosen"]


def test_rejected_includes_all_steps():
    pair = make_pair(chosen=["Good step"], rejected=["Bad step 1", "Bad step 2"])
    record = format_pair(pair, make_task())
    assert "Step 1:" in record["rejected"]
    assert "Step 2:" in record["rejected"]


def test_empty_chosen_trace_raises():
    pair = make_pair(chosen=[], rejected=["Something"])
    with pytest.raises(ValueError, match="chosen"):
        format_pair(pair, make_task())


def test_empty_rejected_trace_raises():
    pair = make_pair(chosen=["Something"], rejected=[])
    with pytest.raises(ValueError, match="rejected"):
        format_pair(pair, make_task())


def test_prompt_contains_task_question():
    task = make_task()
    pair = make_pair(chosen=["Step A"], rejected=["Bad step"])
    record = format_pair(pair, task)
    assert task.question in record["prompt"]


def test_prompt_contains_context_type():
    task = make_task()
    pair = make_pair(chosen=["Step A"], rejected=["Bad step"])
    record = format_pair(pair, task)
    assert task.context_type in record["prompt"]


def test_chosen_and_rejected_are_different():
    pair = make_pair(chosen=["Correct reasoning"], rejected=["Wrong reasoning"])
    record = format_pair(pair, make_task())
    assert record["chosen"] != record["rejected"]


def test_format_pair_all_values_are_strings():
    pair = make_pair(chosen=["Step A", "Step B"], rejected=["Bad step"])
    record = format_pair(pair, make_task())
    for key, value in record.items():
        assert isinstance(value, str), f"Key '{key}' has non-string value: {type(value)}"


def test_step_numbering_is_one_indexed():
    pair = make_pair(chosen=["First", "Second", "Third"], rejected=["Only"])
    record = format_pair(pair, make_task())
    # Should be "Step 1:", "Step 2:", "Step 3:" — not "Step 0:"
    assert "Step 0:" not in record["chosen"]
    assert "Step 1:" in record["chosen"]


# ---------------------------------------------------------------------------
# build_dpo_dataset — tested with mocked repo and clusterer
# ---------------------------------------------------------------------------

async def test_build_dpo_dataset_returns_hf_dataset():
    from unittest.mock import AsyncMock, MagicMock

    import numpy as np

    from deltaloop.training.data_builder import build_dpo_dataset

    task = make_task()
    pairs = [
        make_pair(["Correct step 1", "Correct step 2"], ["Wrong step"]),
        make_pair(["Better approach"], ["Flawed reasoning"]),
    ]
    # Assign cluster labels
    pairs[0].cluster_label = 0
    pairs[1].cluster_label = 1

    mock_repo = AsyncMock()
    mock_repo.get_all_pairs = AsyncMock(return_value=pairs)
    mock_repo.get_task = AsyncMock(return_value=task)

    mock_clusterer = MagicMock()
    mock_clusterer.sample_balanced = MagicMock(return_value=pairs)

    dataset = await build_dpo_dataset(mock_repo, n=2)

    assert dataset is not None
    assert len(dataset) == 2
    assert set(dataset.column_names) == {"prompt", "chosen", "rejected"}


async def test_build_dpo_dataset_correct_format():
    from unittest.mock import AsyncMock

    from deltaloop.training.data_builder import build_dpo_dataset

    task = make_task()
    pair = make_pair(["Correct step"], ["Wrong step"])
    pair.cluster_label = 0

    mock_repo = AsyncMock()
    mock_repo.get_all_pairs = AsyncMock(return_value=[pair])
    mock_repo.get_task = AsyncMock(return_value=task)

    dataset = await build_dpo_dataset(mock_repo, n=1)

    row = dataset[0]
    assert "Step 1:" in row["chosen"]
    assert task.question in row["prompt"]
