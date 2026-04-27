"""Unit tests for benchmark/loader.py — task loading and schema validation."""
import json
import textwrap
from pathlib import Path

import pytest

from deltaloop.benchmark.loader import _parse_task, load_tasks
from deltaloop.storage.models import BenchmarkTask


# ------------------------------------------------------------------
# _parse_task (unit — no file I/O)
# ------------------------------------------------------------------

def make_raw(overrides: dict | None = None) -> dict:
    base = {
        "id": "test_001",
        "category": "multimodal_qa",
        "question": "What is shown?",
        "context_type": "image",
        "context_path": "data/benchmark/assets/chart_001.png",
        "ground_truth": "A bar chart.",
        "scoring_rubric": "partial_match",
        "difficulty": "easy",
    }
    if overrides:
        base.update(overrides)
    return base


def test_parse_valid_task():
    task = _parse_task(make_raw(), "test.jsonl")
    assert isinstance(task, BenchmarkTask)
    assert task.id == "test_001"
    assert task.scoring_rubric == "partial_match"


def test_parse_missing_required_field_raises():
    raw = make_raw()
    del raw["ground_truth"]
    with pytest.raises(ValueError, match="missing required fields"):
        _parse_task(raw, "test.jsonl")


def test_parse_invalid_scoring_rubric_raises():
    with pytest.raises(ValueError, match="invalid scoring_rubric"):
        _parse_task(make_raw({"scoring_rubric": "fuzzy_match"}), "test.jsonl")


def test_parse_invalid_difficulty_raises():
    with pytest.raises(ValueError, match="invalid difficulty"):
        _parse_task(make_raw({"difficulty": "very_hard"}), "test.jsonl")


def test_parse_invalid_context_type_raises():
    with pytest.raises(ValueError, match="invalid context_type"):
        _parse_task(make_raw({"context_type": "video"}), "test.jsonl")


def test_parse_optional_context_path_can_be_absent():
    raw = make_raw()
    del raw["context_path"]
    task = _parse_task(raw, "test.jsonl")
    assert task.context_path is None


# ------------------------------------------------------------------
# load_tasks (integration with real JSONL files)
# ------------------------------------------------------------------

async def test_load_all_tasks_returns_100():
    tasks = await load_tasks()
    assert len(tasks) == 100


async def test_load_multimodal_qa_returns_40():
    tasks = await load_tasks(category="multimodal_qa")
    assert len(tasks) == 40
    assert all(t.category == "multimodal_qa" for t in tasks)


async def test_load_document_comprehension_returns_35():
    tasks = await load_tasks(category="document_comprehension")
    assert len(tasks) == 35
    assert all(t.category == "document_comprehension" for t in tasks)


async def test_load_web_navigation_returns_25():
    tasks = await load_tasks(category="web_navigation")
    assert len(tasks) == 25
    assert all(t.category == "web_navigation" for t in tasks)


async def test_load_with_limit_caps_results():
    tasks = await load_tasks(limit=10)
    assert len(tasks) == 10


async def test_load_unknown_category_raises():
    with pytest.raises(ValueError, match="Unknown category"):
        await load_tasks(category="nonexistent_category")


async def test_load_tasks_all_ids_unique():
    tasks = await load_tasks()
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "Task IDs must be unique across all JSONL files"


async def test_load_tasks_malformed_json(tmp_path: Path, monkeypatch):
    """Malformed JSON line should raise ValueError."""
    bad_file = tmp_path / "multimodal_qa.jsonl"
    bad_file.write_text('NOT_JSON\n{"id": "x"}\n')

    import deltaloop.benchmark.loader as loader_module
    monkeypatch.setattr(loader_module, "_BENCHMARK_DIR", tmp_path)

    with pytest.raises(ValueError, match="invalid JSON"):
        await load_tasks(category="multimodal_qa")


async def test_load_tasks_missing_file_raises(tmp_path: Path, monkeypatch):
    import deltaloop.benchmark.loader as loader_module
    monkeypatch.setattr(loader_module, "_BENCHMARK_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        await load_tasks(category="multimodal_qa")
