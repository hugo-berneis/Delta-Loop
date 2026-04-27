import json
from pathlib import Path

from loguru import logger

from deltaloop.storage.models import BenchmarkTask

_BENCHMARK_DIR = Path(__file__).parent.parent.parent / "data" / "benchmark"

_CATEGORY_FILES = {
    "multimodal_qa": "multimodal_qa.jsonl",
    "document_comprehension": "document_comprehension.jsonl",
    "web_navigation": "web_navigation.jsonl",
}

_VALID_RUBRICS = {"exact_match", "partial_match"}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_CONTEXT_TYPES = {"image", "document", "html"}


def _parse_task(raw: dict, source_file: str) -> BenchmarkTask:
    """Parse and validate a raw dict into a BenchmarkTask. Raises on invalid data."""
    missing = [k for k in ("id", "category", "question", "context_type", "ground_truth",
                           "scoring_rubric", "difficulty") if k not in raw]
    if missing:
        raise ValueError(f"Task in {source_file} missing required fields: {missing}")

    if raw["scoring_rubric"] not in _VALID_RUBRICS:
        raise ValueError(
            f"Task {raw['id']}: invalid scoring_rubric '{raw['scoring_rubric']}'. "
            f"Must be one of {_VALID_RUBRICS}"
        )
    if raw["difficulty"] not in _VALID_DIFFICULTIES:
        raise ValueError(
            f"Task {raw['id']}: invalid difficulty '{raw['difficulty']}'. "
            f"Must be one of {_VALID_DIFFICULTIES}"
        )
    if raw["context_type"] not in _VALID_CONTEXT_TYPES:
        raise ValueError(
            f"Task {raw['id']}: invalid context_type '{raw['context_type']}'. "
            f"Must be one of {_VALID_CONTEXT_TYPES}"
        )

    return BenchmarkTask(
        id=raw["id"],
        category=raw["category"],
        question=raw["question"],
        context_type=raw["context_type"],
        context_path=raw.get("context_path"),
        ground_truth=raw["ground_truth"],
        scoring_rubric=raw["scoring_rubric"],
        difficulty=raw["difficulty"],
    )


async def load_tasks(
    category: str | None = None,
    limit: int | None = None,
) -> list[BenchmarkTask]:
    """Load tasks from JSONL files. Optionally filter by category or cap count (for CI)."""
    if category is not None and category not in _CATEGORY_FILES:
        raise ValueError(
            f"Unknown category '{category}'. Must be one of {list(_CATEGORY_FILES.keys())}"
        )

    categories = [category] if category else list(_CATEGORY_FILES.keys())
    tasks: list[BenchmarkTask] = []

    for cat in categories:
        path = _BENCHMARK_DIR / _CATEGORY_FILES[cat]
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        with path.open() as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path.name} line {line_num}: invalid JSON — {e}") from e

                tasks.append(_parse_task(raw, path.name))

    if limit is not None:
        tasks = tasks[:limit]

    logger.info(f"Loaded {len(tasks)} benchmark tasks (category={category}, limit={limit})")
    return tasks
