from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from deltaloop.api.deps import get_repo
from deltaloop.storage.repository import Repository

router = APIRouter()


class LearningCurveResponse(BaseModel):
    iterations: list[int]
    task_success_rate: list[float]
    avg_score: list[float]
    fine_tuning_events: list[int]


@router.get("/api/learning-curve", response_model=LearningCurveResponse)
async def get_learning_curve(repo: Repository = Depends(get_repo)) -> LearningCurveResponse:
    iteration_numbers = await repo.get_iteration_numbers()

    iterations: list[int] = []
    success_rates: list[float] = []
    avg_scores: list[float] = []
    fine_tuning_events: list[int] = []

    for n in iteration_numbers:
        traces = await repo.get_traces_for_iteration(n)
        if not traces:
            continue

        n_correct = sum(1 for t in traces if t.is_correct)
        avg = sum(t.score for t in traces) / len(traces)

        iterations.append(n)
        success_rates.append(n_correct / len(traces))
        avg_scores.append(avg)

        if Path(f"adapters/iteration_{n}").exists():
            fine_tuning_events.append(n)

    return LearningCurveResponse(
        iterations=iterations,
        task_success_rate=success_rates,
        avg_score=avg_scores,
        fine_tuning_events=fine_tuning_events,
    )
