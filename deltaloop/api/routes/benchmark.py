from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from deltaloop.api.deps import get_repo, get_runner
from deltaloop.benchmark.runner import Runner, RunnerState
from deltaloop.storage.repository import Repository

router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class IterationSummary(BaseModel):
    iteration: int
    task_success_rate: float
    avg_score: float
    pairs_stored: int
    training_triggered: bool
    completed_at: str | None


class TraceResponse(BaseModel):
    id: int | None
    task_id: str
    iteration: int
    reasoning_steps: str
    final_answer: str
    is_correct: bool
    score: float
    created_at: str


class PaginatedTraces(BaseModel):
    page: int
    page_size: int
    total: int
    items: list[TraceResponse]


class RunResponse(BaseModel):
    run_id: str
    status: str


class StatusResponse(BaseModel):
    state: str
    current_iteration: int
    current_task: str | None
    adapter: str | None


# ---------------------------------------------------------------------------
# GET /api/iterations
# ---------------------------------------------------------------------------

@router.get("/api/iterations", response_model=list[IterationSummary])
async def list_iterations(repo: Repository = Depends(get_repo)) -> list[IterationSummary]:
    iteration_numbers = await repo.get_iteration_numbers()
    summaries: list[IterationSummary] = []

    for n in iteration_numbers:
        traces = await repo.get_traces_for_iteration(n)
        if not traces:
            continue

        n_correct = sum(1 for t in traces if t.is_correct)
        avg_score = sum(t.score for t in traces) / len(traces)
        pairs = await repo.get_pairs_for_iteration(n)
        adapter_path = Path(f"adapters/iteration_{n}")
        training_triggered = adapter_path.exists()
        completed_at = max(t.created_at for t in traces).isoformat()

        summaries.append(IterationSummary(
            iteration=n,
            task_success_rate=n_correct / len(traces),
            avg_score=avg_score,
            pairs_stored=len(pairs),
            training_triggered=training_triggered,
            completed_at=completed_at,
        ))

    return summaries


# ---------------------------------------------------------------------------
# GET /api/iterations/{n}/traces
# ---------------------------------------------------------------------------

@router.get("/api/iterations/{n}/traces", response_model=PaginatedTraces)
async def get_iteration_traces(
    n: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    filter: str | None = Query(default=None, pattern="^failed$"),
    repo: Repository = Depends(get_repo),
) -> PaginatedTraces:
    traces = await repo.get_traces_for_iteration(n)

    if filter == "failed":
        traces = [t for t in traces if not t.is_correct]

    total = len(traces)
    start = (page - 1) * page_size
    page_items = traces[start : start + page_size]

    return PaginatedTraces(
        page=page,
        page_size=page_size,
        total=total,
        items=[
            TraceResponse(
                id=t.id,
                task_id=t.task_id,
                iteration=t.iteration,
                reasoning_steps=t.reasoning_steps,
                final_answer=t.final_answer,
                is_correct=t.is_correct,
                score=t.score,
                created_at=t.created_at.isoformat(),
            )
            for t in page_items
        ],
    )


# ---------------------------------------------------------------------------
# POST /api/run
# ---------------------------------------------------------------------------

@router.post("/api/run", response_model=RunResponse)
async def trigger_run(
    background_tasks: BackgroundTasks,
    runner: Runner = Depends(get_runner),
    repo: Repository = Depends(get_repo),
) -> RunResponse:
    if runner.state != RunnerState.IDLE:
        raise HTTPException(409, f"Runner is busy: {runner.state.value}")

    next_iteration = runner.current_iteration + 1
    run_id = f"iter_{next_iteration:03d}"

    async def _run() -> None:
        await runner.run_iteration(next_iteration)

    background_tasks.add_task(_run)
    return RunResponse(run_id=run_id, status="started")


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------

@router.get("/api/status", response_model=StatusResponse)
async def get_status(runner: Runner = Depends(get_runner)) -> StatusResponse:
    adapter = None
    if runner._adapter_manager is not None:
        adapter = runner._adapter_manager.current_adapter

    return StatusResponse(
        state=runner.state.value,
        current_iteration=runner.current_iteration,
        current_task=runner.current_task_id,
        adapter=adapter,
    )
