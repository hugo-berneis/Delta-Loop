from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from deltaloop.api.deps import get_repo
from deltaloop.storage.repository import Repository

router = APIRouter()


class PairResponse(BaseModel):
    id: int | None
    task_id: str
    iteration: int
    failure_mode: str
    failure_explanation: str
    quality_score: float
    cluster_label: int
    chosen_trace: str
    rejected_trace: str
    created_at: str


@router.get("/api/iterations/{n}/pairs", response_model=list[PairResponse])
async def get_iteration_pairs(
    n: int,
    cluster_label: int | None = Query(default=None),
    repo: Repository = Depends(get_repo),
) -> list[PairResponse]:
    pairs = await repo.get_pairs_for_iteration(n)

    if cluster_label is not None:
        pairs = [p for p in pairs if p.cluster_label == cluster_label]

    return [
        PairResponse(
            id=p.id,
            task_id=p.task_id,
            iteration=p.iteration,
            failure_mode=p.failure_mode,
            failure_explanation=p.failure_explanation,
            quality_score=p.quality_score,
            cluster_label=p.cluster_label,
            chosen_trace=p.chosen_trace,
            rejected_trace=p.rejected_trace,
            created_at=p.created_at.isoformat(),
        )
        for p in pairs
    ]
