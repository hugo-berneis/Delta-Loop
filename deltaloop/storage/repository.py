from typing import Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, select

from deltaloop.config import settings
from deltaloop.storage.models import BenchmarkTask, PreferencePair, ReasoningTrace

_engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.db_path}",
    echo=False,
)

_AsyncSession = sessionmaker(  # type: ignore[call-overload]
    _engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    async with _engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


class Repository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # BenchmarkTask
    # ------------------------------------------------------------------

    async def save_task(self, task: BenchmarkTask) -> None:
        self._session.add(task)
        await self._session.commit()

    async def get_task(self, task_id: str) -> Optional[BenchmarkTask]:
        result = await self._session.execute(
            select(BenchmarkTask).where(BenchmarkTask.id == task_id)
        )
        return result.scalars().first()

    async def get_all_tasks(self) -> list[BenchmarkTask]:
        result = await self._session.execute(select(BenchmarkTask))
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # ReasoningTrace
    # ------------------------------------------------------------------

    async def save_trace(self, trace: ReasoningTrace) -> ReasoningTrace:
        self._session.add(trace)
        await self._session.commit()
        await self._session.refresh(trace)
        return trace

    async def get_all_traces(self) -> list[ReasoningTrace]:
        result = await self._session.execute(select(ReasoningTrace))
        return list(result.scalars().all())

    async def get_iteration_numbers(self) -> list[int]:
        """Return sorted list of distinct iteration numbers that have traces."""
        result = await self._session.execute(select(ReasoningTrace.iteration).distinct())
        return sorted(int(r) for r in result.scalars().all())

    async def get_traces_for_iteration(self, iteration: int) -> list[ReasoningTrace]:
        result = await self._session.execute(
            select(ReasoningTrace).where(ReasoningTrace.iteration == iteration)
        )
        return list(result.scalars().all())

    async def get_failed_traces(self, iteration: int) -> list[ReasoningTrace]:
        result = await self._session.execute(
            select(ReasoningTrace).where(
                ReasoningTrace.iteration == iteration,
                ReasoningTrace.is_correct == False,  # noqa: E712
            )
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # PreferencePair
    # ------------------------------------------------------------------

    async def save_pair(self, pair: PreferencePair) -> Optional[PreferencePair]:
        """Returns None if quality_score <= 0.4 (too noisy to train on)."""
        if pair.quality_score <= 0.4:
            logger.info(
                f"Skipping low-quality pair for task {pair.task_id} "
                f"(quality_score={pair.quality_score:.3f})"
            )
            return None
        self._session.add(pair)
        await self._session.commit()
        await self._session.refresh(pair)
        return pair

    async def get_all_pairs(self) -> list[PreferencePair]:
        result = await self._session.execute(select(PreferencePair))
        return list(result.scalars().all())

    async def get_pairs_for_iteration(self, iteration: int) -> list[PreferencePair]:
        result = await self._session.execute(
            select(PreferencePair).where(PreferencePair.iteration == iteration)
        )
        return list(result.scalars().all())

    async def update_cluster_labels(self, updates: list[tuple[int, int]]) -> None:
        """Update cluster_label for each (pair_id, label) tuple."""
        for pair_id, label in updates:
            result = await self._session.execute(
                select(PreferencePair).where(PreferencePair.id == pair_id)
            )
            pair = result.scalars().first()
            if pair is not None:
                pair.cluster_label = label
                self._session.add(pair)
        await self._session.commit()


from contextlib import asynccontextmanager


@asynccontextmanager
async def get_session():
    async with _AsyncSession() as session:
        yield session
