from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from deltaloop.benchmark.runner import Runner
from deltaloop.storage.repository import Repository, _AsyncSession

# ---------------------------------------------------------------------------
# Shared runner singleton (lives for the duration of the process)
# ---------------------------------------------------------------------------

_runner: Runner | None = None


def get_runner() -> Runner:
    global _runner
    if _runner is None:
        _runner = Runner()
    return _runner


# ---------------------------------------------------------------------------
# Per-request DB session → Repository
# ---------------------------------------------------------------------------


async def _get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _AsyncSession() as session:
        yield session


async def get_repo(session: AsyncSession = Depends(_get_session)) -> Repository:
    return Repository(session)
