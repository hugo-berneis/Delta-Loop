import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from deltaloop.storage.repository import Repository

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def session():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    _Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)  # type: ignore[call-overload]
    async with _Session() as s:
        yield s

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def repo(session: AsyncSession) -> Repository:
    return Repository(session)
